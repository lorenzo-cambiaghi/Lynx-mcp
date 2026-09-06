"""The ONNX Runtime embedder and the model-file resolution around it.

Everything here runs without a network and without a model, except the
last test, which is skipped unless the default embedding model is already
in the local HuggingFace cache (CI never downloads it).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lynx import config as lynx_config
from lynx import embeddings
from lynx.model_files import ModelFiles, ModelFilesError, resolve_model_files


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------

def test_bge_english_models_get_the_llamaindex_query_prefix():
    assert embeddings.default_query_instruction("BAAI/bge-small-en-v1.5") == (
        "Represent this question for searching relevant passages: ")
    assert embeddings.default_query_instruction("BAAI/bge-base-zh-v1.5").startswith("为")
    assert embeddings.default_query_instruction("sentence-transformers/all-MiniLM-L6-v2") == ""
    assert embeddings.default_query_instruction("/some/local/folder") == ""


def test_cls_pooling_takes_the_first_token_and_mean_pooling_honours_the_mask():
    hidden = np.array([[[1.0, 0.0], [3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)  # third token is padding
    assert embeddings.pool_hidden_states(hidden, mask, "cls").tolist() == [[1.0, 0.0]]
    assert embeddings.pool_hidden_states(hidden, mask, "mean").tolist() == [[2.0, 2.0]]


def test_l2_normalize_yields_unit_vectors_and_survives_zero():
    v = embeddings.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert v[0].tolist() == pytest.approx([0.6, 0.8])
    assert v[1].tolist() == [0.0, 0.0]


def test_build_feed_only_passes_inputs_the_graph_declares():
    class Enc:
        ids = [101, 7592, 102]
        attention_mask = [1, 1, 1]
        type_ids = [0, 0, 0]

    feed = embeddings.build_feed(("input_ids", "attention_mask"), [Enc()])
    assert set(feed) == {"input_ids", "attention_mask"}
    assert feed["input_ids"].dtype == np.int64
    assert feed["input_ids"].shape == (1, 3)


def _fake_model_dir(tmp_path: Path, *, with_graph: bool, sbert_max=None) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")
    if sbert_max is not None:
        (root / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": sbert_max}), encoding="utf-8")
    if with_graph:
        (root / "onnx").mkdir()
        (root / "onnx" / "model.onnx").write_bytes(b"not really a graph")
    return root


def test_read_max_length_prefers_sentence_transformers_then_tokenizer_then_default(tmp_path):
    root = _fake_model_dir(tmp_path, with_graph=True, sbert_max=256)
    files = ModelFiles(root=root, onnx_path=root / "onnx/model.onnx",
                       tokenizer_path=root / "tokenizer.json")
    assert embeddings.read_max_length(files) == 256

    (root / "sentence_bert_config.json").unlink()
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 1000000000000}), encoding="utf-8")  # HF's "no limit"
    assert embeddings.read_max_length(files) == embeddings.DEFAULT_MAX_LENGTH


# ---------------------------------------------------------------------------
# model resolution and the offline probe
# ---------------------------------------------------------------------------

def test_local_folder_without_an_onnx_graph_is_refused_with_the_fix_in_the_message(tmp_path):
    root = _fake_model_dir(tmp_path, with_graph=False)
    with pytest.raises(ModelFilesError) as exc:
        resolve_model_files(str(root))
    msg = str(exc.value)
    assert "onnx/model.onnx" in msg
    assert "optimum-cli" in msg


def test_local_folder_with_graph_and_tokenizer_resolves(tmp_path):
    root = _fake_model_dir(tmp_path, with_graph=True)
    files = resolve_model_files(str(root))
    assert files.onnx_path == root / "onnx" / "model.onnx"
    assert files.tokenizer_path == root / "tokenizer.json"


def test_offline_probe_needs_the_onnx_graph_not_just_any_snapshot(tmp_path, monkeypatch):
    """An install older than 1.9 left PyTorch-only snapshots in the cache.
    They must NOT count as cached, or offline mode would kick in and the
    graph could never be fetched."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"torch weights")
    assert lynx_config._hf_model_cached("BAAI/bge-small-en-v1.5") is False

    (snap / "onnx").mkdir()
    (snap / "onnx" / "model.onnx").write_bytes(b"graph")
    assert lynx_config._hf_model_cached("BAAI/bge-small-en-v1.5") is True


def test_offline_probe_accepts_a_local_model_folder(tmp_path):
    root = _fake_model_dir(tmp_path, with_graph=True)
    assert lynx_config._hf_model_cached(str(root)) is True
    assert lynx_config._hf_model_cached(str(tmp_path / "missing")) is False


def test_reranker_falls_back_to_the_original_order_when_the_model_cannot_load(tmp_path):
    from lynx.reranker import Reranker

    root = _fake_model_dir(tmp_path, with_graph=False)
    rr = Reranker(model_name=str(root))
    results = [{"content": "a", "score": 0.5}, {"content": "b", "score": 0.4}]
    out = rr.rerank("query", results, top_k=1)
    assert out == results[:1]  # untouched, truncated to top_k, no exception


# ---------------------------------------------------------------------------
# the real model, only when it is already on this machine
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_needs_model = pytest.mark.skipif(
    not lynx_config._hf_model_cached(_DEFAULT_MODEL),
    reason="default embedding model not in the local HF cache",
)


@_needs_model
def test_real_model_vectors_are_unit_length_384d_and_queries_differ_from_texts(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = embeddings.OnnxEmbedding(model_name=_DEFAULT_MODEL)

    text = "def clamp(value, lo, hi):\n    return max(lo, min(hi, value))"
    t = model.get_text_embedding(text)
    q = model.get_query_embedding(text)
    assert len(t) == 384 and len(q) == 384
    assert math.isclose(sum(x * x for x in t), 1.0, rel_tol=1e-4)

    # Same string, but the query path prepends the BGE instruction: the two
    # vectors must be close cousins, not identical.
    cos = sum(a * b for a, b in zip(t, q))
    assert 0.7 < cos < 0.999

    # Batch and single paths agree, for texts and for queries.
    batch_t = model.embed_texts([text, "unrelated words"])
    batch_q = model.embed_queries([text])
    assert batch_t[0] == pytest.approx(t, abs=1e-5)
    assert batch_q[0] == pytest.approx(q, abs=1e-5)

    # Edge cases the indexer will hit: empty text, very long text.
    assert len(model.get_text_embedding("")) == 384
    assert len(model.get_text_embedding("x" * 20000)) == 384
