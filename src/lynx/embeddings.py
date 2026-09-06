"""Embeddings on ONNX Runtime, plugged into LlamaIndex.

Lynx used to embed through `llama-index-embeddings-huggingface`, which
means sentence-transformers, which means PyTorch. For a CPU-only search
server that was the wrong trade: torch was 480 MB installed on Windows,
about 4 GB of download on Linux (the default wheel carries the CUDA
libraries), and a couple of seconds of import on every start. The same
model exported to ONNX gives the same vectors from a runtime that is a
15 MB wheel and imports in well under a second.

What this module does, and what it does not:

  - `OnnxEmbedding` is a `BaseEmbedding` for LlamaIndex: same interface the
    rest of Lynx already used, so the index pipeline and Chroma are untouched
    and existing indexes stay valid (same model, same vectors).
  - The model is a folder with `onnx/model.onnx` and `tokenizer.json`
    (`model_files.py` resolves and downloads it). BGE models pool the
    `[CLS]` token and L2-normalise, which is read from `1_Pooling/config.json`
    like sentence-transformers does, with mean pooling as the other option.
  - BGE English models prepend an instruction to *queries* only. LlamaIndex
    did that for us; we do it here with the same string, which is why a
    query vector from this class matches one from the old build.
  - One loaded runtime per model, shared by every source (`_RUNTIMES`).
    `onnxruntime.InferenceSession.run` is thread-safe, so the watcher's
    re-index and a search can overlap on it.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from .model_files import ModelFiles, resolve_model_files

DEFAULT_MAX_LENGTH = 512
DEFAULT_EMBED_BATCH_SIZE = 8

# Query instructions LlamaIndex's HuggingFaceEmbedding applied for these exact
# model ids (`llama_index.embeddings.huggingface.utils`). Reproduced verbatim:
# the vectors stored in every existing index were computed against them.
QUERY_INSTRUCTION_BGE_EN = "Represent this question for searching relevant passages: "
QUERY_INSTRUCTION_BGE_ZH = "为这个句子生成表示以用于检索相关文章："
_BGE_MODELS = frozenset({
    "BAAI/bge-small-en", "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en", "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en", "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-zh", "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-base-zh", "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-large-zh", "BAAI/bge-large-zh-v1.5",
})


def default_query_instruction(model_name: str) -> str:
    """The query prefix a model expects, or "" when it has none."""
    if model_name in _BGE_MODELS:
        return QUERY_INSTRUCTION_BGE_ZH if "zh" in model_name else QUERY_INSTRUCTION_BGE_EN
    return ""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def pool_hidden_states(hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Collapse `[batch, tokens, dim]` to `[batch, dim]`: first token or
    attention-masked mean."""
    if mode == "cls":
        return hidden[:, 0, :]
    m = mask[..., None].astype(hidden.dtype)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def read_max_length(files: ModelFiles, default: int = DEFAULT_MAX_LENGTH) -> int:
    """sentence-transformers' `max_seq_length`, else the tokenizer's
    `model_max_length` when it is a real number, else `default`."""
    sbert = files.read_json("sentence_bert_config.json") or {}
    n = sbert.get("max_seq_length")
    if isinstance(n, int) and 0 < n <= 8192:
        return n
    tok_cfg = files.read_json("tokenizer_config.json") or {}
    n = tok_cfg.get("model_max_length")
    if isinstance(n, int) and 0 < n <= 8192:
        return n
    return default


def read_pad_token(files: ModelFiles, tokenizer) -> tuple:
    """(pad_token, pad_id) from the tokenizer configs, defaulting to [PAD]."""
    pad = None
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        cfg = files.read_json(name) or {}
        value = cfg.get("pad_token")
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str) and value:
            pad = value
            break
    pad = pad or "[PAD]"
    pad_id = tokenizer.token_to_id(pad)
    return pad, (pad_id if pad_id is not None else 0)


def make_session(onnx_path) -> Any:
    """A CPU `InferenceSession` with the noisy per-node warnings silenced."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # errors only; ORT's warnings are for exporters
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def build_feed(session_input_names: Sequence[str], encodings) -> Dict[str, np.ndarray]:
    """Turn `tokenizers` encodings into the int64 tensors the graph declares.
    Only inputs the graph actually has are passed (some exports drop
    `token_type_ids`)."""
    feed = {
        "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.asarray([e.type_ids for e in encodings], dtype=np.int64),
    }
    return {k: v for k, v in feed.items() if k in session_input_names}


@dataclass
class _Runtime:
    """One loaded model: session, tokenizer and how to read its output."""
    files: ModelFiles
    session: Any
    tokenizer: Any
    input_names: tuple
    output_name: str
    output_is_pooled: bool
    pooling: str
    max_length: int
    dimension: Optional[int]

    def encode(self, texts: List[str], normalize: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        encodings = self.tokenizer.encode_batch(texts)
        feed = build_feed(self.input_names, encodings)
        (out,) = self.session.run([self.output_name], feed)
        if self.output_is_pooled:
            vectors = out
        else:
            mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
            vectors = pool_hidden_states(out, mask, self.pooling)
        if normalize:
            vectors = l2_normalize(vectors)
        return np.ascontiguousarray(vectors, dtype=np.float32)


_RUNTIMES: Dict[str, _Runtime] = {}
_RUNTIMES_LOCK = threading.Lock()


def load_runtime(model_name: str) -> _Runtime:
    """Load (once per process) the ONNX session and tokenizer for a model.

    Resolution goes through `model_files.resolve_model_files`, which downloads
    into the HF cache when the graph is missing and the hub is reachable, or
    raises a `ModelFilesError` that says what to run. Two sources configured
    with the same model share the loaded runtime.
    """
    files = resolve_model_files(model_name)
    key = str(files.onnx_path)
    with _RUNTIMES_LOCK:
        rt = _RUNTIMES.get(key)
        if rt is not None:
            return rt

        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(files.tokenizer_path))
        max_length = read_max_length(files)
        tokenizer.enable_truncation(max_length)
        pad_token, pad_id = read_pad_token(files, tokenizer)
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        session = make_session(files.onnx_path)
        input_names = tuple(i.name for i in session.get_inputs())
        outputs = session.get_outputs()
        chosen = next((o for o in outputs if o.name == "last_hidden_state"), outputs[0])
        shape = list(chosen.shape or [])
        output_is_pooled = len(shape) == 2
        dimension = shape[-1] if shape and isinstance(shape[-1], int) else None

        pooling_cfg = files.read_json("1_Pooling/config.json") or {}
        if pooling_cfg.get("pooling_mode_cls_token"):
            pooling = "cls"
        elif pooling_cfg.get("pooling_mode_mean_tokens"):
            pooling = "mean"
        else:
            # No sentence-transformers pooling file. BGE exports pool [CLS];
            # most other sentence encoders (MiniLM, e5, nomic) take the mean.
            pooling = "cls" if "bge" in model_name.lower() else "mean"

        rt = _Runtime(
            files=files, session=session, tokenizer=tokenizer,
            input_names=input_names, output_name=chosen.name,
            output_is_pooled=output_is_pooled, pooling=pooling,
            max_length=max_length, dimension=dimension,
        )
        _RUNTIMES[key] = rt
        _log(f"[embeddings] {model_name}: ONNX Runtime, {pooling} pooling, "
             f"max {max_length} tokens ({files.onnx_path.name})")
        return rt


class OnnxEmbedding(BaseEmbedding):
    """LlamaIndex embedding model backed by ONNX Runtime on CPU.

    `model_name` is a HuggingFace repo id or a local folder holding
    `onnx/model.onnx` and `tokenizer.json`. `query_instruction=None` means
    the model's default prefix (the BGE instruction for BGE English models,
    nothing for the rest); pass "" to disable it. Texts get
    `text_instruction`, empty by default.
    """

    normalize: bool = Field(default=True, description="L2-normalise the vectors.")
    query_instruction: Optional[str] = Field(
        default=None, description="Prefix for queries; None = model default.")
    text_instruction: Optional[str] = Field(
        default=None, description="Prefix for indexed texts; None = none.")
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, description="Token limit per text.")

    _runtime: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        *,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        normalize: bool = True,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        callback_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        runtime = load_runtime(model_name)
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
            max_length=runtime.max_length,
            callback_manager=callback_manager,
            **kwargs,
        )
        self._runtime = runtime

    @classmethod
    def class_name(cls) -> str:
        return "OnnxEmbedding"

    # -- formatting -------------------------------------------------------

    def _query_prefix(self) -> str:
        if self.query_instruction is None:
            return default_query_instruction(self.model_name)
        return self.query_instruction

    def _format_queries(self, queries: Sequence[str]) -> List[str]:
        prefix = self._query_prefix()
        return [prefix + q for q in queries]

    def _format_texts(self, texts: Sequence[str]) -> List[str]:
        prefix = self.text_instruction or ""
        return [prefix + t for t in texts]

    # -- batch entry points (used directly by rag_manager) ---------------

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        """Vectors for N queries in as few graph runs as the batch size allows."""
        return self._encode_batched(self._format_queries(list(queries)))

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode_batched(self._format_texts(list(texts)))

    def _encode_batched(self, items: List[str]) -> List[List[float]]:
        """Encode in batches of `embed_batch_size`, grouping texts of similar
        length first so a batch pads to its longest member and not to the
        longest chunk of the whole call. Measured on 200 real C# chunks:
        16.9 s unsorted at batch 32, 14.1 s sorted at batch 8, which is why
        the default batch is small. The original order is restored."""
        if not items:
            return []
        step = max(1, int(self.embed_batch_size))
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        out: List[Optional[List[float]]] = [None] * len(items)
        for start in range(0, len(order), step):
            idx = order[start:start + step]
            vecs = self._runtime.encode([items[i] for i in idx], self.normalize).tolist()
            for i, v in zip(idx, vecs):
                out[i] = v
        return out  # type: ignore[return-value]

    # -- BaseEmbedding contract ------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embed_queries([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)
