"""Cross-encoder reranker for hybrid search results, on ONNX Runtime.

Hybrid RRF fusion is fast and works well on average, but it ranks based
only on the rank position of each chunk in the two retrievers (dense and
sparse). It never *looks* at the chunk content against the query. A
cross-encoder takes (query, chunk_content) as a pair and emits a relevance
score that does inspect the content, fixing many "the top result is
technically relevant but not the best answer" failures of pure RRF.

The default is `cross-encoder/ms-marco-MiniLM-L-6-v2` (91 MB as an ONNX
graph, tens of milliseconds per query on CPU for 30 candidates). It is the
standard small-and-fast reranker; bigger ones improve quality marginally
at 5-10x the cost. Any cross-encoder whose repo ships `onnx/model.onnx`
works; see `model_files.py` for how a model is resolved.

Design notes:
  - **Lazy model load.** The download and the session creation happen on
    the first `rerank()` call, not at `__init__`. Users with the reranker
    disabled pay nothing; users with it enabled but never querying, the same.
  - **Preserves all result fields.** Only `score` changes (the RRF score is
    kept as `original_score`).
  - **Chunk truncation.** Cross-encoders have a hard token limit (512 for
    MiniLM). We cut the chunk to `max_input_chars` first, then let the
    tokenizer truncate the *document* side of the pair, never the query.
  - **Lazy imports.** `onnxruntime`, `tokenizers` and `numpy` are imported
    inside the loader, so importing `lynx.reranker` stays free.
"""
from __future__ import annotations

import sys
from typing import List, Optional


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# Default cross-encoder model. Trained on MS-MARCO (web Q&A), surprisingly
# strong on code search too because identifier-rich queries look like
# question fragments to the model. Swap via config if you want a bigger /
# domain-specific reranker.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chars (not tokens) to feed the cross-encoder. The MiniLM tokenizer
# rounds about 4 chars to 1 token, so 1600 chars is about 400 tokens,
# comfortably below the model's 512-token window after the query is prepended.
DEFAULT_MAX_INPUT_CHARS = 1600


class _CrossEncoderRuntime:
    """A loaded cross-encoder: pair tokenizer + ONNX session + activation."""

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer
        from .embeddings import make_session, read_max_length, read_pad_token
        from .model_files import resolve_model_files

        files = resolve_model_files(model_name)
        self.tokenizer = Tokenizer.from_file(str(files.tokenizer_path))
        max_length = read_max_length(files)
        # Truncate the document side only: the query must survive whole.
        self.tokenizer.enable_truncation(max_length, strategy="only_second")
        pad_token, pad_id = read_pad_token(files, self.tokenizer)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        self.session = make_session(files.onnx_path)
        self.input_names = tuple(i.name for i in self.session.get_inputs())
        self.output_name = self.session.get_outputs()[0].name

        # sentence-transformers applies a sigmoid to single-logit models unless
        # the repo config says otherwise; the ms-marco models say Identity, and
        # their raw logits (roughly -11..+10) are what Lynx documented as the
        # reranked `score`. Reproduce that rule so scores keep their scale.
        cfg = files.read_json("config.json") or {}
        activation = str(cfg.get("sbert_ce_default_activation_function", ""))
        self.apply_sigmoid = "Sigmoid" in activation

    def predict(self, pairs: List[tuple]) -> List[float]:
        import numpy as np
        from .embeddings import build_feed

        encodings = self.tokenizer.encode_batch(pairs)
        feed = build_feed(self.input_names, encodings)
        (logits,) = self.session.run([self.output_name], feed)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 2:
            # [batch, 1] for a regression head; take the last column for the
            # rare 2-class head (its "relevant" logit).
            logits = logits[:, -1]
        if self.apply_sigmoid:
            logits = 1.0 / (1.0 + np.exp(-logits))
        return [float(x) for x in logits]


class Reranker:
    """Applies a cross-encoder to a list of search results.

    Stateless w.r.t. queries: instantiate once per source, call `rerank()`
    once per search. The underlying model is loaded the first time
    `rerank()` runs, not in `__init__`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ):
        self.model_name = model_name
        # Kept for callers that still pass it; the runtime is CPU-only.
        self.device = device
        self.max_input_chars = int(max_input_chars)
        self._model: Optional[_CrossEncoderRuntime] = None

    def _ensure_loaded(self) -> None:
        """Load the cross-encoder on demand (download included, when the
        model is not cached yet and the hub is reachable)."""
        if self._model is not None:
            return
        _log(f"[reranker] loading model {self.model_name!r} (ONNX Runtime, CPU) on first use")
        self._model = _CrossEncoderRuntime(self.model_name)

    def rerank(self, query: str, results: list, top_k: Optional[int] = None) -> list:
        """Rerank `results` by cross-encoder relevance and return top_k.

        Each result dict keeps every field (file, content, symbol_name,
        etc.). The reranker:
          - replaces `score` with the cross-encoder score (a float in
            roughly [-11, 10] for MS-MARCO models, NOT comparable to
            the RRF scale)
          - sets `original_score` to the previous value, so callers can
            tell whether a result moved up or down

        No-op fast paths (no model load) when:
          - `results` is empty
          - `len(results) <= 1` (nothing to reorder)

        With non-trivial `top_k <= len(results)`, we still rerank because
        the cross-encoder can change the ordering of the top results
        (which is the whole point).
        """
        n = len(results)
        if n == 0 or n == 1:
            return results

        # Loading can fail too (model not in HF cache + offline mode, bad
        # model name, no ONNX export in the repo). Treat it the same as a
        # predict failure: log + return original ranking so search still
        # works.
        try:
            self._ensure_loaded()
        except Exception as e:
            _log(f"[reranker] model load failed ({e}); falling back to original ranking")
            return results[:top_k] if top_k is not None else results

        pairs = [(query, self._prepare_text(r)) for r in results]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            # Don't break search if the model fails. Log and keep the
            # original ranking so users still get usable results.
            _log(f"[reranker] predict failed ({e}); falling back to original ranking")
            return results[:top_k] if top_k is not None else results

        # Pair scores with results, sort desc, slice.
        scored = list(zip(scores, results))
        scored.sort(key=lambda t: float(t[0]), reverse=True)

        out = []
        for new_score, r in scored:
            r_out = dict(r)
            r_out["original_score"] = r.get("score")
            r_out["score"] = float(new_score)
            r_out["reranked"] = True
            out.append(r_out)
        return out[:top_k] if top_k is not None else out

    def _prepare_text(self, result: dict) -> str:
        """Extract and truncate the text fed to the cross-encoder.

        We prefer `content` (the chunk body) but fall back to `text` in
        case a caller passes a different shape. Truncation to
        `max_input_chars` keeps the (query, doc) pair under the model's
        token window even on chunks of arbitrary length.
        """
        text = result.get("content") or result.get("text") or ""
        if len(text) > self.max_input_chars:
            text = text[: self.max_input_chars]
        return text
