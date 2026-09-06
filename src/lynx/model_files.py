"""Locate the files an embedding / reranker model needs at run time.

Lynx runs its transformer models on ONNX Runtime, not PyTorch. A model is
therefore a folder holding an ONNX graph plus the tokenizer that goes with
it, and this module is the single place that knows which files those are.
Three callers share it so they can never disagree:

  - `lynx.embeddings` and `lynx.reranker` load the files;
  - `lynx manager install --model` downloads exactly these files and no
    other weight format (`MODEL_ALLOW_PATTERNS`);
  - `lynx manager doctor` and the offline-mode probe in `config.py` check
    that a cached snapshot is actually usable.

`model_name` can be a HuggingFace repo id (`BAAI/bge-small-en-v1.5`) or a
local directory in the same layout. HuggingFace repos are resolved through
the hub cache (`HF_HOME` / `HF_HUB_CACHE`), so the offline policy, the
mirror settings and the `--from-archive` import all keep working unchanged.

Only the standard library is imported at module level. `huggingface_hub` is
imported inside the one function that downloads, so importing this module
from the CLI parser or the doctor stays cheap.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ONNX_GRAPH_CANDIDATES, snapshot_has_runtime_files

# Files fetched from a model repo. Everything else the repo may hold
# (PyTorch, TensorFlow, Flax, OpenVINO, CoreML weights, quantised or
# graph-optimised ONNX variants) is skipped: it is dead weight for this
# runtime. For bge-small the download drops from 267 MB (both Torch formats)
# to 134 MB (the ONNX graph plus the tokenizer).
MODEL_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",   # external tensors, used by the bigger exports
    "model.onnx",
    "model.onnx_data",
]

_NO_ONNX_HINT = (
    "Lynx runs models on ONNX Runtime and needs an `onnx/model.onnx` in the "
    "model repo. The BGE family (BAAI/bge-small-en-v1.5, bge-base-en-v1.5, "
    "bge-m3), sentence-transformers/all-MiniLM-L6-v2 and the ms-marco "
    "cross-encoders ship one. For another model, export it once with "
    "`optimum-cli export onnx --model <repo> <folder>` and point the config "
    "at that folder."
)


class ModelFilesError(RuntimeError):
    """The model folder cannot be used: files missing, or no ONNX export."""


@dataclass(frozen=True)
class ModelFiles:
    """Resolved paths of one model."""
    root: Path
    onnx_path: Path
    tokenizer_path: Path

    def read_json(self, relative: str) -> Optional[dict]:
        """Parse a small JSON side file, or None when it is absent."""
        p = self.root / relative
        if not p.is_file():
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None


def find_onnx_graph(model_dir: Path) -> Optional[Path]:
    """The ONNX graph inside `model_dir`, or None."""
    for rel in ONNX_GRAPH_CANDIDATES:
        p = model_dir / rel
        if p.is_file():
            return p
    return None


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false")


def resolve_model_files(model_name: str) -> ModelFiles:
    """Return the on-disk files for `model_name`, downloading them if allowed.

    A local directory is used as is. A HuggingFace repo id goes through
    `snapshot_download` with `MODEL_ALLOW_PATTERNS`: with the hub reachable
    it fetches only the files still missing from the cache, with
    `HF_HUB_OFFLINE=1` it returns the cached snapshot without touching the
    network. Either way the result is checked for the ONNX graph and the
    tokenizer, and a `ModelFilesError` names what is missing and how to get
    it, instead of a stack trace from deep inside the runtime.
    """
    local = Path(model_name)
    if local.is_dir():
        return _check(local, model_name, local=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ModelFilesError(f"huggingface_hub is not installed: {e}") from e

    try:
        snapshot = Path(snapshot_download(repo_id=model_name,
                                          allow_patterns=MODEL_ALLOW_PATTERNS))
    except Exception as e:
        if _offline():
            raise ModelFilesError(
                f"model {model_name!r} is not in the local HuggingFace cache "
                f"and HF_HUB_OFFLINE is set. Fetch it once with "
                f"`lynx manager install --model {model_name}` (or import an "
                f"archive with `--from-archive`), then retry."
            ) from e
        raise ModelFilesError(
            f"could not download {model_name!r} from HuggingFace "
            f"({type(e).__name__}: {e}). If this machine cannot reach "
            f"huggingface.co, run `lynx manager install --model {model_name}` "
            f"which falls back to the project's GitHub release, or import an "
            f"archive with `--from-archive`."
        ) from e
    return _check(snapshot, model_name, local=False)


def _check(root: Path, model_name: str, *, local: bool) -> ModelFiles:
    onnx_path = find_onnx_graph(root)
    tokenizer_path = root / "tokenizer.json"
    if onnx_path is not None and tokenizer_path.is_file():
        return ModelFiles(root=root, onnx_path=onnx_path,
                          tokenizer_path=tokenizer_path)

    missing = []
    if onnx_path is None:
        missing.append("onnx/model.onnx")
    if not tokenizer_path.is_file():
        missing.append("tokenizer.json")
    where = "folder" if local else "cached snapshot"
    fix = (
        _NO_ONNX_HINT if local else
        f"Run `lynx manager install --model {model_name}` to fetch the missing "
        f"files (installs older than 1.9 downloaded only the PyTorch weights). "
        + _NO_ONNX_HINT
    )
    raise ModelFilesError(
        f"model {model_name!r}: the {where} at {root} lacks "
        f"{', '.join(missing)}. {fix}"
    )


def snapshot_is_usable(model_dir: Path) -> bool:
    """Stdlib check shared with `config.py`: graph + tokenizer both present."""
    return snapshot_has_runtime_files(model_dir)
