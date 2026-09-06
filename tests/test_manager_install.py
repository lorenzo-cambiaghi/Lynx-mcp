"""Unit tests for `lynx manager install`.

We mock `subprocess.run` (pip) and `huggingface_hub.snapshot_download`
(HF) so the tests run in milliseconds and never touch the network or
modify the system.

Scenarios:
  1. list_extras prints something + exits 0
  2. install_extra with unknown name → exit 2
  3. install_extra calls `python -m pip install lynx[extra]`
  4. install_extra: pip exits non-zero → returns 2
  5. install_extra: pip OK but import still fails → returns 1
  6. download_model: clears OFFLINE flags, calls snapshot_download, restores
  7. download_model: snapshot fails → exit 2 + env restored
  8. download_models_for_config without config → falls back to defaults
  9. download_models_for_config + with_reranker → 2 downloads
 10. run_install dispatches correctly based on args
 11. _normalize_archive_source classifies url vs local path
 12. export_model_archive → import_model_archive round-trips via the HF cache
 13. import_model_archive from an http(s) URL (mocked urlopen)
 14. run_install dispatches --from-archive / --export-archive
 15. export_model_archive fails cleanly when the model isn't cached
 16. import rejects an HTML body (auth page / Drive interstitial) clearly
 17. export skips blobs/ (no double-zip) and still round-trips
 18. download_model falls back to the GitHub archive when HF is unreachable
 19. LYNX_MODEL_ARCHIVE_BASE_URL overrides the fallback host
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock


def _make_args(**kw):
    """Build an argparse.Namespace with all install args defaulted to None/False."""
    return argparse.Namespace(
        list=kw.get("list", False),
        model=kw.get("model", None),
        extra=kw.get("extra", None),
        with_reranker=kw.get("with_reranker", False),
        config=kw.get("config", None),
        from_archive=kw.get("from_archive", None),
        export_archive=kw.get("export_archive", None),
        model_name=kw.get("model_name", None),
    )


def _seed_fake_model_cache(cache_dir: Path, model="BAAI/bge-small-en-v1.5") -> Path:
    """Create a minimal HF hub cache entry for `model` and return its dir.

    "Minimal" means what the ONNX runtime opens (the graph and the
    tokenizer), because that is what the cache probe checks: a snapshot with
    only `config.json` counts as not cached, on purpose."""
    safe = model.replace("/", "--")
    snap = cache_dir / f"models--{safe}" / "snapshots" / "abc123"
    (snap / "onnx").mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snap / "onnx" / "model.onnx").write_bytes(b"graph")
    return cache_dir / f"models--{safe}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-install-"))
    print(f"[test] tempdir: {tmp}")
    try:
        from lynx.manager import install

        # ============================================================
        # 1. list_extras prints + exits 0
        # ============================================================
        rc = install.list_extras()
        if rc != 0:
            print(f"[test] FAIL [1/10]: list_extras returned {rc}")
            return 1
        print(f"[test] OK [1/10] list_extras: exit 0")

        # ============================================================
        # 2. Unknown extra → exit 2
        # ============================================================
        rc = install.install_extra("does-not-exist")
        if rc != 2:
            print(f"[test] FAIL [2/10]: unknown extra should exit 2, got {rc}")
            return 2
        print(f"[test] OK [2/10] unknown extra: exit 2")

        # ============================================================
        # 3. install_extra invokes `python -m pip install lynx[extra]`
        # ============================================================
        # Three probes: pre-check (not installed) → pre-check again (still not)
        # → post-pip re-check (now installed). Anything else takes the
        # "already installed" short-circuit and never calls pip — which fails
        # in dev envs where pymupdf happens to be present.
        with mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(install, "_is_installed",
                               side_effect=[False, False, True]) as mock_check:
            mock_run.return_value = mock.MagicMock(returncode=0)
            rc = install.install_extra("pdf-fast")
        if rc != 0:
            print(f"[test] FAIL [3/10]: successful install returned {rc}")
            return 3
        # Verify the exact command shape
        called_cmd = mock_run.call_args[0][0]
        if called_cmd[0] != sys.executable:
            print(f"[test] FAIL [3/10]: not using current Python: {called_cmd}")
            return 3
        if called_cmd[1:4] != ["-m", "pip", "install"]:
            print(f"[test] FAIL [3/10]: not invoking pip module: {called_cmd}")
            return 3
        # install_extra installs the requirement directly (`pymupdf>=…`), not
        # `lynx[pdf-fast]` — see the comment on KNOWN_EXTRAS["pdf-fast"].
        if not called_cmd[-1].startswith("pymupdf"):
            print(f"[test] FAIL [3/10]: extra spec wrong: {called_cmd}")
            return 3
        print(f"[test] OK [3/10] install_extra: correct pip command")

        # ============================================================
        # 4. pip failure → exit 2
        # ============================================================
        with mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(install, "_is_installed", return_value=False):
            mock_run.return_value = mock.MagicMock(returncode=1)
            rc = install.install_extra("pdf-fast")
        if rc != 2:
            print(f"[test] FAIL [4/10]: pip failure should exit 2, got {rc}")
            return 4
        print(f"[test] OK [4/10] pip failure: exit 2")

        # ============================================================
        # 5. pip OK but import still broken → exit 1
        # ============================================================
        with mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(install, "_is_installed", return_value=False):
            mock_run.return_value = mock.MagicMock(returncode=0)
            rc = install.install_extra("pdf-fast")
        if rc != 1:
            print(f"[test] FAIL [5/10]: pip-OK-but-broken should exit 1, got {rc}")
            return 5
        print(f"[test] OK [5/10] pip OK but import broken: exit 1")

        # ============================================================
        # 6. download_model: clears OFFLINE, calls snapshot, restores
        # ============================================================
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        captured: dict = {}

        def fake_snapshot(repo_id, **kwargs):
            captured["repo_id"] = repo_id
            captured["allow_patterns"] = kwargs.get("allow_patterns")
            captured["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
            captured["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
            return "/fake/path"

        with mock.patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
            rc = install.download_model("test/model")
        if rc != 0:
            print(f"[test] FAIL [6/10]: download returned {rc}")
            return 6
        if captured.get("repo_id") != "test/model":
            print(f"[test] FAIL [6/10]: wrong repo_id: {captured}")
            return 6
        # Only the files the ONNX runtime opens are downloaded: the graph and
        # the tokenizer, never the PyTorch / TF / Flax weights.
        ap = captured.get("allow_patterns") or []
        if "onnx/model.onnx" not in ap or "tokenizer.json" not in ap:
            print(f"[test] FAIL [6/10]: allow_patterns missing runtime files: {ap}")
            return 6
        if any(p.endswith((".safetensors", ".bin", ".h5")) or p.startswith("*") for p in ap):
            print(f"[test] FAIL [6/10]: allow_patterns is not a plain runtime whitelist: {ap}")
            return 6
        # During the call, both flags must be unset
        if captured.get("HF_HUB_OFFLINE") is not None:
            print(f"[test] FAIL [6/10]: HF_HUB_OFFLINE leaked into call")
            return 6
        if captured.get("TRANSFORMERS_OFFLINE") is not None:
            print(f"[test] FAIL [6/10]: TRANSFORMERS_OFFLINE leaked into call")
            return 6
        # After the call, flags restored
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            print(f"[test] FAIL [6/10]: HF_HUB_OFFLINE not restored")
            return 6
        if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            print(f"[test] FAIL [6/10]: TRANSFORMERS_OFFLINE not restored")
            return 6
        print(f"[test] OK [6/10] download_model: temp clear + restore offline flags")

        # ============================================================
        # 7. snapshot raises → GitHub fallback attempted; if it also
        #    fails → exit 2 + env still restored
        # ============================================================
        os.environ["HF_HUB_OFFLINE"] = "1"
        with mock.patch("huggingface_hub.snapshot_download",
                        side_effect=RuntimeError("simulated network failure")), \
             mock.patch.object(install, "import_model_archive",
                               return_value=2) as m_fb7:
            rc = install.download_model("bad/model")
        if rc != 2:
            print(f"[test] FAIL [7/10]: both-fail should exit 2, got {rc}")
            return 7
        if not m_fb7.called:
            print(f"[test] FAIL [7/10]: GitHub fallback was not attempted")
            return 7
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            print(f"[test] FAIL [7/10]: flag not restored after failure")
            return 7
        print(f"[test] OK [7/10] snapshot failure: GitHub fallback + env restored")

        # Cleanup env
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        # ============================================================
        # 8. download_models_for_config without config → defaults
        # ============================================================
        with mock.patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake"
            rc = install.download_models_for_config(None, with_reranker=False)
        if rc != 0:
            print(f"[test] FAIL [8/10]: defaults path returned {rc}")
            return 8
        called_with = mock_dl.call_args[1]["repo_id"]
        if called_with != "BAAI/bge-small-en-v1.5":
            print(f"[test] FAIL [8/10]: default model name wrong: {called_with}")
            return 8
        print(f"[test] OK [8/10] no config: falls back to default model")

        # ============================================================
        # 9. with_reranker → two downloads
        # ============================================================
        with mock.patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake"
            rc = install.download_models_for_config(None, with_reranker=True)
        if rc != 0:
            print(f"[test] FAIL [9/10]: with-reranker returned {rc}")
            return 9
        if mock_dl.call_count != 2:
            print(f"[test] FAIL [9/10]: expected 2 downloads, got {mock_dl.call_count}")
            return 9
        # Reranker model name should appear in the second call
        second_call = mock_dl.call_args_list[1][1]["repo_id"]
        if "cross-encoder" not in second_call:
            print(f"[test] FAIL [9/10]: reranker model wrong: {second_call}")
            return 9
        print(f"[test] OK [9/10] with-reranker: 2 downloads invoked")

        # ============================================================
        # 10. run_install dispatch
        # ============================================================
        # --list path
        rc = install.run_install(_make_args(list=True))
        if rc != 0:
            print(f"[test] FAIL [10/10]: --list dispatch wrong, got {rc}")
            return 10
        # No args → help-ish, returns 2
        rc = install.run_install(_make_args())
        if rc != 2:
            print(f"[test] FAIL [10/10]: empty args should return 2, got {rc}")
            return 10
        # --model with default value
        with mock.patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/fake"
            rc = install.run_install(_make_args(model="__default__"))
        if rc != 0:
            print(f"[test] FAIL [10/10]: --model dispatch wrong, got {rc}")
            return 10
        # extra dispatch
        with mock.patch.object(install, "install_extra", return_value=0) as mock_ie:
            rc = install.run_install(_make_args(extra="pdf-fast"))
        if rc != 0 or mock_ie.call_args[0][0] != "pdf-fast":
            print(f"[test] FAIL [10/10]: extra dispatch wrong: {mock_ie.call_args}")
            return 10
        print(f"[test] OK [10/10] run_install dispatch: all 4 paths")

        # ============================================================
        # 11. _normalize_archive_source: url vs path
        # ============================================================
        cases = [
            ("https://drive.google.com/uc?export=download&id=X", "url"),
            ("http://example.com/m.zip", "url"),
            ("/local/path/m.zip", "path"),
            ("relative/m.tar.gz", "path"),
        ]
        for src, want in cases:
            kind, _ = install._normalize_archive_source(src)
            if kind != want:
                print(f"[test] FAIL [11/15]: {src!r} → {kind}, want {want}")
                return 11
        print(f"[test] OK [11/15] _normalize_archive_source classifies url/path")

        # ============================================================
        # 12. export → import round-trip via temp HF cache
        # ============================================================
        saved_cache_env = os.environ.pop("HF_HUB_CACHE", None)
        try:
            cache = tmp / "cache"
            cache.mkdir()
            os.environ["HF_HUB_CACHE"] = str(cache)
            _seed_fake_model_cache(cache)

            from lynx.config import _hf_model_cached
            zip_path = tmp / "bge.zip"
            rc = install.export_model_archive(None, str(zip_path), None)
            if rc != 0 or not zip_path.is_file():
                print(f"[test] FAIL [12/15]: export rc={rc}, exists={zip_path.is_file()}")
                return 12
            # Wipe the cache so import has to recreate it.
            shutil.rmtree(cache)
            cache.mkdir()
            if _hf_model_cached("BAAI/bge-small-en-v1.5"):
                print(f"[test] FAIL [12/15]: cache not actually wiped")
                return 12
            rc = install.import_model_archive(str(zip_path), None, None)
            if rc != 0:
                print(f"[test] FAIL [12/15]: import rc={rc}")
                return 12
            if not _hf_model_cached("BAAI/bge-small-en-v1.5"):
                print(f"[test] FAIL [12/15]: model not cached after import")
                return 12
            print(f"[test] OK [12/15] export→import round-trip restores cache")

            # ========================================================
            # 13. import from URL (mocked urlopen streams the zip)
            # ========================================================
            import io
            shutil.rmtree(cache)
            cache.mkdir()
            zip_bytes = zip_path.read_bytes()
            with mock.patch("urllib.request.urlopen",
                            return_value=io.BytesIO(zip_bytes)):
                rc = install.import_model_archive(
                    "https://example.com/bge.zip", None, None)
            if rc != 0 or not _hf_model_cached("BAAI/bge-small-en-v1.5"):
                print(f"[test] FAIL [13/15]: URL import rc={rc}")
                return 13
            print(f"[test] OK [13/15] import from URL via mocked urlopen")

            # ========================================================
            # 14. run_install dispatch: --from-archive / --export-archive
            # ========================================================
            with mock.patch.object(install, "import_model_archive",
                                   return_value=0) as m_imp:
                rc = install.run_install(_make_args(from_archive="x.zip"))
            if rc != 0 or m_imp.call_args[0][0] != "x.zip":
                print(f"[test] FAIL [14/15]: from-archive dispatch {m_imp.call_args}")
                return 14
            with mock.patch.object(install, "export_model_archive",
                                   return_value=0) as m_exp:
                rc = install.run_install(_make_args(export_archive="out.zip"))
            if rc != 0 or m_exp.call_args[0][1] != "out.zip":
                print(f"[test] FAIL [14/15]: export-archive dispatch {m_exp.call_args}")
                return 14
            print(f"[test] OK [14/15] run_install dispatch: archive import/export")
        finally:
            os.environ.pop("HF_HUB_CACHE", None)
            if saved_cache_env is not None:
                os.environ["HF_HUB_CACHE"] = saved_cache_env

        # ============================================================
        # 15. export error when model not in cache
        # ============================================================
        saved_cache_env = os.environ.pop("HF_HUB_CACHE", None)
        try:
            empty = tmp / "empty-cache"
            empty.mkdir()
            os.environ["HF_HUB_CACHE"] = str(empty)
            rc = install.export_model_archive(None, str(tmp / "nope.zip"), None)
            if rc != 2:
                print(f"[test] FAIL [15/17]: export of missing model should be 2, got {rc}")
                return 15
        finally:
            os.environ.pop("HF_HUB_CACHE", None)
            if saved_cache_env is not None:
                os.environ["HF_HUB_CACHE"] = saved_cache_env
        print(f"[test] OK [15/17] export of uncached model fails cleanly")

        # ============================================================
        # 16. HTML body (e.g. Google Drive interstitial / auth page) is
        #     rejected with a clear error, not a cryptic BadZipFile.
        # ============================================================
        saved_cache_env = os.environ.pop("HF_HUB_CACHE", None)
        try:
            cache = tmp / "cache-html"
            cache.mkdir()
            os.environ["HF_HUB_CACHE"] = str(cache)
            html = tmp / "not-a-model.zip"  # .zip suffix but HTML content
            html.write_text(
                "<!DOCTYPE html><html><head><title>Google Drive - "
                "Virus scan warning</title></head><body>...</body></html>",
                encoding="utf-8",
            )
            rc = install.import_model_archive(str(html), None, None)
            if rc != 2:
                print(f"[test] FAIL [16/17]: HTML import should fail with 2, got {rc}")
                return 16
            from lynx.config import _hf_model_cached as _cached16
            if _cached16("BAAI/bge-small-en-v1.5"):
                print(f"[test] FAIL [16/17]: HTML import should not cache anything")
                return 16
            print(f"[test] OK [16/17] HTML-not-archive rejected cleanly")
        finally:
            os.environ.pop("HF_HUB_CACHE", None)
            if saved_cache_env is not None:
                os.environ["HF_HUB_CACHE"] = saved_cache_env

        # ============================================================
        # 17. export skips the blobs/ store so content isn't zipped twice.
        #     Real HF caches keep blobs/<sha> + snapshots/<rev>/file→blob;
        #     including both doubles the archive size.
        # ============================================================
        import zipfile as _zip17
        saved_cache_env = os.environ.pop("HF_HUB_CACHE", None)
        try:
            cache = tmp / "cache-blobs"
            cache.mkdir()
            os.environ["HF_HUB_CACHE"] = str(cache)
            model = "BAAI/bge-small-en-v1.5"
            safe = model.replace("/", "--")
            mdir = cache / f"models--{safe}"
            (mdir / "blobs").mkdir(parents=True)
            (mdir / "blobs" / "deadbeef").write_text("BIGWEIGHTS", encoding="utf-8")
            snap = mdir / "snapshots" / "rev1"
            (snap / "onnx").mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
            (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snap / "onnx" / "model.onnx").write_bytes(b"graph")
            (mdir / "refs").mkdir()
            (mdir / "refs" / "main").write_text("rev1", encoding="utf-8")

            out = tmp / "dedup.zip"
            rc = install.export_model_archive(model, str(out), None)
            if rc != 0 or not out.is_file():
                print(f"[test] FAIL [17/17]: export rc={rc}, exists={out.is_file()}")
                return 17
            names = _zip17.ZipFile(out).namelist()
            if any("/blobs/" in n or n.endswith("/blobs") for n in names):
                print(f"[test] FAIL [17/17]: blobs/ should be excluded: {names}")
                return 17
            if not any(n.endswith("snapshots/rev1/config.json") for n in names):
                print(f"[test] FAIL [17/17]: snapshot file missing from zip: {names}")
                return 17
            # And it must still round-trip into a usable cache.
            shutil.rmtree(cache); cache.mkdir()
            from lynx.config import _hf_model_cached as _cached17
            if install.import_model_archive(str(out), model, None) != 0 \
                    or not _cached17(model):
                print(f"[test] FAIL [17/17]: blob-free archive didn't import")
                return 17
            print(f"[test] OK [17/17] export excludes blobs/, still round-trips")
        finally:
            os.environ.pop("HF_HUB_CACHE", None)
            if saved_cache_env is not None:
                os.environ["HF_HUB_CACHE"] = saved_cache_env

        # ============================================================
        # 18. download_model: HF fails → GitHub fallback succeeds (exit 0),
        #     and the fallback URL points at the project's models Release.
        # ============================================================
        with mock.patch("huggingface_hub.snapshot_download",
                        side_effect=RuntimeError("net down")), \
             mock.patch.object(install, "import_model_archive",
                               return_value=0) as m_fb:
            rc = install.download_model("BAAI/bge-small-en-v1.5")
        if rc != 0:
            print(f"[test] FAIL [18/19]: fallback success should exit 0, got {rc}")
            return 18
        fb_url = m_fb.call_args[0][0]
        fb_model = m_fb.call_args[0][1]
        if "releases/download/models" not in fb_url \
                or not fb_url.endswith("BAAI--bge-small-en-v1.5.zip"):
            print(f"[test] FAIL [18/19]: wrong fallback URL: {fb_url!r}")
            return 18
        if fb_model != "BAAI/bge-small-en-v1.5":
            print(f"[test] FAIL [18/19]: fallback model name wrong: {fb_model!r}")
            return 18
        print(f"[test] OK [18/19] download_model: HF→GitHub fallback succeeds")

        # ============================================================
        # 19. LYNX_MODEL_ARCHIVE_BASE_URL overrides the fallback host.
        # ============================================================
        saved_base = os.environ.pop("LYNX_MODEL_ARCHIVE_BASE_URL", None)
        try:
            os.environ["LYNX_MODEL_ARCHIVE_BASE_URL"] = "https://mirror.local/m/"
            url = install._model_archive_url("cross-encoder/ms-marco-MiniLM-L-6-v2")
            if url != "https://mirror.local/m/cross-encoder--ms-marco-MiniLM-L-6-v2.zip":
                print(f"[test] FAIL [19/19]: override URL wrong: {url!r}")
                return 19
        finally:
            os.environ.pop("LYNX_MODEL_ARCHIVE_BASE_URL", None)
            if saved_base is not None:
                os.environ["LYNX_MODEL_ARCHIVE_BASE_URL"] = saved_base
        print(f"[test] OK [19/19] LYNX_MODEL_ARCHIVE_BASE_URL override honored")

        print("\n[test] === SUCCESS: install works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
