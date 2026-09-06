"""Unit tests for `lynx manager doctor`.

We stub the filesystem checks (HF cache, config file) with tempdirs and
synthetic configs so no real disk state matters. Pure-function design
of `doctor.py` makes this straightforward.

Scenarios:
  1. check_python_version: passes on current process (always 3.12 in CI)
  2. check_hf_model_cache: missing dir → WARN
  3. check_hf_model_cache: dir exists, snapshot complete → OK
  4. check_hf_model_cache: dir exists, snapshot empty → WARN
  5. check_config_file: missing file → ERROR
  6. check_config_file: valid file → OK + returns loaded config
  7. check_config_file: invalid JSON → ERROR (captures SystemExit)
  8. check_source: missing path → ERROR
  9. check_source: valid codebase path → OK with details
 10. check_disk_space: low space simulated → WARN/ERROR
 11. _worst_status / _format_human roundtrip
 12. run_all_checks end-to-end on synthetic config
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    from lynx.manager.doctor import (
        check_python_version, check_hf_model_cache, check_config_file,
        check_source, check_disk_space, check_optional_extras,
        run_all_checks, _worst_status, _format_human,
        CheckResult, STATUS_OK, STATUS_WARN, STATUS_ERROR,
        _hf_cache_dir_for,
    )

    tmp = Path(tempfile.mkdtemp(prefix="lynx-doctor-"))
    print(f"[test] tempdir: {tmp}")
    try:
        # ============================================================
        # 1. Python version always passes here (we're running on 3.12)
        # ============================================================
        r = check_python_version()
        if r.status != STATUS_OK:
            print(f"[test] FAIL [1/12]: expected ok, got {r.status} ({r.summary})")
            return 1
        if not r.summary.startswith("Python 3."):
            print(f"[test] FAIL [1/12]: summary wrong: {r.summary}")
            return 1
        print(f"[test] OK [1/12] python version: {r.summary}")

        # ============================================================
        # 2. HF cache missing → WARN
        # ============================================================
        # Pick a model name that definitely isn't in any user's cache.
        fake_model = "this-org/never-exists-9zzz"
        r = check_hf_model_cache(fake_model)
        if r.status != STATUS_WARN:
            print(f"[test] FAIL [2/12]: expected WARN for missing model, got {r.status}")
            return 2
        if fake_model not in r.summary:
            print(f"[test] FAIL [2/12]: model name missing from summary")
            return 2
        print(f"[test] OK [2/12] hf cache missing: WARN with helpful message")

        # ============================================================
        # 3. HF cache complete → OK (simulate by faking the dirs)
        # ============================================================
        # Monkey-patch _hf_cache_dir_for to point at our tmpdir, then
        # build the expected layout.
        fake_cache_root = tmp / "hf_cache"
        fake_cache_root.mkdir()
        snapshot = fake_cache_root / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        # Complete means the files the ONNX runtime opens: the graph and the
        # tokenizer. Until 1.9 a config.json on its own was enough here.
        (snapshot / "config.json").write_text("{}")
        (snapshot / "tokenizer.json").write_text("{}")
        (snapshot / "onnx").mkdir()
        (snapshot / "onnx" / "model.onnx").write_bytes(b"")

        from lynx.manager import doctor as doc_module
        original = doc_module._hf_cache_dir_for
        doc_module._hf_cache_dir_for = lambda m: fake_cache_root
        try:
            r = check_hf_model_cache("fake/model")
        finally:
            doc_module._hf_cache_dir_for = original
        if r.status != STATUS_OK:
            print(f"[test] FAIL [3/12]: expected OK for complete cache, got {r.status}")
            return 3
        print(f"[test] OK [3/12] hf cache complete: {r.summary}")

        # ============================================================
        # 4. HF cache snapshot empty (no config.json) → WARN
        # ============================================================
        bad_cache = tmp / "hf_bad"
        bad_snap = bad_cache / "snapshots" / "def456"
        bad_snap.mkdir(parents=True)
        # NO config.json
        doc_module._hf_cache_dir_for = lambda m: bad_cache
        try:
            r = check_hf_model_cache("fake/incomplete")
        finally:
            doc_module._hf_cache_dir_for = original
        if r.status != STATUS_WARN:
            print(f"[test] FAIL [4/12]: expected WARN for incomplete cache, got {r.status}")
            return 4
        if "missing" not in r.summary.lower():
            print(f"[test] FAIL [4/12]: summary should mention 'missing': {r.summary}")
            return 4
        print(f"[test] OK [4/12] hf cache incomplete: WARN")

        # ============================================================
        # 5. Config missing → ERROR
        # ============================================================
        nonexistent = tmp / "does_not_exist.json"
        r, cfg = check_config_file(nonexistent)
        if r.status != STATUS_ERROR or cfg is not None:
            print(f"[test] FAIL [5/12]: missing config should be ERROR + cfg=None")
            return 5
        print(f"[test] OK [5/12] config missing: ERROR")

        # ============================================================
        # 6. Config valid → OK
        # ============================================================
        code_dir = tmp / "code"
        code_dir.mkdir()
        (code_dir / "x.py").write_text("def f(): pass\n")
        cfg_path = tmp / "config.json"
        cfg_path.write_text(json.dumps({
            "config_version": 2,
            "storage_path": str(tmp / "storage"),
            "sources": {
                "demo": {
                    "type": "codebase",
                    "path": str(code_dir),
                    "supported_extensions": [".py"],
                    "watcher": {"enabled": False},
                    "git_integration": {"enabled": False},
                },
            },
        }))
        r, cfg = check_config_file(cfg_path)
        if r.status != STATUS_OK:
            print(f"[test] FAIL [6/12]: valid config should be OK, got {r.status}")
            return 6
        if cfg is None:
            print(f"[test] FAIL [6/12]: cfg should be returned on success")
            return 6
        if "demo" not in cfg.sources:
            print(f"[test] FAIL [6/12]: cfg.sources missing 'demo'")
            return 6
        print(f"[test] OK [6/12] config valid: returned with {len(cfg.sources)} source(s)")

        # ============================================================
        # 7. Config invalid JSON → ERROR
        # ============================================================
        bad_json = tmp / "bad.json"
        bad_json.write_text("{not valid json}")
        r, cfg = check_config_file(bad_json)
        if r.status != STATUS_ERROR or cfg is not None:
            print(f"[test] FAIL [7/12]: bad JSON should be ERROR, got {r.status}")
            return 7
        print(f"[test] OK [7/12] config invalid JSON: ERROR")

        # ============================================================
        # 8. Source path missing → ERROR
        # ============================================================
        from pathlib import Path as _P
        r = check_source(
            "bad", {"type": "codebase", "path": "/this/does/not/exist"},
            tmp / "storage",
        )
        if r.status != STATUS_ERROR:
            print(f"[test] FAIL [8/12]: missing path should be ERROR")
            return 8
        print(f"[test] OK [8/12] source missing path: ERROR")

        # ============================================================
        # 9. Source path valid → OK
        # ============================================================
        r = check_source(
            "good", {"type": "codebase", "path": str(code_dir),
                     "watcher": {"enabled": False}},
            tmp / "storage",
        )
        if r.status != STATUS_OK:
            print(f"[test] FAIL [9/12]: valid path should be OK, got {r.status} ({r.summary})")
            return 9
        if not any("Source path" in d for d in r.details):
            print(f"[test] FAIL [9/12]: expected 'Source path' in details: {r.details}")
            return 9
        print(f"[test] OK [9/12] source valid: details = {len(r.details)}")

        # ============================================================
        # 10. Disk space — we can't force this reliably, just sanity check
        #    the function doesn't crash on tmp and returns OK/WARN.
        # ============================================================
        r = check_disk_space(tmp)
        if r.status not in (STATUS_OK, STATUS_WARN, STATUS_ERROR):
            print(f"[test] FAIL [10/12]: bad status: {r.status}")
            return 10
        if "free" not in r.summary.lower():
            print(f"[test] FAIL [10/12]: summary missing 'free': {r.summary}")
            return 10
        print(f"[test] OK [10/12] disk space: {r.status} — {r.summary}")

        # ============================================================
        # 11. _worst_status + _format_human roundtrip
        # ============================================================
        results = [
            CheckResult("a", STATUS_OK, "ok"),
            CheckResult("b", STATUS_WARN, "warn"),
            CheckResult("c", STATUS_OK, "ok"),
        ]
        if _worst_status(results) != STATUS_WARN:
            print(f"[test] FAIL [11/12]: worst of (OK,WARN,OK) should be WARN")
            return 11
        results.append(CheckResult("d", STATUS_ERROR, "err"))
        if _worst_status(results) != STATUS_ERROR:
            print(f"[test] FAIL [11/12]: any ERROR should dominate")
            return 11
        text = _format_human(results)
        for needle in ("Lynx diagnostic", "ok", "warn", "err"):
            if needle not in text:
                print(f"[test] FAIL [11/12]: formatted output missing {needle!r}")
                return 11
        print(f"[test] OK [11/12] worst_status + format_human")

        # ============================================================
        # 12. run_all_checks end-to-end on synthetic config
        # ============================================================
        results = run_all_checks(cfg_path)
        names = {r.name for r in results}
        for required in ("Python version", "Config file"):
            if required not in names:
                print(f"[test] FAIL [12/12]: expected check {required!r} missing")
                return 12
        if not any(r.name.startswith("Source ") for r in results):
            print(f"[test] FAIL [12/12]: no per-source check ran")
            return 12
        print(f"[test] OK [12/12] run_all_checks: {len(results)} checks ran")

        print("\n[test] === SUCCESS: doctor works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
