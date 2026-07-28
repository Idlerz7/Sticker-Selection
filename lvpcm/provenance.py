"""Command provenance shared by every LVPCM command-line program."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterator

from .io import append_jsonl


def _run(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def environment_snapshot() -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _run(["git", "status", "--porcelain"]),
    }


@contextlib.contextmanager
def command_record(args: argparse.Namespace, artifact_root: str = "artifacts/lvpcm") -> Iterator[dict]:
    start = dt.datetime.now(dt.timezone.utc)
    record = {
        "event": "start",
        "utc": start.isoformat(),
        "cwd": str(Path.cwd()),
        "argv": [str(item) for item in sys.argv],
        "arguments": vars(args),
        "environment": environment_snapshot(),
    }
    log_path = Path(artifact_root) / "logs" / "command_history.jsonl"
    append_jsonl(log_path, record)
    outcome = {"exit_code": 1}
    try:
        yield outcome
        outcome["exit_code"] = 0
    except BaseException as exc:
        outcome["error_type"] = type(exc).__name__
        outcome["error"] = str(exc)
        raise
    finally:
        end = dt.datetime.now(dt.timezone.utc)
        append_jsonl(log_path, {
            "event": "finish",
            "utc": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            **outcome,
            "argv": [str(item) for item in sys.argv],
        })
