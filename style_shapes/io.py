"""Atomic I/O, hashes, and command provenance for Style Shapes."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_write_bytes(path: os.PathLike, value: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: os.PathLike, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def atomic_write_text(path: os.PathLike, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def _capture(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def environment_snapshot() -> dict:
    gpu = _capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_head": _capture(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _capture(["git", "status", "--porcelain"]),
        "gpus": gpu.splitlines(),
    }


def append_jsonl(path: os.PathLike, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def command_record(arguments: Any, artifact_root: str = "artifacts/style_shapes") -> Iterator[dict]:
    start = dt.datetime.now(dt.timezone.utc)
    argv = [str(item) for item in sys.argv]
    append_jsonl(
        Path(artifact_root) / "logs" / "command_history.jsonl",
        {
            "event": "start",
            "utc": start.isoformat(),
            "cwd": str(Path.cwd()),
            "argv": argv,
            "normalized_command": " ".join(argv),
            "arguments": vars(arguments) if hasattr(arguments, "__dict__") else arguments,
            "environment": environment_snapshot(),
        },
    )
    result = {"exit_code": 1}
    try:
        yield result
        result["exit_code"] = 0
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        end = dt.datetime.now(dt.timezone.utc)
        append_jsonl(
            Path(artifact_root) / "logs" / "command_history.jsonl",
            {
                "event": "finish",
                "utc": end.isoformat(),
                "duration_seconds": (end - start).total_seconds(),
                "argv": argv,
                **result,
            },
        )

