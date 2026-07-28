"""Deterministic, atomic I/O and content-addressed manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def load_yaml(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping: %s" % path)
    return value


def atomic_write_bytes(path: os.PathLike, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
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
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def atomic_write_text(path: os.PathLike, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_torch_save(path: os.PathLike, value: Any) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: os.PathLike, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(value) + "\n"
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def file_records(paths: Iterable[os.PathLike]) -> list:
    records = []
    for raw in paths:
        path = Path(raw)
        records.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def refuse_incompatible_manifest(path: os.PathLike, expected_hash: str) -> bool:
    """Return True when a compatible completed artifact can be resumed."""
    target = Path(path)
    if not target.exists():
        return False
    with open(target, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    actual = existing.get("input_config_hash")
    if actual != expected_hash:
        raise RuntimeError("refusing to overwrite incompatible artifact: %s (%s != %s)" % (target, actual, expected_hash))
    return existing.get("status") == "complete"
