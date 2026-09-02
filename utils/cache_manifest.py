"""Sidecar provenance manifests for pickle/npy result caches (P1-06).

Every cache-worthy result file `X` gets a companion `X.manifest.json`
recording the parameters it was produced with. This intentionally does not
change the format of `X` itself, so existing caches stay byte-for-byte
readable by old and new code alike.

Manifests cannot prove a file's *historical* correctness — only the person
who produced it can assert that. What they protect against is *future*
confusion: reusing a cache that was actually computed with different
parameters, a different seed, or different inputs than the current run asks
for.

Usage at a cache read site:

    status = check_cache(cache_path, metadata)
    if status == "missing":
        ... compute, save cache_path, write_manifest(cache_path, metadata) ...
    elif status == "legacy":
        kprint(f"Upgrading legacy cache {cache_path} to provenance-tracked "
               "format (trusted as-is, not recomputed)")
        ... load cache_path as before ...
        write_manifest(cache_path, metadata)
    elif status == "match":
        ... load cache_path as before ...
    elif status == "mismatch":
        ... caller decides: warn/recompute/require --force-recompute ...
"""

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

MANIFEST_SCHEMA_VERSION = 1

CacheStatus = Literal["missing", "legacy", "match", "mismatch"]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def manifest_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".manifest.json")


def read_manifest(cache_path: Path) -> dict[str, Any] | None:
    path = manifest_path(cache_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


def check_cache(cache_path: Path, metadata: dict[str, Any]) -> CacheStatus:
    """Classify a cache file against the metadata the current run expects."""
    if not cache_path.is_file():
        return "missing"

    recorded = read_manifest(cache_path)
    if recorded is None:
        return "legacy"

    if recorded.get("metadata") != _json_safe(metadata):
        return "mismatch"
    return "match"


def write_manifest(cache_path: Path, metadata: dict[str, Any]) -> Path:
    """Atomically write/update the sidecar manifest for `cache_path`."""
    target = manifest_path(cache_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": _json_safe(metadata),
    }
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)
    return target


def path_fingerprint(
    directory: Path, *, suffix: str | None = None
) -> dict[str, Any]:
    """Cheap content fingerprint of a directory: sorted file names and sizes.

    Used to detect "the input directory changed" without hashing large
    binary contents.
    """
    entries = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and (suffix is None or path.name.endswith(suffix))
    )
    sizes = [(directory / name).stat().st_size for name in entries]
    return {
        "directory": str(directory),
        "file_count": len(entries),
        "file_names": entries,
        "file_sizes": sizes,
    }


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap content fingerprint of one file: path, size, mtime."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }
