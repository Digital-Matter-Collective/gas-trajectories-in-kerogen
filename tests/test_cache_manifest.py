from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.cache_manifest import (
    check_cache,
    file_fingerprint,
    path_fingerprint,
    write_manifest,
)


@dataclass(frozen=True)
class DummyParams:
    seed: int
    scale: float


def test_missing_cache_reports_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "result.npy"
    assert check_cache(cache_path, {"seed": 1}) == "missing"


def test_cache_without_manifest_is_legacy(tmp_path: Path) -> None:
    cache_path = tmp_path / "result.npy"
    cache_path.write_bytes(b"data")
    assert check_cache(cache_path, {"seed": 1}) == "legacy"


def test_write_then_matching_metadata_is_match(tmp_path: Path) -> None:
    cache_path = tmp_path / "result.npy"
    cache_path.write_bytes(b"data")
    metadata = {"seed": 1, "params": DummyParams(seed=1, scale=0.5)}

    write_manifest(cache_path, metadata)

    assert check_cache(cache_path, metadata) == "match"
    assert not cache_path.with_name(".result.npy.manifest.json.tmp").exists()


def test_mismatched_metadata_is_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "result.npy"
    cache_path.write_bytes(b"data")
    write_manifest(cache_path, {"seed": 1})

    assert check_cache(cache_path, {"seed": 2}) == "mismatch"


def test_write_manifest_json_serializes_dataclasses_and_arrays(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "result.npy"
    cache_path.write_bytes(b"data")
    metadata = {
        "params": DummyParams(seed=7, scale=1.5),
        "grid": np.array([1, 2, 3]),
        "path": tmp_path / "input",
    }

    write_manifest(cache_path, metadata)
    manifest_file = cache_path.with_name("result.npy.manifest.json")
    assert manifest_file.is_file()

    # A fresh metadata dict built the same way must compare equal.
    assert check_cache(cache_path, metadata) == "match"


def test_path_fingerprint_detects_added_file(tmp_path: Path) -> None:
    (tmp_path / "a_node1.dat").write_text("1")
    fp1 = path_fingerprint(tmp_path, suffix=".dat")

    (tmp_path / "b_node1.dat").write_text("22")
    fp2 = path_fingerprint(tmp_path, suffix=".dat")

    assert fp1 != fp2
    assert fp2["file_count"] == 2


def test_file_fingerprint_detects_size_change(tmp_path: Path) -> None:
    target = tmp_path / "trj.gro"
    target.write_text("abc")
    fp1 = file_fingerprint(target)

    target.write_text("abcdef")
    fp2 = file_fingerprint(target)

    assert fp1["size_bytes"] != fp2["size_bytes"]
