"""Build and verify a checksum manifest for a Zenodo data release.

Usage:
    python -m scripts.build_data_release_manifest build <data_dir> \\
        --license CC-BY-4.0 --code-url https://github.com/... --code-version v1.0.0

    python -m scripts.build_data_release_manifest verify <data_dir> \\
        --manifest <data_dir>/data_manifest.json
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
CHUNK_SIZE = 1 << 20


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_data_files(data_dir: Path, manifest_name: str):
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir).as_posix()
        if relative in (manifest_name, f"{manifest_name}.tmp"):
            continue
        yield path, relative


def build_manifest(
    data_dir: Path,
    *,
    license_name: str,
    code_url: str | None,
    code_version: str | None,
    description: str | None,
    manifest_name: str = "data_manifest.json",
) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path, relative in _iter_data_files(data_dir, manifest_name):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": _sha256sum(path),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
        "license": license_name,
        "code_repository": code_url,
        "code_version": code_version,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def save_manifest(
    manifest: dict[str, Any], data_dir: Path, manifest_name: str = "data_manifest.json"
) -> Path:
    target = data_dir / manifest_name
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)
    return target


def verify_manifest(
    data_dir: Path, manifest_path: Path
) -> tuple[list[str], list[str], list[str]]:
    """Return (missing, corrupted, unexpected) relative to the manifest."""
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    on_disk = {
        relative: path
        for path, relative in _iter_data_files(data_dir, manifest_path.name)
    }

    missing = sorted(set(expected) - set(on_disk))
    unexpected = sorted(set(on_disk) - set(expected))
    corrupted = sorted(
        relative
        for relative in set(expected) & set(on_disk)
        if _sha256sum(on_disk[relative]) != expected[relative]
    )
    return missing, corrupted, unexpected


def _build_command(args: argparse.Namespace) -> None:
    manifest = build_manifest(
        args.data_dir,
        license_name=args.license,
        code_url=args.code_url,
        code_version=args.code_version,
        description=args.description,
        manifest_name=args.manifest_name,
    )
    path = save_manifest(manifest, args.data_dir, manifest_name=args.manifest_name)
    print(
        f"Wrote {path} ({manifest['file_count']} files, "
        f"{manifest['total_bytes']} bytes)"
    )


def _verify_command(args: argparse.Namespace) -> None:
    manifest_path = args.manifest or (args.data_dir / "data_manifest.json")
    missing, corrupted, unexpected = verify_manifest(args.data_dir, manifest_path)
    if not missing and not corrupted and not unexpected:
        print(f"OK: {args.data_dir} matches {manifest_path}")
        return
    if missing:
        print(f"Missing ({len(missing)}):")
        for relative in missing:
            print(f"  {relative}")
    if corrupted:
        print(f"Checksum mismatch ({len(corrupted)}):")
        for relative in corrupted:
            print(f"  {relative}")
    if unexpected:
        print(f"Unexpected extra files ({len(unexpected)}):")
        for relative in unexpected:
            print(f"  {relative}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a SHA-256 checksum manifest for a Zenodo data "
            "release directory (reference PNM outputs, publication inputs)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Hash every file under data_dir and write data_manifest.json"
    )
    build_parser.add_argument("data_dir", type=Path)
    build_parser.add_argument(
        "--license",
        default="CC-BY-4.0",
        help="SPDX license identifier for the data release",
    )
    build_parser.add_argument(
        "--code-url",
        default=None,
        help="URL of the code repository/release this data was produced with",
    )
    build_parser.add_argument(
        "--code-version",
        default=None,
        help="Tag, release name, or commit hash of the code that produced this data",
    )
    build_parser.add_argument("--description", default=None)
    build_parser.add_argument("--manifest-name", default="data_manifest.json")
    build_parser.set_defaults(func=_build_command)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Recompute checksums under data_dir and compare against a manifest",
    )
    verify_parser.add_argument("data_dir", type=Path)
    verify_parser.add_argument("--manifest", type=Path, default=None)
    verify_parser.set_defaults(func=_verify_command)

    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
