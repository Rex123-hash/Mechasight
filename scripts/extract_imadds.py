"""Extract the verified IMAD-DS archives into data/raw/extracted/.

Refuses to extract an archive that has not passed checksum verification, so the
audit can never run against a truncated or corrupted download.

Usage:
    python scripts/extract_imadds.py --data-root data/raw
    python scripts/extract_imadds.py --data-root data/raw --only BrushlessMotor.7z
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import py7zr


def load_verified(provenance_path: Path) -> set[str]:
    if not provenance_path.is_file():
        return set()
    data = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        entry["filename"]
        for entry in data.get("files", [])
        if entry.get("status") == "VERIFIED"
    }


def directory_size(path: Path) -> tuple[int, int]:
    total, count = 0, 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--dest", default=None, type=Path,
                        help="extraction directory (default <data-root>/extracted)")
    parser.add_argument("--provenance",
                        default="reports/dataset_audit/download_provenance.json",
                        type=Path)
    parser.add_argument("--only", default=None, help="extract just this archive")
    parser.add_argument("--force", action="store_true",
                        help="re-extract even if the destination already exists")
    args = parser.parse_args()

    dest_root = args.dest or (args.data_root / "extracted")
    dest_root.mkdir(parents=True, exist_ok=True)

    verified = load_verified(args.provenance)
    if not verified:
        print("ERROR: no verified archives in provenance; run verify_download.py first")
        return 1

    archives = sorted(args.data_root.glob("*.7z"))
    if args.only:
        archives = [a for a in archives if a.name == args.only]
    if not archives:
        print("ERROR: no matching archives found")
        return 1

    report = []
    for archive in archives:
        if archive.name not in verified:
            print(f"SKIP {archive.name}: not checksum-verified")
            report.append({"archive": archive.name, "status": "SKIPPED_UNVERIFIED"})
            continue

        target = dest_root / archive.stem
        if target.exists() and not args.force:
            size, count = directory_size(target)
            print(f"SKIP {archive.name}: already extracted "
                  f"({count} files, {size / 2**30:.2f} GiB) -- use --force to redo")
            report.append({"archive": archive.name, "status": "ALREADY_EXTRACTED",
                           "extracted_bytes": size, "extracted_files": count})
            continue

        print(f"extracting {archive.name} -> {target} ...", flush=True)
        started = time.time()
        target.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as handle:
            handle.extractall(path=target)
        elapsed = time.time() - started
        size, count = directory_size(target)
        print(f"  done in {elapsed:.0f}s: {count} files, {size / 2**30:.2f} GiB "
              f"(compression ratio {archive.stat().st_size / size:.3f})")
        report.append({
            "archive": archive.name,
            "status": "EXTRACTED",
            "compressed_bytes": archive.stat().st_size,
            "extracted_bytes": size,
            "extracted_files": count,
            "seconds": round(elapsed, 1),
        })

    out = args.provenance.parent / "extraction_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.is_file():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    merged = {entry["archive"]: entry for entry in existing if isinstance(entry, dict)}
    merged.update({entry["archive"]: entry for entry in report})
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
    print(f"\nextraction report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
