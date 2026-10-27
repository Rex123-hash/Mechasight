"""Verify downloaded IMAD-DS archives against the official Zenodo checksums.

Records full provenance (URL, DOI, byte sizes, official and computed checksums,
verification timestamp) so a reviewer can confirm we audited the official
release and not some mirror or a truncated download.

Usage:
    python scripts/verify_download.py --data-root data/raw
    python scripts/verify_download.py --data-root data/raw --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ZENODO_RECORD = "12665499"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Official values captured from the Zenodo REST API on 2026-08-21. Kept here so
# verification still works offline, and so any future change to the record is
# visible as a mismatch rather than silently accepted.
EXPECTED = {
    "BrushlessMotor.7z": {
        "size": 1577018050,
        "checksum": "md5:d5f1a49b928b1b0b01c3d6de930cd277",
    },
    "RoboticArm.7z": {
        "size": 3545453763,
        "checksum": "md5:a64498b7dcc297946a7fb8366e38ba33",
    },
}


def file_md5(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_official() -> dict[str, dict]:
    """Re-fetch the record so the pinned constants above are themselves checked."""
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        record = json.load(response)
    return {
        f["key"]: {"size": f["size"], "checksum": f.get("checksum")}
        for f in record.get("files", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--out", default="reports/dataset_audit/download_provenance.json",
                        type=Path)
    parser.add_argument("--offline", action="store_true",
                        help="skip re-fetching the Zenodo record; use pinned values")
    args = parser.parse_args()

    official = EXPECTED
    live_matches_pinned = None
    if not args.offline:
        try:
            live = fetch_official()
            live_matches_pinned = live == EXPECTED
            if not live_matches_pinned:
                print("WARNING: live Zenodo record differs from pinned expectations",
                      file=sys.stderr)
                print(f"  live   : {live}", file=sys.stderr)
                print(f"  pinned : {EXPECTED}", file=sys.stderr)
            official = live
        except Exception as exc:  # network unavailable is not a hard failure
            print(f"NOTE: could not re-fetch Zenodo record ({exc}); using pinned values")

    results, all_ok = [], True
    for name, expected in sorted(official.items()):
        path = args.data_root / name
        entry: dict = {
            "filename": name,
            "source_url": f"{ZENODO_API}/files/{name}/content",
            "expected_size": expected["size"],
            "expected_checksum": expected["checksum"],
            "present": path.exists(),
        }
        if not path.exists():
            entry.update(status="MISSING")
            all_ok = False
        else:
            actual_size = path.stat().st_size
            entry["actual_size"] = actual_size
            entry["size_match"] = actual_size == expected["size"]
            print(f"hashing {name} ({actual_size / 2**30:.2f} GiB) ...", flush=True)
            actual = f"md5:{file_md5(path)}"
            entry["actual_checksum"] = actual
            entry["checksum_match"] = actual == expected["checksum"]
            ok = entry["size_match"] and entry["checksum_match"]
            entry["status"] = "VERIFIED" if ok else "MISMATCH"
            all_ok &= ok
        print(f"  {name}: {entry['status']}")
        results.append(entry)

    provenance = {
        "zenodo_record": ZENODO_RECORD,
        "zenodo_record_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "doi": "10.5281/zenodo.12665499",
        "concept_doi": "10.5281/zenodo.12636236",
        "licence": "CC BY-SA 4.0",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "live_record_matches_pinned_values": live_matches_pinned,
        "files": results,
        "all_verified": all_ok,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nprovenance written to {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
