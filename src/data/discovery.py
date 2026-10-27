"""Filesystem discovery for an extracted IMAD-DS release.

Discovery-first by design: this module walks what is actually on disk and
reports it, rather than asserting an expected layout. Anything unexpected is
surfaced as a finding instead of being ignored or raising.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Sensor tokens as they appear in IMAD-DS filenames and metadata column names.
# Used for *matching*, never as an assumption that all three are present.
SENSOR_TOKENS = {
    "mic": "imp23absu_mic",
    "acc": "ism330dhcx_acc",
    "gyro": "ism330dhcx_gyro",
}

METADATA_GLOB = "attributes_*.csv"

# attributes_<anomaly>_<domain>_<split>.csv
METADATA_NAME_RE = re.compile(
    r"^attributes_(?P<anomaly>normal|anomaly)_(?P<domain>source|target)_(?P<split>train|test)\.csv$",
    re.IGNORECASE,
)


@dataclass
class MachineLayout:
    """What was actually found on disk for one machine."""

    machine: str
    root: Path
    metadata_files: list[Path] = field(default_factory=list)
    parquet_files: list[Path] = field(default_factory=list)
    other_files: list[Path] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)
    zero_byte_files: list[Path] = field(default_factory=list)
    unparsed_metadata_names: list[Path] = field(default_factory=list)
    duplicate_basenames: dict[str, list[Path]] = field(default_factory=dict)
    parquet_by_sensor: dict[str, int] = field(default_factory=dict)
    parquet_unclassified: list[Path] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(
            p.stat().st_size
            for p in (*self.metadata_files, *self.parquet_files, *self.other_files)
            if p.is_file()
        )


def find_machine_roots(data_root: Path) -> dict[str, Path]:
    """Locate each machine's directory under an extraction root.

    Accepts either <root>/<Machine>/{train,test} or a nested
    <root>/<Machine>/<Machine>/{train,test}, because archive layouts vary.
    Returns {machine_name: directory containing train/ and/or test/}.
    """
    found: dict[str, Path] = {}
    if not data_root.exists():
        return found

    for candidate in sorted(data_root.iterdir()):
        if not candidate.is_dir():
            continue
        machine = candidate.name
        # Walk down while the directory holds exactly one subdirectory and no
        # train/test pair of its own -- handles archives with a nested top level.
        node = candidate
        for _ in range(3):
            if any((node / split).is_dir() for split in ("train", "test")):
                break
            subdirs = [d for d in node.iterdir() if d.is_dir()]
            if len(subdirs) != 1:
                break
            node = subdirs[0]
        found[machine] = node
    return found


def classify_parquet(path: Path) -> str | None:
    """Map a parquet filename to a sensor key, or None if unrecognised."""
    name = path.name.lower()
    for key, token in SENSOR_TOKENS.items():
        if token in name:
            return key
    return None


def parse_metadata_name(path: Path) -> dict[str, str] | None:
    """Derive (anomaly, domain, split) from an attributes_*.csv filename.

    Returned for cross-checking against the in-file `split_label` column. A
    disagreement between the two is a real finding, so the two sources are kept
    separate rather than merged.
    """
    match = METADATA_NAME_RE.match(path.name)
    return {k: v.lower() for k, v in match.groupdict().items()} if match else None


def scan_machine(machine: str, root: Path) -> MachineLayout:
    """Inventory every file under one machine root."""
    layout = MachineLayout(machine=machine, root=root)
    basenames: dict[str, list[Path]] = defaultdict(list)
    sensor_counts: Counter[str] = Counter()

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            layout.directories.append(path)
            continue
        if not path.is_file():
            continue

        try:
            if path.stat().st_size == 0:
                layout.zero_byte_files.append(path)
        except OSError:
            layout.other_files.append(path)
            continue

        suffix = path.suffix.lower()
        if suffix == ".csv" and path.name.lower().startswith("attributes_"):
            layout.metadata_files.append(path)
            if parse_metadata_name(path) is None:
                layout.unparsed_metadata_names.append(path)
        elif suffix == ".parquet":
            layout.parquet_files.append(path)
            sensor = classify_parquet(path)
            if sensor is None:
                layout.parquet_unclassified.append(path)
            else:
                sensor_counts[sensor] += 1
        else:
            layout.other_files.append(path)

        basenames[path.name].append(path)

    layout.parquet_by_sensor = dict(sensor_counts)
    layout.duplicate_basenames = {
        name: paths for name, paths in basenames.items() if len(paths) > 1
    }
    return layout


def summarise_layout(layout: MachineLayout) -> dict:
    """Flat, JSON-serialisable summary of one machine's layout."""
    return {
        "machine": layout.machine,
        "root": str(layout.root),
        "n_directories": len(layout.directories),
        "n_metadata_csv": len(layout.metadata_files),
        "metadata_csv_names": sorted(p.name for p in layout.metadata_files),
        "n_parquet": len(layout.parquet_files),
        "parquet_by_sensor": layout.parquet_by_sensor,
        "n_parquet_unclassified": len(layout.parquet_unclassified),
        "parquet_unclassified_examples": [
            p.name for p in layout.parquet_unclassified[:10]
        ],
        "n_other_files": len(layout.other_files),
        "other_file_names": sorted({p.name for p in layout.other_files})[:50],
        "other_file_suffixes": dict(
            Counter(p.suffix.lower() or "<none>" for p in layout.other_files)
        ),
        "n_zero_byte_files": len(layout.zero_byte_files),
        "zero_byte_examples": [str(p) for p in layout.zero_byte_files[:10]],
        "n_duplicate_basenames": len(layout.duplicate_basenames),
        "duplicate_basename_examples": sorted(layout.duplicate_basenames)[:10],
        "n_unparsed_metadata_names": len(layout.unparsed_metadata_names),
        "unparsed_metadata_names": [p.name for p in layout.unparsed_metadata_names],
        "total_bytes": layout.total_bytes,
    }
