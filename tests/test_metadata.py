"""Tests for metadata parsing, sensor-file matching and consistency checks.

Every fixture is synthetic and built in a tmp_path, so these tests catch wrong
data handling rather than restating anything published about IMAD-DS.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.discovery import (
    classify_parquet,
    find_machine_roots,
    parse_metadata_name,
    scan_machine,
)
from src.data.metadata import (
    check_label_consistency,
    load_machine_metadata,
    load_metadata_file,
    resolve_sensor_paths,
)

SENSOR_COLUMNS = {
    "mic": "imp23absu_mic",
    "acc": "ism330dhcx_acc",
    "gyro": "ism330dhcx_gyro",
}


def _row(seg: str, anomaly: str, domain: str, split: str, op: str, env: str) -> dict:
    return {
        "imp23absu_mic": f"imp23absu_mic_{seg}.parquet",
        "ism330dhcx_acc": f"ism330dhcx_acc_{seg}.parquet",
        "ism330dhcx_gyro": f"ism330dhcx_gyro_{seg}.parquet",
        "split_label": f"{anomaly.capitalize()}_{domain.capitalize()}_{split.capitalize()}",
        "anomaly_label": "normal" if anomaly == "normal" else "magnet",
        "domain_shift_op": op,
        "domain_shift_env": env,
    }


def build_machine(tmp_path, *, make_files=True, machine="TestMachine"):
    """Create a miniature IMAD-DS-shaped tree."""
    root = tmp_path / machine
    specs = [
        ("normal", "source", "train", ["s01", "s02"]),
        ("normal", "target", "train", ["t01"]),
        ("normal", "source", "test", ["s10"]),
        ("anomaly", "source", "test", ["s11"]),
        ("normal", "target", "test", ["t10"]),
        ("anomaly", "target", "test", ["t11"]),
    ]
    for anomaly, domain, split, segments in specs:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _row(seg, anomaly, domain, split, op="load20", env="noiseA")
            for seg in segments
        ]
        pd.DataFrame(rows).to_csv(
            split_dir / f"attributes_{anomaly}_{domain}_{split}.csv"
        )
        if make_files:
            for seg in segments:
                for column in SENSOR_COLUMNS.values():
                    path = split_dir / f"{column}_{seg}.parquet"
                    pd.DataFrame({"Time": [0.0, 1.0], "x": [1.0, 2.0]}).to_parquet(path)
    return root


class TestFilenameParsing:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("attributes_normal_source_train.csv",
             {"anomaly": "normal", "domain": "source", "split": "train"}),
            ("attributes_anomaly_target_test.csv",
             {"anomaly": "anomaly", "domain": "target", "split": "test"}),
        ],
    )
    def test_parses_known_shapes(self, tmp_path, name, expected):
        assert parse_metadata_name(tmp_path / name) == expected

    def test_returns_none_for_unrecognised_name(self, tmp_path):
        """Unrecognised names must be reported, not silently mapped to a default."""
        assert parse_metadata_name(tmp_path / "attributes_weird_thing.csv") is None

    def test_sensor_classification(self, tmp_path):
        assert classify_parquet(tmp_path / "imp23absu_mic_a1.parquet") == "mic"
        assert classify_parquet(tmp_path / "ism330dhcx_acc_a1.parquet") == "acc"
        assert classify_parquet(tmp_path / "ism330dhcx_gyro_a1.parquet") == "gyro"
        assert classify_parquet(tmp_path / "something_else.parquet") is None


class TestDiscovery:
    def test_finds_machine_root_and_inventories_files(self, tmp_path):
        build_machine(tmp_path)
        roots = find_machine_roots(tmp_path)
        assert set(roots) == {"TestMachine"}

        layout = scan_machine("TestMachine", roots["TestMachine"])
        assert len(layout.metadata_files) == 6
        assert len(layout.parquet_files) == 7 * 3
        assert layout.parquet_by_sensor == {"mic": 7, "acc": 7, "gyro": 7}
        assert layout.unparsed_metadata_names == []

    def test_descends_through_a_nested_top_level_directory(self, tmp_path):
        """Some archives wrap the payload in an extra directory level."""
        nested = tmp_path / "BrushlessMotor"
        build_machine(nested, machine="BrushlessMotor")
        roots = find_machine_roots(tmp_path)
        assert (roots["BrushlessMotor"] / "train").is_dir()

    def test_zero_byte_and_foreign_files_are_flagged(self, tmp_path):
        root = build_machine(tmp_path)
        (root / "train" / "empty.parquet").write_bytes(b"")
        (root / "README.txt").write_text("notes")
        layout = scan_machine("TestMachine", root)
        assert len(layout.zero_byte_files) == 1
        assert any(p.name == "README.txt" for p in layout.other_files)
        assert len(layout.parquet_unclassified) == 1

    def test_duplicate_basenames_across_splits_are_detected(self, tmp_path):
        root = build_machine(tmp_path)
        duplicate = root / "test" / "imp23absu_mic_s01.parquet"  # also exists in train/
        pd.DataFrame({"Time": [0.0], "x": [0.0]}).to_parquet(duplicate)
        layout = scan_machine("TestMachine", root)
        assert "imp23absu_mic_s01.parquet" in layout.duplicate_basenames


class TestMetadataLoading:
    def test_segment_id_strips_sensor_prefix_and_suffix(self, tmp_path):
        root = build_machine(tmp_path)
        frame, info = load_metadata_file(
            root / "train" / "attributes_normal_source_train.csv"
        )
        assert sorted(frame["segment_id"]) == ["s01", "s02"]
        assert info["n_null_segment_id"] == 0
        assert info["sensor_columns_missing"] == []

    def test_missing_label_columns_are_reported_not_raised(self, tmp_path):
        path = tmp_path / "attributes_normal_source_train.csv"
        pd.DataFrame([{"imp23absu_mic": "imp23absu_mic_a.parquet"}]).to_csv(path)
        frame, info = load_metadata_file(path)
        assert info["readable"] is True
        assert set(info["label_columns_missing"]) == {
            "split_label", "anomaly_label", "domain_shift_op", "domain_shift_env"
        }
        assert frame["segment_id"].tolist() == ["a"]

    def test_unreadable_file_is_reported_not_raised(self, tmp_path):
        path = tmp_path / "attributes_normal_source_train.csv"
        path.write_bytes(b"\x00\x01\x02 not,a;csv\x00")
        frame, info = load_metadata_file(path)
        assert info["readable"] is False or info["n_rows"] == 0

    def test_load_machine_metadata_concatenates_all_files(self, tmp_path):
        root = build_machine(tmp_path)
        layout = scan_machine("TestMachine", root)
        frame, infos = load_machine_metadata(layout.metadata_files)
        assert len(frame) == 7
        assert len(infos) == 6
        assert set(frame["file_split"]) == {"train", "test"}


class TestReleasedSegmentIdColumn:
    """Regression: the released CSVs carry `segment_id` as their first column.

    Reading with index_col=0 consumed it into the index, so it vanished from
    `raw.columns` and the loader silently fell back to a filename-derived id of
    a different shape. Nothing errored -- downstream analysis just quietly
    received the wrong identifier.
    """

    def test_released_segment_id_column_is_used_verbatim(self, tmp_path):
        path = tmp_path / "attributes_normal_source_train.csv"
        pd.DataFrame([{
            "segment_id": "20240423_15_30_05_segment_0",
            "split_label": "Normal_Source_Train",
            "anomaly_label": "normal",
            "imp23absu_mic": "imp23absu_mic_20240423_15_30_05_DataLog_0.parquet",
            "ism330dhcx_acc": "ism330dhcx_acc_20240423_15_30_05_DataLog_0.parquet",
            "ism330dhcx_gyro": "ism330dhcx_gyro_20240423_15_30_05_DataLog_0.parquet",
        }]).to_csv(path, index=False)

        frame, info = load_metadata_file(path)
        assert info["segment_id_source"] == "released_column"
        assert frame["segment_id"].iloc[0] == "20240423_15_30_05_segment_0"
        # The filename-derived form differs and is kept separately for comparison.
        assert frame["segment_id_from_mic"].iloc[0] == "20240423_15_30_05_DataLog_0"

    def test_falls_back_to_filename_only_when_column_absent(self, tmp_path):
        path = tmp_path / "attributes_normal_source_train.csv"
        pd.DataFrame([{
            "imp23absu_mic": "imp23absu_mic_20240423_15_30_05_DataLog_0.parquet"
        }]).to_csv(path, index=False)
        frame, info = load_metadata_file(path)
        assert info["segment_id_source"] == "derived_from_mic_filename"
        assert frame["segment_id"].iloc[0] == "20240423_15_30_05_DataLog_0"


class TestLabelConsistency:
    def test_consistent_metadata_reports_no_mismatch(self, tmp_path):
        root = build_machine(tmp_path)
        layout = scan_machine("TestMachine", root)
        frame, _ = load_machine_metadata(layout.metadata_files)
        result = check_label_consistency(frame)
        assert result["anomaly_mismatch_count"] == 0
        assert result["domain_mismatch_count"] == 0
        assert result["split_mismatch_count"] == 0
        assert result["n_duplicate_segment_ids"] == 0

    def test_detects_split_label_contradicting_its_filename(self, tmp_path):
        """The core validity check: the release's two label sources must agree."""
        root = build_machine(tmp_path)
        path = root / "test" / "attributes_anomaly_source_test.csv"
        frame = pd.read_csv(path, index_col=0)
        frame.loc[frame.index[0], "split_label"] = "Normal_Target_Train"
        frame.to_csv(path)

        layout = scan_machine("TestMachine", root)
        combined, _ = load_machine_metadata(layout.metadata_files)
        result = check_label_consistency(combined)
        assert result["anomaly_mismatch_count"] == 1
        assert result["domain_mismatch_count"] == 1
        assert result["split_mismatch_count"] == 1

    def test_detects_anomaly_label_contradicting_its_filename(self, tmp_path):
        root = build_machine(tmp_path)
        path = root / "test" / "attributes_normal_source_test.csv"
        frame = pd.read_csv(path, index_col=0)
        frame.loc[frame.index[0], "anomaly_label"] = "magnet"
        frame.to_csv(path)

        layout = scan_machine("TestMachine", root)
        combined, _ = load_machine_metadata(layout.metadata_files)
        result = check_label_consistency(combined)
        assert result["anomaly_vs_filename_mismatch_count"] == 1

    def test_detects_duplicate_segment_ids(self, tmp_path):
        root = build_machine(tmp_path)
        path = root / "train" / "attributes_normal_source_train.csv"
        frame = pd.read_csv(path, index_col=0)
        frame.loc[frame.index[1], "imp23absu_mic"] = "imp23absu_mic_s01.parquet"
        frame.to_csv(path)

        layout = scan_machine("TestMachine", root)
        combined, _ = load_machine_metadata(layout.metadata_files)
        result = check_label_consistency(combined)
        assert result["n_duplicate_segment_ids"] == 2
        assert "s01" in result["duplicate_segment_id_examples"]


class TestSensorPathResolution:
    def test_resolves_every_sensor_when_files_exist(self, tmp_path):
        root = build_machine(tmp_path)
        layout = scan_machine("TestMachine", root)
        frame, _ = load_machine_metadata(layout.metadata_files)
        resolved = resolve_sensor_paths(frame, root)
        for key in ("mic", "acc", "gyro"):
            assert resolved[f"{key}_exists"].all()

    def test_missing_sensor_file_is_flagged_not_fabricated(self, tmp_path):
        root = build_machine(tmp_path)
        (root / "train" / "ism330dhcx_gyro_s02.parquet").unlink()
        layout = scan_machine("TestMachine", root)
        frame, _ = load_machine_metadata(layout.metadata_files)
        resolved = resolve_sensor_paths(frame, root)

        missing = resolved.loc[resolved["segment_id"] == "s02"]
        assert not missing["gyro_exists"].iloc[0]
        assert pd.isna(missing["gyro_path"].iloc[0])
        assert missing["mic_exists"].iloc[0]

    def test_metadata_row_with_no_files_at_all(self, tmp_path):
        root = build_machine(tmp_path, make_files=False)
        layout = scan_machine("TestMachine", root)
        frame, _ = load_machine_metadata(layout.metadata_files)
        resolved = resolve_sensor_paths(frame, root)
        assert not resolved["mic_exists"].any()


class TestWarningClassification:
    """Informational release properties must not be counted as data defects."""

    def test_quantisation_is_informational_not_a_defect(self):
        from src.data.audit import classify_warnings

        info, defect = classify_warnings([
            "mic:timestamps_quantised:2_distinct_dt",
            "acc:timestamps_quantised:3_distinct_dt",
        ])
        assert (info, defect) == (2, 0)

    def test_real_problems_count_as_defects(self):
        from src.data.audit import classify_warnings

        info, defect = classify_warnings([
            "mic:timestamps_quantised:2_distinct_dt",
            "acc:nan:A_x [g]:17",
            "gyro:constant_channel:G_z [mdps]",
            "no_path:gyro",
        ])
        assert info == 1
        assert defect == 3
