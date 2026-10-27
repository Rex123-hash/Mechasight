"""Tests for recording-time structure analysis (Q8 / risk R9).

Synthetic segment ids with known temporal layouts. The point is that the
confound detector must fire when a confound exists and stay quiet when it does
not -- a detector that always fires would be as useless as one that never does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.temporal import analyse_temporal_structure, parse_segment_times


def _sid(hour: int, minute: int, second: int = 0, index: int = 0) -> str:
    return f"20240423_{hour:02d}_{minute:02d}_{second:02d}_segment_{index}"


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (segment_id, anomaly_label, file_split)"""
    return pd.DataFrame(
        {
            "machine": "M",
            "segment_id": [r[0] for r in rows],
            "anomaly_label": [r[1] for r in rows],
            "file_split": [r[2] for r in rows],
            "file_domain": "source",
        }
    )


class TestParsing:
    def test_extracts_stamp_and_index(self):
        parsed = parse_segment_times(pd.Series([_sid(15, 30, 5), _sid(9, 1, 2, index=3)]))
        assert parsed["recording_stamp"].tolist() == ["20240423_15_30_05", "20240423_09_01_02"]
        assert parsed["segment_index"].tolist() == [0, 3]
        assert parsed["recording_time"].iloc[0] == pd.Timestamp("2024-04-23 15:30:05")

    def test_accepts_both_released_id_shapes(self):
        """Metadata uses `_segment_N`; sensor filenames use `_DataLog_N`."""
        parsed = parse_segment_times(pd.Series([
            "20240423_15_30_05_segment_0",
            "20240423_15_30_05_DataLog_0",
        ]))
        assert parsed["recording_time"].nunique() == 1
        assert parsed["recording_time"].notna().all()

    def test_accepts_robotic_arm_id_shapes_including_corr_marker(self):
        """RoboticArm ids use `dataLog` and some carry an undocumented `corr` tag.

        Discovered from the release: 2218 plain, 77 `corr#`, 8 `corr_#`.
        A parser that only knew the BrushlessMotor shape silently rejected all
        2303 of them and reported the machine as un-analysable.
        """
        parsed = parse_segment_times(pd.Series([
            "20240418_14_01_48_dataLog_12",
            "20240418_12_46_21_corr3_dataLog_7",
            "20240418_12_46_21_corr_3_dataLog_7",
        ]))
        assert parsed["recording_time"].notna().all()
        assert parsed["segment_index"].tolist() == [12, 7, 7]
        assert parsed["has_corr_marker"].tolist() == [False, True, True]

    def test_unparseable_ids_become_nat_not_an_exception(self):
        parsed = parse_segment_times(pd.Series(["not_a_segment_id", _sid(10, 0)]))
        assert pd.isna(parsed["recording_time"].iloc[0])
        assert not pd.isna(parsed["recording_time"].iloc[1])


class TestConfoundDetection:
    def test_detects_class_blocked_in_time(self):
        """All normals before all anomalies -- the confound must be reported."""
        rows = [(_sid(10, m), "normal", "train") for m in range(0, 30, 2)]
        rows += [(_sid(14, m), "belt", "test") for m in range(0, 20, 2)]
        result = analyse_temporal_structure(_frame(rows))

        assert result["classes_temporally_disjoint"] is True
        assert result["normal_anomaly_overlap_s"] < 0
        assert result["n_normal_after_first_anomaly"] == 0
        assert result["n_anomaly_before_last_normal"] == 0

    def test_interleaved_classes_are_not_flagged_as_disjoint(self):
        """The detector must stay quiet when classes are properly interleaved."""
        rows = []
        for m in range(0, 40, 4):
            rows.append((_sid(10, m), "normal", "train"))
            rows.append((_sid(10, m + 2), "belt", "test"))
        result = analyse_temporal_structure(_frame(rows))

        assert result["classes_temporally_disjoint"] is False
        assert result["normal_anomaly_overlap_s"] > 0
        assert result["n_normal_after_first_anomaly"] > 0
        assert result["n_anomaly_before_last_normal"] > 0

    def test_anomaly_types_blocked_relative_to_each_other(self):
        rows = [(_sid(10, m), "normal", "train") for m in range(0, 20, 2)]
        rows += [(_sid(14, m), "belt", "test") for m in range(0, 10, 2)]
        rows += [(_sid(16, m), "magnetic", "test") for m in range(0, 10, 2)]
        result = analyse_temporal_structure(_frame(rows))

        overlaps = result["anomaly_type_overlap_s"]
        assert len(overlaps) == 1
        assert next(iter(overlaps.values())) < 0


class TestRecordingStructure:
    def test_one_segment_per_recording_is_detected(self):
        rows = [(_sid(10, m), "normal", "train") for m in range(5)]
        result = analyse_temporal_structure(_frame(rows))
        assert result["one_segment_per_recording"] is True
        assert result["n_stamps_reused_across_segments"] == 0

    def test_multiple_segments_per_recording_is_detected(self):
        """Several segments cut from one recording is a parent-recording leak risk."""
        rows = [(_sid(10, 0, index=i), "normal", "train") for i in range(4)]
        result = analyse_temporal_structure(_frame(rows))
        assert result["one_segment_per_recording"] is False
        assert result["segment_index_values"] == [0, 1, 2, 3]
        assert result["n_stamps_reused_across_segments"] == 3

    def test_normal_train_test_interleaving_measured(self):
        rows = []
        for m in range(0, 40, 4):
            rows.append((_sid(10, m), "normal", "train"))
            rows.append((_sid(10, m + 2), "normal", "test"))
        result = analyse_temporal_structure(_frame(rows))
        assert result["normal_train_test_overlap_s"] > 0
        assert result["normal_train_test_alternating_blocks"] == 20

    def test_session_span_reported(self):
        rows = [(_sid(10, 0), "normal", "train"), (_sid(18, 0), "belt", "test")]
        result = analyse_temporal_structure(_frame(rows))
        assert result["session_hours"] == pytest.approx(8.0)
        assert result["recording_dates"] == ["2024-04-23"]

    def test_all_unparseable_reports_not_analysable(self):
        frame = _frame([("garbage", "normal", "train")])
        result = analyse_temporal_structure(frame)
        assert result["analysable"] is False
        assert result["n_unparsed_segment_ids"] == 1


class TestParentRecordingLeakage:
    """Risk R9: several segments cut from one recording must not cross splits.

    When one recording yields many segments, splitting by segment id is not
    enough -- neighbouring segments from the same recording are near-duplicates.
    """

    def _multi(self, rows):
        return _frame(rows)

    def test_recording_straddling_train_and_test_is_flagged(self):
        rows = [
            ("20240418_10_00_00_dataLog_0", "normal", "train"),
            ("20240418_10_00_00_dataLog_1", "normal", "test"),
            ("20240418_11_00_00_dataLog_0", "normal", "train"),
        ]
        result = analyse_temporal_structure(self._multi(rows))
        assert result["n_recordings_spanning_train_and_test"] == 1
        assert "20240418_10_00_00" in result["recordings_spanning_splits_examples"]
        assert result["n_segments_in_straddling_recordings"] == 2

    def test_clean_recording_level_split_is_not_flagged(self):
        rows = [
            ("20240418_10_00_00_dataLog_0", "normal", "train"),
            ("20240418_10_00_00_dataLog_1", "normal", "train"),
            ("20240418_11_00_00_dataLog_0", "normal", "test"),
            ("20240418_11_00_00_dataLog_1", "normal", "test"),
        ]
        result = analyse_temporal_structure(self._multi(rows))
        assert result["n_recordings_spanning_train_and_test"] == 0
        assert result["n_segments_in_straddling_recordings"] == 0

    def test_recording_containing_both_classes_is_flagged(self):
        rows = [
            ("20240418_10_00_00_dataLog_0", "normal", "test"),
            ("20240418_10_00_00_dataLog_1", "loosescrewsA", "test"),
        ]
        result = analyse_temporal_structure(self._multi(rows))
        assert result["n_recordings_spanning_multiple_labels"] == 1

    def test_segments_per_recording_statistics(self):
        rows = [(f"20240418_10_00_00_dataLog_{i}", "normal", "train") for i in range(5)]
        rows += [("20240418_11_00_00_dataLog_0", "normal", "train")]
        result = analyse_temporal_structure(self._multi(rows))
        assert result["segments_per_recording"]["max"] == 5
        assert result["segments_per_recording"]["min"] == 1
        assert result["segments_per_recording"]["n_recordings"] == 2
        assert result["one_segment_per_recording"] is False
