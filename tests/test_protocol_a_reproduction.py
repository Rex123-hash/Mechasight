"""Phase-3.3: Protocol-A fidelity to the pinned upstream IMAD-DS baseline.

Every expected value here is derived from the upstream source at commit
592bcbbed58893dc26bbab3240c5d7a3105a1a7f, by hand or by an independent
formulation, so the tests do not merely restate the implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.auc import official_pairwise_auc
from src.experiment.ledger_history import verify_ledger_history
from src.reproduction.protocol_a import (
    GYROSCOPE_WARM_UP_TIME,
    LABEL_NAMES,
    OFFICIAL_PARAMS,
    OFFICIAL_SENSOR_SPEC,
    SENSORS,
    UPSTREAM_COMMIT,
    UPSTREAM_DEFECTS,
    WINDOW_SIZE,
    group_by_segment_id_median,
    mse_per_sample,
    official_segment_bounds,
    official_single_auc,
    official_window,
    official_window_length,
    sensor_slices,
    standardize_window,
)

CONFIG = Path("configs/phase3/protocol_a_official_reproduction_v1.json")
SPLIT_V3 = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
PROTOCOL_B_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")


def _frame(times_s, values):
    f = pd.DataFrame({"Time": pd.to_datetime(np.asarray(times_s), unit="s")})
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    for i in range(arr.shape[1]):
        f[f"c{i}"] = arr[:, i]
    return f.set_index("Time").sort_index()


# ---------------------------------------------- fixed upstream constants
def test_official_rate_constants_are_the_upstream_fixed_values():
    """16000/7063/7063 -- NOT the measured machine-specific effective rates."""

    assert OFFICIAL_SENSOR_SPEC["imp23absu_mic"]["fs"] == 16000
    assert OFFICIAL_SENSOR_SPEC["ism330dhcx_acc"]["fs"] == 7063
    assert OFFICIAL_SENSOR_SPEC["ism330dhcx_gyro"]["fs"] == 7063
    # the measured effective rates must never leak into Protocol A
    for measured in (7063.294, 7135.716, 7135.331, 16003.077):
        assert measured not in {s["fs"] for s in OFFICIAL_SENSOR_SPEC.values()}


def test_window_lengths_follow_int_fs_times_one_tenth_second():
    assert official_window_length("imp23absu_mic") == 1600
    assert official_window_length("ism330dhcx_acc") == 706
    assert official_window_length("ism330dhcx_gyro") == 706
    # int() truncation, not rounding: 7063 * 0.1 = 706.3 -> 706
    assert int(7063 * 0.1) == 706


def test_channel_counts_and_sensor_order_match_upstream():
    assert SENSORS == ("imp23absu_mic", "ism330dhcx_acc", "ism330dhcx_gyro")
    assert [OFFICIAL_SENSOR_SPEC[s]["number_of_channel"] for s in SENSORS] == [1, 3, 3]
    assert LABEL_NAMES == ("segment_id", "split_label", "anomaly_label",
                           "domain_shift_op", "domain_shift_env")


def test_upstream_training_hyperparameters_are_pinned():
    assert OFFICIAL_PARAMS["layer_dims"] == [2048, 2048, 2048, 16]
    assert OFFICIAL_PARAMS["lr"] == 0.0001
    assert OFFICIAL_PARAMS["batch_size"] == 1024
    assert OFFICIAL_PARAMS["num_epochs"] == 1000
    assert OFFICIAL_PARAMS["patience"] == 3
    assert OFFICIAL_PARAMS["valid_size"] == 0.1
    assert OFFICIAL_PARAMS["seed"] == 1995
    assert OFFICIAL_PARAMS["normalisation"] == "std_window"
    assert OFFICIAL_PARAMS["criterion"] == "MSE"


def test_gyro_warmup_and_window_size_constants():
    assert GYROSCOPE_WARM_UP_TIME == pd.to_timedelta("35ms")
    assert WINDOW_SIZE == pd.to_timedelta("100ms")


# ------------------------------------------------ right-inclusive slicing
def test_window_slicing_is_right_inclusive():
    """pandas label slicing includes the right edge, so windows share a sample."""

    times = np.arange(0, 0.25, 0.05)          # 0.00 .. 0.20 s
    frame = _frame(times, np.arange(len(times), dtype=float))
    start = pd.to_datetime(0.0, unit="s")
    sliced = frame[start:start + WINDOW_SIZE]
    # 0.00, 0.05 and the right edge 0.10 are all included -> 3 rows, not 2
    assert len(sliced) == 3
    assert list(sliced.iloc[:, 0]) == [0.0, 1.0, 2.0]
    # the boundary sample is shared with the next window
    nxt = frame[start + WINDOW_SIZE:start + 2 * WINDOW_SIZE]
    assert nxt.iloc[0, 0] == sliced.iloc[-1, 0] == 2.0


def test_short_window_is_zero_padded_to_the_fixed_length():
    times = np.arange(0, 0.02, 0.01)
    frame = _frame(times, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = official_window(frame, pd.to_datetime(0.0, unit="s"), "ism330dhcx_acc")
    assert out.shape == (3, 706)
    assert np.array_equal(out[:, 0], [1.0, 2.0, 3.0])
    assert np.array_equal(out[:, 1], [4.0, 5.0, 6.0])
    assert np.all(out[:, 2:] == 0.0)          # zero padded, not edge padded


def test_long_window_is_truncated_not_resampled():
    n = 3000
    times = np.linspace(0, 0.1, n)
    frame = _frame(times, np.arange(n, dtype=float))
    out = official_window(frame, pd.to_datetime(0.0, unit="s"), "imp23absu_mic")
    assert out.shape == (1, 1600)
    # first 1600 raw samples kept in order; no interpolation
    assert np.array_equal(out[0], np.arange(1600, dtype=float))


def test_segment_bounds_apply_the_gyro_warm_up_and_common_interval():
    mic = _frame(np.arange(0, 1.0, 0.001), np.zeros(1000))
    acc = _frame(np.arange(0.010, 0.900, 0.001), np.zeros((890, 3)))
    gyro = _frame(np.arange(0.000, 0.950, 0.001), np.zeros((950, 3)))
    start, end, n = official_segment_bounds(
        {"imp23absu_mic": mic, "ism330dhcx_acc": acc, "ism330dhcx_gyro": gyro})
    # gyro first (0.000) + 35 ms = 0.035 beats max(first) = 0.010
    assert start == pd.to_datetime(0.035, unit="s")
    assert end == acc.index[-1]               # min of the last timestamps
    assert n == int((end - start) // WINDOW_SIZE)
    assert n == 8


# ----------------------------------------------------- standardisation
def test_standardize_window_puts_epsilon_in_the_denominator():
    """The published path is std_window, where the epsilon is applied correctly."""

    data = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    out = standardize_window(data.copy())
    mean, std = 2.5, np.std([1.0, 2.0, 3.0, 4.0])
    expected = (np.array([1.0, 2.0, 3.0, 4.0]) - mean) / (std + 1e-5)
    assert np.allclose(out[0, 0], expected)


def test_standardize_window_is_per_window_and_per_channel():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(5, 3, 40)) * 10 + 4
    out = standardize_window(data.copy())
    # each (sample, channel) row is standardised on its own statistics
    assert np.allclose(out.mean(axis=2), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=2), 1.0, atol=1e-3)


def test_constant_window_does_not_divide_by_zero():
    data = np.full((1, 1, 8), 7.0)
    out = standardize_window(data.copy())
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


# ------------------------------------------------- reconstruction error
def test_mse_is_per_sample_mean_over_features():
    actual = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    estimate = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    got = mse_per_sample(actual, estimate)
    assert got.shape == (2,)
    assert got[0] == pytest.approx(1.0)       # mean of (1,1,1)
    assert got[1] == pytest.approx(0.0)


def test_sensor_slices_follow_the_concatenation_order():
    lengths = [official_window_length(s) for s in SENSORS]
    channels = [OFFICIAL_SENSOR_SPEC[s]["number_of_channel"] for s in SENSORS]
    bounds = sensor_slices(lengths, channels)
    assert bounds == [(0, 1600), (1600, 3718), (3718, 5836)]
    assert bounds[-1][1] == 1600 * 1 + 706 * 3 + 706 * 3 == 5836


# ------------------------------------------------- median aggregation
def test_segment_score_is_the_median_of_window_scores():
    labels = pd.DataFrame({
        "segment_id": ["a", "a", "a", "b", "b"],
        "split_label": ["Normal_Source_Test"] * 5,
        "anomaly_label": ["normal"] * 5,
        "domain_shift_op": ["w0"] * 5,
        "domain_shift_env": ["BckgA"] * 5,
        "total_loss": [1.0, 100.0, 3.0, 5.0, 7.0],
    })
    grouped = group_by_segment_id_median(labels, ["total_loss"])
    got = dict(zip(grouped.segment_id, grouped.total_loss))
    assert got["a"] == 3.0                     # median, so the 100 outlier does not win
    assert got["b"] == 6.0
    assert list(grouped.columns[:1]) == ["segment_id"]


# ------------------------------------------------------------- AUC
def test_official_auc_gives_ties_zero_credit():
    """Strict `normal < anomaly`; an exact tie contributes 0, not 0.5."""

    assert official_pairwise_auc([1.0], [1.0]) == 0.0
    assert official_pairwise_auc([1.0], [2.0]) == 1.0
    assert official_pairwise_auc([2.0], [1.0]) == 0.0
    # half the pairs tied -> upstream 0.5 where sklearn ROC AUC would give 0.75
    assert official_pairwise_auc([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.75)
    assert official_pairwise_auc([0.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_official_auc_matches_an_independent_double_loop():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = rng.normal(size=int(rng.integers(2, 25)))
        a = rng.normal(size=int(rng.integers(2, 25)))
        expected = sum(int(x < y) for x in n for y in a) / (len(n) * len(a))
        assert official_pairwise_auc(n, a) == pytest.approx(expected, abs=1e-12)


def test_official_single_auc_reports_all_source_and_target_slices():
    frame = pd.DataFrame({
        "segment_id": list("abcdefgh"),
        "split_label": ["Normal_Source_Test", "Anomaly_Source_Test"] * 2 +
                       ["Normal_Target_Test", "Anomaly_Target_Test"] * 2,
        "anomaly_label": ["normal", "loose"] * 4,
        "total_loss": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    })
    out = official_single_auc(frame, "total_loss")
    assert set(out) == {"all", "source", "target"}
    assert out["source"] == pytest.approx(1.0)
    assert out["target"] == pytest.approx(1.0)
    assert out["all"] == pytest.approx(
        official_pairwise_auc([0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]))


def test_official_single_auc_requires_both_classes_in_a_slice():
    frame = pd.DataFrame({
        "split_label": ["Normal_Source_Test", "Normal_Target_Test"],
        "anomaly_label": ["normal", "normal"],
        "total_loss": [0.1, 0.2],
    })
    with pytest.raises(ValueError, match="lacks one of the two classes"):
        official_single_auc(frame, "total_loss")


def test_higher_score_means_more_anomalous():
    """Reconstruction error is the score, so anomalies must score higher."""

    frame = pd.DataFrame({
        "split_label": ["Normal_Source_Test", "Anomaly_Source_Test",
                        "Normal_Target_Test", "Anomaly_Target_Test"],
        "anomaly_label": ["normal", "loose", "normal", "loose"],
        "total_loss": [0.01, 5.0, 0.02, 6.0],
    })
    out = official_single_auc(frame, "total_loss")
    assert out["source"] == 1.0 and out["target"] == 1.0 and out["all"] == 1.0

    # inverted orientation must score 0, proving the direction is not symmetric
    inverted = frame.assign(total_loss=[5.0, 0.01, 6.0, 0.02])
    assert official_single_auc(inverted, "total_loss")["all"] == 0.0


# ------------------------------------- no Protocol-B contamination
def test_protocol_a_module_does_not_import_protocol_b_machinery():
    source = Path("src/reproduction/protocol_a.py").read_text(encoding="utf-8")
    body = source.split("FIDELITY RULES")[1].split('"""', 1)[1]
    for forbidden in ("src.preprocessing.splits", "src.experiment.phase31",
                      "src.experiment.phase32", "src.models.classical",
                      "IsolationForest", "OneClassSVM", "few_shot_adaptation",
                      "protocol_b", "common_evaluation", "q90", "quantile"):
        assert forbidden not in body, forbidden


def test_upstream_defects_are_declared_and_reproduced():
    ids = {d["id"] for d in UPSTREAM_DEFECTS}
    assert "best_model_state_is_a_live_reference" in ids
    assert "right_inclusive_window_slicing" in ids
    assert "window_level_train_validation_split" in ids
    assert "fixed_imu_rate_constant" in ids
    for defect in UPSTREAM_DEFECTS:
        assert defect["reproduced"].startswith(("yes", "behaviour preserved"))


# ------------------------------------------------- config identity
def test_protocol_a_config_exists_and_pins_the_upstream_commit():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["upstream"]["commit_sha"] == UPSTREAM_COMMIT
    assert cfg["upstream"]["repository"].endswith("IMAD-DS_baseline")
    assert cfg["protocol"] == "official_baseline_reproduction"
    assert cfg["regime"] == "T"
    # published reference values are recorded as non-tuning references
    assert cfg["published_reference_values"]["status"] == "REFERENCE_ONLY_NOT_A_TUNING_TARGET"
    assert cfg["published_reference_values"]["RoboticArm"]["overall"] == 91.62
    assert cfg["published_reference_values"]["BrushlessMotor"]["overall"] == 58.95
    # the RA overlap disclosure must be present and explicit
    disclosure = cfg["robotic_arm_overlap_disclosure"]
    assert disclosure["parent_recordings"] == 101
    assert disclosure["parents_crossing_official_split"] == 62
    assert disclosure["normal_test_segments_with_training_sibling"] == 212
    assert disclosure["retained_deliberately"] is True


def test_protocol_a_config_identity_is_deterministic():
    from src.experiment.provenance import content_sha256

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    recorded = cfg.pop("config_identity")
    assert recorded == f"p33-protocol-a-{content_sha256(cfg)[:24]}"


def test_protocol_a_uses_the_released_split_not_the_protocol_b_split():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["split"]["source"] == "released_official_train_test_folders"
    assert cfg["split"]["parent_disjoint"] is False
    assert "protocol_b_few_shot_adaptation_v3" not in json.dumps(cfg)


# --------------------------------------- Protocol-B evidence untouched
def test_protocol_b_ledger_still_has_its_original_28_entries():
    """Protocol A only appended; the 28 Phase-3.2 Protocol-B looks it found are intact.

    Stage note: later Protocol-B looks (Single-Modality Benchmarking) belong to
    their own authorised batch, so the original 28 are identified by the
    Phase-3.2 evaluator SHA rather than by counting the protocol value.
    """
    history = verify_ledger_history(PROTOCOL_B_LEDGER)
    original = [e for e in history.rows
                if e["git_sha"] == "99f6055156c314bbf26a847713d26792979fd050"]
    assert original == list(history.rows[:28])
    assert {e["protocol"] for e in original} == {"mechasight_experimental_protocol"}
    protocol_a = history.segment("phase3_3_protocol_a_reproduction")
    assert (protocol_a.start, protocol_a.end) == (28, 30)


def test_protocol_b_split_manifest_is_unchanged():
    digest = hashlib.sha256(SPLIT_V3.read_bytes()).hexdigest()
    assert digest == "68e5406d63e96cf5ee418ae58808f5c3e7790889ba1b8497420642d1f0d8409e"


def test_phase32_artifact_hashes_are_unchanged():
    manifest = json.loads(
        Path("reports/phase3_2/artifact_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for name, expected in manifest["files"].items():
        path = Path(name) if Path(name).exists() else Path("reports/phase3_2") / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        checked += 1
    assert checked == 155


# ------------------------------------------- ledger pre-score requirement
def test_evaluator_refuses_to_score_without_a_protocol_a_ledger_entry(tmp_path, monkeypatch):
    """Scoring the official test set must be gated on a durable ledger entry."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_p33_runner", Path("scripts/run_protocol_a_reproduction.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    empty = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(runner, "LEDGER", empty)
    assert runner._ledger_has_protocol_a("RoboticArm", "p33-protocol-a-anything") is False

    # a Protocol-B entry must not authorise a Protocol-A test look
    empty.write_text(json.dumps({
        "protocol": "mechasight_experimental_protocol", "machine": "RoboticArm",
        "model_config_identity": "p33-protocol-a-anything"}) + "\n", encoding="utf-8")
    assert runner._ledger_has_protocol_a("RoboticArm", "p33-protocol-a-anything") is False

    # the matching Protocol-A entry does authorise it
    with empty.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "protocol": "official_baseline_reproduction", "machine": "RoboticArm",
            "model_config_identity": "p33-protocol-a-anything"}) + "\n")
    assert runner._ledger_has_protocol_a("RoboticArm", "p33-protocol-a-anything") is True
    # ... but only for the machine and config it names
    assert runner._ledger_has_protocol_a("BrushlessMotor", "p33-protocol-a-anything") is False
    assert runner._ledger_has_protocol_a("RoboticArm", "p33-protocol-a-other") is False


def test_protocol_a_ledger_entries_use_the_distinct_protocol_value():
    """Protocol-A test looks must never be indistinguishable from Protocol-B ones."""

    entries = [json.loads(l) for l in
               PROTOCOL_B_LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    protocols = {e.get("protocol") for e in entries}
    assert "mechasight_experimental_protocol" in protocols
    # Stage note: the Phase-3.4 protocol value joined the ledger at step D (c10fdfc);
    # Protocol-A entries remain distinguishable from every other protocol.
    assert protocols <= {"mechasight_experimental_protocol", "official_baseline_reproduction",
                         "roboticarm_focal_parent_sibling_toggle_v1"}
    for entry in entries:
        if entry.get("protocol") == "official_baseline_reproduction":
            assert entry["regime"] == "T"
            assert entry["reason"].startswith("first locked Protocol-A")
            assert entry["model_config_identity"].startswith("p33-protocol-a-")
