"""Phase 5.4 sensor-integrity-aware fusion (L3): frozen gates, fallback decision, corruption generator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fusion import l3
from src.fusion import l3_falsification as falsification

RECORDS = Path("reports/phase5_development/l2/l2_calibration_records.json")


# ------------------------------------------------------------------ G2 flatline
def test_flatline_run_qualifies_at_20_ms_measured_from_the_streams_own_timestamps():
    t = np.arange(100) * 0.001                       # 1 kHz
    x = np.arange(100, dtype=np.float64)
    x[10:31] = 5.0                                   # 21 samples, t_last - t_first = 0.020 s -> qualifies
    x[50:70] = 7.0                                   # 20 samples, 0.019 s -> does not qualify
    fraction, longest = l3.flatline_profile(t, x)
    assert fraction == pytest.approx(21 / 100)
    assert longest == pytest.approx(0.020)


def test_irregular_timestamps_are_used_not_a_nominal_period():
    t = np.array([0.0, 0.001, 0.002, 0.030, 0.031])  # two samples 28 ms apart
    x = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    assert l3.flatline_profile(t, x) == (pytest.approx(2 / 5), pytest.approx(0.028))


def test_nan_breaks_a_run_and_a_constant_channel_is_fully_flat():
    t = np.arange(50) * 0.001
    x = np.full(50, 2.0)
    assert l3.flatline_profile(t, x)[0] == 1.0
    x[25] = np.nan
    assert l3.flatline_profile(t, x)[0] == pytest.approx(49 / 50)   # two 24-sample runs of 0.024 s


def _signal(values: np.ndarray, rate: float = 1000.0, channels=("x", "y")) -> l3.SensorSignal:
    t = np.arange(len(values)) / rate
    return l3.SensorSignal("acc", t, values, tuple(channels), len(values), len(values))


def test_g1_and_g2_verdicts_inside_the_common_coverage_only():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1000, 2))
    assert l3.g1_g2_values(_signal(values), 0.1, 0.9)["verdict"] == l3.VALID
    bad = values.copy()
    bad[50, 0] = np.nan                                     # outside [0.1, 0.9)
    assert l3.g1_g2_values(_signal(bad), 0.1, 0.9)["verdict"] == l3.VALID
    bad[500, 1] = np.inf
    assert l3.g1_g2_values(_signal(bad), 0.1, 0.9)["verdict"] == l3.INVALID_NON_FINITE
    flat = values.copy()
    flat[100:500, 1] = 0.0                                  # exactly 50 % of the 800 coverage samples
    assert l3.g1_g2_values(_signal(flat), 0.1, 0.9)["verdict"] == l3.INVALID_FLATLINE
    partial = values.copy()
    partial[100:499, 1] = 0.0                               # 49.9 %: below the frozen 0.50
    out = l3.g1_g2_values(_signal(partial), 0.1, 0.9)
    assert out["verdict"] == l3.VALID and out["flatline_per_channel"]["y"]["flatline_fraction"] < 0.5


# ------------------------------------------------------------------ G3 timing / D absence
def _write_stream(path: Path, t: np.ndarray, n_channels: int = 3) -> Path:
    frame = pd.DataFrame({"Time": t, **{f"c{i}": np.sin(t * (i + 1)) for i in range(n_channels)}})
    frame.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("mutate, reason", [
    (lambda t: np.concatenate([t[:100], [t[101], t[100]], t[102:]]), "non-increasing or duplicate timestamp"),
    (lambda t: np.concatenate([t[:101], [t[100]], t[102:]]), "non-increasing or duplicate timestamp"),
    (lambda t: np.where(np.arange(len(t)) == 100, np.nan, t), "non-finite timestamp"),
    (lambda t: t[t <= t[0] + 0.5 * (t[-1] - t[0])], "coverage below 0.90 nominal"),
])
def test_g3_reads_the_stored_order_before_the_frozen_reader_could_sort_it(tmp_path, mutate, reason):
    t = np.arange(0, 5.2, 1 / 1000)
    clean, _ = l3.load_raw_stream(_write_stream(tmp_path / "clean.parquet", t), "acc")
    assert l3.g3_timing(clean, "BrushlessMotor")["passed"]
    broken, _ = l3.load_raw_stream(_write_stream(tmp_path / "bad.parquet", mutate(t)), "acc")
    verdict = l3.g3_timing(broken, "BrushlessMotor")
    assert not verdict["passed"] and reason in verdict["reasons"]


def test_a_larger_reversed_section_is_caught_although_the_frozen_reader_would_sort_it_away(tmp_path):
    t = np.arange(0, 5.2, 1 / 1000)
    t_rev = t.copy()
    t_rev[1000:1500] = t[1000:1500][::-1]
    path = _write_stream(tmp_path / "reversed.parquet", t_rev)
    stream, _ = l3.load_raw_stream(path, "acc")
    assert not l3.g3_timing(stream, "BrushlessMotor")["passed"]
    # The frozen reader silently restores the order: the post-sort view looks clean.
    sorted_view = l3.read_sensor_parquet(path, "acc")
    assert np.all(np.diff(sorted_view.timestamps) > 0)


def test_a_nan_timestamp_tail_fails_g3_although_the_frozen_signal_type_admits_padding(tmp_path):
    # SensorSignal admits NaN-timestamp *padding* at the tail: that is the Protocol-A window
    # representation produced by _official_shape, never content of a raw IMAD-DS file (the dataset
    # audit and the clean scan find no non-finite raw timestamp).  In a raw file it is a G3 failure.
    t = np.arange(0, 5.2, 1 / 1000)
    t_tail = t.copy()
    t_tail[-3:] = np.nan
    path = _write_stream(tmp_path / "tail.parquet", t_tail)
    assert np.isnan(l3.read_sensor_parquet(path, "acc").timestamps[-1])     # accepted post-sort
    stream, _ = l3.load_raw_stream(path, "acc")
    assert l3.g3_timing(stream, "BrushlessMotor")["reasons"] == ["non-finite timestamp"]


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"x": [1.0, 2.0]}),                                              # no timestamp column
    pd.DataFrame({"Time": [0.0, 1.0], "t": [0.0, 1.0], "x": [1.0, 2.0]}),          # two timestamp columns
    pd.DataFrame({"Time": ["a", "b"], "x": [1.0, 2.0]}),                          # non-numeric timestamps
    pd.DataFrame({"Time": pd.Series([], dtype=float), "x": pd.Series([], dtype=float)}),  # no samples
    pd.DataFrame({"Time": [0.0, 1.0]}),                                           # no channels
])
def test_malformed_streams_are_type_d_not_an_exception(tmp_path, frame):
    path = tmp_path / "malformed.parquet"
    frame.to_parquet(path, index=False)
    stream, reason = l3.load_raw_stream(path, "acc")
    assert stream is None and reason


def test_a_truncated_file_is_type_d_and_a_short_stream_fails_coverage(tmp_path):
    full = _write_stream(tmp_path / "full.parquet", np.arange(0, 5.2, 1 / 1000))
    cut = tmp_path / "cut.parquet"
    cut.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    assert l3.load_raw_stream(cut, "acc")[1].startswith("unreadable")
    one, _ = l3.load_raw_stream(_write_stream(tmp_path / "one.parquet", np.array([0.0])), "acc")
    assert l3.g3_timing(one, "BrushlessMotor")["reasons"] == ["coverage below 0.90 nominal"]


def test_any_c_or_d_failure_abstains_for_the_whole_segment_with_every_modality_accounted(tmp_path):
    from src.fusion.assembly import PREPROCESSING_CONFIGS
    config = l3.load_regime_configs(PREPROCESSING_CONFIGS)
    t = np.arange(0, 5.2, 1 / 1000)
    paths = {m: _write_stream(tmp_path / f"{m}.parquet", t, 1 if m == "mic" else 3) for m in l3.MODALITIES}
    t_bad = t.copy()
    t_bad[[200, 201]] = t[[201, 200]]
    paths["gyro"] = _write_stream(tmp_path / "gyro_bad.parquet", t_bad)
    gate = l3.run_gate({**paths, "mic": None}, "BrushlessMotor", config)
    assert gate.alignment_failure and gate.integrity_flag and gate.survivors == ()
    assert gate.verdicts == {"mic": l3.FAIL_ABSENT, "acc": l3.NOT_EVALUATED, "gyro": l3.FAIL_TIMING}


def test_absent_or_unreadable_streams_are_type_d(tmp_path):
    assert l3.load_raw_stream(None, "mic") == (None, "file absent")
    assert l3.load_raw_stream(tmp_path / "missing.parquet", "mic")[1] == "file absent"
    (tmp_path / "junk.parquet").write_bytes(b"not parquet")
    assert l3.load_raw_stream(tmp_path / "junk.parquet", "mic")[1].startswith("unreadable")


# ------------------------------------------------------------------ cohort guard and selection
def test_development_rows_never_admit_a_protected_segment():
    catalog = pd.DataFrame({"machine": ["M"] * 5, "segment_id": list("abcde"), "parent_recording": list("ppqqr")})
    assignments = pd.DataFrame({
        "machine": ["M"] * 10, "segment_id": list("abcde") * 2, "regime": ["S"] * 5 + ["T"] * 5,
        "split": ["train", "validation", "test", "reserved_adaptation", "blocked_sibling",
                  "train", "validation", "test", "train", "blocked_sibling"]})
    rows = l3.development_segment_rows(catalog, assignments)
    assert list(rows["segment_id"]) == ["a", "b", "d"]
    assert set(rows["split_t"]) == {"train", "validation"}


def test_segment_selection_follows_the_design_rule():
    ids = [f"s{i:03d}" for i in range(186)]
    rows = pd.DataFrame({"machine": "RoboticArm", "segment_id": ids, "split_t": "validation"})
    picks = l3.selected_validation_segments(rows, "RoboticArm")
    assert picks == tuple(ids[round(k * 185 / 11)] for k in range(12))
    assert picks[0] == "s000" and picks[-1] == "s185"


# ------------------------------------------------------------------ decision policy
@pytest.fixture(scope="module")
def records():
    return l3.load_l2_records(RECORDS)


def _unit():
    from src.fusion.assembly import load_design_registry
    return load_design_registry()[0]


def _bank(unit, thresholds):
    bundles = {unit["phase4_single_modality_ids"][m]: SimpleNamespace(threshold=SimpleNamespace(value=v))
               for m, v in thresholds.items()}
    return SimpleNamespace(bundles=bundles)


def test_fallback_uses_the_survivor_subset_threshold_and_never_compares_robust_z_with_a_raw_threshold(records):
    unit = _unit()
    key = l3.unit_key(unit)
    bank = _bank(unit, {"mic": 1.0, "acc": 2.0, "gyro": 3.0})
    scores = {"mic": 0.9, "acc": 5.0, "gyro": 1.1}
    full = l3.decide_unit(unit, ("mic", "acc", "gyro"), scores, records, bank, False)
    assert full["status"] == l3.DECISION_ALL
    assert full["threshold_source"] == records[(key, ("mic", "acc", "gyro"))]["calibration_id"]
    pair = l3.decide_unit(unit, ("mic", "gyro"), {"mic": 0.9, "gyro": 1.1}, records, bank, False)
    assert pair["status"] == l3.DECISION_PAIR
    assert pair["threshold"] == records[(key, ("mic", "gyro"))]["threshold"]["value"]
    single = l3.decide_unit(unit, ("acc",), {"acc": 5.0}, records, bank, False)
    assert single["status"] == l3.DECISION_SINGLE and single["score"] == 5.0 and single["threshold"] == 2.0
    assert single["anomaly"] is True
    below = l3.decide_unit(unit, ("acc",), {"acc": 2.0}, records, bank, False)   # score > threshold, not >=
    assert below["anomaly"] is False


def test_abstentions_carry_no_decision(records):
    unit = _unit()
    bank = _bank(unit, {})
    none = l3.decide_unit(unit, (), {}, records, bank, False)
    assert none["status"] == l3.ABSTAIN_SENSOR_INVALID and none["anomaly"] is None and none["score"] is None
    aligned = l3.decide_unit(unit, ("mic",), {"mic": 1.0}, records, bank, True)
    assert aligned["status"] == l3.ABSTAIN_ALIGNMENT_FAILURE and aligned["anomaly"] is None


def test_l3_on_clean_scores_is_exactly_l2_all_three(records):
    from src.fusion.l2 import fuse, robust_z
    unit = _unit()
    record = records[(l3.unit_key(unit), ("mic", "acc", "gyro"))]
    scores = {"mic": 0.71, "acc": 1.9, "gyro": 0.4}
    out = l3.decide_unit(unit, ("mic", "acc", "gyro"), scores, records, _bank(unit, {}), False)
    calibrated = {m: robust_z(np.array([scores[m]]), l3._calibration(record, m)) for m in scores}
    assert out["score"] == float(fuse(calibrated, ("mic", "acc", "gyro"))[0])


@pytest.mark.parametrize("pair", [("acc", "gyro"), ("mic", "gyro"), ("mic", "acc")])
def test_l3_pair_fallback_matches_l2_pair_scores(records, pair):
    """The fallback selects the same calibrated pair score and threshold as L2."""

    from src.fusion.l2 import ModalityCalibration, fuse, robust_z

    unit = _unit()
    record = records[(l3.unit_key(unit), pair)]
    calibration = {m: ModalityCalibration(**record["modality_calibration"][m]) for m in pair}
    # Synthetic probes span both sides of the operating threshold. No detector or raw file is used.
    z_probes = np.array([[-3.0, -1.0], [0.125, 2.75], [5.0, 7.0]])
    raw = {m: calibration[m].median + calibration[m].iqr * z_probes[:, i]
           for i, m in enumerate(pair)}
    expected = fuse({m: robust_z(raw[m], calibration[m]) for m in pair}, pair)
    for i, expected_score in enumerate(expected):
        decision = l3.decide_unit(
            unit, tuple(reversed(pair)), {m: float(raw[m][i]) for m in pair},
            records, _bank(unit, {}), False)
        assert decision["status"] == l3.DECISION_PAIR
        assert decision["score"] == pytest.approx(float(expected_score), rel=0, abs=1e-12)
        assert decision["threshold"] == record["threshold"]["value"]
        assert decision["threshold_source"] == record["calibration_id"]
        assert decision["anomaly"] == bool(expected_score > record["threshold"]["value"])


# ------------------------------------------------------------------ scenarios and corruption
def test_the_catalogue_is_the_frozen_83_runs_per_segment():
    scenarios = falsification.scenario_catalogue()
    assert len(scenarios) == 83 and 83 * 12 * 2 == 1992
    codes = [s.code for s in scenarios]
    assert codes.count("M1") == 3 and codes.count("M2") == 1 and codes.count("CLEAN") == 1
    by_code = {s.scenario_id: s for s in scenarios}
    assert by_code["B4|acc"].expected_status == l3.DECISION_ALL
    assert by_code["A2|gyro"].expected_verdicts["gyro"] == l3.INVALID_NON_FINITE
    assert by_code["D1|mic"].expected_verdicts == {"mic": l3.FAIL_ABSENT, "acc": l3.NOT_EVALUATED,
                                                   "gyro": l3.NOT_EVALUATED}
    assert all(s.expected_status is None for s in scenarios if s.code.startswith("S"))


def _reference(tmp_path):
    t = np.arange(0, 7.5, 1 / 1000)
    frame = pd.DataFrame({"Time": t, "A_x [g]": np.sin(t), "A_y [g]": np.cos(t), "A_z [g]": t})
    path = tmp_path / "acc.parquet"
    frame.to_parquet(path, index=False)
    stats = pd.DataFrame({"machine": "RoboticArm", "sensor": "acc", "channel": ["A_x [g]", "A_y [g]", "A_z [g]"],
                          "training_sd": [1.0, 2.0, 3.0], "training_signed_extreme": [-4.0, 5.0, 6.0],
                          "training_max_abs": [4.0, 5.0, 6.0]})
    reference = falsification.CleanReference("RoboticArm", "seg", {"acc": str(path)}, (1.0, 7.0))
    return frame, stats, reference


@pytest.mark.parametrize("code, fraction", [("A3", 0.10), ("B3", 0.60), ("B4", 0.40), ("B5", 0.60)])
def test_partial_corruptions_are_centred_in_the_coverage(tmp_path, code, fraction):
    frame, stats, reference = _reference(tmp_path)
    out, _ = falsification.corrupt_frame(frame, code=code, sensor="acc", reference=reference,
                                         scenario_id=f"{code}|acc", stats=stats)
    changed = out["A_x [g]"].to_numpy() != frame["A_x [g]"].to_numpy()
    t = frame["Time"].to_numpy()
    lo, hi = 1.0 + (1 - fraction) / 2 * 6.0, 1.0 + (1 + fraction) / 2 * 6.0
    inside = (t >= lo) & (t < hi)
    assert not changed[~inside].any()
    assert changed[inside].mean() > 0.99


def test_value_corruptions_never_touch_samples_outside_the_coverage_or_the_timestamps(tmp_path):
    frame, stats, reference = _reference(tmp_path)
    outside = (frame["Time"] < 1.0) | (frame["Time"] >= 7.0)
    for code in ("A1", "B1", "B2", "N1", "N4", "N5", "N8", "N9", "S4"):
        out, _ = falsification.corrupt_frame(frame, code=code, sensor="acc", reference=reference,
                                             scenario_id=f"{code}|acc", stats=stats)
        pd.testing.assert_frame_equal(out.loc[outside], frame.loc[outside])
        assert out["Time"].equals(frame["Time"])


def test_timestamp_corruptions_and_absence(tmp_path):
    frame, stats, reference = _reference(tmp_path)
    kwargs = dict(sensor="acc", reference=reference, stats=stats)
    swapped, p = falsification.corrupt_frame(frame, code="C1", scenario_id="C1|acc", **kwargs)
    i = p["row"]
    t = frame["Time"].to_numpy()
    assert swapped["Time"].iloc[i] == t[i + 1] and swapped["Time"].iloc[i + 1] == t[i]
    truncated, _ = falsification.corrupt_frame(frame, code="C4", scenario_id="C4|acc", **kwargs)
    assert truncated["Time"].iloc[-1] - truncated["Time"].iloc[0] <= 0.5 * (t[-1] - t[0])
    assert falsification.corrupt_frame(frame, code="D1", scenario_id="D1|acc", **kwargs)[0] is None


def test_noise_is_deterministic(tmp_path):
    frame, stats, reference = _reference(tmp_path)
    a, _ = falsification.corrupt_frame(frame, code="N6", sensor="acc", reference=reference, scenario_id="N6|acc",
                                       stats=stats)
    b, _ = falsification.corrupt_frame(frame, code="N6", sensor="acc", reference=reference, scenario_id="N6|acc",
                                       stats=stats)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ end to end on one real development segment
@pytest.fixture(scope="module")
def brushless():
    from src.fusion.assembly import PREPROCESSING_CONFIGS, load_design_registry
    from src.preprocessing.pipeline import load_raw_catalog
    rows = l3.development_segment_rows(load_raw_catalog("data/raw/extracted"),
                                       pd.read_csv("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"))
    bank = l3.DetectorBank.load("BrushlessMotor", "reports/phase4_pretest/experiments", "data/interim/phase4/models")
    segment = l3.selected_validation_segments(rows, "BrushlessMotor")[0]
    row = rows[rows["segment_id"].eq(segment)].iloc[0]
    return SimpleNamespace(rows=rows, bank=bank, segment=segment,
                           paths={m: str(row[f"{m}_path"]) for m in l3.MODALITIES},
                           config=l3.load_regime_configs(PREPROCESSING_CONFIGS), units=load_design_registry(),
                           records=l3.load_l2_records(RECORDS))


def test_clean_runner_reproduces_every_phase4_validation_score(brushless):
    run = l3.run_segment(brushless.paths, machine="BrushlessMotor", config=brushless.config, units=brushless.units,
                         bank=brushless.bank, records=brushless.records)
    assert not run.gate.integrity_flag and len(run.raw_scores) == 78 and len(run.decisions) == 26
    for experiment_id, score in run.raw_scores.items():
        recorded = pd.read_csv(brushless.bank.bundles[experiment_id].bundle_path / "validation_scores.csv",
                               dtype={"segment_id": str}, float_precision="round_trip").set_index("segment_id")
        assert abs(score - recorded.loc[brushless.segment, "score"]) <= 1e-9
    assert {d["status"] for d in run.decisions} == {l3.DECISION_ALL}


def test_a_dead_modality_is_gated_flagged_and_the_survivors_score_as_if_clean(brushless, tmp_path):
    clean = l3.run_segment(brushless.paths, machine="BrushlessMotor", config=brushless.config,
                           units=brushless.units, bank=brushless.bank, records=brushless.records)
    gate = l3.run_gate(brushless.paths, "BrushlessMotor", brushless.config)
    reference = falsification.CleanReference("BrushlessMotor", brushless.segment, brushless.paths, gate.coverage)
    stats = pd.DataFrame(columns=["machine", "sensor", "channel"])
    scenario = next(s for s in falsification.scenario_catalogue() if s.scenario_id == "B1|acc")
    paths, record = falsification.write_scenario_files(scenario, reference, stats, tmp_path)
    assert record["files"]["acc"]["sha256"]
    run = l3.run_segment(paths, machine="BrushlessMotor", config=brushless.config, units=brushless.units,
                         bank=brushless.bank, records=brushless.records)
    assert run.gate.verdicts == {"mic": l3.VALID, "acc": l3.INVALID_FLATLINE, "gyro": l3.VALID}
    assert all(d["integrity_flag"] and d["status"] == l3.DECISION_PAIR and d["survivors"] == "mic+gyro"
               for d in run.decisions)
    for experiment_id, score in run.raw_scores.items():
        assert brushless.bank.bundles[experiment_id].modality != "acc"
        assert score == clean.raw_scores[experiment_id]
