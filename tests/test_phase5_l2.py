"""Phase 5.3 L2 score-level fusion: source-validation calibration and fused q90 thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fusion import l2
from src.fusion.assembly import load_design_registry
from src.fusion.l2 import L2CalibrationError

PRETEST = Path("reports/phase4_pretest/experiments")


@pytest.fixture(scope="module")
def calibrated():
    return l2.calibrate_all(PRETEST)


def test_design_subsets_are_the_four_frozen_subsets():
    assert l2.design_subsets() == (("mic", "acc", "gyro"), ("acc", "gyro"), ("mic", "gyro"), ("mic", "acc"))


def test_robust_z_uses_median_and_linear_iqr():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    cal = l2.calibrate_modality(values)
    assert (cal.median, cal.q25, cal.q75, cal.iqr) == (3.0, 2.0, 4.0, 2.0)
    assert np.array_equal(l2.robust_z(values, cal), (values - 3.0) / 2.0)


@pytest.mark.parametrize("values, message", [
    (np.array([5.0, 5.0, 5.0, 5.0]), "IQR"),
    (np.array([1.0, np.nan, 3.0]), "finite"),
    (np.array([1.0, np.inf, 3.0]), "finite"),
    (np.array([]), "non-empty"),
])
def test_degenerate_or_non_finite_calibration_inputs_are_refused_not_repaired(values, message):
    with pytest.raises(L2CalibrationError, match=message):
        l2.calibrate_modality(values)


def test_fusion_is_the_mean_of_the_subset_in_fixed_order_and_refuses_bad_subsets():
    z = {"mic": np.array([1.0, 4.0]), "acc": np.array([2.0, 5.0]), "gyro": np.array([3.0, 6.0])}
    assert np.array_equal(l2.fuse(z, ("mic", "acc", "gyro")), np.array([2.0, 5.0]))
    assert np.array_equal(l2.fuse(z, ("gyro", "acc")), np.array([2.5, 5.5]))
    with pytest.raises(L2CalibrationError):
        l2.fuse(z, ("acc",))
    with pytest.raises(L2CalibrationError):
        l2.fuse({**z, "gyro": np.array([1.0])}, ("acc", "gyro"))


def test_single_modality_robust_z_threshold_is_decision_identical_to_the_raw_threshold():
    """Monotone transform + 'higher' quantile: the decision rule L3 relies on for one survivor."""

    rng = np.random.default_rng(5)
    scores = rng.gamma(2.0, size=186)
    cal = l2.calibrate_modality(scores)
    z = l2.robust_z(scores, cal)
    raw_t = np.quantile(scores, 0.9, method="higher")
    z_t = np.quantile(z, 0.9, method="higher")
    probe = rng.gamma(2.0, size=1000)
    assert np.array_equal(probe > raw_t, l2.robust_z(probe, cal) > z_t)


def test_all_208_records_map_exactly_to_the_v2_registry(calibrated):
    records, fused = calibrated
    units = load_design_registry()
    assert len(records) == 208 and len({r["calibration_id"] for r in records}) == 208
    by_unit = {}
    for r in records:
        by_unit.setdefault(r["unit"], []).append(r)
    assert set(by_unit) == {l2.unit_key(u) for u in units}
    for unit in units:
        unit_records = by_unit[l2.unit_key(unit)]
        assert [tuple(r["subset"]) for r in unit_records] == list(l2.design_subsets())
        for r in unit_records:
            assert r["component_experiment_ids"] == {m: unit["phase4_single_modality_ids"][m] for m in r["subset"]}
            assert r["test_evaluated"] is False and r["validation_segment_count"] in (126, 186)


def test_every_threshold_is_its_own_subsets_q90_higher(calibrated):
    records, fused = calibrated
    grouped = dict(tuple(fused.groupby("calibration_id", sort=False)))
    for r in records:
        values = grouped[r["calibration_id"]]["fused_score"].to_numpy()
        assert r["threshold"]["value"] == float(np.quantile(values, 0.9, method="higher"))
        assert np.isfinite(values).all() and len(values) == r["validation_segment_count"]
    # a two-sensor subset never inherits the three-sensor threshold
    for unit in {r["unit"] for r in records}:
        thresholds = [r["threshold"]["value"] for r in records if r["unit"] == unit]
        assert len(set(thresholds)) == 4


def test_calibration_reproduces_from_persisted_artifacts_and_detects_tampering(tmp_path, calibrated):
    records, fused = calibrated
    l2.write_calibration_artifacts(tmp_path, records, fused, code_git_sha="test", code_source_hash="test")
    assert l2.verify_calibration_artifacts(tmp_path, PRETEST) == {"records_reproduced": 208, "fused_rows_reproduced": len(fused)}
    with pytest.raises(FileExistsError):
        l2.write_calibration_artifacts(tmp_path, records, fused, code_git_sha="test", code_source_hash="test")
    path = tmp_path / l2.FUSED_NAME
    path.write_text(path.read_text(encoding="utf-8").replace("p5l2-", "p5l2x", 1), encoding="utf-8")
    with pytest.raises(L2CalibrationError, match="hash mismatch"):
        l2.verify_calibration_artifacts(tmp_path, PRETEST)


def test_calibration_never_touches_a_protected_reader(monkeypatch):
    import src.preprocessing.protected_access as pa

    def refuse(*_a, **_k):
        raise AssertionError("a protected reader was reached during L2 calibration")
    monkeypatch.setattr(pa, "require_protected_read", refuse)
    for module in ("src.fusion.assembly", "src.experiment.phase4", "src.experiment.store"):
        mod = __import__(module, fromlist=["x"])
        for name in ("require_protected_read", "require_sealed_score_read"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, refuse)
    records, _ = l2.calibrate_all(PRETEST)
    assert len(records) == 208


def test_a_misaligned_modality_is_refused(monkeypatch):
    from src.fusion import assembly as fa

    unit = load_design_registry()[3]
    real = pd.read_csv

    def shifted(path, *args, **kwargs):
        frame = real(path, *args, **kwargs)
        if Path(path).parent.name == unit["phase4_single_modality_ids"]["gyro"]:
            frame = frame.assign(segment_id=list(frame["segment_id"])[1:] + ["foreign-segment"])
        return frame

    monkeypatch.setattr(fa.pd, "read_csv", shifted)
    with pytest.raises(fa.FusionAssemblyError):
        l2.calibrate_unit(fa.assemble_unit_validation_scores(unit, PRETEST), l2.design_subsets())


# ------------------------------------------------ implementation clarifications
def test_iqr_uses_numpy_linear_which_is_the_library_default_and_differs_from_higher():
    """Clarification recorded before any calibration: IQR = q75 - q25 with method='linear'
    (NumPy's default, i.e. the (n-1)p interpolation of numpy.percentile); q90 stays 'higher'."""

    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0])
    linear = np.quantile(values, [0.25, 0.75], method="linear")
    assert np.array_equal(linear, np.quantile(values, [0.25, 0.75]))           # library default
    assert np.array_equal(linear, np.percentile(values, [25, 75]))
    assert list(linear) == [1.25, 8.25]                                          # (n-1)p = 1.25, 3.75
    assert list(np.quantile(values, [0.25, 0.75], method="higher")) == [2.0, 10.0]
    assert l2.IQR_QUANTILE_METHOD == "linear" and l2.THRESHOLD_METHOD == "higher"
    cal = l2.calibrate_modality(values)
    assert cal.iqr == 7.0 and cal.median == 2.5


def test_round_trip_parsing_reconstructs_every_persisted_phase4_validation_score_exactly():
    """Read-only over the 156 Phase-4 pre-test bundles: round_trip parsing equals Python's
    correctly rounded float(text) for every stored value, and each bundle's q90 'higher'
    recomputed from those values equals its recorded threshold exactly."""

    import csv
    files = sorted(PRETEST.glob("*/validation_scores.csv"))
    assert len(files) == 156
    default_mismatches = 0
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            exact = np.array([float(row["score"]) for row in csv.DictReader(handle)])
        parsed = pd.read_csv(path, float_precision="round_trip")["score"].to_numpy()
        assert np.array_equal(parsed, exact), path
        default_mismatches += int((pd.read_csv(path)["score"].to_numpy() != exact).sum())
        recorded = json.loads((path.parent / "threshold.json").read_text(encoding="utf-8"))["value"]
        assert float(np.quantile(exact, 0.9, method="higher")) == recorded, path
    assert default_mismatches > 0        # the reason round_trip is required
