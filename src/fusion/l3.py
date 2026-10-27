"""Sensor-integrity-aware fusion (L3): frozen gates, raw end-to-end runner and fallback decision.

Implements ``arms.L3_sensor_integrity_fallback`` and ``integrity_gate`` of the
frozen design v2 (``phase5-integrity-gate-v2``) in the design's order of
operations:

1. load each modality from its raw file; missing or unreadable -> type D;
2. G3 on the *stored* timestamp order of each stream (finite, strictly
   increasing, own-stream duration >= 0.90 x nominal) -> type C.  The frozen
   reader sorts by timestamp, which would hide a swapped pair and accept a NaN
   tail, so G3 inspects the file before the frozen reader runs;
3. frozen value-independent startup transforms;
4. frozen shared alignment over all three modalities (timestamps only);
5. G1 (non-finite) and G2 (flatline) per modality on the samples inside the
   common coverage [start, end);
6. surviving modalities only: the frozen Phase-3.1 window features (segment
   median) and Phase-4 log-power spectra, scored by the frozen Phase-4 detectors;
7. robust-z calibration from the persisted L2 records, mean over survivors,
   survivor-subset threshold; one survivor uses its raw Phase-4 score against
   its own raw Phase-4 q90 threshold.

No weight, threshold or calibration is fitted here.  Any C/D failure abstains
(``ABSTAIN_ALIGNMENT_FAILURE``); no valid modality abstains
(``ABSTAIN_SENSOR_INVALID``); every non-valid modality raises ``integrity_flag``
whatever the anomaly decision, so a gated segment is never passed silently.

Only training and source-validation segments can be loaded: ``development_segment_rows``
refuses every segment that is not train/validation in Regime T (which covers
the Regime-S training and validation cohorts and the released target normals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.experiment.phase4 import (
    FrozenBundle,
    ModelStore,
    load_frozen_bundles,
    modality_feature_columns,
    standardize_float32,
)
from src.features.extraction import flatten_features, frequency_features, time_features
from src.fusion.l2 import ModalityCalibration, fuse, robust_z, unit_key
from src.preprocessing.config import PreprocessingConfig, load_config
from src.preprocessing.pipeline import _normalize
from src.preprocessing.signals import (
    SENSOR_KEYS,
    SensorSignal,
    apply_startup_transformations,
    common_coverage,
    iter_aligned_windows,
    read_sensor_parquet,
    resample_regular,
)
from src.preprocessing.spectral_windows import log_power_spectrum_input


GATE_VERSION = "phase5-integrity-gate-v2"
L3_RUNNER_VERSION = "phase5-l3-runner-v1"
MODALITIES = ("mic", "acc", "gyro")
FLATLINE_MIN_RUN_S = 0.020
FLATLINE_INVALID_FRACTION = 0.50
COVERAGE_MIN_FRACTION = 0.90
NOMINAL_DURATION_S = {"RoboticArm": 7.5, "BrushlessMotor": 5.2}
LOW_MARGIN_FLATLINE_FRACTION = 0.25
# Implementation clarification: run durations are differences of float64 timestamps, so a run
# spanning exactly 20 ms can evaluate to 0.0199999...; 1e-9 s absorbs representation error only
# (more than four orders of magnitude below the shortest sample period, 62.5 us).
DURATION_TOLERANCE_S = 1e-9
TIME_COLUMN_NAMES = {"time", "timestamp", "t"}

VALID = "VALID"
INVALID_NON_FINITE = "INVALID_G1_NON_FINITE"
INVALID_FLATLINE = "INVALID_G2_FLATLINE"
FAIL_ABSENT = "FAIL_D_ABSENT"
FAIL_TIMING = "FAIL_C_TIMING"
NOT_EVALUATED = "NOT_EVALUATED_ALIGNMENT_FAILED"

DECISION_ALL = "DECISION_L2_ALL_THREE"
DECISION_PAIR = "DECISION_L2_PAIR"
DECISION_SINGLE = "DECISION_PHASE4_SINGLE"
ABSTAIN_SENSOR_INVALID = "ABSTAIN_SENSOR_INVALID"
ABSTAIN_ALIGNMENT_FAILURE = "ABSTAIN_ALIGNMENT_FAILURE"


class L3Error(RuntimeError):
    """A technical defect of the runner (never a gate verdict)."""


# ------------------------------------------------------------------ development cohort guard
def development_segment_rows(catalog: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Catalog rows of the permitted development segments only (train/validation, never test).

    A segment is permitted iff its Regime-T split is ``train`` or ``validation``
    and no regime assigns it to ``test``.  Adds ``split_t`` and ``split_s``.
    """

    pivot = assignments.pivot_table(index=["machine", "segment_id"], columns="regime",
                                    values="split", aggfunc="first").reset_index()
    pivot = pivot.rename(columns={"S": "split_s", "T": "split_t"})
    permitted = pivot[pivot["split_t"].isin(["train", "validation"])
                      & ~pivot["split_s"].eq("test") & ~pivot["split_t"].eq("test")]
    rows = catalog.merge(
        permitted, on=["machine", "segment_id"], how="inner", validate="one_to_one")
    if rows["split_t"].eq("test").any() or rows["split_s"].eq("test").any():  # pragma: no cover - guard
        raise L3Error("a protected segment entered the development cohort")
    return rows.sort_values(["machine", "segment_id"]).reset_index(drop=True)


def selected_validation_segments(rows: pd.DataFrame, machine: str, count: int = 12) -> tuple[str, ...]:
    """Design rule: sort source-validation ids and take indices round(k (n-1) / 11), k = 0..11."""

    ids = sorted(rows[rows["machine"].eq(machine) & rows["split_t"].eq("validation")]["segment_id"].astype(str))
    n = len(ids)
    picks = [ids[int(round(k * (n - 1) / (count - 1)))] for k in range(count)]
    if len(set(picks)) != count:
        raise L3Error("segment selection produced duplicates")
    return tuple(picks)


# ------------------------------------------------------------------ raw loading + G3
@dataclass(frozen=True)
class RawStream:
    """One modality's file exactly as stored (row order untouched)."""

    sensor: str
    path: Path
    time_column: str
    channels: tuple[str, ...]
    timestamps: np.ndarray
    values: np.ndarray


def load_raw_stream(path: str | Path | None, sensor: str) -> tuple[RawStream | None, str | None]:
    """(stream, None) or (None, reason) for an absent/unreadable/ill-formed file (type D)."""

    if path is None or not Path(path).is_file():
        return None, "file absent"
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # noqa: BLE001 - any read failure is an absent stream by definition
        return None, f"unreadable: {type(error).__name__}"
    times = [c for c in frame.columns if str(c).lower() in TIME_COLUMN_NAMES]
    if len(times) != 1:
        return None, f"expected one timestamp column, found {times}"
    channels = tuple(str(c) for c in frame.columns if c != times[0])
    if not channels or len(frame) == 0:
        return None, "no channels or no samples"
    try:
        timestamps = pd.to_numeric(frame[times[0]], errors="raise").to_numpy(np.float64)
        values = frame[list(channels)].to_numpy(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        return None, f"non-numeric content: {type(error).__name__}"
    return RawStream(sensor, Path(path), str(times[0]), channels, timestamps, values), None


def g3_timing(stream: RawStream, machine: str) -> dict:
    """G3 on the stored order: finite, strictly increasing (no duplicates), duration >= 0.90 nominal."""

    t = stream.timestamps
    finite = bool(np.all(np.isfinite(t)))
    increasing = bool(finite and (len(t) < 2 or np.all(np.diff(t) > 0)))
    duration = float(t[-1] - t[0]) if finite and len(t) else float("nan")
    required = COVERAGE_MIN_FRACTION * NOMINAL_DURATION_S[machine]
    coverage_ok = bool(finite and duration >= required)
    reasons = []
    if not finite:
        reasons.append("non-finite timestamp")
    else:
        if not increasing:
            reasons.append("non-increasing or duplicate timestamp")
        if not coverage_ok:
            reasons.append("coverage below 0.90 nominal")
    return {"passed": not reasons, "reasons": reasons, "duration_s": duration,
            "required_duration_s": required, "samples": int(len(t))}


# ------------------------------------------------------------------ G1 / G2
def flatline_profile(timestamps: np.ndarray, column: np.ndarray) -> tuple[float, float]:
    """(flatline fraction, longest run duration) of one channel.

    A run is a maximal sequence of consecutive samples with exactly equal stored
    float64 values; its duration is t_last - t_first from the stream's own
    timestamps; a run qualifies at >= 0.020 s.  NaN never equals NaN, so a NaN
    sample breaks a run (non-finite values are G1's concern).
    """

    n = len(column)
    if n == 0:
        raise L3Error("empty channel inside the common coverage")
    if n == 1:
        return 0.0, 0.0
    boundaries = np.flatnonzero(column[1:] != column[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n])) - 1          # inclusive last index of each run
    durations = timestamps[ends] - timestamps[starts]
    lengths = ends - starts + 1
    qualifying = durations >= FLATLINE_MIN_RUN_S - DURATION_TOLERANCE_S
    return float(lengths[qualifying].sum() / n), float(durations.max())


def g1_g2_values(signal: SensorSignal, start_s: float, end_s: float) -> dict:
    """G1 and G2 for one modality on its samples inside [start, end)."""

    mask = (signal.timestamps >= start_s) & (signal.timestamps < end_s)
    t, values = signal.timestamps[mask], signal.values[mask]
    if len(t) == 0:
        raise L3Error(f"{signal.sensor}: no samples inside the common coverage")
    non_finite = {ch: int(np.count_nonzero(~np.isfinite(values[:, i]))) for i, ch in enumerate(signal.channels)}
    flat = {}
    for i, ch in enumerate(signal.channels):
        fraction, longest = flatline_profile(t, values[:, i])
        flat[ch] = {"flatline_fraction": fraction, "longest_run_s": longest}
    g1 = any(non_finite.values())
    g2 = any(v["flatline_fraction"] >= FLATLINE_INVALID_FRACTION for v in flat.values())
    verdict = INVALID_NON_FINITE if g1 else INVALID_FLATLINE if g2 else VALID
    return {"verdict": verdict, "g1_triggered": g1, "g2_triggered": g2, "samples_in_coverage": int(len(t)),
            "non_finite_per_channel": non_finite, "flatline_per_channel": flat}


# ------------------------------------------------------------------ gate (steps 1-5)
@dataclass
class GateOutcome:
    machine: str
    verdicts: dict[str, str]
    details: dict[str, dict]
    coverage: tuple[float, float] | None
    transformed: dict[str, SensorSignal] | None
    alignment_failure: bool
    alignment_reason: str | None = None

    @property
    def survivors(self) -> tuple[str, ...]:
        return tuple(m for m in MODALITIES if self.verdicts[m] == VALID)

    @property
    def integrity_flag(self) -> bool:
        return any(v != VALID for v in self.verdicts.values())


def run_gate(paths: Mapping[str, str | Path | None], machine: str, config: PreprocessingConfig) -> GateOutcome:
    """Steps 1-5 of the frozen order of operations for one segment."""

    verdicts: dict[str, str] = {}
    details: dict[str, dict] = {}
    for sensor in MODALITIES:
        stream, reason = load_raw_stream(paths.get(sensor), sensor)
        if stream is None:
            verdicts[sensor], details[sensor] = FAIL_ABSENT, {"reason": reason}
            continue
        g3 = g3_timing(stream, machine)
        details[sensor] = {"g3": g3}
        verdicts[sensor] = VALID if g3["passed"] else FAIL_TIMING
    failed = [m for m in MODALITIES if verdicts[m] in (FAIL_ABSENT, FAIL_TIMING)]
    if failed:
        for m in MODALITIES:
            if m not in failed:
                verdicts[m] = NOT_EVALUATED
        return GateOutcome(machine, verdicts, details, None, None, True,
                           "; ".join(f"{m}: {verdicts[m]}" for m in failed))
    # G3 passed for every stream, so the frozen reader's sort is the identity.
    raw = {sensor: read_sensor_parquet(paths[sensor], sensor) for sensor in SENSOR_KEYS}
    try:
        transformed, _log = apply_startup_transformations(raw, machine, config)
        start, end = common_coverage(transformed)
    except ValueError as error:
        for m in MODALITIES:
            verdicts[m] = NOT_EVALUATED
        return GateOutcome(machine, verdicts, details, None, None, True,
                           f"frozen alignment refused after G3: {error}")
    for sensor in MODALITIES:
        values_check = g1_g2_values(transformed[sensor], start, end)
        details[sensor]["values"] = values_check
        verdicts[sensor] = values_check["verdict"]
    return GateOutcome(machine, verdicts, details, (start, end), transformed, False)


# ------------------------------------------------------------------ step 6: frozen processing
def _check_regime_invariant_processing(configs: Sequence[PreprocessingConfig]) -> None:
    """The S and T preprocessing configs differ only in ``regime``, so one processing pass serves both."""

    canon = [{k: v for k, v in c.canonical_dict().items() if k != "regime"} for c in configs]
    if any(c != canon[0] for c in canon[1:]):
        raise L3Error("regime preprocessing configs differ beyond the regime label")


@dataclass
class ModalityInputs:
    """Frozen per-modality detector inputs of one segment."""

    features: dict[str, float]          # segment-median Phase-3.1 features of this modality
    spectra: np.ndarray                 # (windows, bins) float32, as stored in the Phase-4 cache
    window_count: int


def process_modalities(transformed: Mapping[str, SensorSignal], modalities: Iterable[str],
                       config: PreprocessingConfig, machine: str) -> dict[str, ModalityInputs]:
    """Frozen Phase-3.1 features and Phase-4 spectra for ``modalities`` only.

    The shared window grid comes from the frozen ``iter_aligned_windows`` over all
    three (timestamp-only) streams; each requested modality is then processed by
    the same calls, in the same order, as ``pipeline._iter_processed_windows`` and
    the spectral-cache builder.  Other modalities' values are never read.
    """

    wanted = [m for m in MODALITIES if m in set(modalities)]
    rows: dict[str, list[dict[str, float]]] = {m: [] for m in wanted}
    spectra: dict[str, list[np.ndarray]] = {m: [] for m in wanted}
    for window in iter_aligned_windows(dict(transformed), config):
        normalized = _normalize({m: window.signals[m] for m in wanted}, config, None)
        for sensor in wanted:
            signal = normalized[sensor]
            features: dict[str, float] = {}
            if "time" in config.features:
                features.update(flatten_features(sensor, signal.channels, time_features(signal.values)))
            if "frequency" in config.features:
                target = config.resampling.target_rates_hz[machine][sensor]
                regular = resample_regular(signal, target, method=config.resampling.method)
                features.update(flatten_features(sensor, regular.channels,
                                                 frequency_features(regular.values, target)))
            rows[sensor].append(features)
            spectra[sensor].append(np.asarray(log_power_spectrum_input(signal, config, machine),
                                              dtype=np.float32).reshape(-1))
    output = {}
    for sensor in wanted:
        if not rows[sensor]:
            raise L3Error("the common coverage holds no complete window")
        names = tuple(sorted(rows[sensor][0]))
        if any(tuple(sorted(r)) != names for r in rows[sensor]):
            raise L3Error("window feature schemas differ within a segment")
        matrix = np.asarray([[float(r[n]) for n in names] for r in rows[sensor]], dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise L3Error(f"non-finite window feature for surviving modality {sensor}")
        output[sensor] = ModalityInputs(dict(zip(names, (float(v) for v in np.median(matrix, axis=0)))),
                                        np.vstack(spectra[sensor]), len(rows[sensor]))
    return output


# ------------------------------------------------------------------ frozen detectors + calibrations
@dataclass
class DetectorBank:
    """The 78 frozen Phase-4 detectors of one machine (both regimes), loaded once and hash-verified."""

    machine: str
    bundles: dict[str, FrozenBundle]
    models: dict[str, tuple] = field(default_factory=dict)

    @classmethod
    def load(cls, machine: str, pretest_root: str | Path, model_root: str | Path) -> "DetectorBank":
        bundles = {k: b for k, b in load_frozen_bundles(pretest_root).items() if b.config.machine == machine}
        if len(bundles) != 78:
            raise L3Error(f"expected 78 Phase-4 detectors for {machine}, found {len(bundles)}")
        store = ModelStore(model_root)
        bank = cls(machine, bundles)
        for experiment_id, bundle in bundles.items():
            model, standardizer, _ = store.load(experiment_id, expected_manifest_sha256=bundle.model_manifest_sha256)
            bank.models[experiment_id] = (model, standardizer)
        return bank

    def score(self, experiment_id: str, inputs: ModalityInputs) -> float:
        bundle = self.bundles[experiment_id]
        model, standardizer = self.models[experiment_id]
        if bundle.config.model_family == "dense_autoencoder":
            window_scores = model.score_windows(standardize_float32(standardizer, inputs.spectra))
            return float(pd.Series(np.asarray(window_scores, dtype=np.float64)).median())
        columns = modality_feature_columns(inputs.features, bundle.modality)
        X = np.asarray([[inputs.features[c] for c in columns]], dtype=np.float64)
        return float(model.score(standardizer.transform(X))[0])


def load_l2_records(path: str | Path) -> dict[tuple[str, tuple[str, ...]], dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(records) != 208:
        raise L3Error("expected the 208 persisted L2 calibration records")
    return {(r["unit"], tuple(r["subset"])): r for r in records}


def _calibration(record: Mapping, modality: str) -> ModalityCalibration:
    c = record["modality_calibration"][modality]
    return ModalityCalibration(float(c["median"]), float(c["q25"]), float(c["q75"]), float(c["iqr"]))


# ------------------------------------------------------------------ step 7: decision
def decide_unit(unit: Mapping, survivors: Sequence[str], raw_scores: Mapping[str, float],
                records: Mapping[tuple[str, tuple[str, ...]], dict], bank: DetectorBank,
                alignment_failure: bool) -> dict:
    """The frozen fallback for one unit.  ``raw_scores`` holds Phase-4 scores of survivors only."""

    key = unit_key(unit)
    base = {"unit": key, "survivors": "+".join(survivors)}
    if alignment_failure:
        return {**base, "status": ABSTAIN_ALIGNMENT_FAILURE, "score": None, "threshold": None,
                "anomaly": None, "threshold_source": None}
    if not survivors:
        return {**base, "status": ABSTAIN_SENSOR_INVALID, "score": None, "threshold": None,
                "anomaly": None, "threshold_source": None}
    if set(raw_scores) != set(survivors):
        raise L3Error("raw scores must cover exactly the surviving modalities")
    if len(survivors) == 1:
        modality = survivors[0]
        experiment_id = unit["phase4_single_modality_ids"][modality]
        threshold = float(bank.bundles[experiment_id].threshold.value)
        score = float(raw_scores[modality])
        return {**base, "status": DECISION_SINGLE, "score": score, "threshold": threshold,
                "anomaly": bool(score > threshold), "threshold_source": f"phase4:{experiment_id}"}
    subset = tuple(m for m in MODALITIES if m in survivors)
    record = records[(key, subset)]
    calibrated = {m: robust_z(np.asarray([raw_scores[m]]), _calibration(record, m)) for m in subset}
    score = float(fuse(calibrated, subset)[0])
    threshold = float(record["threshold"]["value"])
    status = DECISION_ALL if len(subset) == 3 else DECISION_PAIR
    return {**base, "status": status, "score": score, "threshold": threshold,
            "anomaly": bool(score > threshold), "threshold_source": record["calibration_id"]}


@dataclass
class SegmentRun:
    gate: GateOutcome
    raw_scores: dict[str, float]                      # experiment id -> Phase-4 score (survivors only)
    decisions: list[dict]
    window_counts: dict[str, int]


def run_segment(paths: Mapping[str, str | Path | None], *, machine: str, config: PreprocessingConfig,
                units: Sequence[Mapping], bank: DetectorBank,
                records: Mapping[tuple[str, tuple[str, ...]], dict]) -> SegmentRun:
    """The complete L3 path for one segment of ``machine`` across its 26 units."""

    gate = run_gate(paths, machine, config)
    survivors = gate.survivors if not gate.alignment_failure else ()
    inputs = (process_modalities(gate.transformed, survivors, config, machine)
              if survivors else {})
    raw_scores: dict[str, float] = {}
    for experiment_id, bundle in bank.bundles.items():
        if bundle.modality in inputs:
            raw_scores[experiment_id] = bank.score(experiment_id, inputs[bundle.modality])
    decisions = []
    for unit in units:
        if unit["machine"] != machine:
            continue
        unit_scores = {m: raw_scores[unit["phase4_single_modality_ids"][m]] for m in survivors}
        decision = decide_unit(unit, survivors, unit_scores, records, bank, gate.alignment_failure)
        decision["integrity_flag"] = gate.integrity_flag
        decisions.append(decision)
    return SegmentRun(gate, raw_scores, decisions, {m: v.window_count for m, v in inputs.items()})


def load_regime_configs(paths: Mapping[str, str | Path]) -> PreprocessingConfig:
    """The S config (used for processing) after asserting S and T process identically."""

    configs = [load_config(paths[r]) for r in ("S", "T")]
    _check_regime_invariant_processing(configs)
    return configs[0]
