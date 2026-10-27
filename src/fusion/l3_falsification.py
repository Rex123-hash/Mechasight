"""Development-only falsification of the L3 integrity gate (design v2 ``development_falsification``).

Two parts, on permitted training and source-validation recordings only:

* ``clean_scan``: G1-G3 on every clean training and source-validation
  segment-channel of both machines, the per-channel flatline distribution, and
  the clean-training channel statistics the corruption generator needs
  (pooled SD for N4, signed extreme for B5, max |value| for the emulated
  full scale of S1-S4);
* the end-to-end suite: 12 deterministic source-validation segments per
  machine x 83 scenarios, each a corrupted copy of the raw files written to a
  development directory and passed through the complete L3 runner.

Constants the design leaves open are fixed here before any run and recorded
in every artifact (``CORRUPTION_PARAMETERS``): where a partial corruption sits
(centred in the common coverage), the dead value (0.0), the impulse train and
tone, and the noise power reference.  They are implementation clarifications,
not tuned values.  Synthetic perturbations are negative controls for gate
behaviour, not physical fault evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.fusion import l3
from src.preprocessing.config import PreprocessingConfig


ACC_FULL_SCALE_G = 15.988832
CORRUPTION_VERSION = "phase5-l3-corruptions-v1"
CORRUPTION_PARAMETERS = {
    "version": CORRUPTION_VERSION,
    "region_reference": "the clean segment's common coverage [start, end) after the frozen startup transforms; "
                        "samples outside it are never changed",
    "partial_region": "fraction p of the coverage duration, centred: [start + (1-p)/2 L, start + (1+p)/2 L)",
    "A2_channel": "the modality's first stored channel",
    "B1_dead_value": 0.0,
    "B2_stuck": "from start + 0.25 L to end, every channel holds its value at the last sample before start + 0.25 L",
    "B3_dropout_value": 0.0,
    "B4_flatline": "centred 40 %, every channel holds its value at the first sample of the region",
    "B5_plateau_value": "the signed clean-training extreme of that machine-channel (largest |value|)",
    "C_position": "stored row i = first row with t >= start + 0.5 L; C1 swaps t[i] and t[i+1]; C2 sets t[i+1] = t[i]; "
                  "C3 sets t[i] = NaN; C4 keeps rows with t <= t0 + 0.5 (t_last - t0)",
    "D1": "the modality's file is not written",
    "M_dead": "M1 and M2 apply B1 to two or three modalities",
    "N4_offset_sd": "pooled population SD of the machine-channel over all clean-training coverage samples",
    "noise_power_reference": "population variance of the channel's coverage samples in that segment",
    "N8_impulses": "every 0.25 s from start, +10 x the channel's segment SD added to the first sample at or after "
                   "each instant",
    "N9_tone": "113 Hz sinusoid, amplitude = the channel's segment SD, phase 0 at start",
    "S_full_scale": {"acc": ACC_FULL_SCALE_G, "gyro": "clean-training max |value| of the machine-channel",
                     "mic": "clean-training max |value| of the machine-channel"},
    "noise_seed": "first 16 hex digits of sha256('machine|segment|scenario|modality|channel') as an integer",
}
GAINS = {"N1": 0.5, "N2": 2.0, "N3": 3.0}
SNR_DB = {"N5": 20.0, "N6": 10.0, "N7": 0.0}
SATURATION_GAINS = {"S1": 1.5, "S2": 2.0, "S3": 3.0, "S4": 5.0}
IMPULSE_PERIOD_S, IMPULSE_SD = 0.25, 10.0
TONE_HZ = 113.0
VALUE_FAILURE_CODES = ("A1", "A2", "A3", "B1", "B2", "B3", "B5")
SINGLE_CODES = ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "D1")
NEGATIVE_CODES = tuple(f"N{i}" for i in range(1, 10))
SATURATION_CODES = tuple(SATURATION_GAINS)
RUNS_PER_SEGMENT = 83


# ------------------------------------------------------------------ scenario catalogue
@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    code: str
    targets: tuple[str, ...]
    expected_verdicts: Mapping[str, str] | None     # None: characterisation only (S1-S4)
    expected_status: str | None


def _expected(code: str, targets: Sequence[str]) -> tuple[dict[str, str] | None, str | None]:
    others = {m: l3.VALID for m in l3.MODALITIES}
    if code in ("A1", "A2", "A3"):
        return {**others, targets[0]: l3.INVALID_NON_FINITE}, l3.DECISION_PAIR
    if code in ("B1", "B2", "B3", "B5"):
        return {**others, targets[0]: l3.INVALID_FLATLINE}, l3.DECISION_PAIR
    if code == "B4" or code in NEGATIVE_CODES or code == "CLEAN":
        return others, l3.DECISION_ALL
    if code in ("C1", "C2", "C3", "C4", "D1"):
        failed = l3.FAIL_ABSENT if code == "D1" else l3.FAIL_TIMING
        return ({m: failed if m in targets else l3.NOT_EVALUATED for m in l3.MODALITIES},
                l3.ABSTAIN_ALIGNMENT_FAILURE)
    if code == "M1":
        return ({m: l3.INVALID_FLATLINE if m in targets else l3.VALID for m in l3.MODALITIES},
                l3.DECISION_SINGLE)
    if code == "M2":
        return {m: l3.INVALID_FLATLINE for m in l3.MODALITIES}, l3.ABSTAIN_SENSOR_INVALID
    if code in SATURATION_CODES:
        return None, None
    raise ValueError(f"unknown scenario code {code!r}")


def scenario_catalogue() -> tuple[Scenario, ...]:
    """The 83 frozen scenarios of one segment, in a fixed order."""

    items: list[tuple[str, tuple[str, ...]]] = [("CLEAN", ())]
    items += [(code, (m,)) for code in SINGLE_CODES for m in l3.MODALITIES]
    items += [("M1", pair) for pair in (("mic", "acc"), ("mic", "gyro"), ("acc", "gyro"))]
    items += [("M2", l3.MODALITIES)]
    items += [(code, (m,)) for code in NEGATIVE_CODES for m in l3.MODALITIES]
    items += [(code, (m,)) for code in SATURATION_CODES for m in l3.MODALITIES]
    scenarios = []
    for code, targets in items:
        verdicts, status = _expected(code, targets)
        sid = code if not targets else f"{code}|{'+'.join(targets)}"
        scenarios.append(Scenario(sid, code, targets, verdicts, status))
    if len(scenarios) != RUNS_PER_SEGMENT or len({s.scenario_id for s in scenarios}) != RUNS_PER_SEGMENT:
        raise l3.L3Error("the scenario catalogue must hold 83 unique scenarios")
    return tuple(scenarios)


# ------------------------------------------------------------------ clean scan
@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    extreme: float = 0.0
    max_abs: float = 0.0

    def add(self, values: np.ndarray) -> None:
        n = len(values)
        mean = float(values.mean())
        m2 = float(np.square(values - mean).sum())
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.m2 += m2 + delta * delta * self.count * n / total
        self.count = total
        index = int(np.argmax(np.abs(values)))
        if abs(values[index]) > self.max_abs:
            self.max_abs, self.extreme = float(abs(values[index])), float(values[index])


def clean_scan(rows: pd.DataFrame, config: PreprocessingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per segment-channel gate table, per machine-channel clean-training statistics)."""

    records, moments = [], {}
    for row in rows.itertuples(index=False):
        paths = {m: getattr(row, f"{m}_path") for m in l3.MODALITIES}
        gate = l3.run_gate(paths, row.machine, config)
        for sensor in l3.MODALITIES:
            detail = gate.details[sensor]
            g3 = detail.get("g3", {})
            values = detail.get("values")
            channels = values["flatline_per_channel"] if values else {"<stream>": None}
            for channel, flat in channels.items():
                records.append({
                    "machine": row.machine, "segment_id": row.segment_id, "split_t": row.split_t,
                    "split_s": row.split_s, "sensor": sensor, "channel": channel,
                    "verdict": gate.verdicts[sensor], "g3_passed": bool(g3.get("passed", False)),
                    "stream_duration_s": g3.get("duration_s"), "stream_samples": g3.get("samples"),
                    "samples_in_coverage": values["samples_in_coverage"] if values else None,
                    "non_finite_samples": values["non_finite_per_channel"][channel] if values else None,
                    "flatline_fraction": flat["flatline_fraction"] if flat else None,
                    "longest_run_s": flat["longest_run_s"] if flat else None,
                })
        if row.split_t == "train" and not gate.alignment_failure:
            start, end = gate.coverage
            for sensor in l3.MODALITIES:
                signal = gate.transformed[sensor]
                mask = (signal.timestamps >= start) & (signal.timestamps < end)
                for i, channel in enumerate(signal.channels):
                    moments.setdefault((row.machine, sensor, channel), _Moments()).add(signal.values[mask, i])
    stats = pd.DataFrame([
        {"machine": m, "sensor": s, "channel": c, "training_samples": v.count,
         "training_mean": v.mean, "training_sd": float(np.sqrt(v.m2 / v.count)),
         "training_signed_extreme": v.extreme, "training_max_abs": v.max_abs}
        for (m, s, c), v in sorted(moments.items())
    ])
    return pd.DataFrame(records), stats


def clean_scan_summary(table: pd.DataFrame) -> dict:
    """Acceptance (zero triggers) and the per machine x sensor x channel distribution."""

    triggers = table[table["verdict"].ne(l3.VALID)]
    distribution = []
    for (machine, sensor, channel), group in table.groupby(["machine", "sensor", "channel"], sort=True):
        fractions = group["flatline_fraction"].astype(float)
        runs = group["longest_run_s"].astype(float)
        distribution.append({
            "machine": machine, "sensor": sensor, "channel": channel, "segments": int(len(group)),
            "flatline_fraction_max": float(fractions.max()), "flatline_fraction_p99": float(fractions.quantile(0.99)),
            "flatline_fraction_median": float(fractions.median()),
            "segments_with_any_qualifying_run": int((fractions > 0).sum()),
            "longest_run_s_max": float(runs.max()), "longest_run_s_median": float(runs.median()),
            "stream_duration_s_min": float(group["stream_duration_s"].min()),
        })
    low_margin = sorted({f"{d['machine']}|{d['sensor']}" for d in distribution
                         if d["flatline_fraction_max"] > l3.LOW_MARGIN_FLATLINE_FRACTION})
    segments = table.drop_duplicates(["machine", "segment_id"])
    return {
        "gate_version": l3.GATE_VERSION,
        "segments": {m: int(n) for m, n in segments.groupby("machine").size().items()},
        "segment_channels": int(len(table)),
        "by_split_t": {f"{m}|{s}": int(n) for (m, s), n in segments.groupby(["machine", "split_t"]).size().items()},
        "triggers": int(len(triggers)),
        "trigger_rows": triggers.to_dict(orient="records"),
        "accepted": bool(len(triggers) == 0),
        "low_margin_sensors": low_margin,
        "distribution": distribution,
    }


# ------------------------------------------------------------------ corruption
@dataclass(frozen=True)
class CleanReference:
    """The clean segment's coverage, from the gate on the unmodified files."""

    machine: str
    segment_id: str
    paths: Mapping[str, str]
    coverage: tuple[float, float]


def _seed(*parts: str) -> int:
    return int(sha256("|".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def _region(t: np.ndarray, start: float, end: float, fraction: float) -> np.ndarray:
    length = end - start
    lo, hi = start + (1 - fraction) / 2 * length, start + (1 + fraction) / 2 * length
    return (t >= lo) & (t < hi)


def corrupt_frame(frame: pd.DataFrame, *, code: str, sensor: str, reference: CleanReference,
                  scenario_id: str, stats: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    """A corrupted copy of one modality's raw frame (stored order kept) and its parameters."""

    times = [c for c in frame.columns if str(c).lower() in l3.TIME_COLUMN_NAMES]
    tcol = times[0]
    channels = [c for c in frame.columns if c != tcol]
    out = frame.copy(deep=True)
    t = out[tcol].to_numpy(np.float64)
    start, end = reference.coverage
    cover = (t >= start) & (t < end)
    params: dict = {"code": code, "sensor": sensor}
    machine_stats = stats[stats["machine"].eq(reference.machine) & stats["sensor"].eq(sensor)].set_index("channel")

    def put(mask: np.ndarray, column, values) -> None:
        series = out[column].to_numpy(np.float64, copy=True)
        series[mask] = values
        out[column] = series

    if code in ("D1",):
        return None, params
    if code in ("C1", "C2", "C3"):
        i = int(np.flatnonzero(t >= start + 0.5 * (end - start))[0])
        new_t = t.copy()
        if code == "C1":
            new_t[i], new_t[i + 1] = t[i + 1], t[i]
        elif code == "C2":
            new_t[i + 1] = t[i]
        else:
            new_t[i] = np.nan
        out[tcol] = new_t
        params["row"] = i
        return out, params
    if code == "C4":
        keep = t <= t[0] + 0.5 * (t[-1] - t[0])
        params["rows_kept"] = int(keep.sum())
        return out.loc[keep].reset_index(drop=True), params
    for index, channel in enumerate(channels):
        x = out[channel].to_numpy(np.float64)
        xc = x[cover]
        seg_sd = float(np.sqrt(np.mean(np.square(xc - xc.mean()))))
        if code == "A1" or (code == "A2" and index == 0):
            put(cover, channel, np.nan)
        elif code == "A3":
            put(_region(t, start, end, 0.10), channel, np.nan)
        elif code in ("B1", "M1", "M2"):
            put(cover, channel, l3_dead_value())
        elif code == "B2":
            cut = start + 0.25 * (end - start)
            held = x[np.flatnonzero(cover & (t < cut))[-1]]
            put(cover & (t >= cut), channel, held)
        elif code == "B3":
            put(_region(t, start, end, 0.60), channel, 0.0)
        elif code == "B4":
            region = _region(t, start, end, 0.40)
            put(region, channel, x[np.flatnonzero(region)[0]])
        elif code == "B5":
            value = float(machine_stats.loc[str(channel), "training_signed_extreme"])
            put(_region(t, start, end, 0.60), channel, value)
            params.setdefault("plateau_value", {})[str(channel)] = value
        elif code in GAINS:
            put(cover, channel, xc * GAINS[code])
        elif code == "N4":
            offset = 3.0 * float(machine_stats.loc[str(channel), "training_sd"])
            put(cover, channel, xc + offset)
            params.setdefault("offset", {})[str(channel)] = offset
        elif code in SNR_DB:
            sigma = float(np.sqrt(seg_sd ** 2 / 10 ** (SNR_DB[code] / 10)))
            seed = _seed(reference.machine, reference.segment_id, scenario_id, sensor, str(channel))
            noise = np.random.default_rng(seed).normal(0.0, sigma, size=len(xc))
            put(cover, channel, xc + noise)
            params.setdefault("noise_sigma", {})[str(channel)] = sigma
            params.setdefault("noise_seed", {})[str(channel)] = seed
        elif code == "N8":
            instants = np.arange(start, end, IMPULSE_PERIOD_S)
            idx = np.unique(np.searchsorted(t, instants, side="left"))
            idx = idx[(idx < len(t)) & cover[np.minimum(idx, len(t) - 1)]]
            mask = np.zeros(len(t), dtype=bool)
            mask[idx] = True
            put(mask, channel, x[mask] + IMPULSE_SD * seg_sd)
            params.setdefault("impulses", {})[str(channel)] = int(len(idx))
        elif code == "N9":
            put(cover, channel, xc + seg_sd * np.sin(2 * np.pi * TONE_HZ * (t[cover] - start)))
        elif code in SATURATION_GAINS:
            full_scale = (ACC_FULL_SCALE_G if sensor == "acc"
                          else float(machine_stats.loc[str(channel), "training_max_abs"]))
            put(cover, channel, np.clip(xc * SATURATION_GAINS[code], -full_scale, full_scale))
            params.setdefault("full_scale", {})[str(channel)] = full_scale
        elif code not in ("A2",):
            raise ValueError(f"no corruption rule for {code!r}")
    return out, params


def l3_dead_value() -> float:
    return float(CORRUPTION_PARAMETERS["B1_dead_value"])


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_scenario_files(scenario: Scenario, reference: CleanReference, stats: pd.DataFrame,
                         work_root: Path) -> tuple[dict[str, str | None], dict]:
    """Paths for the run (clean originals for untouched modalities) and the corruption record."""

    paths: dict[str, str | None] = dict(reference.paths)
    record: dict = {"targets": list(scenario.targets), "files": {}, "parameters": {}}
    for sensor in scenario.targets:
        frame = pd.read_parquet(reference.paths[sensor])
        corrupted, params = corrupt_frame(frame, code=scenario.code, sensor=sensor, reference=reference,
                                          scenario_id=scenario.scenario_id, stats=stats)
        record["parameters"][sensor] = params
        if corrupted is None:
            paths[sensor] = None
            record["files"][sensor] = None
            continue
        destination = work_root / reference.machine / reference.segment_id / scenario.scenario_id.replace("|", "__")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{sensor}.parquet"
        corrupted.to_parquet(target, index=False)
        paths[sensor] = str(target)
        record["files"][sensor] = {"path": target.as_posix(), "sha256": file_sha256(target)}
    return paths, record


# ------------------------------------------------------------------ ungated reference
def ungated_scores(paths: Mapping[str, str | None], *, machine: str, config: PreprocessingConfig,
                   bank: l3.DetectorBank, modality: str) -> dict[str, float | str]:
    """Phase-4 scores of ``modality`` with the gate bypassed (descriptive demonstration only)."""

    raw = {s: l3.read_sensor_parquet(paths[s], s) for s in l3.SENSOR_KEYS}
    transformed, _ = l3.apply_startup_transformations(raw, machine, config)
    output: dict[str, float | str] = {}
    try:
        inputs = l3.process_modalities(transformed, (modality,), config, machine)[modality]
    except (l3.L3Error, ValueError) as error:
        return {eid: f"REFUSED: {error}" for eid, b in bank.bundles.items() if b.modality == modality}
    for eid, bundle in bank.bundles.items():
        if bundle.modality == modality:
            try:
                output[eid] = bank.score(eid, inputs)
            except ValueError as error:
                output[eid] = f"REFUSED: {error}"
    return output



# ------------------------------------------------------------------ suite execution
@dataclass
class SuiteOutput:
    runs: list[dict] = field(default_factory=list)             # one row per scenario run
    decisions: list[dict] = field(default_factory=list)        # one row per run x unit
    detector_scores: list[dict] = field(default_factory=list)  # one row per run x surviving detector
    ungated: list[dict] = field(default_factory=list)          # dead-input scores (B1 runs), descriptive


def run_machine_suite(machine: str, *, rows: pd.DataFrame, config: PreprocessingConfig, units: Sequence[Mapping],
                      bank: l3.DetectorBank, records: Mapping, stats: pd.DataFrame, work_root: Path,
                      output: SuiteOutput, progress=None, segment_limit: int | None = None) -> None:
    """All 12 x 83 runs of one machine.  An exception is recorded on its run, never raised (E6)."""

    scenarios = scenario_catalogue()
    for segment_id in l3.selected_validation_segments(rows, machine)[:segment_limit]:
        row = rows[rows["machine"].eq(machine) & rows["segment_id"].eq(segment_id)].iloc[0]
        clean_paths = {m: str(row[f"{m}_path"]) for m in l3.MODALITIES}
        clean_gate = l3.run_gate(clean_paths, machine, config)
        if clean_gate.alignment_failure or clean_gate.integrity_flag:
            raise l3.L3Error(f"selected clean segment {segment_id} does not pass the gate")
        reference = CleanReference(machine, segment_id, clean_paths, clean_gate.coverage)
        for scenario in scenarios:
            run_id = f"{machine}|{segment_id}|{scenario.scenario_id}"
            entry = {"run_id": run_id, "machine": machine, "segment_id": segment_id,
                     "parent_recording": str(row["parent_recording"]), "scenario_id": scenario.scenario_id,
                     "code": scenario.code, "targets": "+".join(scenario.targets),
                     "coverage_start_s": reference.coverage[0], "coverage_end_s": reference.coverage[1],
                     "expected_status": scenario.expected_status,
                     "expected_verdicts": None if scenario.expected_verdicts is None
                     else "|".join(f"{m}={scenario.expected_verdicts[m]}" for m in l3.MODALITIES)}
            try:
                paths, corruption = write_scenario_files(scenario, reference, stats, work_root)
                result = l3.run_segment(paths, machine=machine, config=config, units=units, bank=bank,
                                        records=records)
                entry.update({
                    "exception": None, "corruption": corruption,
                    "alignment_failure": result.gate.alignment_failure,
                    "alignment_reason": result.gate.alignment_reason,
                    "integrity_flag": result.gate.integrity_flag,
                    "survivors": "+".join(result.gate.survivors) if not result.gate.alignment_failure else "",
                    **{f"verdict_{m}": result.gate.verdicts[m] for m in l3.MODALITIES},
                    **{f"flatline_max_{m}": _max_flatline(result.gate.details[m]) for m in l3.MODALITIES},
                    **{f"non_finite_{m}": _non_finite(result.gate.details[m]) for m in l3.MODALITIES},
                    "window_count": max(result.window_counts.values()) if result.window_counts else None,
                })
                for decision in result.decisions:
                    output.decisions.append({"run_id": run_id, **decision})
                for experiment_id, score in result.raw_scores.items():
                    output.detector_scores.append({"run_id": run_id, "experiment_id": experiment_id,
                                                   "modality": bank.bundles[experiment_id].modality,
                                                   "score": score})
                if scenario.code == "B1":
                    dead = scenario.targets[0]
                    for experiment_id, score in ungated_scores(paths, machine=machine, config=config, bank=bank,
                                                               modality=dead).items():
                        output.ungated.append({"run_id": run_id, "machine": machine, "segment_id": segment_id,
                                               "modality": dead, "experiment_id": experiment_id,
                                               "dead_input_score": score})
            except Exception as error:  # noqa: BLE001 - E6: every escape is recorded and fails the suite
                entry.update({"exception": f"{type(error).__name__}: {error}"})
            output.runs.append(entry)
            if progress:
                progress(entry)


def _max_flatline(detail: Mapping) -> float | None:
    values = detail.get("values")
    return None if not values else max(v["flatline_fraction"] for v in values["flatline_per_channel"].values())


def _non_finite(detail: Mapping) -> int | None:
    values = detail.get("values")
    return None if not values else int(sum(values["non_finite_per_channel"].values()))


# ------------------------------------------------------------------ assertions
ATOL = 1e-9
FULL = ("mic", "acc", "gyro")


def _recorded_validation(bank: l3.DetectorBank) -> dict[str, pd.Series]:
    return {eid: pd.read_csv(b.bundle_path / "validation_scores.csv", dtype={"segment_id": str},
                             float_precision="round_trip").set_index("segment_id")["score"]
            for eid, b in bank.bundles.items()}


def _unit_key(unit: Mapping) -> str:
    return f"{unit['machine']}|{unit['regime']}|{unit['family']}|{unit['seed']}"


def evaluate_assertions(runs: pd.DataFrame, decisions: pd.DataFrame, scores: pd.DataFrame, *,
                        banks: Mapping[str, l3.DetectorBank], units: Sequence[Mapping], records: Mapping,
                        persisted_fused: pd.DataFrame) -> dict:
    """E0-E6 and bridging conditions B1-B4 (B5 is an assumption), recomputed independently of the runner."""

    unit_by_key = {_unit_key(u): u for u in units}
    recorded = {m: _recorded_validation(b) for m, b in banks.items()}
    run_index = runs.set_index("run_id")
    score_map = {(r.run_id, r.experiment_id): float(r.score) for r in scores.itertuples(index=False)}
    clean_ids = runs[runs["code"].eq("CLEAN")].set_index(["machine", "segment_id"])["run_id"]
    machines = sorted(banks)
    result: dict = {}

    # E6 -- no exception escaped.
    exceptions = runs[runs["exception"].notna()]
    result["E6"] = {"passed": bool(exceptions.empty), "exceptions": int(len(exceptions)),
                    "examples": exceptions[["run_id", "exception"]].head(10).to_dict(orient="records")}

    # E0 -- the clean runner reproduces every Phase-4 recorded validation score.
    worst, compared = 0.0, 0
    for (machine, segment), run_id in clean_ids.items():
        for eid in banks[machine].bundles:
            diff = abs(score_map[(run_id, eid)] - float(recorded[machine][eid][segment]))
            worst, compared = max(worst, diff), compared + 1
    result["E0"] = {"passed": bool(compared == 12 * 78 * len(machines) and worst <= ATOL),
                    "comparisons": compared, "max_abs_difference": worst, "tolerance": ATOL}

    # E1 -- gate verdicts and status equal the scenario expectation.
    status_by_run = decisions.groupby("run_id")["status"].agg(lambda s: sorted(set(s)))
    expected_runs = runs[runs["expected_status"].notna()]
    mismatches = []
    for row in expected_runs.itertuples(index=False):
        if isinstance(row.exception, str):
            mismatches.append({"run_id": row.run_id, "exception": row.exception})
            continue
        observed = "|".join(f"{m}={getattr(row, f'verdict_{m}')}" for m in l3.MODALITIES)
        statuses = status_by_run.get(row.run_id, [])
        if observed != row.expected_verdicts or statuses != [row.expected_status]:
            mismatches.append({"run_id": row.run_id, "code": row.code, "expected": row.expected_verdicts,
                               "observed": observed, "expected_status": row.expected_status,
                               "observed_status": statuses})
    by_code = {}
    for code, group in expected_runs.groupby("code"):
        bad = {m["run_id"] for m in mismatches}
        by_code[code] = {"runs": int(len(group)), "matching": int((~group["run_id"].isin(bad)).sum())}
    result["E1"] = {"passed": not mismatches, "runs_checked": int(len(expected_runs)),
                    "mismatches": len(mismatches), "by_code": by_code, "examples": mismatches[:10]}

    # E2 / B2 -- untouched surviving modalities score exactly as on the clean run.
    e2_worst, e2_count, e2_runs = 0.0, 0, 0
    value_runs = runs[runs["code"].isin(VALUE_FAILURE_CODES + ("B4", "M1")) & runs["exception"].isna()]
    for row in value_runs.itertuples(index=False):
        clean_run = clean_ids[(row.machine, row.segment_id)]
        untouched = [m for m in l3.MODALITIES
                     if getattr(row, f"verdict_{m}") == l3.VALID and m not in row.targets.split("+")]
        for eid, bundle in banks[row.machine].bundles.items():
            if bundle.modality in untouched:
                diff = abs(score_map[(row.run_id, eid)] - score_map[(clean_run, eid)])
                e2_worst, e2_count = max(e2_worst, diff), e2_count + 1
        e2_runs += 1
    result["E2"] = {"passed": bool(e2_count > 0 and e2_worst <= ATOL), "runs": e2_runs, "comparisons": e2_count,
                    "max_abs_difference": e2_worst, "tolerance": ATOL,
                    "scope": "untouched surviving modalities of A1-A3, B1-B5 and M1 runs vs the same segment's "
                             "clean run, every frozen Phase-4 detector of those modalities"}

    # E3 / B3 -- survivor-subset threshold; two-failure decisions equal the single Phase-4 decision.
    violations, pair_n, all_n, single_n, monotone = [], 0, 0, 0, 0
    scored = decisions[decisions["status"].isin([l3.DECISION_ALL, l3.DECISION_PAIR, l3.DECISION_SINGLE])]
    for row in scored.itertuples(index=False):
        unit = unit_by_key[row.unit]
        survivors = tuple(row.survivors.split("+"))
        machine, segment = run_index.loc[row.run_id, "machine"], run_index.loc[row.run_id, "segment_id"]
        if row.status == l3.DECISION_SINGLE:
            (modality,) = survivors
            eid = unit["phase4_single_modality_ids"][modality]
            frozen_threshold = float(banks[machine].bundles[eid].threshold.value)
            clean_score = float(recorded[machine][eid][segment])
            ok = (row.threshold == frozen_threshold and bool(row.anomaly) == (clean_score > frozen_threshold)
                  and abs(row.score - clean_score) <= ATOL)
            c = records[(row.unit, FULL)]["modality_calibration"][modality]
            z_score, z_threshold = (row.score - c["median"]) / c["iqr"], (row.threshold - c["median"]) / c["iqr"]
            monotone += int((z_score > z_threshold) == bool(row.anomaly))
            single_n += 1
        else:
            record = records[(row.unit, survivors)]
            ok = (row.threshold_source == record["calibration_id"]
                  and row.threshold == float(record["threshold"]["value"])
                  and (len(survivors) == 3) == (row.status == l3.DECISION_ALL))
            pair_n += int(row.status == l3.DECISION_PAIR)
            all_n += int(row.status == l3.DECISION_ALL)
        if not ok:
            violations.append({k: v for k, v in row._asdict().items()})
    result["E3"] = {"passed": bool(not violations and single_n > 0 and pair_n > 0 and monotone == single_n),
                    "all_three_decisions": all_n, "pair_decisions": pair_n, "single_decisions": single_n,
                    "single_robust_z_decision_identical": monotone, "violations": len(violations),
                    "examples": violations[:10]}

    # E4 -- C, D and M2 abstain with the defined status; every unit row is retained and counted.
    abstain_runs = runs[runs["code"].isin(("C1", "C2", "C3", "C4", "D1", "M2"))].set_index("run_id")
    subset = decisions[decisions["run_id"].isin(abstain_runs.index)]
    bad_runs = []
    for run_id, expected_status in abstain_runs["expected_status"].items():
        group = subset[subset["run_id"].eq(run_id)]
        if not (len(group) == 26 and set(group["status"]) == {expected_status} and group["integrity_flag"].all()
                and group["anomaly"].isna().all() and group["score"].isna().all()):
            bad_runs.append(run_id)
    result["E4"] = {"passed": bool(not bad_runs and abstain_runs["exception"].isna().all()),
                    "abstaining_runs": int(len(abstain_runs)), "unit_rows": int(len(subset)),
                    "unit_rows_by_status": {k: int(v) for k, v in subset["status"].value_counts().items()},
                    "failing_runs": bad_runs[:10]}

    # E5 -- clean and N1-N9: L3 equals an independently computed L2(all three) in score and decision.
    fused_index = persisted_fused.set_index(["calibration_id", "segment_id"])["fused_score"]
    e5_runs = runs[runs["code"].isin(("CLEAN",) + NEGATIVE_CODES)]["run_id"]
    e5_bad, e5_count, e5_persisted = [], 0, 0.0
    for row in decisions[decisions["run_id"].isin(e5_runs)].itertuples(index=False):
        unit = unit_by_key[row.unit]
        record = records[(row.unit, FULL)]
        total = 0.0
        for m in FULL:
            c = record["modality_calibration"][m]
            total = total + (score_map[(row.run_id, unit["phase4_single_modality_ids"][m])] - c["median"]) / c["iqr"]
        expected = total / 3
        if (row.status != l3.DECISION_ALL or row.score != expected
                or bool(row.anomaly) != (expected > float(record["threshold"]["value"]))):
            e5_bad.append({k: v for k, v in row._asdict().items()})
        if run_index.loc[row.run_id, "code"] == "CLEAN":
            persisted = float(fused_index[(record["calibration_id"], run_index.loc[row.run_id, "segment_id"])])
            e5_persisted = max(e5_persisted, abs(row.score - persisted))
        e5_count += 1
    result["E5"] = {"passed": bool(not e5_bad and e5_count == 12 * 26 * 28 * len(machines)
                                   and e5_persisted <= ATOL),
                    "unit_decisions_checked": e5_count, "violations": len(e5_bad),
                    "clean_vs_persisted_l2_max_abs_difference": e5_persisted, "tolerance": ATOL,
                    "examples": e5_bad[:10]}

    value_codes_ok = all(by_code.get(c, {}).get("runs") == by_code.get(c, {}).get("matching")
                         for c in VALUE_FAILURE_CODES)
    result["bridging_conditions"] = {
        "B1": {"evidence": "E1 restricted to A1-A3, B1-B3, B5", "holds": bool(value_codes_ok)},
        "B2": {"evidence": "E2", "holds": result["E2"]["passed"]},
        "B3": {"evidence": "E3", "holds": result["E3"]["passed"]},
        "B4": {"evidence": "E4: C and D abstain, so only A/B failures can map to the pair tables",
               "holds": result["E4"]["passed"]},
        "B5": {"status": "ASSUMPTION - not testable on this dataset",
               "statement": "the surviving physical sensors are unaffected by whatever failed the other one; every "
                            "synthetic corruption alters only its target file, so the suite cannot test it"},
    }
    result["all_pass_criteria_passed"] = all(result[k]["passed"] for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"))
    return result


def saturation_characterisation(runs: pd.DataFrame, decisions: pd.DataFrame) -> list[dict]:
    """S1-S4: how often the gate excludes the saturated modality and what the decision becomes."""

    out = []
    sat = runs[runs["code"].isin(SATURATION_CODES) & runs["exception"].isna()]
    for (code, target), group in sat.groupby(["code", "targets"], sort=True):
        d = decisions[decisions["run_id"].isin(group["run_id"])]
        alarms = d["anomaly"].dropna().astype(bool)
        out.append({"code": code, "gain": SATURATION_GAINS[code], "modality": target, "runs": int(len(group)),
                    "modality_gated_runs": int(group[f"verdict_{target}"].ne(l3.VALID).sum()),
                    "flatline_max_median": float(group[f"flatline_max_{target}"].median()),
                    "flatline_max_max": float(group[f"flatline_max_{target}"].max()),
                    "unit_status_counts": {k: int(v) for k, v in d["status"].value_counts().items()},
                    "unit_alarm_rate": float(alarms.mean()) if len(alarms) else None})
    return out


def ungated_demonstration(ungated: pd.DataFrame, *, banks: Mapping[str, l3.DetectorBank], units: Sequence[Mapping],
                          records: Mapping, persisted_fused: pd.DataFrame) -> list[dict]:
    """Descriptive: source-validation FPR of ungated L2(all three) with one dead modality vs L3's pair.

    The dead-input score of each detector is taken from the B1 runs (zero input over the
    coverage); the other two modalities keep their recorded clean validation scores.
    In-sample on the validation cohort that set the thresholds: illustrative only.
    """

    rows = []
    recorded = {m: _recorded_validation(b) for m, b in banks.items()}
    fused = persisted_fused.set_index("calibration_id")
    for unit in units:
        key = _unit_key(unit)
        full = records[(key, FULL)]
        for dead in l3.MODALITIES:
            values = ungated[ungated["experiment_id"].eq(unit["phase4_single_modality_ids"][dead])]["dead_input_score"]
            numeric = pd.to_numeric(values, errors="coerce")
            base = {"unit": key, "dead_modality": dead, "dead_input_runs": int(len(values))}
            if len(values) == 0 or numeric.isna().any():
                rows.append({**base, "ungated_status": "FROZEN_PIPELINE_REFUSED" if len(values) else "NOT_RUN",
                             "refusal": str(values[numeric.isna()].iloc[0]) if len(values) else None})
                continue
            dead_score = float(numeric.median())
            segments = recorded[unit["machine"]][unit["phase4_single_modality_ids"]["mic"]].index
            total = np.zeros(len(segments))
            for m in FULL:
                c = full["modality_calibration"][m]
                raw = (np.full(len(segments), dead_score) if m == dead else
                       recorded[unit["machine"]][unit["phase4_single_modality_ids"][m]].loc[segments].to_numpy())
                total = total + (raw - c["median"]) / c["iqr"]
            pair_record = records[(key, tuple(m for m in FULL if m != dead))]
            pair_scores = fused.loc[pair_record["calibration_id"], "fused_score"].to_numpy(np.float64)
            c_dead = full["modality_calibration"][dead]
            rows.append({**base, "ungated_status": "SCORED", "dead_input_score": dead_score,
                         "dead_input_score_range": float(numeric.max() - numeric.min()),
                         "dead_input_robust_z": (dead_score - c_dead["median"]) / c_dead["iqr"],
                         "validation_segments": int(len(segments)),
                         "ungated_l2_all_three_fpr": float(np.mean(total / 3 > full["threshold"]["value"])),
                         "l3_pair_fpr": float(np.mean(pair_scores > pair_record["threshold"]["value"]))})
    return rows
