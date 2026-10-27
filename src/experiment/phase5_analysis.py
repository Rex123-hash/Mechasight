"""SYNTHETIC analyse-once wiring to the unchanged V2 statistical machinery.

No reader or protected scope lives here. Only Executor's completed, explicitly
authorized synthetic analysis stage calls this function.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math

import numpy as np
import pandas as pd

from src.experiment.phase32 import BootstrapDistribution, make_common_bootstrap_draws, paired_distribution_difference
from src.experiment.phase4_analysis import experiment_metrics, family_bootstrap
from src.experiment.phase5_fixture import NORMAL, ANOMALY
from src.experiment.phase5_inputs import SYNTHETIC, require
from src.fusion.statistics import classify_metric, joint_category, operating_axis, threshold_metric_bootstrap

METRICS = ("auroc", "healthy_target_fpr", "recall", "average_precision", "standardized_pauroc_fpr_0_10")
PRIMARY = METRICS[:3]


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def abstention_accounting(decisions):
    """Development decision accounting only; never create protected L3 runs."""
    return {"total": len(decisions), "decisions": sum(d is not None for d in decisions),
            "abstentions": sum(d is None for d in decisions), "dropped": 0}


def analyse_synthetic(executor, access):
    executor._verify_capability(access, "analysis")
    state = executor.verify_complete()
    package = executor.package
    labels = {}
    draws = {}
    for machine in ("RoboticArm", "BrushlessMotor"):
        frame = executor.inputs.labels(access, machine).sort_values("segment_id").reset_index(drop=True)
        require(set(frame["fake_label"]) <= {NORMAL, ANOMALY}, "unrecognized synthetic label")
        require(frame["segment_id"].is_unique and len(frame) == len(executor.fixture["cohorts"][machine]), "label cohort mismatch")
        expected = pd.DataFrame(executor.fixture["cohorts"][machine]).sort_values("segment_id").reset_index(drop=True)
        require(frame[list(expected.columns)].equals(expected), "label parent/domain mismatch")
        frame["is_anomaly"] = frame["fake_label"].eq(ANOMALY).astype(int)
        frame["machine"] = machine
        labels[machine] = frame
        draws[machine] = make_common_bootstrap_draws(frame, replicates=10_000, seed=1995)
    frames, thresholds, points = {}, {}, {}
    groups = {}
    for unit in package.registry["units"]:
        cell = (unit["machine"], unit["regime"], unit["family"])
        arms = {**unit["L0"], "L1": unit["L1"]["experiment_id"], **{"L2:" + k: v for k, v in unit["L2"].items()}}
        groups.setdefault(cell, {})[unit["seed"]] = arms
        for arm, eid in arms.items():
            if eid in state["tables"]:
                table = pd.DataFrame(executor.sandbox.json(state["tables"][eid]["file"])["rows"])
            else:
                table = executor.inputs.historical(access, eid)
            table = executor._aligned(table, unit["machine"])
            merged = table.merge(labels[unit["machine"]][["segment_id", "is_anomaly"]], on="segment_id", validate="one_to_one")
            key = (*cell, unit["seed"], arm)
            frames[key] = merged
            threshold = (package.registry["calibrations"][eid]["threshold"]["value"] if arm.startswith("L2:")
                         else package.registry["detectors"][eid]["threshold"]["value"])
            thresholds[key] = threshold
            points[eid] = experiment_metrics(merged, threshold=threshold)
    cache = {}

    def distribution(cell, arm, metric, domain="complete"):
        key = (*cell, arm, metric, domain)
        if key in cache:
            return cache[key]
        seeds = tuple(sorted(groups[cell]))
        by_seed = {s: frames[(*cell, s, arm)] for s in seeds}
        shared = draws[cell[0]]
        if domain != "complete":
            mask = labels[cell[0]]["domain"].eq(domain).to_numpy()
            by_seed = {s: f.loc[mask].reset_index(drop=True) for s, f in by_seed.items()}
            shared = replace(shared, sample_weights=shared.sample_weights[:, mask], identity=shared.identity + ":" + domain)
        if metric in {"healthy_target_fpr", "recall"}:
            result = threshold_metric_bootstrap(by_seed, {s: thresholds[(*cell, s, arm)] for s in seeds},
                                                seeds=seeds, metric=metric, draws=shared)
        elif by_seed[seeds[0]]["is_anomaly"].nunique() < 2:
            result = BootstrapDistribution(metric, float("nan"), np.full(10_000, np.nan), 10_000, 0, 10_000,
                                           None, None, shared.identity)
        else:
            result = family_bootstrap(by_seed, seeds=seeds, metric=metric, draws=shared)
        cache[key] = result
        return result

    intervals, joint = [], []

    def contrast(cell, first, second, name, *, domain="complete", metrics=METRICS, other_cell=None):
        categories = {}
        for metric in metrics:
            a, b = distribution(cell, first, metric, domain), distribution(other_cell or cell, second, metric, domain)
            difference = paired_distribution_difference(a, b, orientation="first_minus_second")
            category = asdict(classify_metric(difference, metric=metric)) if metric != METRICS[-1] else {"category": "SECONDARY_NO_PRACTICAL_MARGIN", "tag": None}
            categories[metric] = category
            intervals.append({"scope": SYNTHETIC, "cell": list(cell), "contrast": name, "first": first, "second": second,
                              "domain": domain, "nominal": True, **difference, **category})
        if name in {"P1:mic", "P1:acc", "P1:gyro", "P2", "P3:acc+gyro", "P3:mic+gyro", "P3:mic+acc"} and domain == "complete":
            from src.fusion.statistics import MetricCategory
            c = {m: MetricCategory(**categories[m]) for m in PRIMARY}
            joint.append({"cell": list(cell), "contrast": name,
                          "category": joint_category(c["auroc"], operating_axis(c["healthy_target_fpr"], c["recall"]))})

    fusion = ("L1", "L2:mic+acc+gyro", "L2:acc+gyro", "L2:mic+gyro", "L2:mic+acc")
    for cell in groups:
        primary = [("L2:mic+acc+gyro", m, "P1:" + m) for m in ("mic", "acc", "gyro")]
        primary += [("L1", "L2:mic+acc+gyro", "P2")]
        primary += [("L2:" + pair, "L2:mic+acc+gyro", "P3:" + pair) for pair in ("acc+gyro", "mic+gyro", "mic+acc")]
        for first, second, name in primary:
            contrast(cell, first, second, name)
            for domain in ("source", "target"):
                contrast(cell, first, second, "S4:" + name, domain=domain)
        for m in ("mic", "acc", "gyro"):
            contrast(cell, "L1", m, "S1")
        for pair in ("acc+gyro", "mic+gyro", "mic+acc"):
            for m in pair.split("+"):
                contrast(cell, "L2:" + pair, m, "S2")
        best = max(("mic", "acc", "gyro"), key=lambda m: distribution(cell, m, "auroc").point_estimate)
        for arm in fusion:
            contrast(cell, arm, best, "S3_POST_HOC_BEST_SINGLE")
        # Slice distributions are not needed after this cell; retain only full
        # cohort arrays for paired T-S contrasts, limiting memory use.
        for key in list(cache):
            if key[-1] != "complete":
                del cache[key]
    for machine, regime, family in groups:
        if regime == "T":
            for arm in fusion:
                contrast((machine, regime, family), arm, arm, "S6:T-S", other_cell=(machine, "S", family))
    primary_rows = [r for r in intervals if r["contrast"].startswith("P") and r["metric"] in PRIMARY]
    require(len(primary_rows) == 420 and len(joint) == 140, "frozen primary contrast count mismatch")
    yardstick = {}
    for name in ("P1", "P2", "P3"):
        for metric in PRIMARY:
            rows = [r for r in primary_rows if r["contrast"].startswith(name) and r["metric"] == metric]
            yardstick[name + ":" + metric] = {"intervals": len(rows), "expected_null_exclusions_approx": .05 * len(rows),
                "observed_exclusions": sum(r["ci_lower"] is not None and (r["ci_lower"] > 0 or r["ci_upper"] < 0) for r in rows)}
    return finite_json({"scope": SYNTHETIC, "not_real_performance": True, "replicates": 10_000, "seed": 1995,
                        "draws": {m: d.identity for m, d in draws.items()}, "unit_points": points,
                        "intervals": intervals, "primary_intervals": 420, "primary_contrasts": 140, "joint_categories": joint,
                        "multiplicity": package.contract["contrasts"]["multiplicity"], "null_yardstick": yardstick,
                        "protected_L3_executions": 0, "abstentions": 0, "missing_scores": 0})
