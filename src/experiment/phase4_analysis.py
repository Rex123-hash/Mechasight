"""Single-Modality Benchmarking: analyse-once metric and contrast layer.

Runs only after ``ScoreStaging.verify_complete`` has accepted all 156 hashed
score tables.  Labels are joined here for the first time.  Reporting
conventions follow the Phase-3.2.1 correction: count-identity F1, undefined
precision when nothing was flagged, AP with its slice prevalence.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.experiment.metrics import protocol_b_ranking_metrics, protocol_b_threshold_metrics
from src.experiment.phase32 import (
    BootstrapDistribution,
    CommonBootstrapDraws,
    make_common_bootstrap_draws,
    paired_distribution_difference,
    ranking_metric_value,
    weighted_bootstrap_metric,
)
from src.experiment.provenance import canonical_json, file_sha256
from src.experiment.reporting_correction import (
    PRECISION_UNDEFINED,
    count_identity_f1,
    false_positive_rate_or_undefined,
    precision_or_undefined,
    prevalence_context,
    produced_no_alarms,
    recall_or_undefined,
)


CONTRAST_METRICS = ("auroc", "average_precision", "standardized_pauroc_fpr_0_10")
MODALITY_PAIRS = (("acc", "mic"), ("gyro", "mic"), ("acc", "gyro"))
LABEL_COLUMNS = (
    "segment_id", "parent_recording", "domain", "anomaly_label",
    "operating_condition", "background_condition",
)


def join_labels(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach frozen labels to a label-free score table by segment identity."""

    missing = set(LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"label table lacks {sorted(missing)}")
    left = scores.copy()
    left["segment_id"] = left["segment_id"].astype(str)
    right = labels.loc[:, list(LABEL_COLUMNS)].copy()
    right["segment_id"] = right["segment_id"].astype(str)
    if set(left["segment_id"]) != set(right["segment_id"]) or left["segment_id"].duplicated().any():
        raise ValueError("score table segment identities differ from the frozen label cohort")
    joined = left.merge(right, on="segment_id", how="inner", suffixes=("", "_label"))
    if not joined["domain"].astype(str).eq(joined["domain_label"].astype(str)).all():
        raise ValueError("score table domain differs from frozen label domain")
    if not joined["parent_recording"].astype(str).eq(joined["parent_recording_label"].astype(str)).all():
        raise ValueError("score table parent recording differs from frozen label parent")
    joined = joined.drop(columns=["domain_label", "parent_recording_label"])
    joined["is_anomaly"] = joined["anomaly_label"].astype(str).str.lower().ne("normal").astype(int)
    return joined.sort_values("segment_id").reset_index(drop=True)


def _slice_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["is_anomaly"].to_numpy()
    scores = frame["score"].to_numpy(dtype=np.float64)
    anomaly = int(labels.sum())
    normal = int(len(labels) - anomaly)
    if anomaly == 0 or normal == 0:
        return {
            "status": "UNDEFINED - ONE CLASS PRESENT", "sample_count": int(len(labels)),
            "anomaly_count": anomaly, "normal_count": normal,
            "auroc": None, "average_precision": None, "standardized_pauroc_fpr_0_10": None,
            "prevalence": anomaly / len(labels) if len(labels) else None, "ap_minus_prevalence": None,
        }
    ranking = protocol_b_ranking_metrics(labels, scores)
    context = prevalence_context(
        anomaly_count=anomaly, normal_count=normal, average_precision=ranking["average_precision"]
    )
    return {
        "status": "defined",
        "sample_count": int(len(labels)),
        "anomaly_count": anomaly,
        "normal_count": normal,
        "auroc": ranking["auroc"],
        "average_precision": ranking["average_precision"],
        "standardized_pauroc_fpr_0_10": ranking["standardized_pauroc_fpr_0_10"],
        "prevalence": context.prevalence,
        "ap_minus_prevalence": context.ap_minus_prevalence,
    }


def _threshold_block(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    raw = protocol_b_threshold_metrics(
        frame["is_anomaly"].to_numpy(), frame["score"].to_numpy(dtype=np.float64), threshold=threshold
    )
    precision = precision_or_undefined(raw.true_positive, raw.false_positive)
    return {
        "threshold": float(threshold),
        "true_negative": raw.true_negative,
        "false_positive": raw.false_positive,
        "false_negative": raw.false_negative,
        "true_positive": raw.true_positive,
        "false_positive_rate": false_positive_rate_or_undefined(raw.false_positive, raw.true_negative),
        "recall": recall_or_undefined(raw.true_positive, raw.false_negative),
        "precision": PRECISION_UNDEFINED if precision is None else precision,
        "phase321_count_identity_f1": count_identity_f1(
            raw.true_positive, raw.false_positive, raw.false_negative
        ),
        "produced_no_positive_predictions": produced_no_alarms(raw.true_positive, raw.false_positive),
    }


def experiment_metrics(joined: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    complete = _slice_metrics(joined)
    complete["threshold"] = _threshold_block(joined, threshold)
    domains = joined["domain"].astype(str)
    source = _slice_metrics(joined[domains.eq("source")])
    target = _slice_metrics(joined[domains.eq("target")])
    healthy = joined[domains.eq("target") & joined["is_anomaly"].eq(0)]
    false_positive = int((healthy["score"].to_numpy(dtype=np.float64) > threshold).sum())
    healthy_n = int(len(healthy))
    return {
        "complete": complete,
        "source": source,
        "target": target,
        "healthy_target": {
            "healthy_target_sample_count": healthy_n,
            "healthy_target_parent_count": int(healthy["parent_recording"].nunique()),
            "false_positive": false_positive,
            "false_positive_rate": false_positive / healthy_n if healthy_n else None,
        },
    }


def family_bootstrap(
    frames_by_seed: Mapping[int, pd.DataFrame],
    *,
    seeds: tuple[int, ...],
    metric: str,
    draws: CommonBootstrapDraws,
) -> BootstrapDistribution:
    """Seed-averaged family metric per replicate (frozen R8 rule)."""

    if set(frames_by_seed) != set(seeds):
        raise ValueError("family frames do not contain the frozen seed set")
    ordered = {seed: frames_by_seed[seed].sort_values("segment_id").reset_index(drop=True) for seed in seeds}
    base = ordered[seeds[0]]
    for seed in seeds[1:]:
        if not ordered[seed]["segment_id"].equals(base["segment_id"]) or not ordered[seed]["is_anomaly"].equals(base["is_anomaly"]):
            raise ValueError("family seed frames do not share identical evaluation units")
    labels = base["is_anomaly"].to_numpy()
    point = float(np.mean([ranking_metric_value(metric, labels, ordered[seed]["score"].to_numpy()) for seed in seeds]))
    per_seed = np.vstack([
        weighted_bootstrap_metric(labels, ordered[seed]["score"].to_numpy(dtype=np.float64), draws.sample_weights, metric=metric)
        for seed in seeds
    ])
    values = np.mean(per_seed, axis=0)
    valid = values[np.isfinite(values)]
    lower, upper = (None, None)
    if len(valid):
        lower, upper = map(float, np.quantile(valid, [0.025, 0.975]))
    return BootstrapDistribution(
        metric=metric, point_estimate=point, values=values,
        replicates_requested=draws.replicates, valid_replicates=int(len(valid)),
        invalid_replicates=int(draws.replicates - len(valid)),
        ci_lower=lower, ci_upper=upper, draw_identity=draws.identity,
    )


def paired_contrast(first: BootstrapDistribution, second: BootstrapDistribution, *, orientation: str) -> dict[str, Any]:
    return paired_distribution_difference(first, second, orientation=orientation)


def seed_summary(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    values = np.asarray([values_by_seed[seed] for seed in sorted(values_by_seed)], dtype=np.float64)
    return {
        "seed_count": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "per_seed": {int(seed): float(values_by_seed[seed]) for seed in sorted(values_by_seed)},
    }


def dominance_table(contrasts: pd.DataFrame, *, families: Iterable[str]) -> pd.DataFrame:
    """Pre-registered rule: same-sign CI excluding zero in *every* family."""

    families = tuple(families)
    rows = []
    for (machine, regime, pair, metric), group in contrasts.groupby(["machine", "regime", "pair", "metric"], sort=True):
        by_family = group.set_index("family")
        if set(by_family.index) != set(families):
            raise ValueError("dominance rule requires one contrast per family")
        positive = int(((by_family["ci_lower"] > 0)).sum())
        negative = int(((by_family["ci_upper"] < 0)).sum())
        if positive == len(families):
            verdict = "first_dominates"
        elif negative == len(families):
            verdict = "second_dominates"
        else:
            verdict = "no_consistent_dominance"
        rows.append({
            "machine": machine, "regime": regime, "pair": pair, "metric": metric,
            "family_count": len(families),
            "families_excluding_zero": positive + negative,
            "families_positive": positive,
            "families_negative": negative,
            "verdict": verdict,
            "mean_point_delta": float(by_family["point_delta"].mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ orchestration

def analyse_once(
    *,
    staging,
    bundles: Mapping[str, Any],
    labels_by_machine: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    families: tuple[str, ...],
    seeds_for: Any,
    modalities: tuple[str, ...],
    replicates: int = 10_000,
    bootstrap_seed: int = 1995,
) -> dict[str, Any]:
    """Compute every metric exactly once over the complete, verified batch."""

    manifest = staging.verify_complete(bundles)
    if manifest.get("metrics_computed"):
        raise RuntimeError("metrics were already computed for this batch; a completed analysis is final")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined: dict[str, pd.DataFrame] = {}
    individual_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    healthy_rows: list[dict[str, Any]] = []
    for experiment_id, bundle in sorted(bundles.items()):
        config = bundle.config
        frame = join_labels(staging.read_scores(experiment_id), labels_by_machine[config.machine])
        joined[experiment_id] = frame
        metrics = experiment_metrics(frame, threshold=bundle.threshold.value)
        identity = {
            "experiment_id": experiment_id, "machine": config.machine, "regime": config.regime,
            "modality": bundle.modality, "family": config.model_family, "seed": config.model_seed,
        }
        individual_rows.append({
            **identity,
            **{key: value for key, value in metrics["complete"].items() if key != "threshold"},
            **{f"threshold_{key}": value for key, value in metrics["complete"]["threshold"].items()},
        })
        for domain in ("source", "target"):
            slice_rows.append({**identity, "domain": domain, **metrics[domain]})
        healthy_rows.append({**identity, **metrics["healthy_target"]})
        (out / "experiments").mkdir(exist_ok=True)
        (out / "experiments" / f"{experiment_id}.json").write_text(
            canonical_json({**identity, "threshold": asdict(bundle.threshold), "metrics": metrics}) + "\n",
            encoding="utf-8",
        )

    draws = {
        machine: make_common_bootstrap_draws(
            next(f for e, f in joined.items() if bundles[e].config.machine == machine).assign(machine=machine),
            replicates=replicates, seed=bootstrap_seed,
        )
        for machine in sorted({b.config.machine for b in bundles.values()})
    }

    family_rows: list[dict[str, Any]] = []
    distributions: dict[tuple, BootstrapDistribution] = {}
    groups: dict[tuple, dict[int, str]] = {}
    for experiment_id, bundle in bundles.items():
        key = (bundle.config.machine, bundle.config.regime, bundle.modality, bundle.config.model_family)
        groups.setdefault(key, {})[bundle.config.model_seed] = experiment_id
    for key in sorted(groups):
        machine, regime, modality, family = key
        seeds = seeds_for(family)
        frames = {seed: joined[groups[key][seed]] for seed in seeds}
        row: dict[str, Any] = {"machine": machine, "regime": regime, "modality": modality, "family": family, "seed_count": len(seeds)}
        for metric in CONTRAST_METRICS:
            distribution = family_bootstrap(frames, seeds=seeds, metric=metric, draws=draws[machine])
            distributions[(*key, metric)] = distribution
            per_seed = {seed: ranking_metric_value(metric, frames[seed]["is_anomaly"].to_numpy(), frames[seed]["score"].to_numpy()) for seed in seeds}
            summary = seed_summary(per_seed)
            row.update({
                f"{metric}": distribution.point_estimate,
                f"{metric}_ci_lower": distribution.ci_lower,
                f"{metric}_ci_upper": distribution.ci_upper,
                f"{metric}_seed_sd": summary["sd"],
                f"{metric}_valid_replicates": distribution.valid_replicates,
            })
        row["draw_identity"] = draws[machine].identity
        family_rows.append(row)

    modality_rows: list[dict[str, Any]] = []
    for machine in draws:
        for regime in sorted({b.config.regime for b in bundles.values()}):
            for family in families:
                for first, second in MODALITY_PAIRS:
                    for metric in CONTRAST_METRICS:
                        a = distributions[(machine, regime, first, family, metric)]
                        b = distributions[(machine, regime, second, family, metric)]
                        contrast = paired_contrast(a, b, orientation=f"{first}_minus_{second}")
                        modality_rows.append({
                            "machine": machine, "regime": regime, "family": family,
                            "pair": f"{first}_minus_{second}", **contrast,
                        })
    modality_contrasts = pd.DataFrame(modality_rows)

    regime_rows: list[dict[str, Any]] = []
    for machine in draws:
        for modality in modalities:
            for family in families:
                for metric in CONTRAST_METRICS:
                    t = distributions[(machine, "T", modality, family, metric)]
                    s = distributions[(machine, "S", modality, family, metric)]
                    regime_rows.append({
                        "machine": machine, "modality": modality, "family": family,
                        **paired_contrast(t, s, orientation="T_minus_S"),
                    })
    regime_contrasts = pd.DataFrame(regime_rows)
    dominance = dominance_table(modality_contrasts, families=families)

    tables = {
        "individual_experiment_metrics.csv": pd.DataFrame(individual_rows),
        "domain_slice_metrics_individual.csv": pd.DataFrame(slice_rows),
        "healthy_target_false_positives_individual.csv": pd.DataFrame(healthy_rows),
        "family_bootstrap_metrics.csv": pd.DataFrame(family_rows),
        "modality_paired_contrasts.csv": modality_contrasts,
        "regime_paired_contrasts.csv": regime_contrasts,
        "modality_dominance.csv": dominance,
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False, lineterminator="\n")
    artifact_manifest = {
        "batch_id": manifest["batch_id"],
        "experiment_count": len(bundles),
        "files": {name: file_sha256(out / name) for name in tables},
        "score_manifest_sha256": file_sha256(staging.manifest_path),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["metrics_computed"] = True
    staging._write_manifest(manifest)
    return artifact_manifest
