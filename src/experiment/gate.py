"""Official provenance-aware validation and test evaluation entry point."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from src.experiment.config import ExperimentConfig
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.constants import METRIC_POLICY_VERSION
from src.experiment.metrics import (
    RANKING_METRICS,
    protocol_b_threshold_metrics,
    ranking_metric_value,
)
from src.experiment.scores import (
    ScoreBatch,
    ScoreProvenance,
    ThresholdSelection,
    assert_matching_score_provenance,
)


class EvaluationGate:
    """Engineering guardrail for official evaluation paths, not a security boundary."""

    def __init__(self, config: ExperimentConfig, expected_scores: ScoreProvenance):
        if config.experiment_id != expected_scores.experiment_id:
            raise ValueError("experiment config and score provenance identities differ")
        self.config = config
        self.expected_scores = expected_scores

    def _validate_batch(self, batch: ScoreBatch, required_role: str) -> None:
        assert_matching_score_provenance(batch.provenance, self.expected_scores)
        if set(batch.cohort_roles) != {required_role}:
            raise ValueError(f"evaluation requires a pure {required_role} score cohort")
        cohort_key = f"{required_role}_cohort_hash"
        expected_hash = self.expected_scores.data_provenance.get(cohort_key)
        if not expected_hash or batch.cohort_hash != expected_hash:
            raise ValueError(f"evaluation ids do not match the frozen {required_role} cohort")

    def evaluate_validation(
        self,
        batch: ScoreBatch,
        *,
        threshold: ThresholdSelection | None = None,
    ) -> dict:
        self._validate_batch(batch, "source_validation")
        result: dict = {
            "evaluation_scope": "validation",
            "cohort_hash": batch.cohort_hash,
        }
        if threshold is not None:
            result["threshold_metrics"] = asdict(protocol_b_threshold_metrics(
                batch.is_anomaly, batch.scores, threshold=threshold.value
            ))
        return result

    def evaluate_test(
        self,
        batch: ScoreBatch,
        *,
        threshold: ThresholdSelection,
        ledger: TestLookLedger,
        reason: str,
        metrics_requested: tuple[str, ...] = (
            "auroc", "average_precision", "standardized_pauroc_fpr_0_10",
            "threshold_metrics",
        ),
    ) -> dict:
        self._validate_batch(batch, "common_evaluation")
        if threshold.source_experiment_id != self.config.experiment_id:
            raise ValueError("threshold provenance belongs to another experiment")
        allowed = {*RANKING_METRICS, "threshold_metrics"}
        unknown = set(metrics_requested) - allowed
        if unknown or not metrics_requested:
            raise ValueError(f"unsupported test metrics requested: {sorted(unknown)}")
        # The durable test-look record is appended before metrics are computed.
        ledger.append(TestLookRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.config.experiment_id,
            git_sha=self.config.code_git_sha,
            machine=self.config.machine,
            protocol=self.config.protocol,
            regime=self.config.regime,
            model_config_identity=self.expected_scores.model_id,
            test_cohort_hash=batch.cohort_hash,
            reason=reason,
            metrics_requested=metrics_requested,
        ))
        result = {
            "evaluation_scope": "test",
            "cohort_hash": batch.cohort_hash,
            "metric_policy_version": METRIC_POLICY_VERSION,
            "threshold_provenance": asdict(threshold),
        }
        for metric in metrics_requested:
            if metric in RANKING_METRICS:
                result[metric] = ranking_metric_value(
                    metric, batch.is_anomaly, batch.scores
                )
            elif metric == "threshold_metrics":
                result[metric] = asdict(protocol_b_threshold_metrics(
                    batch.is_anomaly, batch.scores, threshold=threshold.value
                ))
        return result
