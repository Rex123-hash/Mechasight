"""Exclusive, atomic result-bundle persistence for future experiments."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.constants import RESULT_SCHEMA_VERSION, SCORE_DIRECTION
from src.experiment.provenance import canonical_json, file_sha256, ordered_id_hash
from src.preprocessing.protected_access import require_sealed_score_read


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read_scores(self, experiment_id: str) -> pd.DataFrame:
        """The one sanctioned reader of a result bundle's sealed test scores.

        Every bundle this store writes carries a declared test-cohort hash, so its
        ``scores.parquet`` is a protected test output.  Refused outside an
        authorized protected-read scope, before any file is opened; inside one,
        the table must match its bundle-manifest hash and declared cohort.
        """

        require_sealed_score_read(f"{experiment_id}/scores.parquet", reader="ResultStore.read_scores")
        bundle = self.root / experiment_id
        manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != experiment_id:
            raise ValueError("result bundle identity mismatch")
        path = bundle / "scores.parquet"
        if file_sha256(path) != manifest["files"]["scores.parquet"]:
            raise ValueError(f"sealed score table hash mismatch for {experiment_id}")
        scores = pd.read_parquet(path)
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        if ordered_id_hash(tuple(scores["segment_id"].astype(str))) != provenance["test_cohort_hash"]:
            raise ValueError(f"sealed score table cohort mismatch for {experiment_id}")
        return scores

    def write_bundle(
        self,
        *,
        config: ExperimentConfig,
        provenance: Mapping[str, Any],
        metrics: Mapping[str, Any],
        scores: pd.DataFrame,
    ) -> Path:
        required_provenance = {
            "test_cohort_hash", "score_direction", "threshold_provenance",
        }
        missing = required_provenance - set(provenance)
        if missing:
            raise ValueError(f"result provenance missing fields: {sorted(missing)}")
        if provenance["score_direction"] != SCORE_DIRECTION:
            raise ValueError("result score direction is incompatible")
        required_scores = {"segment_id", "parent_recording", "score"}
        if not required_scores <= set(scores.columns):
            raise ValueError("score table lacks required traceability fields")
        if scores.empty or scores["segment_id"].duplicated().any():
            raise ValueError("score table must contain unique segment results")
        observed_cohort_hash = ordered_id_hash(
            tuple(scores["segment_id"].astype(str))
        )
        if provenance["test_cohort_hash"] != observed_cohort_hash:
            raise ValueError("score identities do not match the declared test cohort hash")
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / config.experiment_id
        if final.exists():
            raise FileExistsError(f"experiment result already exists: {final}")
        stage = Path(tempfile.mkdtemp(prefix=f".{config.experiment_id}.", dir=self.root))
        try:
            config_path = stage / "config.json"
            provenance_path = stage / "provenance.json"
            metrics_path = stage / "metrics.json"
            scores_path = stage / "scores.parquet"
            config_path.write_text(canonical_json(config.to_dict()) + "\n", encoding="utf-8")
            provenance_payload = {
                **dict(provenance),
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "experiment_id": config.experiment_id,
                "scientific_limitations": list(config.scientific_limitations),
                "scientific_limitation_ids": list(config.scientific_limitation_ids),
                "cohort_provenance": config.cohort_provenance,
            }
            provenance_path.write_text(
                canonical_json(provenance_payload) + "\n", encoding="utf-8"
            )
            metrics_path.write_text(
                canonical_json({"result_schema_version": RESULT_SCHEMA_VERSION, **dict(metrics)})
                + "\n",
                encoding="utf-8",
            )
            scores.to_parquet(scores_path, index=False)
            manifest = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "experiment_id": config.experiment_id,
                "files": {
                    path.name: file_sha256(path)
                    for path in (config_path, provenance_path, metrics_path, scores_path)
                },
            }
            (stage / "bundle_manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8"
            )
            # Rename within one filesystem is atomic; an existing destination is
            # rechecked immediately before promotion.
            if final.exists():
                raise FileExistsError(f"experiment result already exists: {final}")
            stage.rename(final)
            return final
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise
