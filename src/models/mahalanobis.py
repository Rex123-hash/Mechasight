"""Single-Modality Benchmarking: Ledoit-Wolf shrinkage Mahalanobis baseline.

Closed-form Gaussian density baseline on the training-standardized engineered
features (research definition §11, row 1).  Score = squared Mahalanobis
distance to the training-normal mean under the shrinkage covariance; larger
is more anomalous.  Deterministic: seed 0 is a canonical ledger field only.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf

from src.experiment.provenance import content_sha256
from src.models.classical import BaselineSpec


MAHALANOBIS_IMPLEMENTATION_VERSION = "phase4-mahalanobis-shrinkage-v1"

MAHALANOBIS_SPEC = BaselineSpec(
    family="mahalanobis_shrinkage",
    implementation="sklearn.covariance.LedoitWolf.mahalanobis",
    hyperparameters={
        "covariance_estimator": "sklearn.covariance.LedoitWolf",
        "assume_centered": False,
        "store_precision": True,
        "block_size": 1000,
    },
    stochastic=False,
    score_transform="squared Mahalanobis distance to the training mean; higher-is-more-anomalous",
)


class ShrinkageMahalanobis:
    """Fit/score wrapper with the same surface as ``ClassicalAnomalyBaseline``."""

    spec = MAHALANOBIS_SPEC

    def __init__(self, *, seed: int = 0):
        if seed != 0:
            raise ValueError("mahalanobis_shrinkage is deterministic; its canonical seed is 0")
        self.seed = 0
        self.estimator: LedoitWolf | None = None
        self.n_features_in_: int | None = None
        self.n_training_samples_: int | None = None
        self.fit_seconds_: float | None = None

    @property
    def model_id(self) -> str:
        payload = {
            "version": MAHALANOBIS_IMPLEMENTATION_VERSION,
            "family": self.spec.family,
            "hyperparameters": dict(self.spec.hyperparameters),
            "seed": "deterministic",
        }
        return f"p4-model-{content_sha256(payload)[:20]}"

    @staticmethod
    def _matrix(values: np.ndarray, *, n_features: int | None = None) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("model matrix must be non-empty and two-dimensional")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("model matrix must contain finite values only")
        if n_features is not None and matrix.shape[1] != n_features:
            raise ValueError("feature-schema mismatch")
        return matrix

    def fit(self, standardized_training_X: np.ndarray) -> "ShrinkageMahalanobis":
        X = self._matrix(standardized_training_X)
        started = perf_counter()
        hyper = dict(self.spec.hyperparameters)
        hyper.pop("covariance_estimator")
        self.estimator = LedoitWolf(**hyper).fit(X)
        self.n_features_in_ = int(X.shape[1])
        self.n_training_samples_ = int(X.shape[0])
        self.fit_seconds_ = perf_counter() - started
        return self

    def score(self, standardized_X: np.ndarray) -> np.ndarray:
        if self.estimator is None or self.n_features_in_ is None:
            raise RuntimeError("model must be fit before scoring")
        X = self._matrix(standardized_X, n_features=self.n_features_in_)
        scores = np.asarray(self.estimator.mahalanobis(X), dtype=np.float64).reshape(-1)
        if len(scores) != len(X) or not np.all(np.isfinite(scores)):
            raise ValueError("mahalanobis produced invalid anomaly scores")
        return scores

    def save(self, path: str | Path) -> Path:
        if self.estimator is None:
            raise RuntimeError("cannot serialize an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "implementation_version": MAHALANOBIS_IMPLEMENTATION_VERSION,
                "estimator": self.estimator,
                "n_features_in": self.n_features_in_,
                "n_training_samples": self.n_training_samples_,
                "fit_seconds": self.fit_seconds_,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ShrinkageMahalanobis":
        payload = joblib.load(Path(path))
        if payload.get("implementation_version") != MAHALANOBIS_IMPLEMENTATION_VERSION:
            raise ValueError("unsupported serialized mahalanobis version")
        model = cls()
        model.estimator = payload["estimator"]
        model.n_features_in_ = int(payload["n_features_in"])
        model.n_training_samples_ = int(payload["n_training_samples"])
        model.fit_seconds_ = float(payload["fit_seconds"])
        return model
