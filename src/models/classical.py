"""Phase-3.1 classical anomaly baselines with one score convention.

Inputs to these estimators are expected to have been standardized through the
provenance-aware, training-only normalization path.  Every public ``score``
method returns a MechaSight anomaly score where a larger value means more
anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.experiment.provenance import content_sha256


MODEL_IMPLEMENTATION_VERSION = "phase3.1-classical-baselines-v1"


@dataclass(frozen=True)
class BaselineSpec:
    family: str
    implementation: str
    hyperparameters: Mapping[str, Any]
    stochastic: bool
    score_transform: str

    @property
    def identity(self) -> str:
        payload = {
            "version": MODEL_IMPLEMENTATION_VERSION,
            "family": self.family,
            "implementation": self.implementation,
            "hyperparameters": dict(self.hyperparameters),
            "stochastic": self.stochastic,
            "score_transform": self.score_transform,
        }
        return f"p31-model-{content_sha256(payload)[:20]}"


PRIMARY_BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        family="statistical_reference",
        implementation="mechasight.standardized_rms_distance",
        hyperparameters={"distance": "root_mean_square_z_distance"},
        stochastic=False,
        score_transform="sqrt(mean(z_feature ** 2)); already higher-is-more-anomalous",
    ),
    BaselineSpec(
        family="isolation_forest",
        implementation="sklearn.ensemble.IsolationForest",
        hyperparameters={
            "n_estimators": 200,
            "max_samples": "auto",
            "contamination": "auto",
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": 1,
        },
        stochastic=True,
        score_transform="negative sklearn score_samples",
    ),
    BaselineSpec(
        family="one_class_svm",
        implementation="sklearn.svm.OneClassSVM",
        hyperparameters={
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "tol": 0.001,
            "nu": 0.05,
            "shrinking": True,
            "cache_size": 512.0,
            "max_iter": -1,
        },
        stochastic=False,
        score_transform="negative sklearn decision_function",
    ),
)


def primary_spec(family: str) -> BaselineSpec:
    for spec in PRIMARY_BASELINE_SPECS:
        if spec.family == family:
            return spec
    raise ValueError(f"unknown Phase-3.1 baseline family: {family!r}")


class ClassicalAnomalyBaseline:
    """Small reusable fit/score wrapper without hidden normalization state."""

    def __init__(self, spec: BaselineSpec, *, seed: int = 0):
        if seed not in (0, 1, 2, 3, 4):
            raise ValueError("Phase-3.1 model seed must be one of 0, 1, 2, 3, 4")
        self.spec = spec
        self.seed = int(seed)
        self.estimator: IsolationForest | OneClassSVM | None = None
        self.n_features_in_: int | None = None
        self.n_training_samples_: int | None = None
        self.fit_seconds_: float | None = None

    @property
    def model_id(self) -> str:
        seed_payload = self.seed if self.spec.stochastic else "deterministic"
        return f"{self.spec.identity}-{content_sha256({'seed': seed_payload})[:8]}"

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

    def fit(self, standardized_training_X: np.ndarray) -> "ClassicalAnomalyBaseline":
        X = self._matrix(standardized_training_X)
        started = perf_counter()
        if self.spec.family == "statistical_reference":
            estimator = None
        elif self.spec.family == "isolation_forest":
            estimator = IsolationForest(
                **dict(self.spec.hyperparameters), random_state=self.seed
            ).fit(X)
        elif self.spec.family == "one_class_svm":
            estimator = OneClassSVM(**dict(self.spec.hyperparameters)).fit(X)
        else:  # pragma: no cover - protected by primary_spec/config tests
            raise ValueError(f"unsupported baseline family: {self.spec.family}")
        self.estimator = estimator
        self.n_features_in_ = int(X.shape[1])
        self.n_training_samples_ = int(X.shape[0])
        self.fit_seconds_ = perf_counter() - started
        return self

    def score(self, standardized_X: np.ndarray) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("baseline must be fit before scoring")
        X = self._matrix(standardized_X, n_features=self.n_features_in_)
        if self.spec.family == "statistical_reference":
            scores = np.sqrt(np.mean(np.square(X), axis=1))
        elif self.spec.family == "isolation_forest":
            assert isinstance(self.estimator, IsolationForest)
            scores = -self.estimator.score_samples(X)
        elif self.spec.family == "one_class_svm":
            assert isinstance(self.estimator, OneClassSVM)
            scores = -self.estimator.decision_function(X)
        else:  # pragma: no cover
            raise ValueError(f"unsupported baseline family: {self.spec.family}")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(X) or not np.all(np.isfinite(scores)):
            raise ValueError("baseline produced invalid anomaly scores")
        return scores

    def save(self, path: str | Path) -> Path:
        if self.n_features_in_ is None:
            raise RuntimeError("cannot serialize an unfitted baseline")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "implementation_version": MODEL_IMPLEMENTATION_VERSION,
                "spec": self.spec,
                "seed": self.seed,
                "estimator": self.estimator,
                "n_features_in": self.n_features_in_,
                "n_training_samples": self.n_training_samples_,
                "fit_seconds": self.fit_seconds_,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalAnomalyBaseline":
        payload = joblib.load(Path(path))
        if payload.get("implementation_version") != MODEL_IMPLEMENTATION_VERSION:
            raise ValueError("unsupported serialized baseline version")
        model = cls(payload["spec"], seed=int(payload["seed"]))
        model.estimator = payload["estimator"]
        model.n_features_in_ = int(payload["n_features_in"])
        model.n_training_samples_ = int(payload["n_training_samples"])
        model.fit_seconds_ = float(payload["fit_seconds"])
        return model
