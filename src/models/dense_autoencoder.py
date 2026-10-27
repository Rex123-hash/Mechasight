"""Single-Modality Benchmarking: small Protocol-B dense autoencoder.

This is a MechaSight model, not the upstream reproduction: inputs are the
frozen log-power-spectrum windows after training-fitted per-bin z-scoring,
early stopping watches source-validation window MSE only, and the best
validation checkpoint is the frozen scientific object.  Window score = mean
squared reconstruction error; larger is more anomalous.  torch is imported
lazily so the classical path never depends on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.experiment.provenance import content_sha256


AUTOENCODER_IMPLEMENTATION_VERSION = "phase4-dense-autoencoder-protocol-b-v1"
# Intra-op thread count.  It affects floating-point summation order, so the
# frozen scientific object is the saved checkpoint, never a retrain; the count
# used is recorded in every training history.
TORCH_THREADS = int(os.environ.get("MECHASIGHT_TORCH_THREADS", "8"))


@dataclass(frozen=True)
class DenseAutoencoderSpec:
    hidden: tuple[int, ...] = (256, 64)
    bottleneck: int = 16
    activation: str = "relu"
    output: str = "linear"
    loss: str = "mse"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10
    checkpoint_selection: str = "best_source_validation_window_mse"
    shuffle_training_windows: bool = True

    def __post_init__(self) -> None:
        if not self.hidden or self.bottleneck <= 0 or self.batch_size <= 0:
            raise ValueError("autoencoder layer sizes and batch size must be positive")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if (self.activation, self.output, self.loss, self.optimizer) != ("relu", "linear", "mse", "adam"):
            raise ValueError("the dense autoencoder freezes relu/linear/mse/adam")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden"] = list(self.hidden)
        return payload


DENSE_AE_SPEC = DenseAutoencoderSpec()


def _torch():
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    return torch


def _build_network(torch, input_dim: int, spec: DenseAutoencoderSpec):
    nn = torch.nn
    layers: list = []
    last = input_dim
    for width in spec.hidden:
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers += [nn.Linear(last, spec.bottleneck)]
    last = spec.bottleneck
    for width in reversed(spec.hidden):
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers += [nn.Linear(last, input_dim)]
    return nn.Sequential(*layers)


class DenseAutoencoder:
    def __init__(self, *, input_dim: int, seed: int, spec: DenseAutoencoderSpec = DENSE_AE_SPEC):
        if seed not in (0, 1, 2, 3, 4):
            raise ValueError("dense autoencoder seed must be one of 0, 1, 2, 3, 4")
        if input_dim <= 0:
            raise ValueError("input dimension must be positive")
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.spec = spec
        self.history: dict[str, Any] = {}
        torch = _torch()
        torch.manual_seed(self.seed)
        self.network = _build_network(torch, self.input_dim, spec)
        self.network.eval()

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.network.parameters()))

    @property
    def model_id(self) -> str:
        payload = {
            "version": AUTOENCODER_IMPLEMENTATION_VERSION,
            "family": "dense_autoencoder",
            "spec": self.spec.to_dict(),
            "input_dim": self.input_dim,
            "seed": self.seed,
        }
        return f"p4-model-{content_sha256(payload)[:20]}"

    def _matrix(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("autoencoder input must be a non-empty (windows, bins) matrix")
        if matrix.shape[1] != self.input_dim:
            raise ValueError(
                f"input dimension {matrix.shape[1]} differs from the frozen {self.input_dim}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("autoencoder input must be finite")
        return matrix

    def _mean_window_mse(self, torch, X: np.ndarray) -> float:
        return float(np.mean(self.score_windows(X)))

    def fit(self, training_X: np.ndarray, validation_X: np.ndarray) -> "DenseAutoencoder":
        torch = _torch()
        train = self._matrix(training_X)
        validation = self._matrix(validation_X)
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        network = self.network
        optimizer = torch.optim.Adam(
            network.parameters(), lr=self.spec.learning_rate, weight_decay=self.spec.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        train_tensor = torch.from_numpy(train)
        started = perf_counter()
        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_loss = float("inf")
        best_epoch = 0
        best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
        epochs_without_improvement = 0
        stopped_early = False
        for epoch in range(1, self.spec.max_epochs + 1):
            network.train()
            order = (
                torch.randperm(len(train_tensor), generator=generator)
                if self.spec.shuffle_training_windows
                else torch.arange(len(train_tensor))
            )
            total = 0.0
            for start in range(0, len(order), self.spec.batch_size):
                batch = train_tensor[order[start:start + self.spec.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(network(batch), batch)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
            train_losses.append(total / len(train_tensor))
            network.eval()
            validation_loss = self._mean_window_mse(torch, validation)
            validation_losses.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.spec.patience:
                    stopped_early = True
                    break
        network.load_state_dict(best_state)
        network.eval()
        self.history = {
            "epochs_run": len(validation_losses),
            "best_epoch": best_epoch,
            "best_validation_window_mse": best_loss,
            "stopped_early": stopped_early,
            "training_window_mse": train_losses,
            "validation_window_mse": validation_losses,
            "training_windows": int(len(train)),
            "validation_windows": int(len(validation)),
            "fit_seconds": perf_counter() - started,
            "parameter_count": self.parameter_count,
            "torch_threads": int(torch.get_num_threads()),
            "torch_version": str(torch.__version__),
        }
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        matrix = self._matrix(X)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(matrix), 4096):
                batch = torch.from_numpy(matrix[start:start + 4096])
                outputs.append(self.network(batch).numpy())
        return np.vstack(outputs).astype(np.float32)

    def score_windows(self, X: np.ndarray) -> np.ndarray:
        matrix = self._matrix(X)
        error = self.reconstruct(matrix).astype(np.float64) - matrix.astype(np.float64)
        scores = np.mean(np.square(error), axis=1)
        if not np.all(np.isfinite(scores)):
            raise ValueError("autoencoder produced non-finite window scores")
        return scores

    def save(self, path: str | Path) -> Path:
        torch = _torch()
        if not self.history:
            raise RuntimeError("cannot serialize an untrained autoencoder")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "implementation_version": AUTOENCODER_IMPLEMENTATION_VERSION,
                "spec": self.spec.to_dict(),
                "input_dim": self.input_dim,
                "seed": self.seed,
                "history": self.history,
                "state_dict": self.network.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DenseAutoencoder":
        torch = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("implementation_version") != AUTOENCODER_IMPLEMENTATION_VERSION:
            raise ValueError("unsupported serialized autoencoder version")
        spec_payload = dict(payload["spec"])
        spec_payload["hidden"] = tuple(spec_payload["hidden"])
        model = cls(input_dim=int(payload["input_dim"]), seed=int(payload["seed"]),
                    spec=DenseAutoencoderSpec(**spec_payload))
        model.network.load_state_dict(payload["state_dict"])
        model.network.eval()
        model.history = dict(payload["history"])
        return model
