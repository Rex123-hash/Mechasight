"""Versioned preprocessing configuration and cache identity.

The protocol and regime are deliberately required rather than inferred.  The
canonical JSON representation is hashed and becomes part of every cache and
manifest, so changing any preprocessing choice produces a new identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


OFFICIAL_PROTOCOL = "official_baseline_reproduction"
MECHASIGHT_PROTOCOL = "mechasight_experimental_protocol"
VALID_PROTOCOLS = frozenset({OFFICIAL_PROTOCOL, MECHASIGHT_PROTOCOL})
VALID_REGIMES = frozenset({"S", "T"})


@dataclass(frozen=True)
class ResamplingConfig:
    """Regular-grid conversion used only by consumers that require it."""

    enabled_for: tuple[str, ...] = ("frequency_features",)
    method: str = "linear"
    edge_handling: str = "intersection_only"
    anti_aliasing: str = "not_required_at_near-native_target_rate"
    target_rates_hz: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str
    protocol: str
    regime: str
    machines: tuple[str, ...]
    window_length_s: float
    stride_s: float
    gyro_warmup_s: float
    drop_initial_acc_samples: dict[str, int]
    normalization: str
    resampling: ResamplingConfig
    features: tuple[str, ...]
    cache_format: str = "hdf5_ragged_v1"
    window_boundary: str = "half_open"
    official_fixed_rates_hz: dict[str, float] = field(default_factory=dict)
    official_pad_or_truncate: bool = False
    split_version: str = "protocol_b_v1"

    def __post_init__(self) -> None:
        if self.protocol not in VALID_PROTOCOLS:
            raise ValueError(f"protocol must be one of {sorted(VALID_PROTOCOLS)}")
        if self.regime not in VALID_REGIMES:
            raise ValueError("regime must be 'S' or 'T'")
        if self.window_length_s <= 0 or self.stride_s <= 0:
            raise ValueError("window length and stride must be positive")
        if self.gyro_warmup_s < 0:
            raise ValueError("gyro warm-up cannot be negative")
        if any(value < 0 for value in self.drop_initial_acc_samples.values()):
            raise ValueError("initial-sample removal counts cannot be negative")
        unknown_features = set(self.features) - {"time", "frequency"}
        if unknown_features:
            raise ValueError(f"unknown feature families: {sorted(unknown_features)}")
        if self.protocol == OFFICIAL_PROTOCOL:
            if self.window_length_s != 0.1 or self.stride_s != 0.1:
                raise ValueError("Protocol A is fixed to non-overlapping 100 ms windows")
            if self.normalization != "std_window":
                raise ValueError("Protocol A must retain std_window normalization")
            required = {"mic": 16000.0, "acc": 7063.0, "gyro": 7063.0}
            if self.official_fixed_rates_hz != required:
                raise ValueError("Protocol A must retain the authors' 16000/7063/7063 constants")
            if not self.official_pad_or_truncate:
                raise ValueError("Protocol A must retain zero-pad/truncate behavior")
            if any(self.drop_initial_acc_samples.values()):
                raise ValueError("Protocol A must not add the MechaSight startup correction")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return f"{self.version}-{sha256(payload).hexdigest()[:16]}"


def _tuple(value: Any) -> tuple:
    return tuple(value) if not isinstance(value, tuple) else value


def config_from_dict(raw: dict[str, Any]) -> PreprocessingConfig:
    data = dict(raw)
    data["machines"] = _tuple(data["machines"])
    data["features"] = _tuple(data.get("features", ()))
    resampling = dict(data.get("resampling", {}))
    resampling["enabled_for"] = _tuple(resampling.get("enabled_for", ()))
    data["resampling"] = ResamplingConfig(**resampling)
    return PreprocessingConfig(**data)


def load_config(path: str | Path) -> PreprocessingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))
