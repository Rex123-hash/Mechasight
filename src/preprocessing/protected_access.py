"""Read-side guard for protected common-evaluation (test) data.

Building the protected inputs was always refused outside a protected stage
(``assert_pretest_cache_role``, ``CONSUMING_SPLITS``), but Single-Modality
Benchmarking materialised them on disk: the common-evaluation spectral caches
and the common-test feature artifacts now exist.  Path non-existence no longer
protects them, so every repository reader of test-role contents asks this
module first.

A read is allowed only inside ``authorized_protected_read``, which a protected
stage enters after its own authorization checks and which names the authority
(a batch id or the stage's authorization identity) and the purpose.  Anywhere
else, ``require_protected_read`` raises ``ProtectedReadRefused`` before any
matrix, feature frame or window is returned.  The scope is a guard against
accidental development reads, not a security boundary against deliberate
bypass; direct ``h5py``/``parquet`` reads outside the sanctioned readers are
caught by a repository tripwire test instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


PROTECTED_ROLE = "common_evaluation"
PROTECTED_SPLIT = "test"
PROTECTED_SPECTRAL_CACHE_NAME = f"{PROTECTED_ROLE}.spectral.h5"

# Sealed test-score tables written by completed protected stages.  Their
# scores are protected outputs: reading them is a protected read even though
# they carry no label.  Each has exactly one sanctioned reader.  The Phase-3.2
# and Phase-4 readers call ``require_sealed_score_read`` before opening a file.
# The Phase-3.4 reader belongs to a closed analysis whose source is frozen at
# 64694e6 by a standing test, so it is left unguarded; no new code may
# reference its tables (repository tripwire).
SEALED_SCORE_TABLES = {
    "phase3_2_result_store": ("reports/phase3_2/experiments/<id>/scores.parquet",
                              "src.experiment.store.ResultStore.read_scores", "guarded"),
    "phase3_4_focal_toggle": ("reports/phase3_4_results/scores/<id>.csv",
                              "src.experiment.focal_toggle_execution.load_persisted_scores",
                              "closed-stage reader frozen at 64694e6; tripwire only"),
    "phase4_score_staging": ("reports/phase4_protected/staging/scores/<id>.csv",
                             "src.experiment.phase4.ScoreStaging.read_scores", "guarded"),
}


class ProtectedReadRefused(PermissionError):
    """A protected common-evaluation read was attempted outside an authorized scope."""


@dataclass(frozen=True)
class ProtectedReadScope:
    authority: str
    purpose: str


_ACTIVE: ContextVar[ProtectedReadScope | None] = ContextVar("protected_read_scope", default=None)


@contextmanager
def authorized_protected_read(*, authority: str, purpose: str) -> Iterator[ProtectedReadScope]:
    """Admit protected reads for the duration of one authorized protected stage."""

    if not str(authority).strip() or not str(purpose).strip():
        raise ValueError("an authorized protected read needs a non-empty authority and purpose")
    if _ACTIVE.get() is not None:
        raise RuntimeError("authorized protected-read scopes do not nest")
    token = _ACTIVE.set(ProtectedReadScope(str(authority), str(purpose)))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def active_protected_read() -> ProtectedReadScope | None:
    return _ACTIVE.get()


def require_protected_read(what: str) -> ProtectedReadScope:
    scope = _ACTIVE.get()
    if scope is None:
        raise ProtectedReadRefused(
            f"refusing to read protected {PROTECTED_ROLE} data ({what}) outside an authorized "
            "protected-read scope; development and pre-authorization code may read "
            "training/source-validation data only"
        )
    return scope


def require_sealed_score_read(table: str, *, reader: str) -> ProtectedReadScope:
    """Refuse a sealed test-score table read outside an authorized scope."""

    return require_protected_read(f"sealed test-score table {table} via {reader}")


def is_protected_split(split: object) -> bool:
    return str(split) == PROTECTED_SPLIT


def is_protected_role(role: object) -> bool:
    return str(role) == PROTECTED_ROLE
