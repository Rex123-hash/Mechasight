"""Deterministic parent-recording split construction and leakage assertions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.data.temporal import parse_segment_times


SPLIT_ORDER = ("train", "validation", "test")

# Splits whose members are read by some computation: model fitting, learned
# preprocessing statistics, early stopping, threshold calibration, or metric
# computation.  Only these can carry information across a parent recording.
CONSUMING_SPLITS = frozenset({"train", "validation", "test"})

# Splits whose members are deliberately consumed by nothing.  ``reserved_adaptation``
# records the Regime-S counterfactual; ``blocked_sibling`` records segments withheld
# purely to keep an adaptation parent out of evaluation (Phase 2.2).  Membership is
# bookkeeping, so a parent may carry these alongside exactly one consuming split.
NON_CONSUMING_SPLITS = frozenset({"reserved_adaptation", "blocked_sibling"})

BLOCKED_SIBLING_SPLIT = "blocked_sibling"
BLOCKED_SIBLING_POOL = "adaptation_sibling_blocked"

PROTOCOL_B_V3_SPLIT_VERSION = "protocol_b_few_shot_adaptation_v3"
PROTOCOL_B_V3_PREPROCESSING_VERSION = "phase2.2-v3"

#: Every segment carries exactly one of these primary protocol roles.
PROTOCOL_ROLES = (
    "source_training",
    "source_validation",
    "target_adaptation",
    BLOCKED_SIBLING_POOL,
    "common_evaluation",
)


def attach_parent_recording(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = parse_segment_times(out["segment_id"])
    out["parent_recording"] = parsed["recording_stamp"].astype("string")
    if out["parent_recording"].isna().any():
        examples = out.loc[out["parent_recording"].isna(), "segment_id"].head().tolist()
        raise ValueError(f"unparseable parent recording ids: {examples}")
    return out


def _stable_order(values: list[str], seed: int) -> list[str]:
    """Seeded SHA-256 ordering, independent of input row order/library RNG."""

    return sorted(
        values,
        key=lambda value: (sha256(f"{seed}|{value}".encode()).hexdigest(), value),
    )


def _three_way_counts(n: int, fraction: float = 0.10) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"need at least 3 recordings for train/validation/test, got {n}")
    n_validation = max(1, round(n * fraction))
    # A single test parent would make a condition-specific result inseparable
    # from that recording's idiosyncrasies.  Two is still a thin margin, but is
    # feasible in every measured RA stratum and avoids inventing grouped CV.
    n_test = max(2, round(n * fraction))
    if n - n_validation - n_test < 1:
        raise ValueError(f"stratum with {n} recordings cannot satisfy minimum split sizes")
    return n - n_validation - n_test, n_validation, n_test


def _check_parent_homogeneity(frame: pd.DataFrame) -> None:
    fields = ["file_domain", "domain_shift_op", "anomaly_label"]
    counts = frame.groupby("parent_recording")[fields].nunique(dropna=False)
    bad = counts[(counts > 1).any(axis=1)]
    if not bad.empty:
        raise AssertionError(f"parent recordings cross design cells: {bad.index.tolist()[:5]}")


def robotic_arm_assignments(
    frame: pd.DataFrame, *, seed: int = 1995, regime: str = "T"
) -> pd.DataFrame:
    """Partition complete RA recordings within (domain, operating condition).

    Every anomalous recording is test-only.  Normal recordings use an
    approximately 80/10/10 recording split in every stratum, subject to at
    least two test parents so no condition is evaluated from one recording.
    For Regime S, target normals are all unseen and therefore assigned to test.
    """

    if regime not in {"S", "T"}:
        raise ValueError("regime must be S or T")
    data = attach_parent_recording(frame)
    _check_parent_homogeneity(data)
    data["split"] = pd.NA
    is_normal = data["anomaly_label"].astype("string").str.lower().eq("normal")
    data.loc[~is_normal, "split"] = "test"

    normals = data[is_normal]
    for (domain, condition), part in normals.groupby(
        ["file_domain", "domain_shift_op"], sort=True
    ):
        parents = sorted(part["parent_recording"].unique().tolist())
        if regime == "S" and domain == "target":
            assignment = {parent: "test" for parent in parents}
        else:
            n_train, n_val, _ = _three_way_counts(len(parents))
            ordered = _stable_order(parents, seed)
            assignment = {
                parent: (
                    "train" if i < n_train
                    else "validation" if i < n_train + n_val
                    else "test"
                )
                for i, parent in enumerate(ordered)
            }
        mask = is_normal & data["parent_recording"].isin(assignment)
        data.loc[mask, "split"] = data.loc[mask, "parent_recording"].map(assignment)

    data["regime"] = regime
    data["protocol"] = "mechasight_experimental_protocol"
    data["split_seed"] = seed
    assert_no_parent_leakage(data)
    return data


def brushless_motor_assignments(
    frame: pd.DataFrame, *, seed: int = 1995, regime: str = "T"
) -> pd.DataFrame:
    """Keep the released BM test set and carve grouped validation from train.

    BM has exactly one segment per recording, so no official recording sibling
    leakage exists.  The class/session confound cannot be fixed by re-splitting.
    """

    if regime not in {"S", "T"}:
        raise ValueError("regime must be S or T")
    data = attach_parent_recording(frame)
    _check_parent_homogeneity(data)
    data["split"] = "test"
    candidates = data[
        data["file_split"].eq("train")
        & data["anomaly_label"].astype("string").str.lower().eq("normal")
    ]
    for (domain, condition), part in candidates.groupby(
        ["file_domain", "domain_shift_op"], sort=True
    ):
        parents = sorted(part["parent_recording"].unique().tolist())
        if regime == "S" and domain == "target":
            assignment = {parent: "test" for parent in parents}
        else:
            n_val = max(1, round(0.10 * len(parents)))
            ordered = _stable_order(parents, seed)
            assignment = {
                parent: "validation" if i < n_val else "train"
                for i, parent in enumerate(ordered)
            }
        mask = data["parent_recording"].isin(assignment)
        data.loc[mask, "split"] = data.loc[mask, "parent_recording"].map(assignment)

    data["regime"] = regime
    data["protocol"] = "mechasight_experimental_protocol"
    data["split_seed"] = seed
    assert_no_parent_leakage(data)
    return data


def build_protocol_b_assignments(
    frame: pd.DataFrame, *, seed: int = 1995, regime: str = "T"
) -> pd.DataFrame:
    outputs = []
    for machine, part in frame.groupby("machine", sort=True):
        if machine == "RoboticArm":
            outputs.append(robotic_arm_assignments(part, seed=seed, regime=regime))
        elif machine == "BrushlessMotor":
            outputs.append(brushless_motor_assignments(part, seed=seed, regime=regime))
        else:
            raise ValueError(f"unsupported machine {machine!r}")
    result = pd.concat(outputs, ignore_index=True)
    assert_no_parent_leakage(result)
    return result


def _source_assignments_v2(
    frame: pd.DataFrame, *, seed: int, machine: str
) -> pd.DataFrame:
    """Reuse the Phase-2 source split while decoupling target adaptation."""

    if machine == "RoboticArm":
        base = robotic_arm_assignments(frame, seed=seed, regime="T")
    elif machine == "BrushlessMotor":
        base = brushless_motor_assignments(frame, seed=seed, regime="T")
    else:
        raise ValueError(f"unsupported machine {machine!r}")
    return base[base["file_domain"].eq("source")].copy()


def matched_common_test_assignments(
    frame: pd.DataFrame, *, seed: int = 1995, regime: str
) -> pd.DataFrame:
    """Protocol-B v2: identical S/T test, parent-safe adaptation reserve.

    The target adaptation pool is defined by every parent containing a released
    target-normal training example.  All normal siblings of those examples are
    part of the pool.  S reserves the pool; T may fit on it.  Neither regime
    evaluates on it.

    **Superseded by** :func:`few_shot_adaptation_assignments` (Phase 2.2), and
    retained only to reproduce the frozen Phase-2.1 evidence.  Treating the whole
    parent closure as the Regime-T training pool would let RoboticArm adapt on
    123 target segments when the release exposes 27; parent independence requires
    the siblings to leave *evaluation*, not to enter *training*.
    """

    if regime not in {"S", "T"}:
        raise ValueError("regime must be S or T")
    outputs = []
    for machine, machine_frame in frame.groupby("machine", sort=True):
        data = attach_parent_recording(machine_frame)
        _check_parent_homogeneity(data)
        source = _source_assignments_v2(data, seed=seed, machine=machine)
        source["data_pool"] = source["split"].map({
            "train": "source_training",
            "validation": "source_validation",
            "test": "common_evaluation",
        })

        target = data[data["file_domain"].eq("target")].copy()
        is_normal = target["anomaly_label"].astype("string").str.lower().eq("normal")
        released_adaptation = target[is_normal & target["file_split"].eq("train")]
        adaptation_parents = set(released_adaptation["parent_recording"])
        in_adaptation = is_normal & target["parent_recording"].isin(adaptation_parents)
        target["split"] = "test"
        target["data_pool"] = "common_evaluation"
        target.loc[in_adaptation, "split"] = (
            "reserved_adaptation" if regime == "S" else "train"
        )
        target.loc[in_adaptation, "data_pool"] = "target_adaptation"

        combined = pd.concat([source, target], ignore_index=True)
        combined["regime"] = regime
        combined["protocol"] = "mechasight_experimental_protocol"
        combined["split_seed"] = seed
        combined["split_version"] = "protocol_b_common_test_v2"
        combined["preprocessing_version"] = "phase2.1-v2"
        assert_no_parent_leakage(combined)
        outputs.append(combined)

    result = pd.concat(outputs, ignore_index=True)
    assert_no_parent_leakage(result)
    return result


def few_shot_adaptation_assignments(
    frame: pd.DataFrame, *, seed: int = 1995, regime: str
) -> pd.DataFrame:
    """Protocol-B v3: identical S/T test, released-only adaptation budget.

    Phase 2.1 established that a parent recording containing a released
    target-normal training example may not also supply evaluation segments.
    Phase 2.2 separates the two consequences of that rule, which v2 conflated:

    * **target_adaptation** - the segments the release actually offers as
      target-normal training data (RA 27, BM 18).  Regime T consumes exactly
      these; Regime S reserves exactly these.
    * **adaptation_sibling_blocked** - the remaining normal segments of the same
      physical parent recordings (RA 96, BM 0).  Parent independence requires
      these to leave evaluation; it does not require them to become training
      data.  They are consumed by nothing in either regime.

    The common evaluation cohort is bit-for-bit the Phase-2.1 cohort.  Only the
    training/adaptation role labels change.
    """

    if regime not in {"S", "T"}:
        raise ValueError("regime must be S or T")
    outputs = []
    for machine, machine_frame in frame.groupby("machine", sort=True):
        data = attach_parent_recording(machine_frame)
        _check_parent_homogeneity(data)
        source = _source_assignments_v2(data, seed=seed, machine=machine)
        source["data_pool"] = source["split"].map({
            "train": "source_training",
            "validation": "source_validation",
            "test": "common_evaluation",
        })

        target = data[data["file_domain"].eq("target")].copy()
        is_normal = target["anomaly_label"].astype("string").str.lower().eq("normal")
        released = is_normal & target["file_split"].eq("train")
        adaptation_parents = set(target.loc[released, "parent_recording"])
        blocked = (
            is_normal
            & ~released
            & target["parent_recording"].isin(adaptation_parents)
        )

        target["split"] = "test"
        target["data_pool"] = "common_evaluation"
        # Released few-shot allocation: the only target data Regime T consumes.
        target.loc[released, "split"] = "reserved_adaptation" if regime == "S" else "train"
        target.loc[released, "data_pool"] = "target_adaptation"
        # Same-parent siblings: withheld from evaluation, consumed by neither regime.
        target.loc[blocked, "split"] = BLOCKED_SIBLING_SPLIT
        target.loc[blocked, "data_pool"] = BLOCKED_SIBLING_POOL

        combined = pd.concat([source, target], ignore_index=True)
        combined["regime"] = regime
        combined["protocol"] = "mechasight_experimental_protocol"
        combined["split_seed"] = seed
        combined["split_version"] = PROTOCOL_B_V3_SPLIT_VERSION
        combined["preprocessing_version"] = PROTOCOL_B_V3_PREPROCESSING_VERSION
        assert_no_parent_leakage(combined, ignore_splits=NON_CONSUMING_SPLITS)
        outputs.append(combined)

    result = pd.concat(outputs, ignore_index=True)
    assert_no_parent_leakage(result, ignore_splits=NON_CONSUMING_SPLITS)
    return result


def adaptation_budget_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report the four distinct RoboticArm adaptation quantities separately."""

    required = {"machine", "regime", "data_pool", "segment_id", "parent_recording"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing adaptation-budget fields: {sorted(missing)}")
    rows = []
    for machine, part in frame.groupby("machine", sort=True):
        one = part[part["regime"].eq("T")] if "T" in set(part["regime"]) else part
        consumed = one[one["data_pool"].eq("target_adaptation")]
        blocked = one[one["data_pool"].eq(BLOCKED_SIBLING_POOL)]
        consumed_parents = set(consumed["parent_recording"])
        blocked_parents = set(blocked["parent_recording"])
        rows.append({
            "machine": machine,
            "target_adaptation_observed": int(len(consumed)),
            "target_adaptation_parent_groups": int(len(consumed_parents)),
            "target_adaptation_blocked_siblings": int(len(blocked)),
            "target_adaptation_parent_closure": int(len(consumed) + len(blocked)),
            "blocked_sibling_parent_groups": int(len(blocked_parents)),
            "blocked_parents_subset_of_adaptation_parents": bool(
                blocked_parents <= consumed_parents
            ),
        })
    return pd.DataFrame(rows)


def assert_few_shot_adaptation_cohorts(frame: pd.DataFrame) -> None:
    """Phase-2.2 protocol invariants on a frozen v3 assignment frame."""

    assert_matched_regime_cohorts(frame)
    required = {"machine", "regime", "split", "data_pool", "segment_id",
                "parent_recording", "file_split", "file_domain", "anomaly_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing few-shot cohort fields: {sorted(missing)}")

    roles = set(frame["data_pool"].dropna().unique())
    unknown = roles - set(PROTOCOL_ROLES)
    if unknown:
        raise AssertionError(f"unknown protocol roles: {sorted(unknown)}")
    per_segment = frame.groupby(["machine", "regime", "segment_id"])["data_pool"].nunique()
    if (per_segment > 1).any():
        raise AssertionError("a segment carries more than one primary protocol role")

    for machine, part in frame.groupby("machine", sort=True):
        blocked = part[part["data_pool"].eq(BLOCKED_SIBLING_POOL)]
        # 1. blocked siblings are never consumed by any computation
        consumed_splits = set(blocked["split"]) & CONSUMING_SPLITS
        if consumed_splits:
            raise AssertionError(
                f"{machine} blocked siblings entered consuming splits {sorted(consumed_splits)}"
            )
        if not set(blocked["split"]) <= {BLOCKED_SIBLING_SPLIT}:
            raise AssertionError(f"{machine} blocked siblings carry a non-blocked split")
        blocked_ids = set(blocked["segment_id"])
        for role in ("source_training", "source_validation", "common_evaluation",
                     "target_adaptation"):
            overlap = blocked_ids & set(part.loc[part["data_pool"].eq(role), "segment_id"])
            if overlap:
                raise AssertionError(
                    f"{machine} blocked sibling appears in {role}: {sorted(overlap)[:5]}"
                )

        consumed = part[part["data_pool"].eq("target_adaptation")]
        # 2. only released target-normal training examples are consumed
        if not set(consumed["file_split"]) <= {"train"}:
            raise AssertionError(f"{machine} adaptation pool holds non-released segments")
        if not set(consumed["file_domain"]) <= {"target"}:
            raise AssertionError(f"{machine} adaptation pool holds non-target segments")
        if not set(consumed["anomaly_label"].astype(str).str.lower()) <= {"normal"}:
            raise AssertionError(f"{machine} adaptation pool holds anomalous segments")
        released_ids = set(part.loc[
            part["file_domain"].eq("target")
            & part["file_split"].eq("train")
            & part["anomaly_label"].astype(str).str.lower().eq("normal"),
            "segment_id",
        ])
        if set(consumed["segment_id"]) != released_ids:
            raise AssertionError(
                f"{machine} consumed adaptation set is not exactly the released allocation"
            )

        # 3. every consumed segment belongs to an adaptation parent group, and every
        #    blocked sibling shares a parent with a consumed segment
        adaptation_parents = set(consumed["parent_recording"])
        if not set(blocked["parent_recording"]) <= adaptation_parents:
            raise AssertionError(
                f"{machine} blocked sibling does not belong to an adaptation parent"
            )
        evaluation_parents = set(
            part.loc[part["data_pool"].eq("common_evaluation"), "parent_recording"]
        )
        if adaptation_parents & evaluation_parents:
            raise AssertionError(
                f"{machine} adaptation parent also supplies common-evaluation segments"
            )

    assert_no_parent_leakage(frame, ignore_splits=NON_CONSUMING_SPLITS)


def assert_common_test_unchanged(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> None:
    """Fail unless two split freezes share exactly the same evaluation cohort."""

    machines = sorted(set(reference["machine"]) | set(candidate["machine"]))
    for regime in ("S", "T"):
        for machine in machines:
            for field in ("segment_id", "parent_recording"):
                left = set(reference.loc[
                    reference["machine"].eq(machine)
                    & reference["regime"].eq(regime)
                    & reference["split"].eq("test"), field
                ])
                right = set(candidate.loc[
                    candidate["machine"].eq(machine)
                    & candidate["regime"].eq(regime)
                    & candidate["split"].eq("test"), field
                ])
                if left != right:
                    raise AssertionError(
                        f"{machine} regime {regime} common-test {field} changed"
                    )


def assert_matched_regime_cohorts(frame: pd.DataFrame) -> None:
    """Fail unless S/T differ only in access to the adaptation pool."""

    required = {
        "machine", "regime", "split", "segment_id", "parent_recording",
        "anomaly_label", "file_domain", "data_pool",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing matched-cohort fields: {sorted(missing)}")
    if set(frame["regime"]) != {"S", "T"}:
        raise AssertionError("matched comparison requires both Regime S and T")
    for machine, part in frame.groupby("machine", sort=True):
        s = part[part["regime"].eq("S")]
        t = part[part["regime"].eq("T")]
        for field in ("segment_id", "parent_recording"):
            s_test = set(s.loc[s["split"].eq("test"), field])
            t_test = set(t.loc[t["split"].eq("test"), field])
            if s_test != t_test:
                raise AssertionError(f"{machine} S/T test {field} sets differ")
        s_anomaly = set(s.loc[s["anomaly_label"].ne("normal"), "segment_id"])
        t_anomaly = set(t.loc[t["anomaly_label"].ne("normal"), "segment_id"])
        if s_anomaly != t_anomaly:
            raise AssertionError(f"{machine} S/T anomaly test populations differ")
        s_validation = set(s.loc[s["split"].eq("validation"), "segment_id"])
        t_validation = set(t.loc[t["split"].eq("validation"), "segment_id"])
        if s_validation != t_validation:
            raise AssertionError(f"{machine} S/T validation populations differ")
        adaptation_s = set(s.loc[s["data_pool"].eq("target_adaptation"), "segment_id"])
        adaptation_t = set(t.loc[t["data_pool"].eq("target_adaptation"), "segment_id"])
        if adaptation_s != adaptation_t:
            raise AssertionError(f"{machine} S/T adaptation pools differ")
        if set(s.loc[s["data_pool"].eq("target_adaptation"), "split"]) != {
            "reserved_adaptation"
        }:
            raise AssertionError(f"{machine} Regime S adaptation pool is not reserved")
        if set(t.loc[t["data_pool"].eq("target_adaptation"), "split"]) != {"train"}:
            raise AssertionError(f"{machine} Regime T adaptation pool is not train")


def assert_no_parent_leakage(
    frame: pd.DataFrame, *, ignore_splits: frozenset[str] = frozenset()
) -> None:
    """Fail if one parent recording spans more than one split.

    ``ignore_splits`` drops rows whose split is consumed by no computation before
    the check.  Phase 2.2 needs this: a RoboticArm adaptation parent legitimately
    holds released segments in ``train`` and blocked siblings in
    ``blocked_sibling``, and the blocked siblings enter no fit, no calibration and
    no metric, so they cannot carry information anywhere.  The default stays
    strict so the Phase-2/2.1 call sites are unchanged.
    """

    required = {"machine", "regime", "parent_recording", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing leakage-check fields: {sorted(missing)}")
    checked = frame[~frame["split"].isin(ignore_splits)] if ignore_splits else frame
    counts = checked.groupby(
        ["machine", "regime", "parent_recording"], dropna=False
    )["split"].nunique(dropna=False)
    bad = counts[counts > 1]
    if not bad.empty:
        raise AssertionError(
            "parent-recording leakage across splits: " + str(bad.index.tolist()[:10])
        )


def assert_no_window_sibling_leakage(window_manifest: pd.DataFrame) -> None:
    assert_no_parent_leakage(window_manifest)
    segment_counts = window_manifest.groupby(
        ["machine", "regime", "segment_id"], dropna=False
    )["split"].nunique(dropna=False)
    bad = segment_counts[segment_counts > 1]
    if not bad.empty:
        raise AssertionError(
            "window siblings from a segment cross splits: " + str(bad.index.tolist()[:10])
        )


def split_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "machine", "regime", "file_domain", "domain_shift_op",
        "domain_shift_env", "anomaly_label", "split",
    ]
    rows = []
    for group, part in frame.groupby(keys, dropna=False, sort=True):
        rows.append({
            **dict(zip(keys, group, strict=True)),
            "n_recordings": int(part["parent_recording"].nunique()),
            "n_segments": int(len(part)),
        })
    return pd.DataFrame(rows)


def load_audit_catalog(path: str | Path) -> pd.DataFrame:
    columns = [
        "machine", "segment_id", "source_csv", "file_split", "file_domain",
        "split_label", "anomaly_label", "domain_shift_op", "domain_shift_env",
    ]
    return pd.read_csv(path, usecols=columns)
