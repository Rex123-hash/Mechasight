"""Phase 3.4 execution library for the RoboticArm focal-parent sibling toggle (V3).

Frozen by ``configs/phase3/roboticarm_focal_toggle_execution_v2.json``
(``p34d-exec-e5470b025229f59a06f4c772``). This module provides the machinery for
the audited sequence A..H; importing it performs no training and no inference.

Training mirrors ``scripts/run_protocol_a_reproduction.py::stage_train`` line
for line - same seeding, same window-level stratified validation split, same
per-window standardisation, same loaders, same architecture, same optimiser,
same loss, same patience-3 schedule and the same reproduced upstream
best-checkpoint aliasing. The only additions are (1) subsetting the windowed
training set to the frozen arm membership and (2) capturing the initialisation
hash before the first optimiser step (audit finding A4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import h5py
import numpy as np
import pandas as pd

from src.experiment.provenance import (
    canonical_sha256, content_sha256, file_sha256, initial_state_hash, initialization_hash,
    ordered_id_hash,
)
from src.reproduction.protocol_a import (
    LABEL_NAMES, OFFICIAL_PARAMS, SENSORS, build_official_autoencoder,
    group_by_segment_id_median, official_single_auc, sensor_slices, standardize_window,
)

MACHINE = "RoboticArm"
CONFIG_PATH = Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
# INC-1 (2026-09-11T19:50:25Z): fold-1 PRESENT was consumed before any inference. The
# frozen recovery config names the eight retained looks, their order, and the two
# excluded ids. It is read at score time and at analysis time; it never relaxes a gate.
RECOVERY_CONFIG_PATH = Path("configs/phase3/roboticarm_focal_toggle_inc1_recovery_v1.json")
# Fold-1 technical recovery amendment (versioned; supersedes the NO RETRY decision by explicit
# decision after the four-fold result was known). Recognised ONLY for RecoveryExperiment objects;
# the original INC-1 namespace is never reused. Control/provenance only: the protected
# computation is the same _run_protected_inference and the same row/persistence code.
FOLD1_AMENDMENT_PATH = Path("configs/phase3/roboticarm_focal_toggle_fold1_recovery_amendment_v1.json")
MEMBERSHIP_PATH = Path("reports/phase3_4d_release_gate/final_arm_memberships.csv")
PLAN_PATH = Path("reports/phase3_4d_release_gate/planned_experiments_v3.csv")
WINDOWED_TRAIN = Path("data/interim/protocol_a/RoboticArm/train_dataset_window_0.100s.h5")
WINDOWED_TEST = Path("data/interim/protocol_a/RoboticArm/test_dataset_window_0.100s.h5")
WORK_ROOT = Path("data/interim/phase3_4")
RESULTS_ROOT = Path("reports/phase3_4_results")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
PROTOCOL = "roboticarm_focal_parent_sibling_toggle_v1"
TEST_COHORT_HASH = "83cd11956f8ee94907bcf366605259d4986acca8ff69a5bceb3874ee6a7a8efa"
SCORE_COLUMNS = ("total_loss",) + tuple(f"{p}_{s}" for s in SENSORS for p in ("f", "s"))
CHECKPOINT_RECORD_SCHEMA = "phase3.4-checkpoint-record-v1"
SCORE_ARTIFACT_SCHEMA = "phase3.4C-score-artifact-v1"


# ------------------------------------------------------------------ helpers
def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_recovery_config() -> dict:
    return json.loads(RECOVERY_CONFIG_PATH.read_text(encoding="utf-8"))


def retained_experiments() -> list["PlannedExperiment"]:
    """The eight INC-1-retained experiments in the frozen scoring order."""

    rec = load_recovery_config()
    by_id = {e.experiment_id: e for e in planned_experiments()}
    return [by_id[i] for i in rec["scoring_plan"]["frozen_order_experiment_ids"]]


def _recovery_gate(exp: "PlannedExperiment") -> None:
    """INC-1 recovery gates, evaluated BEFORE any claim. Never touches test data.

    Refuses the two excluded fold-1 ids permanently, and refuses any retained id
    that is not the next one lacking SCORE_COMPLETED in the frozen eight-look
    order. A retained look that is claimed but incomplete therefore blocks the
    whole phase: no skipping, no reordering, no substitution.
    """

    if getattr(exp, "recovery_id", None):
        _technical_recovery_gate(exp)                   # fold-1 amendment path
        return
    rec = load_recovery_config()
    excluded = rec["frozen_decisions"]
    if exp.experiment_id == excluded["fold_1_present"]["experiment_id"]:
        raise SystemExit(
            f"refusing to score {exp.experiment_id}: INC-1 fold-1 PRESENT is permanently "
            f"CONSUMED / INCOMPLETE; the frozen decision is NO RETRY")
    if exp.experiment_id == excluded["fold_1_absent"]["experiment_id"]:
        raise SystemExit(
            f"refusing to score {exp.experiment_id}: fold-1 ABSENT is frozen as "
            f"AUTHORIZED / UNUSED because its paired counterpart is permanently incomplete")
    order = rec["scoring_plan"]["frozen_order_experiment_ids"]
    if exp.experiment_id not in order:
        raise SystemExit(f"refusing to score {exp.experiment_id}: not in the frozen eight-look order")
    by_id = {e.experiment_id: e for e in planned_experiments()}
    for eid in order:
        states = [a["status"] for a in score_attempts(by_id[eid])]
        if SCORE_COMPLETED in states:
            continue
        if eid != exp.experiment_id:
            raise SystemExit(
                f"refusing to score {exp.experiment_id}: the next look in the frozen order is "
                f"{eid} (exposure history {states}); no skipping or reordering")
        return
    raise SystemExit(f"refusing to score {exp.experiment_id}: it already has SCORE_COMPLETED")


def load_plan() -> pd.DataFrame:
    return pd.read_csv(PLAN_PATH).sort_values("execution_position").reset_index(drop=True)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write to a temporary sibling then rename, so a partial file never exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


# ------------------------------------------------------- arm materialisation
def training_segment_ids(fold: int, arm: str) -> list[str]:
    """Exact training membership for one arm: B* plus that arm's contribution.

    B* = official training minus every segment of the focal parents minus every
    segment of the donor parents. PRESENT adds the focal contribution; ABSENT
    adds the donor contribution. Reproduces the audited set equations.
    """

    memb = pd.read_csv(MEMBERSHIP_PATH)
    sub = memb[memb.fold == fold]
    focal_parents = set(sub[sub.role == "focal_contribution"].parent_id)
    donor_parents = set(sub[sub.role == "donor_replacement"].parent_id)
    with h5py.File(WINDOWED_TRAIN, "r") as handle:
        ids = [v.decode("utf-8") for v in handle["segment_id"][:].flatten()]
    official = pd.DataFrame({"segment_id": ids}).drop_duplicates()
    official["parent"] = official.segment_id.str.replace(r"_\d+$", "", regex=True)
    background = official[~official.parent.isin(focal_parents | donor_parents)]
    role = "focal_contribution" if arm == "PRESENT" else "donor_replacement"
    contribution = set(sub[sub.role == role].segment_id)
    return sorted(set(background.segment_id) | contribution)


def membership_hash(segment_ids: list[str]) -> str:
    """Raw SHA-256 over the newline-joined sorted ids, exactly as planned_experiments_v3.csv."""

    from hashlib import sha256
    return sha256("\n".join(sorted(segment_ids)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- experiments
@dataclass(frozen=True)
class PlannedExperiment:
    experiment_id: str
    execution_position: int
    fold: int
    arm: str
    seed: int
    training_membership_sha256: str


def planned_experiments() -> list[PlannedExperiment]:
    plan = load_plan()
    return [PlannedExperiment(r.experiment_id, int(r.execution_position), int(r.fold), r.arm,
                              int(r.seed), r.training_membership_sha256)
            for r in plan.itertuples()]


@dataclass(frozen=True)
class RecoveryExperiment:
    """A planned experiment re-evaluated under a versioned technical-recovery identity.

    Same experiment_id, fold, arm, seed, membership and training record (hence the same
    frozen checkpoint); a distinct recovery_id gives it its own claim / attempt / artifact
    namespace so the original incident record is never touched or reused.
    """

    experiment_id: str
    execution_position: int
    fold: int
    arm: str
    seed: int
    training_membership_sha256: str
    recovery_id: str


def load_fold1_amendment() -> dict:
    return json.loads(FOLD1_AMENDMENT_PATH.read_text(encoding="utf-8"))


def recovery_experiments() -> list[RecoveryExperiment]:
    """The amendment's recovery executions, in frozen order (1P then 1A)."""

    amend = load_fold1_amendment()
    by_id = {e.experiment_id: e for e in planned_experiments()}
    out = []
    for r in sorted(amend["recovery_executions"], key=lambda r: r["order"]):
        e = by_id[r["experiment_id"]]
        out.append(RecoveryExperiment(e.experiment_id, e.execution_position, e.fold, e.arm,
                                      e.seed, e.training_membership_sha256, r["recovery_id"]))
    return out


def _artifact_stem(exp) -> str:
    rid = getattr(exp, "recovery_id", None)
    return f"{exp.experiment_id}__{rid}" if rid else exp.experiment_id


def _technical_recovery_gate(exp: RecoveryExperiment) -> None:
    """Admission for a fold-1 recovery evaluation. Evaluated BEFORE any claim.

    Requires: the amendment is frozen and committed at HEAD with the working copy
    byte-equal to the committed blob; this recovery_id is listed for this exact
    experiment and checkpoint; the original incident namespace is in exactly the
    state the amendment documents (PRESENT: original claim held with history
    [SCORE_STARTED] and no artifact; ABSENT: never claimed); the frozen recovery
    order 1P -> 1A; and at most one protected evaluation per recovery_id.
    """

    import subprocess
    amend = load_fold1_amendment()
    if amend["status"] != "FROZEN_BEFORE_RECOVERY_SCORING":
        raise SystemExit("refusing recovery: amendment is not frozen")
    committed = subprocess.run(["git", "show", f"HEAD:{FOLD1_AMENDMENT_PATH.as_posix()}"],
                               capture_output=True).stdout
    if not committed or committed.replace(b"\r\n", b"\n") != FOLD1_AMENDMENT_PATH.read_bytes().replace(b"\r\n", b"\n"):
        raise SystemExit("refusing recovery: the amendment is not committed at HEAD byte-for-byte")
    execs = sorted(amend["recovery_executions"], key=lambda r: r["order"])
    entry = next((r for r in execs if r["recovery_id"] == exp.recovery_id), None)
    if entry is None or entry["experiment_id"] != exp.experiment_id or entry["arm"] != exp.arm:
        raise SystemExit(f"refusing recovery: {exp.recovery_id} is not an amendment execution for {exp.experiment_id}")
    record = json.loads(training_record_path(exp).read_text(encoding="utf-8"))
    if record["checkpoint_sha256"] != entry["checkpoint_sha256"]:
        raise SystemExit("refusing recovery: amendment checkpoint hash differs from the frozen training record")
    original = PlannedExperiment(exp.experiment_id, exp.execution_position, exp.fold, exp.arm,
                                 exp.seed, exp.training_membership_sha256)
    orig_states = [a["status"] for a in score_attempts(original)]
    if entry["original_incident_claim"]:
        inc = amend["inc1_original_record"]
        claim = read_score_claim(original)
        if claim is None or claim["timestamp_utc"] != inc["claim_timestamp_utc"] \
                or orig_states != inc["attempt_history"] or score_path(original).exists():
            raise SystemExit("refusing recovery: the original INC-1 record is not in its documented state")
    else:
        if read_score_claim(original) is not None or orig_states or score_path(original).exists():
            raise SystemExit("refusing recovery: the original namespace was expected to be unclaimed")
    # frozen order and single evaluation per recovery id
    by_id = {e.experiment_id: e for e in planned_experiments()}
    for r in execs:
        rexp = RecoveryExperiment(r["experiment_id"], by_id[r["experiment_id"]].execution_position,
                                  r["fold"], r["arm"], 1995, by_id[r["experiment_id"]].training_membership_sha256,
                                  r["recovery_id"])
        states = [a["status"] for a in score_attempts(rexp)]
        if SCORE_COMPLETED in states:
            continue
        if r["recovery_id"] != exp.recovery_id:
            raise SystemExit(f"refusing recovery: next in the frozen recovery order is {r['recovery_id']} "
                             f"(history {states}); no skipping or reordering")
        return
    raise SystemExit(f"refusing recovery: {exp.recovery_id} already has SCORE_COMPLETED")


def _recovery_binding(exp) -> dict:
    """Extra provenance fields bound into the claim and SCORE_STARTED of a recovery
    evaluation; empty for ordinary planned experiments."""

    rid = getattr(exp, "recovery_id", None)
    if not rid:
        return {}
    amend = load_fold1_amendment()
    entry = next(r for r in amend["recovery_executions"] if r["recovery_id"] == rid)
    inc = amend["inc1_original_record"]
    return {
        "recovery_id": rid,
        "recovery_of_experiment_id": exp.experiment_id,
        "recovery_amendment": amend["amendment_id"],
        "recovery_amendment_commit": git_head(),
        "recovery_amendment_sha256": canonical_sha256(FOLD1_AMENDMENT_PATH),
        "original_incident_claim": ({"experiment_id": inc["experiment_id"],
                                     "claim_timestamp_utc": inc["claim_timestamp_utc"],
                                     "attempt_history": inc["attempt_history"],
                                     "protected_forward_passes": inc["protected_forward_passes"],
                                     "score_artifact": inc["score_artifact"]}
                                    if entry["original_incident_claim"] else None),
        "zero_forward_evidence": ("original INC-1 attempt ends at SCORE_STARTED; no artifact; sentinel "
                                  "replaced inference" if entry["original_incident_claim"]
                                  else "never claimed before this recovery"),
    }


def checkpoint_path(exp: PlannedExperiment) -> Path:
    return WORK_ROOT / f"fold{exp.fold}" / exp.arm / f"focal_toggle_seed{exp.seed}.pth"


def training_record_path(exp: PlannedExperiment) -> Path:
    return RESULTS_ROOT / "training" / f"{exp.experiment_id}.json"


def next_in_frozen_order() -> PlannedExperiment | None:
    """The next experiment whose training record does not yet exist, in frozen order."""

    for exp in planned_experiments():
        if not training_record_path(exp).exists():
            return exp
    return None


# ------------------------------------------------------------------ train
@dataclass(frozen=True)
class TrainingRecord:
    schema_version: str
    experiment_id: str
    execution_position: int
    fold: int
    arm: str
    seed: int
    config_identity: str
    execution_environment_id: str
    training_membership_sha256: str
    training_segments: int
    training_windows: int
    validation_windows: int
    architecture_identity: str
    training_configuration: dict
    initialization_sha256: str
    initial_state_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    epochs_run: int
    stopped_early: bool
    best_validation_epoch_zero_indexed: int
    best_validation_loss: float
    final_epoch_actually_scored_one_indexed: int
    training_losses: tuple
    valid_losses: tuple
    seconds: float
    torch_version: str
    git_head_at_training: str
    restart_count: int
    started_utc: str
    finished_utc: str


def _architecture_identity(params: dict, window_lengths, num_channels) -> str:
    return "p34-arch-" + content_sha256({
        "layer_dims": params["layer_dims"], "batchnorm_hidden": True,
        "window_lengths": list(map(int, window_lengths)),
        "num_channels": list(map(int, num_channels))})[:20]


def train_experiment(exp: PlannedExperiment, device: str = "cpu",
                     restart_count: int = 0) -> TrainingRecord:
    """Train one planned experiment. Refuses to run out of frozen order or twice."""

    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    cfg = load_config()
    env = cfg["execution_environment"]
    if device != env["device"]:
        raise SystemExit(f"frozen environment requires device={env['device']!r}")
    if torch.__version__ != env["torch"]:
        raise SystemExit(f"frozen environment requires torch {env['torch']}, found {torch.__version__}")
    torch.set_num_threads(int(env["torch_num_threads"]))

    nxt = next_in_frozen_order()
    if nxt is None or nxt.experiment_id != exp.experiment_id:
        raise SystemExit(
            f"refusing to train {exp.experiment_id}: the next experiment in the frozen order is "
            f"{nxt.experiment_id if nxt else 'none (all complete)'}")

    segment_ids = training_segment_ids(exp.fold, exp.arm)
    if membership_hash(segment_ids) != exp.training_membership_sha256:
        raise SystemExit("materialised membership does not match the frozen membership hash")

    params = dict(OFFICIAL_PARAMS)
    if params["seed"] != exp.seed:
        raise SystemExit("seed policy violated")
    started = utc_now()

    # ---- identical to Protocol-A stage_train from here, except the membership mask
    torch.manual_seed(params["seed"])

    with h5py.File(WINDOWED_TRAIN, "r") as f:
        Y_all = pd.DataFrame(
            [[v.decode("utf-8") for v in f[name][:].flatten()] for name in LABEL_NAMES]).T
        Y_all.columns = list(LABEL_NAMES)
    keep = np.flatnonzero(Y_all.segment_id.isin(set(segment_ids)).to_numpy())
    Y = Y_all.iloc[keep].reset_index(drop=True)
    Y["combined_label"] = Y["anomaly_label"] + Y["domain_shift_op"] + Y["domain_shift_env"]

    train_idx, valid_idx, _, _ = train_test_split(
        range(len(Y)), Y, stratify=Y["combined_label"],
        test_size=params["valid_size"], random_state=params["seed"])
    train_rows = keep[np.asarray(train_idx)]
    valid_rows = keep[np.asarray(valid_idx)]

    X_train, X_valid = [], []
    with h5py.File(WINDOWED_TRAIN, "r") as f:
        for sensor in SENSORS:
            raw = f[sensor][:]
            X_train.append(standardize_window(raw[train_rows]))
            X_valid.append(standardize_window(raw[valid_rows]))
            del raw

    num_channels = [x.shape[1] for x in X_train]
    window_lengths = [x.shape[2] for x in X_train]

    class _DS(Dataset):
        def __init__(self, X): self.X = X
        def __len__(self): return len(self.X[0])
        def __getitem__(self, i): return [x[i] for x in self.X]

    to_t = lambda xs: [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in xs]
    train_loader = DataLoader(_DS(to_t(X_train)), batch_size=params["batch_size"], shuffle=True)
    valid_loader = DataLoader(_DS(to_t(X_valid)), batch_size=params["batch_size"], shuffle=False)

    model = build_official_autoencoder(window_lengths, num_channels, params).to(device)
    init_sha = initialization_hash(model)          # A4: before the first optimizer step
    state_sha = initial_state_hash(model)          # F2: full state_dict incl. BatchNorm buffers
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    def mse(a, b):
        return torch.mean((a - b) ** 2, dim=1)

    def evaluate(loader) -> float:
        model.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for xb in loader:
                xb = torch.concat([x.flatten(1) for x in xb], axis=1)
                _, est = model(xb)
                total += torch.mean(mse(xb, est)).item()
                batches += 1
        return total / batches

    training_losses, valid_losses = [], []
    best_valid, best_epoch, patience_counter = float("inf"), -1, 0
    stopped_early = False
    t0 = perf_counter()
    for epoch in range(params["num_epochs"]):
        model.train()
        epoch_loss, batches = 0.0, 0
        for xb in train_loader:
            xb = torch.concat([x.flatten(1) for x in xb], axis=1)
            optimizer.zero_grad()
            _, est = model(xb)
            loss = torch.mean(mse(xb, est))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}; frozen policy: STOP")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        training_losses.append(epoch_loss / batches)
        v = evaluate(valid_loader)
        valid_losses.append(v)
        print(f"  [{exp.experiment_id}] epoch {epoch + 1:3d} | train {training_losses[-1]:.6f} "
              f"| valid {v:.6f}", flush=True)
        patience_counter += 1
        if v < best_valid:
            best_valid, best_epoch, patience_counter = v, epoch, 0
        if patience_counter == params["patience"]:
            stopped_early = True
            break
    # upstream restores best_model_state, which aliases the live tensors: final-epoch weights.

    ckpt = checkpoint_path(exp)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)

    record = TrainingRecord(
        schema_version=CHECKPOINT_RECORD_SCHEMA,
        experiment_id=exp.experiment_id, execution_position=exp.execution_position,
        fold=exp.fold, arm=exp.arm, seed=exp.seed,
        config_identity=cfg["config_identity"],
        execution_environment_id=env["execution_environment_id"],
        training_membership_sha256=exp.training_membership_sha256,
        training_segments=len(segment_ids), training_windows=int(len(keep)),
        validation_windows=int(len(valid_rows)),
        architecture_identity=_architecture_identity(params, window_lengths, num_channels),
        training_configuration={k: params[k] for k in
                                ("layer_dims", "lr", "criterion", "batch_size", "num_epochs",
                                 "patience", "normalisation", "valid_size", "seed")},
        initialization_sha256=init_sha, initial_state_sha256=state_sha,
        checkpoint_path=str(ckpt).replace("\\", "/"), checkpoint_sha256=file_sha256(ckpt),
        epochs_run=len(training_losses), stopped_early=stopped_early,
        best_validation_epoch_zero_indexed=int(best_epoch),
        best_validation_loss=float(best_valid),
        final_epoch_actually_scored_one_indexed=len(training_losses),
        training_losses=tuple(map(float, training_losses)),
        valid_losses=tuple(map(float, valid_losses)),
        seconds=round(perf_counter() - t0, 1), torch_version=torch.__version__,
        git_head_at_training=git_head(), restart_count=restart_count,
        started_utc=started, finished_utc=utc_now())
    atomic_write_text(training_record_path(exp), json.dumps(asdict(record), indent=2))
    return record


# ----------------------------------------------------------- authorisation
def frozen_checkpoint_hashes() -> dict[str, str]:
    """experiment_id -> checkpoint sha256 from the persisted training records (COMMIT B)."""

    out = {}
    for exp in planned_experiments():
        p = training_record_path(exp)
        if p.exists():
            out[exp.experiment_id] = json.loads(p.read_text(encoding="utf-8"))["checkpoint_sha256"]
    return out


def build_authorisations(evaluator_git_sha: str) -> list:
    """The 10 test-look records, each naming the frozen checkpoint hash it authorises.

    Refuses unless every planned checkpoint is frozen, the ledger holds exactly
    the 30 historical entries, and no Phase-3.4 entry exists. Builds records only;
    the caller appends them atomically.
    """

    from src.experiment.ledger import TestLookLedger, TestLookRecord

    cfg = load_config()
    hashes = frozen_checkpoint_hashes()
    plan = planned_experiments()
    missing = [e.experiment_id for e in plan if e.experiment_id not in hashes]
    if missing:
        raise SystemExit(f"cannot authorise: {len(missing)} checkpoints are not frozen yet")
    existing = TestLookLedger(LEDGER).read()
    if len(existing) != 30:
        raise SystemExit(f"expected the 30 historical ledger entries, found {len(existing)}")
    if any(e.get("protocol") == PROTOCOL for e in existing):
        raise SystemExit("Phase-3.4 authorisations already exist; refusing to duplicate")

    stamp = utc_now()
    records = []
    for exp in plan:
        records.append(TestLookRecord(
            timestamp_utc=stamp,
            experiment_id=exp.experiment_id,
            git_sha=evaluator_git_sha,
            machine=MACHINE,
            protocol=PROTOCOL,
            regime="T",
            model_config_identity=cfg["config_identity"],
            test_cohort_hash=TEST_COHORT_HASH,
            reason=(f"first locked Phase-3.4 focal-parent sibling-toggle test look; fold={exp.fold}; "
                    f"arm={exp.arm}; seed={exp.seed}; "
                    f"training_membership_sha256={exp.training_membership_sha256}; "
                    f"checkpoint_sha256={hashes[exp.experiment_id]}"),
            metrics_requested=("focal_parent_concordance_delta", "focal_auroc_delta",
                               "per_segment_scores_persisted"),
        ))
    return records


# ------------------------------------------------------------------ score
SCORE_STARTED = "SCORE_STARTED"
SCORE_COMPLETED = "SCORE_COMPLETED"


def _ledger_authorisation(experiment_id: str, checkpoint_sha: str) -> dict | None:
    """The ledger entry that names this experiment AND this exact checkpoint hash."""

    if not LEDGER.exists():
        return None
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if (e.get("protocol") == PROTOCOL and e.get("experiment_id") == experiment_id
                and f"checkpoint_sha256={checkpoint_sha}" in e.get("reason", "")):
            return e
    return None


def _ledger_authorises(experiment_id: str, checkpoint_sha: str) -> bool:
    return _ledger_authorisation(experiment_id, checkpoint_sha) is not None


def score_path(exp: PlannedExperiment) -> Path:
    return RESULTS_ROOT / "scores" / f"{_artifact_stem(exp)}.csv"


def score_attempt_path(exp: PlannedExperiment) -> Path:
    """Append-only exposure history for one experiment. Never deleted, never reset."""

    return RESULTS_ROOT / "score_attempts" / f"{_artifact_stem(exp)}.jsonl"


def score_attempts(exp: PlannedExperiment) -> list[dict]:
    p = score_attempt_path(exp)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_claim_path(exp: PlannedExperiment) -> Path:
    """Single-use, experiment-specific claim. Created once with O_CREAT|O_EXCL; never removed."""

    return RESULTS_ROOT / "score_claims" / f"{_artifact_stem(exp)}.claim"


def read_score_claim(exp: PlannedExperiment) -> dict | None:
    p = score_claim_path(exp)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


class ClaimAlreadyHeld(SystemExit):
    """Raised when the protected-score claim for an experiment already exists."""


# Test seam (Phase 3.4F.1 finding F3): invoked once, immediately before the atomic
# claim attempt, after every pre-check has passed. The concurrency test binds a
# multiprocessing barrier here so two real processes reach the O_EXCL create
# together. Production leaves it as a no-op.
_pre_claim_hook = None


def acquire_score_claim(exp: PlannedExperiment, claim: dict) -> None:
    """Atomically create the claim file or refuse. The create IS the check.

    ``os.open`` with ``O_CREAT | O_EXCL`` is the operating system's
    create-if-absent primitive: exactly one caller can succeed for a given path,
    however many race. There is no separate "exists?" test that a second process
    could interleave with. On success the claim content is written and fsynced
    through the same descriptor before it is closed, so a durable claim exists
    before anything else happens. On failure nothing is written.
    """

    path = score_claim_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(claim, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o444)
    except FileExistsError:
        raise ClaimAlreadyHeld(
            f"refusing to score {exp.experiment_id}: the protected-score claim is already held "
            f"(claim file exists). The look is consumed; if no SCORE_COMPLETED record exists the "
            f"experiment is INCOMPLETE and must not be re-scored.")
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("short write on score claim")
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_durable(path: Path, record: dict) -> None:
    """Append one JSON line with O_APPEND and fsync so it is durable before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("short write on exposure record")
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_protected_inference(record: dict, device: str) -> pd.DataFrame:
    """The ONLY function that loads the protected test cohort and runs a model on it.

    Returns the segment-median grouped score table. Isolated so the state
    machine around it can be tested with a fake that never touches real test
    data.
    """

    import torch

    params = dict(OFFICIAL_PARAMS)
    with h5py.File(WINDOWED_TEST, "r") as f:
        X_test = [f[s][:] for s in SENSORS]
        Y = pd.DataFrame(
            [[v.decode("utf-8") for v in f[name][:].flatten()] for name in LABEL_NAMES]).T
        Y.columns = list(LABEL_NAMES)
    if ordered_id_hash(sorted(set(Y.segment_id))) != TEST_COHORT_HASH:
        raise SystemExit("official test cohort hash mismatch")
    X_test = [standardize_window(x) for x in X_test]
    num_channels = [x.shape[1] for x in X_test]
    window_lengths = [x.shape[2] for x in X_test]

    model = build_official_autoencoder(window_lengths, num_channels, params).to(device)
    model.load_state_dict(torch.load(Path(record["checkpoint_path"]), map_location=device))
    model.eval()
    bounds = sensor_slices(window_lengths, num_channels)
    tensors = [torch.from_numpy(np.ascontiguousarray(x)) for x in X_test]
    n, batch = len(tensors[0]), params["batch_size"]
    total, fused, masked = [], {s: [] for s in SENSORS}, {s: [] for s in SENSORS}
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.concat([t[i:i + batch].flatten(1) for t in tensors], axis=1).to(device)
            _, est = model(xb)
            total.append(torch.mean((xb - est) ** 2, dim=1).cpu().numpy())
            for (lo, hi), sensor in zip(bounds, SENSORS, strict=True):
                fused[sensor].append(
                    torch.mean((xb[:, lo:hi] - est[:, lo:hi]) ** 2, dim=1).cpu().numpy())
            for (lo, hi), sensor in zip(bounds, SENSORS, strict=True):
                m_in = torch.zeros_like(xb)
                m_in[:, lo:hi] = xb[:, lo:hi]
                _, m_est = model(m_in)
                masked[sensor].append(
                    torch.mean((xb[:, lo:hi] - m_est[:, lo:hi]) ** 2, dim=1).cpu().numpy())
    scores = pd.DataFrame({"total_loss": np.concatenate(total)})
    for sensor in SENSORS:
        scores[f"f_{sensor}"] = np.concatenate(fused[sensor])
        scores[f"s_{sensor}"] = np.concatenate(masked[sensor])
    return group_by_segment_id_median(pd.concat([Y, scores], axis=1), scores.columns)


def score_experiment(exp: PlannedExperiment, device: str = "cpu") -> dict:
    """Score one frozen checkpoint on the protected cohort exactly once.

    State machine (Phase 3.4F.1 finding F1)::

        AUTHORIZED -> CLAIMED (atomic, O_EXCL) -> SCORE_STARTED
                  -> artifact persisted + hashed -> SCORE_COMPLETED

    Gates, all of which must pass before any test data is read: the training
    record exists; the checkpoint on disk matches the frozen hash; a ledger
    entry names that exact hash; no claim and no exposure record exist; no score
    artifact exists. The claim is then acquired ATOMICALLY with O_CREAT|O_EXCL,
    so of any number of concurrent invocations exactly one can proceed (finding
    F3); SCORE_STARTED is appended durably after the claim and BEFORE the first
    protected forward. Once the claim exists the look is consumed: any later
    invocation refuses, whether or not an artifact was ever produced, and the
    claim is never deleted or replaced. Nothing here computes an aggregate.
    """

    cfg = load_config()
    rec_path = training_record_path(exp)
    if not rec_path.exists():
        raise SystemExit("no frozen training record (COMMIT B) for this experiment")
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    ckpt = Path(record["checkpoint_path"])
    if not ckpt.exists() or file_sha256(ckpt) != record["checkpoint_sha256"]:
        raise SystemExit("checkpoint on disk does not match the frozen hash")
    authorisation = _ledger_authorisation(exp.experiment_id, record["checkpoint_sha256"])
    if authorisation is None:
        raise SystemExit("refusing to score: no ledger entry authorises this exact checkpoint")
    _recovery_gate(exp)                                 # INC-1: exclusions and frozen order

    existing_claim = read_score_claim(exp)
    if existing_claim is not None:
        if existing_claim.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            raise SystemExit(
                f"HARD FAILURE for {exp.experiment_id}: an existing protected-score claim names "
                f"checkpoint {existing_claim.get('checkpoint_sha256')} but the frozen checkpoint "
                f"is {record['checkpoint_sha256']}. Claims are never replaced; investigate.")
        states = [a["status"] for a in score_attempts(exp)]
        raise SystemExit(
            f"refusing to score {exp.experiment_id}: the protected-score claim is already held "
            f"(exposure history: {states or ['claim only, no SCORE_STARTED']}). The look is "
            f"consumed; without SCORE_COMPLETED the experiment is INCOMPLETE and must not be "
            f"re-scored.")
    attempts = score_attempts(exp)
    if attempts:
        states = [a["status"] for a in attempts]
        raise SystemExit(
            f"refusing to score {exp.experiment_id}: exposure history exists without a claim "
            f"({states}); this is an inconsistent state and must be investigated, not re-scored.")
    out = score_path(exp)
    if out.exists():
        raise SystemExit("score artifact already exists; a test look may be used exactly once")

    evaluator_sha = git_head()
    claim = {
        "status": "CLAIMED", "timestamp_utc": utc_now(),
        "experiment_id": exp.experiment_id, "fold": exp.fold, "arm": exp.arm, "seed": exp.seed,
        "checkpoint_sha256": record["checkpoint_sha256"], "evaluator_git_sha": evaluator_sha,
        "config_identity": cfg["config_identity"],
        "training_membership_sha256": exp.training_membership_sha256,
        "test_cohort_hash": TEST_COHORT_HASH,
        "ledger_authorization_timestamp_utc": authorisation["timestamp_utc"],
        **_recovery_binding(exp),
    }
    if _pre_claim_hook is not None:
        _pre_claim_hook()
    acquire_score_claim(exp, claim)                     # atomic single-owner claim
    started = {
        "status": SCORE_STARTED, "timestamp_utc": utc_now(),
        "experiment_id": exp.experiment_id, "fold": exp.fold, "arm": exp.arm, "seed": exp.seed,
        "checkpoint_sha256": record["checkpoint_sha256"], "evaluator_git_sha": evaluator_sha,
        "config_identity": cfg["config_identity"],
        "training_membership_sha256": exp.training_membership_sha256,
        "test_cohort_hash": TEST_COHORT_HASH,
        "ledger_authorization": {"experiment_id": authorisation["experiment_id"],
                                 "timestamp_utc": authorisation["timestamp_utc"],
                                 "git_sha": authorisation["git_sha"]},
        "claim_timestamp_utc": claim["timestamp_utc"],
        "claim_checkpoint_sha256": claim["checkpoint_sha256"],
        **_recovery_binding(exp),
    }
    _append_durable(score_attempt_path(exp), started)   # durable BEFORE any test forward

    grouped = _run_protected_inference(record, device)

    memb = pd.read_csv(MEMBERSHIP_PATH)
    focal_fold = {p: int(k) for k, sub in memb.groupby("fold")
                  for p in set(sub[sub.role == "focal_contribution"].parent_id)}
    blocking = {"20240222_11_14_08_dataLog", "20240405_13_09_52_dataLog"}
    grouped = grouped.copy()
    grouped["parent_id"] = grouped.segment_id.str.replace(r"_\d+$", "", regex=True)

    def status(p):
        if p in blocking:
            return "excluded_blocking"
        if p in focal_fold:
            return "focal_this_fold" if focal_fold[p] == exp.fold else "focal_other_fold"
        return "non_focal"

    table = pd.DataFrame({
        "experiment_id": exp.experiment_id, "config_identity": cfg["config_identity"],
        "fold": exp.fold, "arm": exp.arm, "seed": exp.seed,
        "segment_id": grouped.segment_id, "parent_id": grouped.parent_id,
        "focal_parent_status": grouped.parent_id.map(status),
        "focal_fold": grouped.parent_id.map(lambda p: focal_fold.get(p, "")),
        "split_label": grouped.split_label, "anomaly_label": grouped.anomaly_label,
        "domain_shift_op": grouped.domain_shift_op, "domain_shift_env": grouped.domain_shift_env,
    })
    for col in SCORE_COLUMNS:
        table[col] = grouped[col].astype(np.float64)
    table["checkpoint_sha256"] = record["checkpoint_sha256"]
    table["evaluator_git_sha"] = evaluator_sha
    table["test_cohort_hash"] = TEST_COHORT_HASH
    table["training_membership_sha256"] = exp.training_membership_sha256
    table["execution_environment_id"] = cfg["execution_environment"]["execution_environment_id"]
    if len(table) != 464:
        raise SystemExit(f"expected 464 score rows, built {len(table)}")

    atomic_write_text(out, table.to_csv(index=False, lineterminator="\n", float_format="%.17g"))
    digest = canonical_sha256(out)
    atomic_write_text(out.with_suffix(".sha256"), digest + "\n")
    _append_durable(score_attempt_path(exp), {
        "status": SCORE_COMPLETED, "timestamp_utc": utc_now(),
        "experiment_id": exp.experiment_id, "checkpoint_sha256": record["checkpoint_sha256"],
        "score_artifact": str(out).replace("\\", "/"), "score_artifact_sha256": digest,
        "rows": int(len(table)), "expected_rows": 464,
        "claim_timestamp_utc": claim["timestamp_utc"]})
    return {"experiment_id": exp.experiment_id, "rows": int(len(table)),
            "score_artifact": str(out).replace("\\", "/"), "score_artifact_sha256": digest,
            "exposure_history": ["CLAIMED", SCORE_STARTED, SCORE_COMPLETED],
            "aggregates_computed_here": False}


# ---------------------------------------------------------------- analyse
def load_persisted_scores(experiments: list[PlannedExperiment] | None = None) -> pd.DataFrame:
    """Persisted score rows for the given experiments (default: the eight INC-1-retained
    looks), verifying each artifact against its recorded hash first."""

    if experiments is None:
        experiments = retained_experiments()
    frames = []
    for exp in experiments:
        p = score_path(exp)
        if not p.exists():
            raise SystemExit(f"score artifact missing for {exp.experiment_id}")
        recorded = p.with_suffix(".sha256").read_text(encoding="utf-8").strip()
        if canonical_sha256(p) != recorded:
            raise SystemExit(f"score artifact hash mismatch for {exp.experiment_id}")
        frames.append(pd.read_csv(p))
    scores = pd.concat(frames, ignore_index=True)
    if len(scores) != 464 * len(experiments):
        raise SystemExit(f"expected {464 * len(experiments)} persisted score rows, found {len(scores)}")
    return scores


def analyse_from_persisted_scores() -> dict:
    """INC-1 incident-contingent SECONDARY sensitivity analysis, computed ONLY from
    the eight retained persisted score artifacts. Performs no inference.

    The original five-fold primary analysis is NOT FULLY ESTIMABLE (fold-1 PRESENT
    is permanently incomplete; fold-1 ABSENT is intentionally unused). This
    function applies the unchanged estimator - parent-equal-weighted focal
    concordance delta, fixed 232-segment anomaly reference, fold-stratified parent
    bootstrap - restricted to the 48 focal parents of intact folds 2-5.
    """

    from src.experiment.focal_concordance import parent_concordance_delta, segment_weighted_auroc

    cfg = load_config()
    rec = load_recovery_config()
    salvage = rec["salvage_analysis"]
    retained = retained_experiments()
    scores = load_persisted_scores(retained)
    if sorted(int(f) for f in scores.fold.unique()) != salvage["retained_folds"]:
        raise SystemExit("persisted scores do not cover exactly the retained folds")
    unc = cfg["uncertainty"]

    per_fold = {}
    frames = []
    for fold, sub in scores.groupby("fold"):
        focal = sub[sub.focal_parent_status == "focal_this_fold"]
        anomaly = sub[sub.anomaly_label != "normal"]
        if anomaly.segment_id.nunique() != 232:
            raise SystemExit("anomaly reference must be the fixed 232 segments")
        frame = pd.concat([focal[focal.anomaly_label == "normal"], anomaly])
        frame = frame.assign(focal_fold=fold)
        frames.append(frame[["segment_id", "parent_id", "arm", "anomaly_label",
                             "focal_fold", "total_loss"]])
        arms = {}
        for arm, a in frame.groupby("arm"):
            n = a[a.anomaly_label == "normal"].total_loss
            an = a[a.anomaly_label != "normal"].total_loss
            arms[arm] = {"segment_weighted_focal_auroc": segment_weighted_auroc(n, an),
                         "mean_raw_focal_normal_score": float(n.mean()),
                         "mean_raw_anomaly_reference_score": float(an.mean())}
        per_fold[int(fold)] = arms
    pooled = pd.concat(frames, ignore_index=True)
    result = parent_concordance_delta(pooled, replicates=int(unc["replicates"]),
                                      seed=int(unc["seed"]), stratify_by_fold=True)
    if result.n_parents != salvage["expected_focal_parents"]:
        raise SystemExit(f"expected {salvage['expected_focal_parents']} focal parents, "
                         f"found {result.n_parents}")

    pd_ = result.parent_deltas
    return {
        "schema_version": rec["analysis_output"]["schema_version"],
        "analysis_label": salvage["label"],
        "not_to_be_called": salvage["not_to_be_called"],
        "computed_from": "the eight retained persisted per-segment score artifacts only; no inference",
        "config_identity": cfg["config_identity"],
        "recovery_config_identity": rec["config_identity"],
        "original_primary_analysis": rec["original_primary_analysis"],
        "incident": {k: rec["incident"][k] for k in (
            "id", "affected_experiment_id", "affected_fold", "affected_arm",
            "protected_forward_passes", "score_completed_exists", "score_artifact_exists")},
        "fold_1_absent": rec["frozen_decisions"]["fold_1_absent"],
        "retained_experiment_ids": [e.experiment_id for e in retained],
        "retained_folds": salvage["retained_folds"],
        "estimand": salvage["estimand"],
        "salvage": {
            "name": "parent-equal-weighted focal concordance delta (folds 2-5, 48 parents)",
            "point_estimate": result.point_estimate,
            "ci95_low": result.ci_low, "ci95_high": result.ci_high,
            "bootstrap": {"replicates": result.bootstrap_replicates,
                          "seed": result.bootstrap_seed,
                          "fold_stratified": result.bootstrap_stratified_by_fold,
                          "conditional_on": salvage["bootstrap"]["conditional_on"],
                          "does_not_include": salvage["bootstrap"]["does_not_include"]},
            "n_focal_parents": result.n_parents,
            "n_focal_segments_descriptive": result.n_segments,
            "fold_level_deltas_descriptive": {int(k): float(v) for k, v in result.fold_deltas.items()},
            "parents_delta_positive": int((pd_ > 0).sum()),
            "parents_delta_zero": int((pd_ == 0).sum()),
            "parents_delta_negative": int((pd_ < 0).sum()),
        },
        "secondary": {
            "S1_segment_weighted_focal_auroc_delta_by_fold": {
                k: v["PRESENT"]["segment_weighted_focal_auroc"] - v["ABSENT"]["segment_weighted_focal_auroc"]
                for k, v in per_fold.items()},
            "S4_raw_focal_normal_score_shift_by_fold_SCALE_DEPENDENT": {
                k: v["PRESENT"]["mean_raw_focal_normal_score"] - v["ABSENT"]["mean_raw_focal_normal_score"]
                for k, v in per_fold.items()},
            "S5_anomaly_reference_raw_shift_by_fold_specificity": {
                k: v["PRESENT"]["mean_raw_anomaly_reference_score"] - v["ABSENT"]["mean_raw_anomaly_reference_score"]
                for k, v in per_fold.items()},
        },
        "mandatory_qualifiers": cfg["claim_boundary"]["mandatory_qualifiers"],
        "mandatory_report_statements": rec["mandatory_report_statements"],
        "limitations": salvage["limitations"],
        "parent_deltas": {k: float(v) for k, v in pd_.items()},
    }


def analyse_five_fold_recovered() -> dict:
    """RECOVERED ORIGINAL FIVE-FOLD ANALYSIS, computed ONLY from persisted artifacts:
    the two fold-1 recovery artifacts plus the eight frozen E1 artifacts (folds 2-5).
    Performs no inference. The four-fold function above is untouched and its committed
    artifact is never rewritten. Same estimator, same reference cohort, same bootstrap
    parameters; 60 parents, folds 1-5.
    """

    from src.experiment.focal_concordance import parent_concordance_delta, segment_weighted_auroc

    cfg = load_config()
    amend = load_fold1_amendment()
    ff = amend["five_fold_analysis"]
    recovered = recovery_experiments()
    retained = retained_experiments()
    scores = load_persisted_scores(recovered + retained)
    if sorted(int(f) for f in scores.fold.unique()) != ff["folds"]:
        raise SystemExit("persisted scores do not cover exactly folds 1-5")
    unc = cfg["uncertainty"]

    per_fold = {}
    frames = []
    for fold, sub in scores.groupby("fold"):
        focal = sub[sub.focal_parent_status == "focal_this_fold"]
        anomaly = sub[sub.anomaly_label != "normal"]
        if anomaly.segment_id.nunique() != ff["anomaly_reference_segments"]:
            raise SystemExit("anomaly reference must be the fixed 232 segments")
        frame = pd.concat([focal[focal.anomaly_label == "normal"], anomaly])
        frame = frame.assign(focal_fold=fold)
        frames.append(frame[["segment_id", "parent_id", "arm", "anomaly_label",
                             "focal_fold", "total_loss"]])
        arms = {}
        for arm, a in frame.groupby("arm"):
            n = a[a.anomaly_label == "normal"].total_loss
            an = a[a.anomaly_label != "normal"].total_loss
            arms[arm] = {"segment_weighted_focal_auroc": segment_weighted_auroc(n, an),
                         "mean_raw_focal_normal_score": float(n.mean()),
                         "mean_raw_anomaly_reference_score": float(an.mean())}
        per_fold[int(fold)] = arms
    pooled = pd.concat(frames, ignore_index=True)
    result = parent_concordance_delta(pooled, replicates=int(unc["replicates"]),
                                      seed=int(unc["seed"]), stratify_by_fold=True)
    if result.n_parents != ff["focal_parents"]:
        raise SystemExit(f"expected {ff['focal_parents']} focal parents, found {result.n_parents}")

    pd_ = result.parent_deltas
    return {
        "schema_version": "phase3.4-five-fold-recovered-analysis-v1",
        "analysis_label": ff["label"],
        "computed_from": "ten persisted per-segment score artifacts only (2 fold-1 recovery + 8 frozen E1); no inference",
        "config_identity": cfg["config_identity"],
        "amendment_id": amend["amendment_id"],
        "amendment_sha256": canonical_sha256(FOLD1_AMENDMENT_PATH),
        "four_fold_result_preserved": amend["decision_context"]["four_fold_result_known_at_decision"],
        "inc1_original_record": amend["inc1_original_record"],
        "recovery_ids": [e.recovery_id for e in recovered],
        "experiment_ids": [e.experiment_id for e in recovered + retained],
        "folds": ff["folds"],
        "estimand": cfg["primary_estimand"]["name"],
        "primary": {
            "name": "parent-equal-weighted focal concordance delta (folds 1-5, 60 parents; fold 1 recovered)",
            "point_estimate": result.point_estimate,
            "ci95_low": result.ci_low, "ci95_high": result.ci_high,
            "bootstrap": {"replicates": result.bootstrap_replicates, "seed": result.bootstrap_seed,
                          "fold_stratified": result.bootstrap_stratified_by_fold,
                          "conditional_on": "the ten fitted models (five pairs)",
                          "does_not_include": unc["does_not_estimate"]},
            "n_focal_parents": result.n_parents,
            "n_focal_segments_descriptive": result.n_segments,
            "fold_level_deltas_descriptive": {int(k): float(v) for k, v in result.fold_deltas.items()},
            "parents_delta_positive": int((pd_ > 0).sum()),
            "parents_delta_zero": int((pd_ == 0).sum()),
            "parents_delta_negative": int((pd_ < 0).sum()),
        },
        "secondary": {
            "S1_segment_weighted_focal_auroc_delta_by_fold": {
                k: v["PRESENT"]["segment_weighted_focal_auroc"] - v["ABSENT"]["segment_weighted_focal_auroc"]
                for k, v in per_fold.items()},
            "S4_raw_focal_normal_score_shift_by_fold_SCALE_DEPENDENT": {
                k: v["PRESENT"]["mean_raw_focal_normal_score"] - v["ABSENT"]["mean_raw_focal_normal_score"]
                for k, v in per_fold.items()},
            "S5_anomaly_reference_raw_shift_by_fold_specificity": {
                k: v["PRESENT"]["mean_raw_anomaly_reference_score"] - v["ABSENT"]["mean_raw_anomaly_reference_score"]
                for k, v in per_fold.items()},
        },
        "mandatory_qualifiers": cfg["claim_boundary"]["mandatory_qualifiers"],
        "recovery_statement": "completed after a documented zero-exposure technical recovery of fold-1 PRESENT (INC-1); reported regardless of direction; the four-fold incident-contingent result is preserved unchanged",
        "parent_deltas": {k: float(v) for k, v in pd_.items()},
    }


# ------------------------------------------------------------ IMPL-1 correction
IMPL1_CORRECTED_SCHEMA = "phase3.4-impl1-corrected-analysis-v1"


def analyse_corrected(folds: list[int]) -> dict:
    """IMPL-1 CORRECTED analysis over ``folds`` (4 = [2..5], 5 = [1..5]), computed ONLY
    from persisted, hash-verified artifacts. Performs no inference.

    Correction: each fold's focal normal segments are ranked against exactly the 232
    anomaly segments scored by THAT fold's model of the SAME arm
    (``parent_concordance_delta_by_fold``). The historical pooled-reference results
    (four-fold 6441ed66..., five-fold 427b04cb...) are left untouched as artifacts and
    are superseded for scientific interpretation.

    Secondary per-fold quantities (S1/S4/S5) were always computed inside each fold's
    own sub-frame and are therefore UNAFFECTED by IMPL-1; they are recomputed here
    identically so the corrected artifact is self-contained.
    """

    from dataclasses import asdict as _asdict
    from src.experiment.focal_concordance import (parent_concordance_delta_by_fold,
                                                  segment_weighted_auroc, ANOMALY_REFERENCE_SIZE)

    cfg = load_config()
    amend = load_fold1_amendment()
    unc = cfg["uncertainty"]
    experiments = [e for e in recovery_experiments() + retained_experiments() if e.fold in folds]
    scores = load_persisted_scores(experiments)
    if sorted(int(f) for f in scores.fold.unique()) != sorted(folds):
        raise SystemExit(f"persisted scores do not cover exactly folds {folds}")

    per_fold = {}
    frames = []
    for fold, sub in scores.groupby("fold"):
        focal = sub[sub.focal_parent_status == "focal_this_fold"]
        anomaly = sub[sub.anomaly_label != "normal"]
        if anomaly.segment_id.nunique() != ANOMALY_REFERENCE_SIZE:
            raise SystemExit("anomaly reference must be the fixed 232 segments")
        # every row - focal normal AND anomaly reference - is tagged with the fold of the
        # model that scored it; the corrected estimator matches on this tag explicitly
        frame = pd.concat([focal[focal.anomaly_label == "normal"], anomaly]).assign(focal_fold=int(fold))
        frames.append(frame[["segment_id", "parent_id", "arm", "anomaly_label", "focal_fold", "total_loss"]])
        arms = {}
        for arm, a in frame.groupby("arm"):
            n = a[a.anomaly_label == "normal"].total_loss
            an = a[a.anomaly_label != "normal"].total_loss
            arms[arm] = {"segment_weighted_focal_auroc": segment_weighted_auroc(n, an),
                         "mean_raw_focal_normal_score": float(n.mean()),
                         "mean_raw_anomaly_reference_score": float(an.mean())}
        per_fold[int(fold)] = arms
    pooled = pd.concat(frames, ignore_index=True)
    result, audits = parent_concordance_delta_by_fold(
        pooled, replicates=int(unc["replicates"]), seed=int(unc["seed"]))
    expected_parents = 12 * len(folds)
    if result.n_parents != expected_parents:
        raise SystemExit(f"expected {expected_parents} focal parents, found {result.n_parents}")

    pd_ = result.parent_deltas
    label = ("IMPL-1 CORRECTED ORIGINAL FIVE-FOLD ANALYSIS (fold-specific anomaly reference; fold 1 recovered)"
             if sorted(folds) == [1, 2, 3, 4, 5] else
             "IMPL-1 CORRECTED INCIDENT-CONTINGENT FOUR-FOLD SENSITIVITY ANALYSIS (fold-specific anomaly reference)")
    return {
        "schema_version": IMPL1_CORRECTED_SCHEMA,
        "analysis_label": label,
        "correction": {
            "id": "IMPL-1",
            "defect": "historical parent_concordance_delta ranked each focal segment against the anomaly rows of every fold's model of the same arm (232 x n_folds rows) when given a pooled frame",
            "corrected_to": "exactly the 232 anomaly segments scored by the SAME fold's model of the SAME arm, per fold",
            "estimator": "src/experiment/focal_concordance.parent_concordance_delta_by_fold",
            "supersedes_for_interpretation": ["reports/phase3_4_results/phase3_4_inc1_salvage_analysis.json (6441ed66...)",
                                              "reports/phase3_4_results/phase3_4_five_fold_recovered_analysis.json (427b04cb...)"],
            "historical_artifacts_retained_unchanged": True,
        },
        "computed_from": f"{len(experiments)} persisted per-segment score artifacts only; no inference",
        "config_identity": cfg["config_identity"],
        "amendment_id": amend["amendment_id"],
        "experiment_ids": [e.experiment_id for e in experiments],
        "recovery_ids": [getattr(e, "recovery_id", None) for e in experiments if getattr(e, "recovery_id", None)],
        "folds": sorted(folds),
        "reference_audit": [_asdict(a) for a in audits],
        "estimand": cfg["primary_estimand"]["name"],
        "primary": {
            "name": f"parent-equal-weighted focal concordance delta (folds {sorted(folds)}, {expected_parents} parents; fold-specific reference)",
            "point_estimate": result.point_estimate,
            "ci95_low": result.ci_low, "ci95_high": result.ci_high,
            "bootstrap": {"replicates": result.bootstrap_replicates, "seed": result.bootstrap_seed,
                          "fold_stratified": True, "parents_per_fold": 12, "parents_per_replicate": expected_parents,
                          "interval": "percentile 95 percent",
                          "conditional_on": f"the {2 * len(folds)} fitted models ({len(folds)} pairs)",
                          "does_not_include": unc["does_not_estimate"]},
            "n_focal_parents": result.n_parents,
            "n_focal_segments_descriptive": result.n_segments,
            "fold_level_deltas_descriptive": {int(k): float(v) for k, v in result.fold_deltas.items()},
            "parents_delta_positive": int((pd_ > 0).sum()),
            "parents_delta_zero": int((pd_ == 0).sum()),
            "parents_delta_negative": int((pd_ < 0).sum()),
        },
        "secondary": {
            "impl1_status": "UNAFFECTED - each quantity was always computed inside its own fold's sub-frame",
            "S1_segment_weighted_focal_auroc_delta_by_fold": {
                k: v["PRESENT"]["segment_weighted_focal_auroc"] - v["ABSENT"]["segment_weighted_focal_auroc"]
                for k, v in per_fold.items()},
            "S4_raw_focal_normal_score_shift_by_fold_SCALE_DEPENDENT": {
                k: v["PRESENT"]["mean_raw_focal_normal_score"] - v["ABSENT"]["mean_raw_focal_normal_score"]
                for k, v in per_fold.items()},
            "S5_anomaly_reference_raw_shift_by_fold_specificity": {
                k: v["PRESENT"]["mean_raw_anomaly_reference_score"] - v["ABSENT"]["mean_raw_anomaly_reference_score"]
                for k, v in per_fold.items()},
        },
        "mandatory_qualifiers": cfg["claim_boundary"]["mandatory_qualifiers"],
        "parent_deltas": {k: float(v) for k, v in pd_.items()},
    }
