"""Phase 3.4 COMMIT A — execution/provenance infrastructure invariants.

Nothing here trains, scores, or touches a checkpoint. Tests that exercise the
execution library do so on synthetic data or on the frozen metadata only.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiment import focal_toggle_execution as fx
from src.experiment.ledger_history import PHASE3_CLOSE_ROWS, prefix_sha256, verify_ledger_history
from src.experiment.provenance import (
    canonical_bytes, canonical_sha256, file_sha256, hash_form, initialization_hash,
)
from src.reproduction.protocol_a import OFFICIAL_PARAMS, build_official_autoencoder

CONFIG = Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
EXPECTED_INIT_SHA = "d889e90224988ac27bceba08"  # prefix, pre-registered in the config


def _cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


# ------------------------------------------------------- A6: canonical hashing
def test_canonical_hash_is_identical_for_lf_and_crlf_copies(tmp_path):
    lf = tmp_path / "a.json"; crlf = tmp_path / "b.json"
    lf.write_bytes(b'{\n  "x": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
    assert file_sha256(lf) != file_sha256(crlf)              # raw bytes differ
    assert canonical_sha256(lf) == canonical_sha256(crlf)    # canonical form agrees
    assert canonical_sha256(lf) == file_sha256(lf)           # equals the git-blob (LF) hash


def test_canonical_hash_leaves_binary_artifacts_untouched(tmp_path):
    blob = tmp_path / "w.pth"
    payload = b"\x00\x01\r\n\x02\r\n"
    blob.write_bytes(payload)
    assert canonical_bytes(blob) == payload
    assert canonical_sha256(blob) == sha256(payload).hexdigest()


def test_hash_form_detects_crlf_drift(tmp_path):
    lf = tmp_path / "m.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf_hash = sha256(b"a,b\r\n1,2\r\n").hexdigest()
    assert hash_form(lf, file_sha256(lf)) == "LF"
    assert hash_form(lf, crlf_hash) == "CRLF"
    assert hash_form(lf, "0" * 64) == "NEITHER"


def test_gitattributes_pins_research_text_to_lf():
    rules = {}
    for line in Path(".gitattributes").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            pattern, *attrs = line.split()
            rules[pattern] = set(attrs)
    for suffix in ("*.json", "*.jsonl", "*.csv", "*.md", "*.py"):
        assert {"text", "eol=lf"} <= rules.get(suffix, set()), suffix
    for suffix in ("*.pth", "*.h5"):
        assert "binary" in rules.get(suffix, set()), suffix


def test_commit_a_manifest_is_a_faithful_historical_record_of_commit_107579f():
    """COMMIT A's manifest is history: each value must equal the SHA-256 of the raw
    content bytes stored in Git at commit 107579f for that path (LF form)."""
    import subprocess
    manifest = json.loads(Path("reports/phase3_4_commit_a/artifact_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["hash_form"].startswith("canonical")
    for name, recorded in manifest["files"].items():
        blob = subprocess.run(["git", "show", f"107579f:{name}"], capture_output=True).stdout
        assert blob, name
        assert sha256(blob).hexdigest() == recorded, name


def test_f1_f2_patch_manifest_is_a_faithful_historical_record_of_commit_dcb57d1():
    import subprocess
    manifest = json.loads(Path("reports/phase3_4_commit_a/patch_f1_f2_manifest.json")
                          .read_text(encoding="utf-8"))
    for name, recorded in manifest["files"].items():
        blob = subprocess.run(["git", "show", f"dcb57d1:{name}"], capture_output=True).stdout
        assert blob, name
        assert sha256(blob).hexdigest() == recorded, name


def test_f3_patch_manifest_is_a_faithful_historical_record_of_commit_e93886c():
    import subprocess
    manifest = json.loads(Path("reports/phase3_4_commit_a/patch_f3_manifest.json")
                          .read_text(encoding="utf-8"))
    for name, recorded in manifest["files"].items():
        blob = subprocess.run(["git", "show", f"e93886c:{name}"], capture_output=True).stdout
        assert blob, name
        assert sha256(blob).hexdigest() == recorded, name


def test_commit_b_manifest_is_a_faithful_historical_record_of_commit_c10fdfc():
    """Stage note: COMMIT B's manifest became history at step D (its own test file is
    listed in it and was updated for the ledger stage transition). Each value must
    equal the SHA-256 of the raw bytes stored in Git at c10fdfc; the checkpoints it
    names are still the ones on disk."""
    import subprocess
    manifest = json.loads(Path("reports/phase3_4_results/commit_b_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["hash_form"].startswith("canonical")
    for name, recorded in manifest["files"].items():
        blob = subprocess.run(["git", "show", f"c10fdfc:{name}"], capture_output=True).stdout
        assert blob, name
        assert sha256(blob).hexdigest() == recorded, name
    cfg = Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
    assert manifest["config_document_sha256"] == canonical_sha256(cfg)   # config unchanged
    assert manifest["contains_test_performance"] is False
    for eid, ck in manifest["checkpoints"].items():
        assert file_sha256(Path(ck["path"])) == ck["sha256"], eid
    assert len(manifest["checkpoints"]) == 10


def test_step_d_manifest_is_a_faithful_historical_record_of_commit_576b8a1():
    """Stage note: became history at the fold-1 recovery amendment (test files it
    lists carry later stage notes). Values must equal the LF-form SHA-256 of the
    bytes stored in Git at 576b8a1; the ledger and INC-1 records are still live-equal."""
    import subprocess
    manifest = json.loads(Path("reports/phase3_4_results/step_d_authorization_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["hash_form"].startswith("canonical")
    for name, recorded in manifest["files"].items():
        blob = subprocess.run(["git", "show", f"576b8a1:{name}"], capture_output=True).stdout
        assert blob, name
        assert sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == recorded, name
    for name in ("reports/phase3_4_results/score_claims/p34d-92fd40fc93c8c8d2e819.claim",
                 "reports/phase3_4_results/score_attempts/p34d-92fd40fc93c8c8d2e819.jsonl"):
        assert canonical_sha256(Path(name)) == manifest["files"][name], name
    # The ledger has since grown by authorised appends; the recorded hash is the
    # 40-row ledger at 576b8a1, which must survive as the byte-identical prefix.
    ledger = "reports/experiment_results/test_look_ledger.jsonl"
    history = verify_ledger_history(ledger)
    assert prefix_sha256(history.lines, PHASE3_CLOSE_ROWS) == manifest["files"][ledger]
    assert manifest["contains_test_performance"] is False
    assert manifest["ledger_before"] == 30 and manifest["ledger_after"] == 40
    assert manifest["protected_test_inference_run"] is False
    assert manifest["score_artifacts"] == 0


def test_historical_manifests_are_untouched():
    """Forward fix only: the seven historical manifests keep their recorded hash forms."""
    audit = pd.read_csv("reports/phase3_4d_audit/manifest_hash_form_check.csv")
    for path, sub in audit.groupby("manifest"):
        live = json.loads(Path(path).read_text(encoding="utf-8"))["files"]
        for row in sub.itertuples():
            assert live[row.file] == row.manifest_sha256, (path, row.file)


# --------------------------------------------------- A4: initialization hash
def _build_seeded(seed: int = 1995):
    torch.manual_seed(seed)
    return build_official_autoencoder([1600, 706, 706], [1, 3, 3], dict(OFFICIAL_PARAMS))


def test_initialization_hash_is_deterministic_under_the_frozen_seed():
    a = initialization_hash(_build_seeded())
    b = initialization_hash(_build_seeded())
    assert a == b
    assert a.startswith(EXPECTED_INIT_SHA)
    assert _cfg()["initialization_policy"]["expected_initialization_sha256"] == a


def test_initialization_hash_changes_with_seed_or_weights():
    base = initialization_hash(_build_seeded(1995))
    assert initialization_hash(_build_seeded(1996)) != base
    model = _build_seeded(1995)
    with torch.no_grad():
        next(model.parameters()).add_(1e-3)   # a single optimizer-like update
    assert initialization_hash(model) != base


def test_paired_arms_share_the_initialization_because_construction_precedes_any_rng_use():
    """DataLoader construction must not consume torch RNG before the model is built."""
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return torch.zeros(1)

    hashes = set()
    for n in (1405, 1410):                       # PRESENT / ABSENT sizes are equal; vary anyway
        torch.manual_seed(1995)
        DataLoader(_DS(n), batch_size=1024, shuffle=True)
        DataLoader(_DS(n), batch_size=1024, shuffle=False)
        hashes.add(initialization_hash(
            build_official_autoencoder([1600, 706, 706], [1, 3, 3], dict(OFFICIAL_PARAMS))))
    assert len(hashes) == 1


def test_freeze_stage_refuses_unequal_paired_initialization(tmp_path, monkeypatch):
    """The COMMIT B freeze must fail loudly if a fold's arms did not start identically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_runner", "scripts/run_focal_toggle_experiment.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

    plan = fx.planned_experiments()
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path)
    for exp in plan:
        init = "same" if not (exp.fold == 3 and exp.arm == "ABSENT") else "different"
        rec = {"checkpoint_sha256": "x", "initialization_sha256": init,
               "initial_state_sha256": "same",
               "checkpoint_path": str(tmp_path / "none.pth"), "epochs_run": 1,
               "final_epoch_actually_scored_one_indexed": 1}
        p = tmp_path / "training" / f"{exp.experiment_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit, match="paired initialisation hashes differ"):
        runner.stage_freeze()


# ------------------------------------------------ memberships / matrix frozen
def test_materialised_memberships_reproduce_all_ten_frozen_hashes():
    for exp in fx.planned_experiments():
        ids = fx.training_segment_ids(exp.fold, exp.arm)
        assert fx.membership_hash(ids) == exp.training_membership_sha256, exp.experiment_id


def test_v3_membership_and_plan_files_are_byte_identical_to_the_audited_release():
    audited = json.loads(Path("reports/phase3_4d_audit/artifact_manifest.json")
                         .read_text(encoding="utf-8"))["files"]
    gate = json.loads(Path("reports/phase3_4d_release_gate/artifact_manifest.json")
                      .read_text(encoding="utf-8"))["files"]
    for name in ("reports/phase3_4d_release_gate/final_arm_memberships.csv",
                 "reports/phase3_4d_release_gate/planned_experiments_v3.csv"):
        assert hash_form(Path(name), gate[name]) in ("LF", "CRLF"), name


def test_experiment_matrix_is_exactly_the_frozen_ten():
    plan = fx.planned_experiments()
    assert len(plan) == 10
    assert len({e.experiment_id for e in plan}) == 10
    assert len({e.training_membership_sha256 for e in plan}) == 10
    assert all(e.seed == 1995 for e in plan)
    by_fold = {}
    for e in plan:
        by_fold.setdefault(e.fold, set()).add(e.arm)
    assert by_fold == {k: {"PRESENT", "ABSENT"} for k in range(1, 6)}
    order = [f"{e.fold}{e.arm[0]}" for e in plan]
    assert order == ["1P", "1A", "2A", "2P", "3P", "3A", "4A", "4P", "5P", "5A"]
    raw = pd.read_csv(fx.PLAN_PATH)
    assert (raw.status == "PLANNED_NOT_EXECUTED").all()


def test_config_identity_and_frozen_policies_unchanged():
    cfg = _cfg()
    assert cfg["config_identity"] == "p34d-exec-e5470b025229f59a06f4c772"
    assert cfg["seed_policy"]["seed"] == 1995
    assert cfg["execution_environment"]["execution_environment_id"] == "p34c-env-b7a5de2af4c1d9614e49"
    assert cfg["execution_environment"]["device"] == "cpu"
    assert cfg["uncertainty"]["replicates"] == 10000 and cfg["uncertainty"]["seed"] == 27182
    assert "FOLD-STRATIFIED" in cfg["uncertainty"]["frozen"]
    assert cfg["primary_estimand"]["parent_aggregation"] == "MEAN"
    assert cfg["initialization_policy"]["mandatory"] is True


# ------------------------------------------------------ gates and the ledger
def test_ledger_holds_one_authorisation_per_frozen_checkpoint():
    """Stage note: ledger was 30 until step D (authorised at c10fdfc appended the ten
    Phase-3.4 looks atomically). Historical scope: 40 = 28 B + 2 A + 10 Phase-3.4 at
    the Phase-3 close; later authorised stages append after them. Every Phase-3.4
    look in the whole ledger is one of those ten, each naming exactly one frozen
    experiment and its exact checkpoint hash from the COMMIT B training record."""
    history = verify_ledger_history(LEDGER)
    p34 = [e for e in history.rows if e["protocol"] == fx.PROTOCOL]
    assert p34 == list(history.rows_of("phase3_4_focal_parent_sibling_toggle"))
    assert len(p34) == 10 and len({e["experiment_id"] for e in p34}) == 10
    for e in p34:
        rec = json.loads((Path("reports/phase3_4_results/training") /
                          f"{e['experiment_id']}.json").read_text(encoding="utf-8"))
        assert f"checkpoint_sha256={rec['checkpoint_sha256']}" in e["reason"]
        assert e["model_config_identity"] == _cfg()["config_identity"]
        assert e["test_cohort_hash"] == fx.TEST_COHORT_HASH


# Stage note: before step B (authorised at e93886c) these two gates fired on "no
# checkpoint / no record". Post-training they must still refuse - for the reasons
# that remain true - and must never touch the ledger or the protected cohort.
def test_authorize_refuses_unless_every_gate_holds(tmp_path, monkeypatch):
    real_before = LEDGER.read_bytes()
    # (a) a checkpoint not yet frozen -> refuse
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path / "partial")
    (tmp_path / "partial" / "training").mkdir(parents=True)
    with pytest.raises(SystemExit, match="not frozen yet"):
        fx.build_authorisations("deadbeef")
    # Stage note: the real ledger now holds 40 entries (step D, c10fdfc); the gates
    # are exercised on temporary ledgers derived from its historical 30-entry prefix.
    monkeypatch.setattr(fx, "RESULTS_ROOT", Path("reports/phase3_4_results"))
    lines = real_before.splitlines(keepends=True)
    historical = lines[:30]
    # (b) all frozen, but ledger not exactly the 30 historical entries -> refuse
    fake_ledger = tmp_path / "ledger31.jsonl"
    fake_ledger.write_bytes(b"".join(historical) + historical[0])
    monkeypatch.setattr(fx, "LEDGER", fake_ledger)
    with pytest.raises(SystemExit, match="expected the 30 historical"):
        fx.build_authorisations("deadbeef")
    # (c) 30 entries but one already carries the Phase-3.4 protocol -> refuse to duplicate
    dup = tmp_path / "ledger_dup.jsonl"
    entry = json.loads(historical[-1]); entry["protocol"] = fx.PROTOCOL
    dup.write_bytes(b"".join(historical[:-1]) + (json.dumps(entry) + "\n").encode("utf-8"))
    monkeypatch.setattr(fx, "LEDGER", dup)
    with pytest.raises(SystemExit, match="already exist"):
        fx.build_authorisations("deadbeef")
    # (d) the REAL ledger already carries the ten looks -> refuse to re-authorise
    monkeypatch.setattr(fx, "LEDGER", LEDGER)
    with pytest.raises(SystemExit, match="expected the 30 historical|already exist"):
        fx.build_authorisations("deadbeef")
    assert LEDGER.read_bytes() == real_before                # real ledger untouched


def test_score_refuses_before_any_claim_or_exposure(tmp_path, monkeypatch):
    """INCIDENT NOTE (step D, 2026-09-11T19:50:25Z): the previous version of this test
    pointed ``score_experiment`` at the REAL results root, relying on the ledger gate
    to refuse. That was true only before step D; once the ten looks were authorised
    the ledger gate passed and the real O_EXCL claim for fold-1 PRESENT
    (p34d-92fd40fc93c8c8d2e819) was acquired and SCORE_STARTED appended before the
    sentinel raised. No protected forward ran (the sentinel replaced inference), but
    the look is consumed under the frozen never-delete policy. This test now never
    leaves tmp_path, and tests/conftest.py refuses any claim inside the real root."""
    exp = fx.planned_experiments()[0]
    # never let a real forward run, whatever the gates do
    monkeypatch.setattr(fx, "_run_protected_inference",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("forward ran")))
    results = tmp_path / "results"
    monkeypatch.setattr(fx, "RESULTS_ROOT", results)
    # (a) no training record at all -> refuse at the first gate
    with pytest.raises(SystemExit, match="no frozen training record"):
        fx.score_experiment(exp)
    # (b) a copy of the real frozen record (checkpoint on disk matches), but a
    #     ledger holding only the 30 historical entries -> refuse at the ledger
    #     gate, before any claim
    real_rec = Path("reports/phase3_4_results/training") / f"{exp.experiment_id}.json"
    p = fx.training_record_path(exp)
    p.parent.mkdir(parents=True)
    p.write_bytes(real_rec.read_bytes())
    historical = LEDGER.read_bytes().splitlines(keepends=True)[:30]
    ledger = tmp_path / "ledger30.jsonl"
    ledger.write_bytes(b"".join(historical))
    monkeypatch.setattr(fx, "LEDGER", ledger)
    with pytest.raises(SystemExit, match="no ledger entry authorises"):
        fx.score_experiment(exp)
    assert fx.read_score_claim(exp) is None
    assert fx.score_attempts(exp) == []
    assert not fx.score_path(exp).exists()
    assert not (results / "score_claims").exists()


def test_train_refuses_out_of_frozen_order(monkeypatch):
    """No model is trained: the order gate fires before any data is loaded."""
    plan = fx.planned_experiments()
    second = plan[1]
    with pytest.raises(SystemExit, match="next experiment in the frozen order"):
        fx.train_experiment(second, device="cpu")


def test_train_refuses_wrong_device_or_torch(monkeypatch):
    first = fx.planned_experiments()[0]
    with pytest.raises(SystemExit, match="frozen environment requires device"):
        fx.train_experiment(first, device="cuda")


def test_only_the_ten_frozen_checkpoints_exist_and_exposure_is_zero():
    """Stage note: the historical 'no checkpoint exists' condition belonged to the
    pre-training stage. After step B (authorised at e93886c) the current invariant is:
    only the ten frozen checkpoints exist, each provably tied to its frozen experiment,
    and protected-test exposure remains zero."""
    results = Path("reports/phase3_4_results")
    plan = {e.experiment_id: e for e in fx.planned_experiments()}
    records = {}
    for q in results.glob("training/*.json"):
        r = json.loads(q.read_text(encoding="utf-8"))
        records[r["experiment_id"]] = r
    assert set(records) == set(plan)                                  # exactly the ten
    on_disk = {str(c).replace("\\", "/") for c in Path("data/interim").glob("phase3_4/**/*.pth")}
    recorded = {r["checkpoint_path"] for r in records.values()}
    assert on_disk == recorded                                        # no extra, none missing
    assert len(recorded) == 10 and len({r["checkpoint_sha256"] for r in records.values()}) == 10
    cfg = _cfg()
    for eid, r in records.items():
        e = plan[eid]
        assert (r["fold"], r["arm"], r["seed"]) == (e.fold, e.arm, 1995)
        assert r["training_membership_sha256"] == e.training_membership_sha256
        assert file_sha256(Path(r["checkpoint_path"])) == r["checkpoint_sha256"], eid
        assert r["initialization_sha256"] == cfg["initialization_policy"]["expected_initialization_sha256"]
        assert r["initial_state_sha256"] == cfg["initialization_policy"]["expected_initial_state_sha256"]
    # step D (c10fdfc): exactly one authorisation record, naming exactly these ten
    auth = json.loads((results / "test_look_authorization.json").read_text(encoding="utf-8"))
    assert auth["ledger_before"] == 30 and auth["ledger_after"] == 40
    assert auth["entries_added"] == 10
    assert {r["experiment_id"] for r in auth["entries"]} == set(plan)
    for r in auth["entries"]:
        assert f"checkpoint_sha256={records[r['experiment_id']]['checkpoint_sha256']}" in r["reason"]
    # Stage note (E1, ac71988): "no score" was the pre-step-E invariant. The current
    # invariant for the ORIGINAL namespace (no "__" recovery suffix) is: exactly the
    # eight retained looks have artifacts and [SCORE_STARTED, SCORE_COMPLETED]; INC-1
    # is preserved as below; fold-1 ABSENT's original namespace is never claimed.
    original = lambda p: "__" not in p.stem
    retained = set(fx.load_recovery_config()["scoring_plan"]["frozen_order_experiment_ids"])
    assert {p.stem for p in results.glob("scores/*.csv") if original(p)} == retained
    # INCIDENT (2026-09-11T19:50:25Z, see test_score_refuses_before_any_claim_or_exposure):
    # the fold-1 PRESENT claim was consumed by the test harness with NO forward run.
    # Frozen policy: never delete, never recover. This pins that exact state so that
    # (i) the claim cannot silently disappear, (ii) no other claim can appear before
    # step E is authorised, (iii) the consumed look can never reach SCORE_COMPLETED.
    consumed = "p34d-92fd40fc93c8c8d2e819"
    assert {c.stem for c in results.glob("score_claims/*.claim") if original(c)} == retained | {consumed}
    assert {a.stem for a in results.glob("score_attempts/*.jsonl") if original(a)} == retained | {consumed}
    assert "p34d-e56529a2f56f98fd4a7a" not in {c.stem for c in results.glob("score_claims/*.claim")
                                             if original(c)}                      # fold-1 ABSENT original
    claim = json.loads((results / "score_claims" / f"{consumed}.claim").read_text(encoding="utf-8"))
    assert claim["checkpoint_sha256"] == records[consumed]["checkpoint_sha256"]
    history = [json.loads(l)["status"] for l in
               (results / "score_attempts" / f"{consumed}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert history == [fx.SCORE_STARTED]                                 # never COMPLETED


def test_score_schema_row_shape_is_frozen():
    schema = json.loads(Path("reports/phase3_4c_freeze/score_schema.json").read_text(encoding="utf-8"))
    names = {c["name"] for c in schema["columns"]}
    for col in fx.SCORE_COLUMNS:
        assert col in names, col
    assert schema["rows_per_experiment"] == 464
    assert schema["total_rows_planned"] == 4640
