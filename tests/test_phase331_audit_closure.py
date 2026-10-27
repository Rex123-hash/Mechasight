"""Phase 3.3.1 — invariants closing the Phase-3.3 independent-audit findings.

These tests add no new result. They pin the test-look discipline that the audit
found was enforced only by inspection, and they verify that the Phase-3.3
historical artifacts remain byte-identical under the additive correction layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.experiment.ledger_history import PHASE3_CLOSE_PREFIX_SHA256, PHASE3_CLOSE_ROWS, verify_ledger_history

LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
P33 = Path("reports/phase3_3_protocol_a")
P331 = Path("reports/phase3_3_1")
CONFIG = Path("configs/phase3/protocol_a_official_reproduction_v1.json")

PROTOCOL_A = "official_baseline_reproduction"
PROTOCOL_B = "mechasight_experimental_protocol"
CONFIG_IDENTITY = "p33-protocol-a-87fdd8c09d374556a8ff6ad0"
ORIGINAL_PREFIX_SHA256 = "db4bde80099a870ae90686fa463400f02f93434fc11dfce6d8c20f2eac0b0db0"
ORIGINAL_PROTOCOL_B_COUNT = 28

# The two Protocol-A test looks that belong to Phase 3.3, named explicitly so
# that this assertion constrains *those* looks rather than the ledger's future.
PHASE33_PROTOCOL_A_LOOKS = {
    "p33a-1cdb02c0d81c21288e58": "RoboticArm",
    "p33a-65b53c5046de90d1f1a5": "BrushlessMotor",
}


def _entries() -> list[dict]:
    return [json.loads(line) for line in
            LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------- historical, permanent
def test_phase33_protocol_a_looks_are_exactly_the_two_recorded_ones():
    """Phase 3.3 took exactly two Protocol-A test looks. That is now fixed history.

    This is the permanent guarantee. It constrains the Phase-3.3 looks by their
    experiment ids, so a future explicitly authorised protocol may still append
    its own looks without this test having to be relaxed — which is precisely
    the distinction the independent audit asked for.
    """
    protocol_a = [e for e in _entries() if e.get("protocol") == PROTOCOL_A]

    assert len(protocol_a) == 2, (
        f"Phase 3.3 recorded two Protocol-A test looks; found {len(protocol_a)}. "
        "An extra Protocol-A look is an undisclosed test access.")

    ids = [e["experiment_id"] for e in protocol_a]
    assert len(set(ids)) == len(ids), f"duplicate Protocol-A experiment ids: {ids}"
    assert set(ids) == set(PHASE33_PROTOCOL_A_LOOKS), (
        f"Protocol-A experiment ids changed: expected {sorted(PHASE33_PROTOCOL_A_LOOKS)}, "
        f"found {sorted(ids)}")

    by_machine: dict[str, int] = {}
    for entry in protocol_a:
        assert entry["machine"] == PHASE33_PROTOCOL_A_LOOKS[entry["experiment_id"]]
        assert entry["model_config_identity"] == CONFIG_IDENTITY
        assert entry["regime"] == "T"
        by_machine[entry["machine"]] = by_machine.get(entry["machine"], 0) + 1

    assert by_machine == {"RoboticArm": 1, "BrushlessMotor": 1}, (
        f"exactly one Protocol-A look per machine is allowed; found {by_machine}")


def test_original_28_protocol_b_entries_remain_a_byte_identical_prefix():
    """The Phase-3.2 evidence is immutable; later stages may only append.

    Stage note: Single-Modality Benchmarking appended 156 further Protocol-B
    looks, so "exactly 28 Protocol-B rows" became "the first 28 rows are the
    Phase-3.2 batch, and every later Protocol-B row belongs to a later
    authorised batch of its own".
    """
    history = verify_ledger_history(LEDGER)
    prefix = b"".join(history.lines[:ORIGINAL_PROTOCOL_B_COUNT])
    assert hashlib.sha256(prefix).hexdigest() == ORIGINAL_PREFIX_SHA256

    assert all(e["protocol"] == PROTOCOL_B for e in history.rows[:ORIGINAL_PROTOCOL_B_COUNT])
    first = history.segment("phase3_2_first_common_test")
    assert (first.start, first.end) == (0, ORIGINAL_PROTOCOL_B_COUNT)
    later_b = [s for s in history.segments if s.protocol == PROTOCOL_B and s is not first]
    assert all(s.start >= ORIGINAL_PROTOCOL_B_COUNT and s.batch_id for s in later_b)
    assert first.batch_id not in {s.batch_id for s in later_b}
    assert sum(s.end - s.start for s in [first, *later_b]) == \
        sum(1 for e in history.rows if e["protocol"] == PROTOCOL_B)


# ------------------------------------------------------- current snapshot
def test_ledger_composition_at_phase3_close_is_frozen():
    """Historical scope: at the Phase-3 close the ledger held 40 entries =
    28 Protocol-B + 2 Protocol-A + 10 Phase-3.4 focal-parent sibling-toggle
    looks (step D, authorised at c10fdfc).  Those 40 rows are now a frozen,
    byte-identical prefix; later authorised stages append after them."""
    history = verify_ledger_history(LEDGER)
    head = history.rows[:PHASE3_CLOSE_ROWS]
    counts = {p: sum(1 for e in head if e["protocol"] == p) for p in {e["protocol"] for e in head}}
    assert counts == {PROTOCOL_B: 28, PROTOCOL_A: 2,
                      "roboticarm_focal_parent_sibling_toggle_v1": 10}
    assert hashlib.sha256(b"".join(history.lines[:PHASE3_CLOSE_ROWS])).hexdigest() == \
        PHASE3_CLOSE_PREFIX_SHA256


def test_every_ledger_row_belongs_to_an_authorised_stage_and_closed_protocols_never_grow():
    """Current state, valid for any future authorised append.

    ``verify_ledger_history`` proves each row sits in a closed, hash-pinned
    stage or in an atomic batch bound to a committed batch authorization.
    Closed Phase-3 protocols must not grow at all: this also closes the
    Phase-3.3.1 audit gap that no test pinned the number of Protocol-A looks.
    """
    history = verify_ledger_history(LEDGER)
    assert sum(1 for e in history.rows if e["protocol"] == PROTOCOL_A) == 2
    assert sum(1 for e in history.rows
               if e["protocol"] == "roboticarm_focal_parent_sibling_toggle_v1") == 10
    assert [s.stage for s in history.segments if s.closed][:3] == [
        "phase3_2_first_common_test", "phase3_3_protocol_a_reproduction",
        "phase3_4_focal_parent_sibling_toggle"]


# ------------------------------------ Phase-3.3 historical artifacts frozen
def test_phase33_result_artifacts_are_unchanged_by_the_correction_layer():
    """Phase 3.3.1 is additive: it must not rewrite a single Phase-3.3 value."""
    manifest = json.loads((P33 / "artifact_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for name, expected in manifest["files"].items():
        path = Path(name)
        assert path.exists(), name
        assert _sha256(path) == expected, f"Phase-3.3 artifact was modified: {name}"
        checked += 1
    assert checked == 15


def test_correction_layer_writes_only_into_its_own_namespace():
    corrected = {p.name for p in P331.iterdir() if p.is_file()}
    assert corrected, "Phase 3.3.1 artifacts are missing"
    assert P331.resolve() != P33.resolve()
    # the corrected layer must not shadow a historical result filename
    original = {p.name for p in P33.iterdir() if p.is_file()}
    assert not (corrected & (original - {"artifact_manifest.json"})), (
        "Phase 3.3.1 must not restate a Phase-3.3 result filename")


def test_protocol_a_config_hash_supplement_matches_the_committed_config():
    supplement = json.loads(
        (P331 / "protocol_a_provenance_supplement.json").read_text(encoding="utf-8"))
    assert supplement["config"]["path"] == str(CONFIG).replace("\\", "/")
    assert supplement["config"]["sha256"] == _sha256(CONFIG)
    assert supplement["config"]["config_identity"] == CONFIG_IDENTITY


def test_score_artifact_recovery_search_records_an_honest_outcome():
    """The audit's F1 gap may only be closed by recovery, never by regeneration."""
    search = json.loads(
        (P331 / "score_artifact_recovery_search.json").read_text(encoding="utf-8"))
    assert search["authentic_original_score_artifact_found"] is False
    assert search["finding_status"] == "OPEN_ACCEPTED_LIMITATION"
    assert search["scores_regenerated"] is False
    assert search["model_inference_performed"] is False
