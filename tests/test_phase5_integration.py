"""Phase 5.5F integration: label-free window index, frozen-model adapter, JSONL ledger.

Permitted development source-validation inputs and SYNTHETIC labels only, in
temporary sandboxes. No protected recording, cache, label or sealed score is
opened; the real ledger is only read (copied) and never written.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from src.experiment import ledger as ledger_module
from src.experiment import phase5_window_index as wi
from src.experiment.ledger_history import LedgerHistoryError, verify_ledger_history
from src.experiment.phase4 import ModelStore
from src.experiment.phase5_dev_fixture import POISON_COLUMN, create_dev_fixture
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture
from src.experiment.phase5_fixture import ANOMALY, NORMAL
from src.experiment.phase5_inference import FrozenL1Adapter
from src.experiment.phase5_inputs import Access, Refused, Sandbox, SCORE_COLUMNS
from src.fusion.assembly import PREPROCESSING_CONFIGS
from src.preprocessing.config import load_config
from src.preprocessing.spectral_cache import load_spectral_cache, spectral_cache_path

os.environ.setdefault("MECHASIGHT_TORCH_THREADS", "4")
SPLIT = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")


@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


@pytest.fixture(scope="module")
def catalog():
    return wi.development_catalog()


@pytest.fixture(scope="module")
def template(frozen, catalog, tmp_path_factory):
    """One development fixture, copied for every test that mutates its sandbox."""
    root = Path(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-5f-template-"))
    box = Sandbox(root)
    create_dev_fixture(box, frozen, catalog=catalog)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def dev(template, frozen, tmp_path):
    root = tmp_path / "SYNTHETIC-dev"
    shutil.copytree(template, root)
    box = Sandbox(root)
    adapter = FrozenL1Adapter(frozen)
    return box, Executor(box, frozen, predictor=adapter), adapter


def permit(box, package, action="score", **kw):
    return authorize_fixture(box, package, action, **kw)


def _ledger(box):
    return box.path("ledger/test_look_ledger.jsonl"), box.path("ledger/authorizations")


def _verify(box):
    path, auth = _ledger(box)
    return verify_ledger_history(path, authorization_root=auth)


def _sources(machine, regime, catalog):
    cohort = wi.label_free_cohort(SPLIT, machine=machine, regime=regime, split="validation")
    paths = cohort.merge(catalog, on=["machine", "segment_id"], validate="one_to_one")
    return [wi.SegmentSource(r.machine, r.segment_id, r.parent_recording, r.domain,
                             {"mic": r.mic_path, "acc": r.acc_path, "gyro": r.gyro_path}) for r in paths.itertuples()]


# ------------------------------------------------------------------ A. label-free window index
@pytest.mark.parametrize("machine", ["RoboticArm", "BrushlessMotor"])
def test_A1_A3_A4_index_is_deterministic_and_equals_the_frozen_cache_row_order(machine, catalog):
    for regime in ("S", "T"):
        config = load_config(PREPROCESSING_CONFIGS[regime])
        first = wi.build_window_index(_sources(machine, regime, catalog), config, regime=regime)
        second = wi.build_window_index(list(reversed(_sources(machine, regime, catalog))), config, regime=regime)
        pd.testing.assert_frame_equal(first, second)
        _, meta = load_spectral_cache(spectral_cache_path("data/interim/phase4/spectral", machine=machine, regime=regime,
                                                          role="source_validation"), modality="mic", expected_split="validation")
        for column in ("segment_id", "window_index", "parent_recording", "domain"):
            assert first[column].tolist() == meta[column].tolist()        # verification only, permitted cache


def test_A2_A3_fixture_rows_are_the_exact_cache_rows_of_every_modality(template, frozen):
    box = Sandbox(template)
    fixture = box.json("fixture.json")
    for cell, entry in fixture["cells"].items():
        machine, regime = cell.split("|")
        index = pd.read_parquet(box.path(entry["index"]))
        cache = spectral_cache_path("data/interim/phase4/spectral", machine=machine, regime=regime, role="source_validation")
        with h5py.File(box.path(entry["spectral"]), "r") as copy, h5py.File(cache, "r") as source:
            for m in ("mic", "acc", "gyro"):
                assert copy[m].dtype == np.float32 and copy[m].shape[1] == entry["index_manifest"]["representation"]["input_dims"][m]
                np.testing.assert_array_equal(copy[m][()], source[m][index["row"].to_numpy()])
        assert set(index["segment_id"]) == {r["segment_id"] for r in fixture["cohorts"][machine]}


def _cell(template):
    box = Sandbox(template)
    fixture = box.json("fixture.json")
    entry = fixture["cells"]["RoboticArm|S"]
    return pd.read_parquet(box.path(entry["index"])), entry["index_manifest"], fixture["cohorts"]["RoboticArm"]


@pytest.mark.parametrize("mutation, message", [
    ("cohort", "another cohort"), ("hash", "differs from its manifest"), ("missing", "differs from its manifest"),
    ("duplicate", "differs from its manifest"), ("cell", "another cell"), ("label", "outcome-like column"),
])
def test_A5_A7_validate_index_rejects_tampering(template, mutation, message):
    index, manifest, cohort = _cell(template)
    manifest = deepcopy(manifest)
    expected, machine, regime = cohort, "RoboticArm", "S"
    if mutation == "cohort":
        expected = cohort[:-1] + [{**cohort[-1], "segment_id": "SYNTHETIC-other"}]
    elif mutation == "hash":
        index = index.assign(window_end_s=index["window_end_s"] + 1e-6)
    elif mutation == "missing":
        index = index.drop(index.index[5]).reset_index(drop=True)
    elif mutation == "duplicate":
        index = pd.concat([index, index.iloc[[3]]], ignore_index=True)
    elif mutation == "cell":
        regime = "T"
    else:
        index = index.assign(anomaly_label=NORMAL)
    with pytest.raises(wi.IndexRefused, match=message):
        wi.validate_index(index, manifest, expected_segments=expected, machine=machine, regime=regime)


def test_A7_misordered_windows_are_rejected_even_with_a_matching_manifest(template):
    index, manifest, cohort = _cell(template)
    swapped = index.copy()
    swapped.loc[[2, 3], "window_index"] = swapped.loc[[3, 2], "window_index"].to_numpy()
    from src.experiment.provenance import content_sha256
    forged = {**manifest, "content_sha256": content_sha256(swapped.to_dict(orient="list"))}
    with pytest.raises(wi.IndexRefused, match="missing or misordered"):
        wi.validate_index(swapped, forged, expected_segments=cohort, machine="RoboticArm", regime="S")


def test_A8_A9_building_the_scoring_inputs_never_reads_label_bearing_provenance(frozen, catalog, tmp_path, monkeypatch):
    opened, csv_columns = [], []
    real_getitem = h5py.Group.__getitem__
    real_read_csv = pd.read_csv

    def trap_getitem(self, key):
        opened.append(key)
        assert key != "provenance_json", "label-bearing cache provenance was requested"
        return real_getitem(self, key)

    def trap_read_csv(path, *args, **kwargs):
        if Path(str(path)).name == SPLIT.name:
            csv_columns.append(tuple(kwargs.get("usecols") or ()))
            assert kwargs.get("usecols") and not any("label" in c or "anomaly" in c for c in kwargs["usecols"])
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "__getitem__", trap_getitem)
    monkeypatch.setattr(pd, "read_csv", trap_read_csv)
    box = Sandbox(tmp_path / "SYNTHETIC-trap")
    create_dev_fixture(box, frozen, catalog=catalog)
    assert csv_columns and set(opened) <= {"mic", "acc", "gyro"}


def test_A10_unsafe_or_protected_index_construction_is_refused(tmp_path):
    t = np.arange(0, 5.2, 0.001)
    t[100], t[101] = t[101], t[100]
    path = tmp_path / "acc.parquet"
    pd.DataFrame({"Time": t, "A_x [g]": 0.0}).to_parquet(path, index=False)
    with pytest.raises(wi.IndexRefused, match="registered development"):
        wi._timestamps(path, "acc")
    with pytest.raises(wi.IndexRefused, match="disabled"):
        wi.build_protected_window_index()
    with pytest.raises(Refused, match="only development cohorts"):
        wi.label_free_cohort(SPLIT, machine="RoboticArm", regime="S", split="test")


# ------------------------------------------------------------------ B. frozen-model adapter
def test_B_real_frozen_models_reproduce_and_l2_aligns_end_to_end(dev, frozen):
    box, executor, adapter = dev
    state = executor.score_all(permit(box, frozen))
    assert len(state["tables"]) == 232 and len(adapter.calls) == 24
    assert {(c["family"]) for c in adapter.calls} == {"mahalanobis_shrinkage", "dense_autoencoder"}
    for call in adapter.calls:
        detector = frozen.registry["detectors"][call["experiment_id"]]
        got = pd.DataFrame(box.json(state["tables"][call["experiment_id"]]["file"])["rows"]).set_index("segment_id")
        payload = box.json(state["tables"][call["experiment_id"]]["file"])
        assert payload["computation"]["kind"] == "frozen_L1_inference"
        assert payload["predictor"] == "phase5-frozen-l1-inference-adapter-v1"
        assert payload["computation"]["model"] == detector["model_manifest"]
        assert (payload["computation"]["family"], payload["computation"]["seed"]) == (detector["family"], detector["seed"])
        ref = pd.read_csv(Path(detector["bundle_path"]) / "validation_scores.csv", dtype={"segment_id": str},
                          float_precision="round_trip").set_index("segment_id")
        assert float((got["score"] - ref.loc[got.index, "score"]).abs().max()) <= 1e-9
        if "window_count" in ref:
            assert (got["window_count"] == ref.loc[got.index, "window_count"]).all()
        assert (call["machine"], call["regime"], call["seed"]) == (detector["machine"], detector["regime"], detector["seed"])
    fused = pd.read_csv("reports/phase5_development/l2/l2_fused_validation_scores.csv", dtype={"segment_id": str},
                        float_precision="round_trip").set_index(["calibration_id", "segment_id"])["fused_score"]
    l2 = [e for e, op in executor.operations.items() if op["operation"] == "derived_L2_scoring"]
    assert len(l2) == 208
    for eid in l2:
        payload = box.json(state["tables"][eid]["file"])
        rows = payload["rows"]
        assert payload["computation"]["kind"] == "derived_L2_scoring"
        assert payload["computation"]["component_origin"] == "PERMITTED_PHASE4_SOURCE_VALIDATION"
        assert payload["computation"]["calibration_id"] == eid
        assert payload["predictor"] == "phase5-l2-frozen-robust-z-mean-v1"
        assert all(abs(r["score"] - fused[(eid, r["segment_id"])]) <= 1e-12 for r in rows)
    verified = _verify(box)
    assert len(verified.rows) == 428 and verified.segments[-1].stage.startswith("authorized_batch:")


def test_B_adapter_refuses_wrong_identity_record_hash_and_representation(dev, frozen):
    box, executor, adapter = dev
    l1 = sorted(adapter.planned)
    maha = next(e for e in l1 if frozen.registry["detectors"][e]["family"] == "mahalanobis_shrinkage")
    ae = next(e for e in l1 if frozen.registry["detectors"][e]["family"] == "dense_autoencoder")
    l0 = frozen.registry["units"][0]["L0"]["mic"]
    with pytest.raises(Refused, match="not a planned new L1"):
        adapter(frozen.registry["detectors"][l0], pd.DataFrame())
    altered = deepcopy(frozen.registry["detectors"][maha])
    altered["threshold"]["value"] += 1.0
    with pytest.raises(Refused, match="differs from the frozen registry"):
        adapter(altered, pd.DataFrame())
    forged_package = deepcopy(frozen)
    forged_package.registry["detectors"][maha]["model_manifest"]["model_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenL1Adapter(forged_package)(forged_package.registry["detectors"][maha], pd.DataFrame())
    detector = frozen.registry["detectors"][maha]
    columns = ("segment_id", "parent_recording", "domain", "window_count", *detector["data_provenance"]["feature_columns"])
    frame = pd.DataFrame([[f"s{i}", "p", "source", 74, *np.zeros(98)] for i in range(3)], columns=list(columns))
    with pytest.raises(Refused, match="98-feature layout"):
        adapter(detector, frame.iloc[:, :-1])
    with pytest.raises(Refused, match="98-feature layout"):
        adapter(detector, frame.assign(**{POISON_COLUMN: ANOMALY}))
    ae_detector = frozen.registry["detectors"][ae]
    blocks = ae_detector["data_provenance"]["spectral_blocks"]
    index = pd.DataFrame({"segment_id": ["s"] * 2, "parent_recording": "p", "domain": "source", "window_index": [0, 1]})
    good = {m: np.zeros((2, blocks[m][1] - blocks[m][0]), dtype=np.float32) for m in ("mic", "acc", "gyro")}
    with pytest.raises(Refused, match="block width"):
        adapter(ae_detector, {**good, "acc": good["acc"][:, :-1]}, index)
    with pytest.raises(Refused, match="block width"):
        adapter(ae_detector, {**good, "mic": good["mic"].astype(np.float64)}, index)


def test_B_real_computation_provenance_cannot_be_changed_to_stub(dev, frozen):
    from src.experiment.phase5_execution import synthetic_predictor
    from src.experiment.provenance import file_sha256
    box, executor, adapter = dev
    state = executor.score_all(permit(box, frozen))
    eid = next(c["experiment_id"] for c in adapter.calls if c["family"] == "dense_autoencoder")
    path, statepath = box.path(state["tables"][eid]["file"]), box.path("state.json")
    original, originalstate = path.read_bytes(), statepath.read_bytes()
    payload = json.loads(original)
    payload["predictor"] = "SYNTHETIC_STUB_V1"
    payload["computation"]["kind"] = "synthetic_L1_stub"
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
        state["tables"][eid]["sha256"] = file_sha256(path)
        statepath.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(Refused, match="computation/predictor"):
            executor.verify_complete()
    finally:
        path.write_bytes(original)
        statepath.write_bytes(originalstate)
    executor.predictor = synthetic_predictor
    with pytest.raises(Refused, match="synthetic input schema"):
        executor.verify_complete()
    executor.predictor = adapter
    executor.verify_complete()


def test_B_scoring_never_fits_or_changes_models_or_thresholds(dev, frozen):
    import pickle
    box, executor, adapter = dev
    store = ModelStore("data/interim/phase5/models")
    thresholds = {e: d["threshold"]["value"] for e, d in frozen.registry["detectors"].items()}
    executor.score_all(permit(box, frozen))
    assert len(adapter._loaded) == 24
    for eid, (model, standardizer) in adapter._loaded.items():
        manifest = frozen.registry["detectors"][eid]["model_manifest"]
        fresh_model, fresh_standardizer, _ = store.load(eid, expected_manifest_sha256=manifest["model_manifest_sha256"])
        assert pickle.dumps(standardizer) == pickle.dumps(fresh_standardizer)          # never refitted in memory
        if hasattr(model, "network"):
            for (k, a), (k2, b) in zip(model.network.state_dict().items(), fresh_model.network.state_dict().items()):
                assert k == k2 and bool((a == b).all())
        else:
            assert pickle.dumps(model) == pickle.dumps(fresh_model)
    assert {e: d["threshold"]["value"] for e, d in frozen.registry["detectors"].items()} == thresholds
    frozen.verify_memory()


# ------------------------------------------------------------------ C. JSONL ledger
def test_C_exposure_is_in_the_jsonl_ledger_before_every_read_and_poison_is_never_requested(dev, frozen, monkeypatch):
    box, executor, _ = dev
    requested = []
    real = pd.read_parquet

    def trap(path, *args, **kwargs):
        columns = kwargs.get("columns") or []
        assert columns and POISON_COLUMN not in columns and "fake_label" not in columns
        assert len(_verify(box).rows) == 428                            # real verifier, before the read
        requested.append(Path(str(path)).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", trap)
    state = executor.score_all(permit(box, frozen))
    assert len(state["tables"]) == 232 and requested
    assert not any(phase == "analysis" for phase, _ in box.events)


def test_C_duplicate_execution_and_unauthorized_scoring_are_refused(dev, frozen):
    box, executor, _ = dev
    with pytest.raises(Refused, match="authorization"):
        executor.score_all()
    executor.score_all(permit(box, frozen))
    with pytest.raises(Refused):
        executor.score_all(permit(box, frozen))
    assert len(_verify(box).rows) == 428


@pytest.mark.parametrize("stage, after_crash, after_recovery, renamed", [
    ("after_batch_authorization", 196, 428, 0),     # authorization written, no look appended
    ("after_reservation", 428, 660, 232),           # looks appended and exposed, nothing read
    ("after_read", 428, 660, 232),                  # first unit read, not persisted
])
def test_C_crashes_retries_and_identities_keep_the_history_valid(dev, frozen, stage, after_crash, after_recovery, renamed):
    box, executor, _ = dev

    def fail(where, unit):
        if where == stage:
            raise OSError("SYNTHETIC crash at " + stage)

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    executor.fault = None
    assert len(_verify(box).rows) == after_crash
    state = box.json("state.json")
    result = executor.recover(permit(box, frozen, "recovery", recovery_from=state["batch_id"]))
    history = _verify(box)
    assert len(result["tables"]) == 232 and len(history.rows) == after_recovery
    assert sum(e["attempt_id"].endswith("-r1") for e in result["tables"].values()) == renamed
    ids = [r["experiment_id"] for r in history.rows]
    assert len(ids) == len(set(ids))
    assert sum(1 for s in history.segments if not s.closed) == (1 if after_crash == 196 else 2)


def test_C_interrupted_ledger_write_fails_closed_before_any_read(dev, frozen, monkeypatch):
    box, executor, _ = dev
    real_write = os.write

    def torn(fd, data):
        real_write(fd, data[: len(data) // 2])
        raise OSError("SYNTHETIC torn ledger append")

    monkeypatch.setattr(ledger_module.os, "write", torn)
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    monkeypatch.setattr(ledger_module.os, "write", real_write)
    assert box.events == []                                             # nothing was read
    with pytest.raises(LedgerHistoryError, match="partial append"):
        _verify(box)
    state = box.json("state.json")
    with pytest.raises(Refused, match="partial append"):
        executor.recover(permit(box, frozen, "recovery", recovery_from=state["batch_id"]))
    assert REAL_LEDGER.read_bytes().count(b"\n") == 196                 # the real ledger is untouched


def test_C_concurrent_attempt_on_the_jsonl_ledger_is_refused(dev, frozen):
    box, a, _ = dev
    b = Executor(box, frozen, predictor=FrozenL1Adapter(frozen))
    outcome = []

    def contend(stage, unit):
        if stage == "before_reservation" and not outcome:
            try:
                b.score_all(permit(box, frozen))
                outcome.append("B accepted")
            except Refused as exc:
                outcome.append(str(exc))

    a.fault = contend
    a.score_all(permit(box, frozen))
    assert len(outcome) == 1 and ("concurrent" in outcome[0] or "already attempted" in outcome[0])
    assert len(_verify(box).rows) == 428


# ------------------------------------------------------------------ integration + N1/N2/N3 preservation
def test_I_full_integration_analysis_only_after_completion(dev, frozen):
    box, executor, _ = dev
    with pytest.raises(Refused):
        executor.analyse_once(permit(box, frozen, "analysis"))
    executor.score_all(permit(box, frozen))
    report = executor.analyse_once(permit(box, frozen, "analysis"))
    assert report["primary_contrasts"] == 140 and report["primary_intervals"] == 420 and report["replicates"] == 10_000
    assert report["scope"] == "SYNTHETIC_ONLY" and report["not_real_performance"]
    label_reads = [name for phase, name in box.events if "labels" in name]
    assert label_reads and all(phase == "analysis" for phase, name in box.events if "labels" in name)
    with pytest.raises(Refused, match="already claimed"):
        executor.analyse_once(permit(box, frozen, "analysis"))
    assert len(_verify(box).rows) == 428


def test_I_integration_cannot_bypass_n1_n2_n3(dev, frozen):
    box, executor, _ = dev
    from src.experiment.phase5_inputs import ProjectedInputs
    with pytest.raises(Refused, match="bound to its issuing Executor"):
        ProjectedInputs(box, box.json("fixture.json"), lambda a, p: None)
    forged = Access(executor._identity, box.identity, "SYNTHETIC-invented", "analysis", "0" * 64)
    with pytest.raises(Refused, match="unissued"):
        executor.inputs.features(forged, "RoboticArm|S")
    state = executor.score_all(permit(box, frozen))
    score = executor._access(state["batch_id"], "score")
    reused = next(u["L1"]["experiment_id"] for u in frozen.registry["units"] if u["L1"]["source"] != "new_fit_phase5")
    with pytest.raises(Refused, match="analysis capability"):
        executor.inputs.historical(score, reused)                         # N2: reused L1 stays analysis-only
    with pytest.raises(Refused, match="analysis capability"):
        executor.inputs.labels(score, "RoboticArm")
    # N3: recovery admission still runs through the coordinator with the JSONL backend
    # (exercised by test_C_crashes_retries_and_identities_keep_the_history_valid).
