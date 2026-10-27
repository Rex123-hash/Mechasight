"""5.5G: reject untrusted raw sources before I/O; bind actual computation lineage."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.experiment import phase5_window_index as wi
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture, synthetic_predictor
from src.experiment.phase5_fixture import create_fixture
from src.experiment.phase5_inputs import Refused, Sandbox
from src.experiment.provenance import file_sha256
from src.fusion.assembly import PREPROCESSING_CONFIGS
from src.preprocessing.config import load_config


def genuine_source():
    r = wi._development_contract()["sources"][0]
    return wi.SegmentSource(r["machine"], r["segment_id"], r["parent_recording"], r["domain"],
                            {m: str(wi.REPOSITORY / p) for m, p in r["paths"].items()})


@pytest.fixture
def source():
    return genuine_source()


@pytest.fixture
def config():
    return load_config(PREPROCESSING_CONFIGS["S"])


@pytest.fixture
def no_sensor_io(monkeypatch):
    counts = {"timestamp_reads": 0, "parquet_opens": 0, "unauthorized_source_hashes": 0}

    def schema(*a, **kw):
        counts["parquet_opens"] += 1
        raise AssertionError("unauthorized Parquet schema access")

    def parquet(*a, **kw):
        counts["timestamp_reads"] += 1
        raise AssertionError("unauthorized timestamp access")

    original = wi.canonical_sha256

    def hashed(path):
        if "data/raw" in Path(path).as_posix():
            counts["unauthorized_source_hashes"] += 1
            raise AssertionError("unauthorized raw-source hash")
        return original(path)

    monkeypatch.setattr(wi.pq, "read_schema", schema)
    monkeypatch.setattr(wi.pd, "read_parquet", parquet)
    monkeypatch.setattr(wi, "canonical_sha256", hashed)
    yield counts
    assert counts == {"timestamp_reads": 0, "parquet_opens": 0, "unauthorized_source_hashes": 0}


@pytest.mark.parametrize("case", ["original_target", "substituted", "wrong_segment", "machine", "parent", "domain",
                                  "traversal", "external", "modality", "representation", "forged_claim"])
@pytest.mark.parametrize("entry", ["generic", "segment", "timestamps"])
def test_untrusted_sources_are_rejected_before_any_sensor_io(source, config, no_sensor_io, tmp_path, case, entry):
    if case == "original_target":
        source = replace(source, segment_id="SYNTHETIC-unregistered-test-segment", domain="target",
                         paths={m: f"SYNTHETIC-UNOPENED/{m}.parquet" for m in wi.SENSOR_KEYS})
    elif case == "substituted":
        source = replace(source, paths={**source.paths, "mic": source.paths["acc"]})
    elif case == "wrong_segment":
        source = replace(source, segment_id="SYNTHETIC-wrong-segment")
    elif case == "machine":
        source = replace(source, machine="SYNTHETIC-wrong-machine")
    elif case == "parent":
        source = replace(source, parent_recording="SYNTHETIC-wrong-parent")
    elif case == "domain":
        source = replace(source, domain="target")
    elif case == "traversal":
        p = Path(source.paths["mic"])
        source = replace(source, paths={**source.paths, "mic": str(p.parent / ".." / p.parent.name / p.name)})
    elif case == "external":
        source = replace(source, paths={**source.paths, "mic": str(tmp_path / "external.parquet")})
    elif case == "modality":
        source = replace(source, paths={"mic": source.paths["mic"]})
    elif case == "representation":
        config = replace(config, stride_s=config.stride_s / 2)
    else:
        # A caller's claimed development flag cannot make an unregistered source trusted.
        source = replace(source, segment_id="SYNTHETIC-forged-development")
        object.__setattr__(source, "development", True)
    with pytest.raises(Refused):
        if entry == "generic":
            wi.build_window_index([source], config, regime="S")
        elif entry == "segment":
            wi.segment_windows(source, config)
        else:
            wi._timestamps(source.paths["mic"], "mic", source=source, config=config)


def test_whole_request_preflight_rejects_late_bad_source(source, config, no_sensor_io):
    with pytest.raises(Refused):
        wi.build_window_index([source, replace(source, segment_id="zz-SYNTHETIC-unregistered")], config, regime="S")


def test_another_registered_segment_cannot_use_the_original_paths(source, config, no_sensor_io):
    other = next(r for r in wi._development_contract()["sources"]
                 if r["machine"] == source.machine and r["regime"] == "S" and r["segment_id"] != source.segment_id)
    forged = replace(source, segment_id=other["segment_id"], parent_recording=other["parent_recording"])
    with pytest.raises(Refused, match="binding"):
        wi.segment_windows(forged, config)


def test_incorrect_regime_is_rejected_before_sensor_io(source, config, no_sensor_io):
    with pytest.raises(Refused, match="regime differs"):
        wi.build_window_index([source], config, regime="T")


@pytest.mark.parametrize("mutation", ["missing", "modified", "split_modified"])
def test_frozen_manifest_missing_or_changed_fails_closed(source, config, no_sensor_io, tmp_path, monkeypatch, mutation):
    if mutation == "split_modified":
        real = wi.canonical_sha256
        monkeypatch.setattr(wi, "canonical_sha256", lambda p: "0" * 64 if Path(p).suffix == ".csv" else real(p))
    else:
        path = tmp_path / "manifest.json"
        if mutation == "modified":
            path.write_text('{"development": true}', encoding="utf-8")
        monkeypatch.setattr(wi, "DEVELOPMENT_SOURCES", path)
    for call in (lambda: wi.build_window_index([source], config, regime="S"), lambda: wi.segment_windows(source, config)):
        with pytest.raises(Refused, match="manifest"):
            call()


def test_path_link_escape_is_rejected(source, config, no_sensor_io, tmp_path):
    target = tmp_path / "external"
    target.mkdir()
    link = tmp_path / "alias"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("directory symlinks unavailable")
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if result.returncode:
            pytest.skip("neither symlinks nor junctions available")
    with pytest.raises(Refused, match="binding"):
        wi.segment_windows(replace(source, paths={**source.paths, "mic": str(link / "mic.parquet")}), config)


def test_link_at_expected_location_is_rejected(source, config, no_sensor_io, monkeypatch):
    real = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda p: p == Path(source.paths["mic"]).parent or real(p))
    with pytest.raises(Refused, match="alias/link"):
        wi.segment_windows(source, config)


def test_direct_timestamp_and_protected_constructors_refuse_without_authority(source, no_sensor_io):
    with pytest.raises(Refused, match="registered development"):
        wi._timestamps(source.paths["mic"], "mic")
    with pytest.raises(Refused, match="disabled"):
        wi.build_protected_window_index(source, development=True)


def test_genuine_source_preserves_frozen_windows_and_timestamp_projection(source, config, monkeypatch):
    expected = wi.segment_windows(source, config)
    index = wi.build_window_index([source], config, regime="S")
    assert list(zip(index.window_start_s, index.window_end_s)) == expected
    real = pd.read_parquet
    columns = []

    def read(path, **kwargs):
        columns.extend(kwargs["columns"])
        return real(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    assert wi.segment_windows(source, config) == expected
    assert len(columns) == 3 and all(c.lower() in wi.TIME_COLUMN_NAMES for c in columns)


def test_nonmonotonic_timestamps_still_refuse_after_membership(source, config, monkeypatch):
    t = np.array([0., 1., .5, 2.])
    monkeypatch.setattr(pd, "read_parquet", lambda p, columns: pd.DataFrame({columns[0]: t}))
    with pytest.raises(Refused, match="strictly increasing"):
        wi.segment_windows(source, config)


@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


@pytest.fixture(scope="module")
def scored_stub(frozen, tmp_path_factory):
    box = Sandbox(tmp_path_factory.mktemp("SYNTHETIC-5-5g-stub"))
    create_fixture(box, frozen)
    executor = Executor(box, frozen)
    state = executor.score_all(authorize_fixture(box, frozen, "score"))
    return box, executor, state


def test_stub_and_derived_lineage_are_explicit(scored_stub):
    box, executor, state = scored_stub
    counts = {"synthetic_L1_stub": 0, "derived_L2_scoring": 0}
    for eid, entry in state["tables"].items():
        table = box.json(entry["file"])
        c = table["computation"]
        counts[c["kind"]] += 1
        assert table["predictor"] == c["implementation"]
        if c["kind"] == "synthetic_L1_stub":
            assert c["implementation"] == "SYNTHETIC_STUB_V1" and "model" not in c
        else:
            assert c["component_origin"] == "SYNTHETIC_FIXTURE"
            assert c["calibration_id"] == eid
            assert set(c["components"]) == set(executor.package.registry["calibrations"][eid]["component_experiment_ids"])
    assert counts == {"synthetic_L1_stub": 24, "derived_L2_scoring": 208}


@pytest.mark.parametrize("mutation", ["missing", "forged_real", "wrong_predictor", "wrong_component", "wrong_calibration"])
def test_forged_computation_fails_completion_even_with_rehashed_output(scored_stub, mutation):
    box, executor, _ = scored_stub
    state = box.json("state.json")
    l2 = mutation in {"wrong_component", "wrong_calibration"}
    eid = next(e for e, op in executor.operations.items() if (op["operation"] == "derived_L2_scoring") == l2)
    entry = state["tables"][eid]
    path, statepath = box.path(entry["file"]), box.path("state.json")
    original, originalstate = path.read_bytes(), statepath.read_bytes()
    table = json.loads(original)
    if mutation == "missing":
        del table["computation"]
    elif mutation == "forged_real":
        table["computation"]["kind"] = "frozen_L1_inference"
    elif mutation == "wrong_predictor":
        table["predictor"] = "phase5-frozen-l1-inference-adapter-v1"
    elif mutation == "wrong_component":
        next(iter(table["computation"]["components"].values()))["experiment_id"] = "SYNTHETIC-forged"
    else:
        table["computation"]["calibration_id"] = "SYNTHETIC-forged"
    try:
        path.write_text(json.dumps(table), encoding="utf-8")
        state["tables"][eid]["sha256"] = file_sha256(path)
        statepath.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(Refused, match="computation/predictor"):
            executor.verify_complete()
    finally:
        path.write_bytes(original)
        statepath.write_bytes(originalstate)


def test_callable_identity_cannot_be_supplied_as_a_string(scored_stub):
    _, executor, _ = scored_stub
    class PretendReal:
        implementation = "phase5-frozen-l1-inference-adapter-v1"
        def __call__(self, *args):
            return synthetic_predictor(*args)
    try:
        executor.predictor = PretendReal()
        with pytest.raises(Refused, match="actual frozen L1"):
            executor.verify_complete()
    finally:
        executor.predictor = synthetic_predictor
