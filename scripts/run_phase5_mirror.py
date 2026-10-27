"""Phase 5.6C: one development-mirror end-to-end run of the protected-style executor path.

trusted manifests -> label-free count index -> frozen L1 -> L2 -> temporary JSONL
ledger (196 -> 428) -> completion gate -> analyse-once on SYNTHETIC labels.

Development source-validation cohorts only. No protected input is opened, the real
ledger is only read (copied), and nothing here is protected performance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiment import phase5_input_manifest as mi
from src.experiment.ledger_history import verify_ledger_history
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture, execute_real_protected
from src.experiment.phase5_inference import ADAPTER_VERSION, FrozenL1Adapter
from src.experiment.phase5_inputs import (
    MIRROR_SCHEMA, PROTECTED_SCHEMA, SYNTHETIC, Refused, Sandbox, atomic_json, require,
)
from src.experiment.phase5_mirror_fixture import MIRROR_RECIPE, create_mirror_fixture
from src.experiment.provenance import file_sha256

REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
REAL_LEDGER_SHA256 = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"
OUTPUT = Path("reports/phase5_6c_mirror")
MODALITIES = {"mic", "acc", "gyro"}


class _H5Proxy:
    """Records every dataset name a reader requests from an HDF5 handle."""

    def __init__(self, handle, seen):
        self._handle, self._seen = handle, seen

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __getitem__(self, key):
        self._seen.append(key)
        return self._handle[key]

    def __contains__(self, key):
        return key in self._handle

    @property
    def attrs(self):
        return self._handle.attrs


def _tripwires(box, reads):
    """Wrap the trusted readers: each read must follow the durable 428-row exposure."""
    real_parquet, real_h5 = mi.pd.read_parquet, mi.h5py.File

    def rows():
        return len(verify_ledger_history(box.path("ledger/test_look_ledger.jsonl"),
                                         authorization_root=box.path("ledger/authorizations")).rows)

    def parquet(path, *args, **kwargs):
        columns = list(kwargs.get("columns") or [])
        reads["parquet"].append({"columns": len(columns), "ledger_rows": rows(),
                                 "label_or_poison_requested": any("label" in c or "poison" in c for c in columns)})
        require(columns, "unprojected parquet read")
        return real_parquet(path, *args, **kwargs)

    def hdf5(path, *args, **kwargs):
        reads["hdf5_opens"].append(rows())
        return _H5Proxy(real_h5(path, *args, **kwargs), reads["hdf5_datasets"])

    return parquet, hdf5


def _refusals(package):
    """Disabled protected entry points, probed with fake arguments only."""
    from src.experiment.phase5_count_index import build_protected_count_index
    from src.experiment.phase5_ledger import append_protected_batch
    probes = {"execute_real_protected": lambda: execute_real_protected(),
              "read_protected_inputs": lambda: mi.read_protected_inputs(capability=None, batch_id="SYNTHETIC"),
              "build_protected_count_index": lambda: build_protected_count_index(capability=None),
              "append_protected_batch": lambda: append_protected_batch(),
              "protected_manifest_read": lambda: mi.read_features(
                  mi.protected_manifest_template(package, machine="RoboticArm", regime="S"))}
    rejected = []
    for name, call in probes.items():
        try:
            call()
        except Refused:
            rejected.append(name)
        else:
            raise AssertionError(f"disabled entry point accepted: {name}")
    # A protected-schema fixture cannot even construct an executor.
    box = Sandbox(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-protected-schema-"))
    atomic_json(box.path("fixture.json"), {"scope": SYNTHETIC, "schema": PROTECTED_SCHEMA, "files": {}, "cells": {},
                                           "cohorts": {}, "historical": {}, "labels": {}})
    try:
        Executor(box, package)
    except Refused:
        rejected.append("protected_schema_executor")
    else:
        raise AssertionError("protected schema accepted")
    return rejected


def mirror_run():
    from unittest.mock import patch
    package = FrozenPackage.load()
    require(file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256, "real ledger is not at its closed 196-row state")
    box = Sandbox(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-mirror-"))
    fixture = create_mirror_fixture(box, package)
    adapter = FrozenL1Adapter(package)
    executor = Executor(box, package, predictor=adapter)
    refusals = _refusals(package)

    # Forged/unexposed capabilities are exercised in tests/test_phase5_mirror_adapter.py;
    # only the executor constructs capabilities (read-guard tripwire).
    reads = {"parquet": [], "hdf5_opens": [], "hdf5_datasets": []}
    parquet, hdf5 = _tripwires(box, reads)
    with patch.object(mi.pd, "read_parquet", parquet), patch.object(mi.h5py, "File", hdf5):
        with _no_labels_during_scoring(executor):
            state = executor.score_all(authorize_fixture(box, package, "score"))
    scoring_events = list(box.events)
    analysis = executor.analyse_once(authorize_fixture(box, package, "analysis"))
    try:
        executor.analyse_once(authorize_fixture(box, package, "analysis"))
    except Refused:
        analyse_twice_refused = True
    else:
        raise AssertionError("second analysis accepted")
    executor.verify_complete()
    require(file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256, "real ledger changed")

    # Real-model results against the frozen development references.
    l1 = []
    for call in adapter.calls:
        eid = call["experiment_id"]
        got = pd.DataFrame(box.json(state["tables"][eid]["file"])["rows"]).set_index("segment_id")
        ref = pd.read_csv(Path(package.registry["detectors"][eid]["bundle_path"]) / "validation_scores.csv",
                          usecols=["segment_id", "score", "window_count"], dtype={"segment_id": str},
                          float_precision="round_trip").set_index("segment_id")
        require(set(got.index) == set(ref.index), f"{eid}: mirror cohort differs from the frozen reference")
        l1.append({"experiment_id": eid, "family": call["family"], "segments": len(got),
                   "max_abs_score_difference": float((got.score - ref.loc[got.index, "score"]).abs().max()),
                   "window_counts_equal": bool((got.window_count == ref.loc[got.index, "window_count"]).all())})
    fused = pd.read_csv("reports/phase5_development/l2/l2_fused_validation_scores.csv", dtype={"segment_id": str},
                        float_precision="round_trip").set_index(["calibration_id", "segment_id"])["fused_score"]
    l2_diff, l2_rows, l2_units = 0.0, 0, 0
    for eid, op in executor.operations.items():
        if op["operation"] == "derived_L2_scoring":
            l2_units += 1
            for row in box.json(state["tables"][eid]["file"])["rows"]:
                l2_diff, l2_rows = max(l2_diff, abs(row["score"] - fused[(eid, row["segment_id"])])), l2_rows + 1
    thresholds = {e: d["threshold"]["value"] for e, d in package.registry["detectors"].items()}
    package.verify_memory()
    require(thresholds == {e: d["threshold"]["value"] for e, d in FrozenPackage.load().registry["detectors"].items()},
            "frozen thresholds changed")
    history = verify_ledger_history(box.path("ledger/test_look_ledger.jsonl"),
                                    authorization_root=box.path("ledger/authorizations"))
    records = [box.json(t["file"])["computation"] for t in state["tables"].values()]
    return {
        "scope": "DEVELOPMENT_MIRROR_ONLY", "not_protected_performance": True, "recipe": MIRROR_RECIPE,
        "input_schema": MIRROR_SCHEMA, "inference_adapter": ADAPTER_VERSION, "data_origin": fixture["data_origin"],
        "cohort_segments": {m: len(c) for m, c in fixture["cohorts"].items()},
        "cells": {c: e["input_manifest_identity"] for c, e in fixture["cells"].items()},
        "units": len(state["tables"]),
        "operations": dict(Counter(op["operation"] for op in executor.operations.values())),
        "ledger": {"rows_before_after": [196, len(history.rows)],
                   "appended_segments": [s.stage for s in history.segments if not s.closed],
                   "verified_with": "verify_ledger_history (unmodified) + JsonlLedger.validate_batches",
                   "real_ledger_sha256_unchanged": True},
        "real_model_results": {
            "l1_units": len(l1), "l1_segment_scores": sum(r["segments"] for r in l1),
            "l1_max_abs_diff_vs_frozen_bundle_validation_scores": max(r["max_abs_score_difference"] for r in l1),
            "l1_window_counts_equal": all(r["window_counts_equal"] for r in l1),
            "l1_families": dict(Counter(r["family"] for r in l1)),
            "l2_units": l2_units, "l2_rows": l2_rows,
            "l2_max_abs_diff_vs_persisted_l2_fused_validation": l2_diff,
            "thresholds_unchanged": True, "models_refitted": 0},
        "synthetic_results": {"labels": "SYNTHETIC parent-homogeneous fake labels; development validation has no faults",
                              "analysis_primary_contrasts": analysis["primary_contrasts"],
                              "analysis_primary_intervals": analysis["primary_intervals"],
                              "analysis_replicates": analysis["replicates"],
                              "analysis_sha256": file_sha256(box.path("analysis.json")),
                              "analyse_twice_refused": analyse_twice_refused},
        "provenance": {"kinds": dict(Counter(r["kind"] for r in records)),
                       "implementations": dict(Counter(r["implementation"] for r in records)),
                       "l2_component_origins": dict(Counter(r["component_origin"] for r in records if "component_origin" in r))},
        "access": {"forged_capability_tests": "tests/test_phase5_mirror_adapter.py::test_A_forged_capability_reads_nothing",
                   "trusted_parquet_reads": len(reads["parquet"]),
                   "trusted_parquet_reads_after_exposure": all(r["ledger_rows"] == 428 for r in reads["parquet"]),
                   "label_or_poison_columns_requested": any(r["label_or_poison_requested"] for r in reads["parquet"]),
                   "hdf5_opens": len(reads["hdf5_opens"]),
                   "hdf5_opens_after_exposure": all(r == 428 for r in reads["hdf5_opens"]),
                   "hdf5_datasets_requested": sorted(set(reads["hdf5_datasets"])),
                   "provenance_json_requested": "provenance_json" in reads["hdf5_datasets"],
                   "score_phase_label_reads": sum("labels" in n for p, n in scoring_events),
                   "analysis_phase_label_reads": sum("labels" in n for p, n in box.events if p == "analysis"),
                   "count_index_builds": sum(n.startswith("count-index:") for p, n in scoring_events)},
        "disabled_entry_points_refused": refusals,
        "fixture_root": str(box.root),
    }


class _no_labels_during_scoring:
    """During scoring, the analysis gate must stay closed: any label read raises."""

    def __init__(self, executor):
        self.executor = executor

    def __enter__(self):
        inputs = self.executor.inputs
        self.real = inputs.labels
        inputs.labels = lambda *a, **k: (_ for _ in ()).throw(AssertionError("label read during scoring"))
        return self

    def __exit__(self, *exc):
        del self.executor.inputs.labels
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-evidence", action="store_true", help="write reports/phase5_6c_mirror/evidence.json")
    args = parser.parse_args()
    evidence = mirror_run()
    access = evidence["access"]
    require(evidence["units"] == 232 and evidence["ledger"]["rows_before_after"] == [196, 428]
            and evidence["real_model_results"]["l1_max_abs_diff_vs_frozen_bundle_validation_scores"] == 0.0
            and evidence["real_model_results"]["l2_max_abs_diff_vs_persisted_l2_fused_validation"] == 0.0
            and access["trusted_parquet_reads_after_exposure"] and access["hdf5_opens_after_exposure"]
            and not access["label_or_poison_columns_requested"] and not access["provenance_json_requested"]
            and access["hdf5_datasets_requested"] == sorted(MODALITIES) and access["score_phase_label_reads"] == 0,
            "mirror run did not meet its acceptance conditions")
    if args.write_evidence:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8", newline="\n")
    print(json.dumps({k: evidence[k] for k in ("units", "ledger", "real_model_results")}, indent=1))
