# Phase 3.4C — Failure and Restart Policy (frozen before execution)

**Governing principle.** A run may be restarted **only while it is still before test exposure**, and
**only for a mechanical failure** — never because a result was disliked or looked unexpected.

The distinction that matters:

| | |
|---|---|
| **RESTART BEFORE TEST INFERENCE** | permitted, for mechanical failure only, under the rule below |
| **RE-SCORE AFTER TEST EXPOSURE** | **prohibited without exception** |

Once a model's scores exist, that model is finished. No amount of surprise about the numbers
reopens it.

---

## Permitted restarts (before test inference only)

- process crash, power loss, or interruption
- disk full, or checkpoint write failure
- an exception raised **before** the checkpoint was written

**Restart rule.** Delete the incomplete checkpoint. Re-run the **same** `experiment_id` with the
**same** seed, membership, fold, arm and environment. Record the restart and its cause in the
training record. The restart count is reported with the results.

Nothing about the experiment may change on a restart. A restart that alters any frozen input is not
a restart — it is a new experiment and requires a new version.

---

## Never permitted

- re-running any model **after** its test scores exist
- re-running a completed model because its loss curve, epoch count or any metric looked unexpected
- substituting a different seed, membership, fold, arm or environment
- completing the remaining arms on a different backend

---

## Specific failure modes

**NaN or non-finite loss → STOP the phase.** Do not restart into a different seed. Record the
occurrence, investigate the cause, and treat any resulting change as a new versioned experiment. A
NaN is information about the design, not an inconvenience to route around.

**Unexpected early stopping → not a failure.** Halting at any epoch is a legitimate outcome of the
frozen `patience = 3` schedule. It is recorded, not corrected. This includes an arm stopping much
earlier or later than its pair.

**Environment change mid-experiment → STOP the phase.** Completing the remaining arms under a
different environment is prohibited. A restart under another device or torch build is a new
versioned experiment, per the no-device-switch rule.

**Score-artifact write failure.** The artifact is written atomically — temporary file then rename —
so either the complete artifact exists or none does. If no artifact was produced, the run counts as
**not scored**, and re-running the evaluation of an already-frozen checkpoint against the
already-authorised test look is permitted and must be recorded. If an artifact *was* produced, the
look is spent.

**Disk exhaustion.** Ten checkpoints at ~163 MB each need ~1.63 GB, plus score artifacts. Verify
free space before COMMIT A. Exhaustion mid-run is a permitted mechanical restart for the affected
run only.

**Interruption of the ~31-hour sequence.** Completed runs keep their frozen checkpoints. The
sequence resumes at the next position in the frozen training order. The order itself never changes.

---

## Partial completion

If the phase stops after some folds, the completed folds are reported as an **incomplete
experiment**, with the stopping reason stated. They must not be presented as the planned design, and
the primary estimand must not be computed as though all five folds had run.

Reporting an incomplete experiment honestly is always preferable to quietly redefining the design to
fit what finished.
