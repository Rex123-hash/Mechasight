# Evaluation Failure and Recovery Policy

<!-- Legacy roadmap scope: Phase 4 onward -->

This policy was established before protected evaluation for Single-Modality Model Benchmarking.
It follows the technical evaluation interruption in the sibling-exposure study and its documented
recovery of an attempt with zero protected inference. It applies to subsequent research stages.

## Principle

A *technical failure* is an infrastructure, software or harness failure that occurs before a
valid completed result exists. A *model result* is a completed, hash-verified protected
evaluation. The two are never treated alike.

## Recovery and reporting requirements

1. **Resume/repeat after technical failure, same object.** An infrastructure or software failure
   that occurs before a valid completed result may be resumed or repeated using the **exact frozen
   experimental object** (same checkpoint hash, same memberships, same preprocessing, same
   protected cohort, same score definition, same estimator, same bootstrap parameters).
2. **Every failed attempt stays logged.** Claims, `SCORE_STARTED` records, timestamps and failure
   reasons are append-only provenance. They are never deleted, edited, completed after the fact,
   or reused. A repeat runs under a new, versioned recovery identity in its own namespace.
3. **No change because of a result.** No model, config, preprocessing, threshold, membership,
   estimator or bootstrap change is permitted because of partial **or completed**
   protected-test results.
4. **A valid completed result is final.** Once a completed, hash-verified result exists for a
   frozen object, it is the result. It is reported regardless of direction.
5. **Rerunning because a result looks bad is prohibited.** Also prohibited: re-scoring a completed
   look, selecting among repeats, or dropping a completed look.
6. **Line between the two kinds of failure.** A failure *before protected-data access* is
   technical and may be corrected (rule 1). A failure *after protected inference begins* is
   subject to an evaluation halt: the attempt is preserved, an immediate retry is not permitted,
   and independent review precedes any recovery decision.
7. **Decision hygiene.** Any recovery decision is recorded in a versioned amendment **before**
   the recovery runs, stating whether it was made after any related result was known, and that
   the recovered result will be accepted regardless of value.
8. **Harness isolation.** Tests never touch real protected paths (suite-wide guard + child-process
   guard + static audit). This is verified before every protected-evaluation stage.
