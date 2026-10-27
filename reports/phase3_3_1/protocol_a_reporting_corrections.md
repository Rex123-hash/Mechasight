# Phase 3.3.1 — Protocol-A Reporting Corrections

**Layer type:** additive. The Phase-3.3 report is **not** edited. The wording below **supersedes**
the corresponding Phase-3.3 sentences for all future citation; the historical text remains on
record exactly as written, at `8add93c`.

**No number changes.** Every reported AUC, difference and classification stands unaltered.

---

## Correction 1 — nondeterminism (audit finding F3)

**Superseded (Phase-3.3 report §1 and §5):**

> "The differences are of the size **expected from a nondeterministic training run** with
> patience-3 early stopping."

**Preferred wording:**

> The residual differences are **consistent with plausible variation arising from the different
> execution environment and the upstream training procedure**, including patience-based early
> stopping and final-epoch checkpoint aliasing.

Supplementary, where the point needs stating in full:

> The upstream implementation is not fully deterministic, but this single reproduction run does not
> estimate a run-to-run distribution.

**Why this is more defensible.** Three reasons, in increasing order of importance:

1. Only **one** reproduction run per machine exists, so no variance was measured. "Expected" claims
   a distribution that was never estimated.
2. The reproduction ran on **CPU**. The cuDNN nondeterminism cited in the original justification is
   a CUDA-path property and does not explain anything about a CPU run.
3. Device, backend and torch-version differences are **genuine, sufficient, and documented**. They
   change floating-point reduction order, which changes the weights, which — through patience-based
   early stopping — changes the epoch at which training halts, which — through the reproduced
   upstream aliasing — is exactly the checkpoint that gets scored.

The corrected wording attributes the residual to the mechanism actually in evidence.

---

## Correction 2 — input construction (audit finding F4)

**Superseded (Phase-3.3 report §9):**

> "The project's data path, windowing, **feature construction**, model and scoring are therefore
> consistent with the published work."

**Preferred wording:**

> **Dataset handling, input construction/preprocessing, official baseline architecture and scoring
> are consistent with the pinned published implementation.**

**Explicit limit — stated so it cannot be read the other way:**

> This does **NOT** validate the Phase-3.1 98-engineered-feature representation.

Protocol A feeds the autoencoder **per-window standardised raw multi-sensor windows** (input
dimension 5836 = 1600·1 + 706·3 + 706·3). The engineered feature pipeline is not exercised anywhere
in Phase 3.3. "Feature construction" named a component that this reproduction never touched.

---

## Correction 3 — literature comparability (audit finding F5)

**Superseded (Phase-3.3 report §9):**

> "…which makes **MechaSight's own results comparable to the literature** rather than unmoored from
> it."

**Preferred wording:**

> Phase 3.3 establishes a **historical comparability anchor for Protocol-A-style results** and
> independently validates the project's handling of the official IMAD-DS data path and
> released-split baseline.

**And, stated explicitly:**

> It does **not** by itself establish direct numerical comparability between Protocol-B results and
> the published Protocol-A baseline, because the protocols differ in **split, preprocessing,
> representation and model family**.

Reproducing the official baseline demonstrates that this project can handle the released dataset and
evaluation correctly. It does not transfer that standing to a pipeline which shares none of those
four components with it.

---

## What is unchanged

| | |
|---|---|
| All six published comparisons | unchanged |
| All 42 rows of `protocol_a_primary_auc.csv` | unchanged |
| Worst absolute error 0.33, all NEAR_EXACT | unchanged |
| Reproduction tolerance | unchanged, still frozen at `872f8e5` |
| Verdict on the reproduction itself | **ACCEPTED** |

The corrections above are about **what the result licenses us to say**, not about what the result is.
