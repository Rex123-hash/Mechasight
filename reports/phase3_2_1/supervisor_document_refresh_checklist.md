# Supervisor / Synopsis Document Refresh Checklist

**Prepared by:** Phase 3.2.1 reporting correction (`p321-correction-75e1479967f120aff5047360`)
**Purpose:** input to a **separate** supervisor-document refresh phase.
**Status:** checklist only — no document was edited, regenerated or overwritten in Phase 3.2.1.

The two affected artifacts are untracked and predate the recorded model results:

| Artifact | Built | Problem |
|---|---|---|
| `reports/supervisor/MechaSight_Project_Direction_and_Implementation_Report.{pdf,docx}` | 23 Aug 2026 | States Phase 3 not started and that no model has been trained |
| `reports/synopsis/MechaSight_MAIT_Synopsis_2026.{pdf,docx}` | 3 Sep 2026 | Predates the Phase-3.2 results commit; every performance-related statement needs line-by-line re-checking |

> **Do not present either build in its current state.** Both assert a project status that the
> repository has since superseded.

---

## 1. Stale statements to remove or replace

| # | Stale statement | Replacement fact |
|---|---|---|
| 1.1 | "No anomaly-detection model has been trained." | 28 pre-registered classical configurations were fitted in Phase 3.1 and evaluated once on the locked common test in Phase 3.2. |
| 1.2 | "Phase 3 has not started." | Phase 3.0, 3.0.1, 3.1, 3.1.1, 3.2 and 3.2.1 are complete. The next experimental gate is Protocol-A reproduction. |
| 1.3 | "Any ML performance figures will be added only after reproducible experiments have actually been run." | Those experiments have now been run; the figures exist and are cited below. |
| 1.4 | "Every performance figure in this document is a published result by the IMAD-DS authors — none of it is ours." | No longer true. MechaSight now has its own first-test results. Keep the attribution for the *authors'* baseline numbers, but add our own clearly separated. |
| 1.5 | Roadmap showing "→ Phase 3.1 classical baselines" as the next step | Roadmap must advance to "→ Phase 3.2 first locked common test ✓ → Phase 3.2.1 reporting correction ✓ → Protocol-A reproduction (next)". |
| 1.6 | Any statement that the common test is unopened | The common test was opened once, on 3 September 2026, in a single 28-entry batch. |

---

## 2. New Phase-3.2 facts to insert

**Provenance to state once**

- Results HEAD `a576c9df8000dd1c0a5ad6fd73845f97facf98b2`; evaluator commit `99f6055…`;
  batch `p32-test-batch-8f7d48324fd10df97ae29100`.
- Test-look ledger: 0 before, **28 after**, written atomically before the first score. One
  batch — not 28 rounds of adaptive selection.
- Independently audited (`docs/phase32_independent_audit.md`): every score, threshold, confusion
  count, ranking metric, bootstrap interval and paired contrast reproduced exactly; verdict
  YELLOW for reporting only; corrected in Phase 3.2.1.

**Headline results — always per machine, never pooled**

| Machine | Regime | Model | AUROC [95 % CI] |
|---|---|---|---|
| RoboticArm | S | isolation_forest | 0.9137 [0.774, 0.976] |
| RoboticArm | S | one_class_svm | 0.8985 [0.783, 0.959] |
| RoboticArm | S | statistical_reference | 0.8522 [0.652, 0.953] |
| BrushlessMotor | T | one_class_svm | 0.8648 [0.821, 0.905] |
| BrushlessMotor | S | one_class_svm | 0.7054 [0.643, 0.764] |
| BrushlessMotor | S | statistical_reference | 0.5988 [0.534, 0.661] |

**The three findings that matter more than the AUROCs**

1. **Threshold transfer failed for RoboticArm Isolation Forest.** AUROC 0.914 with **recall 0.000**
   and **F1 0.000** — the entire test score distribution fell below the frozen q90 operating point
   calibrated on 6 validation recordings.
2. **Healthy-target false alarms are unacceptable on BrushlessMotor.** Every model flagged
   79–100 % of *healthy* target-domain recordings; One-Class SVM in Regime S flagged **78 of 78**.
   This is the "DIFFERENT ≠ FAULTY" failure the project exists to measure.
3. **Adaptation did not have a consistent sign.** With the released budget (RA 27, BM 18 segments),
   T−S was slightly negative for RA Isolation Forest, clearly negative for BM Isolation Forest and
   clearly positive for BM One-Class SVM. No general claim about domain adaptation is supported.

---

## 3. Sensor terminology corrections

Replace "three sensors" / "three physical sensors" with the hardware-precise form:

> **three sensing modalities from two physical sensor devices, providing seven numeric signal
> channels**

| Level | Count | Detail |
|---|---:|---|
| Physical sensor devices | **2** | IMP23ABSU microphone; ISM330DHCX 6-axis IMU |
| Sensing modalities | **3** | microphone/audio, accelerometer/vibration, gyroscope/angular motion |
| Numeric signal channels | **7** | 1 microphone + 3 accelerometer axes + 3 gyroscope axes |

The accelerometer and gyroscope are functional blocks inside the **same** ISM330DHCX package — not
separate devices. Any diagram showing three discrete sensor boxes should be relabelled as
modalities, or redrawn as two devices feeding three modalities.

---

## 4. Audio / dataset representation clarification

Insert the three-level distinction, and remove any implication that MechaSight digitises audio:

> **Physical sensor acquisition** — transduction, conditioning and analogue-to-digital conversion —
> happened once, during the original IMAD-DS collection on the acquisition node.
> **Digital storage format** — the released archive contains per-sensor Parquet signal files plus
> metadata CSVs; the microphone stream is a two-column table of `Time` and `MIC [Waveform]`
> numerical values. There is no MP3, MP4 or WAV in the release.
> **ML preprocessing** — MechaSight reads those stored numbers, which is software deserialisation,
> not analogue-to-digital conversion. MechaSight performs no ADC.

Also required:

- A future live-sensor deployment would again require physical acquisition hardware with ADC, or a
  digital sensor interface.
- The microphone physical unit remains **UNKNOWN** and must not be invented or implied.

---

## 5. Model-result caveats that must travel with every number

- **Per machine only.** Never pool RoboticArm and BrushlessMotor into a single headline or average.
- **Cluster counts.** RoboticArm's 466 common-test segments come from **33** independent physical
  recordings (source validation 6; healthy target 6). Never present 466 as 466 independent
  observations. The wide RoboticArm intervals are the honest consequence.
- **Inference scope.** Intervals are conditional on this observed evaluation cohort and these fixed
  seeds. They do **not** cover training variability, seed population, or transfer to a new machine
  or site.
- **Paired comparisons only.** Superiority claims come from paired bootstrap contrasts, never from
  non-overlapping marginal intervals.
- **No universal winner.** No system is ranked on a single metric.

---

## 6. BrushlessMotor confound wording

Every BrushlessMotor figure must carry the limitation.

**Permitted:** "benchmark anomaly-discrimination performance on the released BrushlessMotor
cohort."

**Forbidden:** "pure mechanical-fault detection performance"; any claim that a high BrushlessMotor
AUROC demonstrates physical fault sensitivity.

All released normal recordings precede all released anomalous recordings with no overlap, so fault
state and acquisition session/time are perfectly confounded and **not independently identifiable**.
This applies specifically to the two numbers most likely to be quoted alone: **0.977** (S source
slice, One-Class SVM) and **0.8648** (T complete cohort, One-Class SVM).

---

## 7. F1 correction wording

If any table reports F1:

- Use the count identity `F1 = 2TP / (2TP + FP + FN)`.
- RoboticArm Isolation Forest F1 is **0.000** in both regimes — **not** blank, not "N/A", not
  "UNDEFINED".
- State plainly what it means: *the detector flagged none of the 232 anomalies at its frozen
  operating point.*
- Precision for those rows is **"undefined — no positive predictions"** and must **not** be shown
  as 0.
- Do **not** describe the accompanying FPR of 0.0000 as excellent specificity. Use:
  *"No positive alarms were generated at the frozen operating point; therefore zero false positives
  coexist with zero fault detections."*

---

## 8. Average-Precision prevalence warning

Any slide or table showing AP must show the slice prevalence beside it.

| Slice | normal | anomaly | prevalence (AP floor) |
|---|---:|---:|---:|
| RoboticArm source | 214 | 116 | 0.3515 |
| **RoboticArm target** | **20** | **116** | **0.8529** |
| BrushlessMotor source | 78 | 78 | 0.5000 |
| BrushlessMotor target | 78 | 78 | 0.5000 |

**The trap to avoid:** the statistical reference shows source AP 0.6486 and target AP 0.8213. Read
naively that says it improves under domain shift. Against its own floors it is **+0.297 above** in
source and **−0.032 below** in target — the opposite conclusion. Four family-level target-slice AP
values sit at or below their own prevalence.

Present `ap_minus_prevalence` as **context only**; never call it a chance-adjusted AP. Keep AUROC
alongside as prevalence-insensitive ranking context.

---

## 9. Refresh checklist

- [ ] Remove every "no model trained" / "Phase 3 not started" statement (§1)
- [ ] Insert provenance: HEAD, evaluator commit, batch, ledger 0→28, audit reference (§2)
- [ ] Insert per-machine headline results with CIs, never pooled (§2)
- [ ] Insert the three findings that outrank the AUROCs (§2)
- [ ] Replace "three sensors" wording everywhere, including diagrams (§3)
- [ ] Insert the acquisition / storage / preprocessing three-level distinction (§4)
- [ ] Attach cluster counts to every threshold-dependent rate (§5)
- [ ] Attach the confound to every BrushlessMotor number (§6)
- [ ] Report F1 = 0.000 with its plain-language meaning; precision undefined (§7)
- [ ] Attach prevalence to every AP; add the source-vs-target worked example (§8)
- [ ] Update the roadmap and state Protocol-A reproduction as the next gate
- [ ] Re-verify every remaining performance figure against `reports/phase3_2_1/`
- [ ] Confirm no figure implies fusion, single-modality or robustness results — none exist yet

---

## 10. Out of scope for the refresh

Do not, while refreshing these documents:

- change any score, threshold, model configuration or split;
- introduce results that do not exist (fusion, single-modality, deep models, corruption, Protocol A);
- present the reliability-aware fusion hypothesis as a demonstrated outcome;
- claim any factory, live-sensor or real-time capability;
- soften the BrushlessMotor confound or the RoboticArm cluster caveat to make results look cleaner.
