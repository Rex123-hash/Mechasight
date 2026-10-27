# MechaSight — complete repository state recovery

> **Historical repository snapshot.** This document records the state inspected on 4 September
> 2026, not the current project status. Its roadmap, test counts and pending decisions belong to
> that snapshot. Benchmark Integrity and Evaluation Validation is now complete; the
> [final evaluation report](phase34_final_report.md) and [current roadmap](../README.md#roadmap)
> provide the handoff for Single-Modality Model Benchmarking. The evidence below is retained
> for historical traceability.

Recovery date: 4 September 2026. Repository: the MechaSight development repository.
Branch: `master`. Verified HEAD: `a576c9df8000dd1c0a5ad6fd73845f97facf98b2`.

## 1. Status and how to use this handoff

The repository is at **Phase 3.2: first locked common-test results recorded**. This recovery is an audit, not a new experimental phase. No model was trained, retuned or scored during recovery; no threshold, preprocessing, split membership, ledger entry or historical artifact was changed. The existing test suite was run, and persisted evidence was read and integrity-checked. The two recovery documents are new, uncommitted handoff artifacts; HEAD remains unchanged.

Read this narrative first, then the exact evidence appendices. The companion `reports/recovery/project_state_recovery.json` embeds full frozen experiment configurations, threshold provenance, all existing result CSV tables, ledger records, Git history/file changes and local artifact inventory. It does not replace the original bundles.

Evidence precedence: current frozen v3 manifests and scoped Phase-3.1/3.1.1 configs, immutable bundles and Phase-3.2 reports/implementation outrank stale overview prose. Git establishes chronology; commit dates do not by themselves establish review approval or test execution. Historical report claims are identified as such. Raw dataset audit measurements outrank paper assumptions about the release. External sources were not newly fetched or revalidated in this recovery.

There are 16 commits, no configured remotes and no tags. Initial tracked worktree was clean; pre-existing untracked directories were `reports/supervisor/`, `reports/synopsis/` and `tmp/`. Ignored raw data, interim features, PDFs and renders are local-only, not recoverable through Git alone. The exact file inventory and commit-to-file map appear below.

## 2. Research definition: frozen, open and superseded

**FROZEN problem:** DIFFERENT != FAULTY. Healthy load, speed or acoustic-background changes can move a machine's measurements; actual faults also move them. The research must distinguish healthy domain shift from anomalous behavior without using test anomaly labels for training or model selection.

**Aim:** build reproducible evidence about robust, normal-only multi-sensor machine anomaly detection under operating/environmental shifts, not a product demo or guaranteed fusion success.

**Primary research question:** can inference-time reliability-aware weighting of microphone, accelerometer and gyroscope evidence improve robustness under domain shift and degraded/missing sensors relative to genuine single-modality models, naive equal-weight fusion and established baselines, trained under the same normal-only protocol and compared with uncertainty? Report each machine independently.

Objectives are to establish a defensible baseline ladder; quantify source/target generalization and healthy-target false alarms; assess sensor corruption/missingness robustness; and explain sensor contributions. Reliability-weighted fusion is established prior art, not an invented mechanism. The potentially useful contribution is the combination of normal-only learning, heterogeneous multi-rate physical sensors and explicit domain-shift/robustness evaluation. It remains a hypothesis, not an achieved result.

**Scope recovered from the untracked supervisor report:** public-dataset offline research and saved-data replay; an anomaly score supports a human maintenance decision. No automatic shutdown or safety-critical actuation. Future factory acquisition would add suitably installed/calibrated microphone, accelerometer and gyroscope channels, synchronized acquisition/interface/DAQ, edge processing and separately validated maintenance procedures. No installed factory system, live multi-sensor trial, validated latency or universal factory transfer has been demonstrated. Optional live audio is a separately validated audio-branch demonstration, not proof of the multi-sensor hypothesis. The supervisor report predates real model results and its no-training statement is now stale.

**OPEN:** exact deep/temporal/fusion implementations, reliability estimator, justified tree objective, corruption severities, external confirmation data, and practical deployment validation. Genuine single-modality and fusion comparisons have not been run. MIMII DUE/ToyADMOS2 are optional separate audio-only checks, never pooled with IMAD-DS or used to prove multi-sensor robustness. k-NN/Mahalanobis comparator source justification remains unverified in the recorded source log.

**SUPERSEDED:** nominal6.7kHz treated as measured rate; assumed6–10s clips; segment independence on RA; marginal-CI winner rules; unequal S/T test populations; RA 123-segment adaptation budget; domain×class×condition bootstrap strata; universal five-seed repetition of deterministic methods; primary classical models taking median window scores. Phase 3.1 instead takes median **features** before scoring one vector per segment. Old alternatives remain historical or sensitivity candidates, not competing primary configurations.

**POST-TEST EXPLORATORY ONLY:** any new design choice inspired by known Phase 3.2 outcomes, unless independently confirmed on a new untouched cohort. A prior roadmap is evidence of prior intent, not proof that every future parameter choice was preregistered.

## 3. Dataset state and evidence strength

Published source record: IMAD-DS, DOI `10.5281/zenodo.12665499`, concept DOI `10.5281/zenodo.12636236`, CC BY-SA4.0. Historical downloads were checksum-verified: RoboticArm.7z3,545,453,763 bytes / MD5 `a64498b7dcc297946a7fb8366e38ba33`; BrushlessMotor.7z1,577,018,050 bytes / MD5 `d5f1a49b928b1b0b01c3d6de930cd277`. A claimed dataset version number is UNKNOWN: the recorded API version field was null. The historical audit covered all 3,896 segments /11,688 sensor parquet files plus12 metadata CSVs, about 10.53GiB extracted. Recovery counted files and inspected saved audit evidence; it did not repeat the full signal audit or hash the multi-gigabyte archives.

| Published/released partition | RA | BM |
| --- | ---: | ---: |
| Source normal train |1812|1263|
| Target normal train |27|18|
| Source normal test |116|78|
| Source anomaly test |116|78|
| Target normal test |116|78|
| Target anomaly test |116|78|
| Total |2303|1593|

Each segment has microphone1 channel, accelerometer3 axes and gyroscope3 axes. Published acquisition platform is STEVAL-STWINBX1; nominal rates are 16kHz microphone and 6.7kHz IMU. Historical measured mean span-derived rates, in Hz: RA 16000.000/7063.294/7063.294 and BM 16003.077/7135.716/7135.331 (mic/acc/gyro). Frozen **median reference rates**, slightly different from these means, are RA 16000/7063.280051/7063.280051 and BM 16003.076013/7135.714148/7135.329526. Rate estimation uses `(N-1)/(last_time-first_time)`, not inverse median timestamp interval: timestamps are quantized to1 microsecond.

RA operational shifts are source loads00/10/15/20 and target25/30/35. BM source has nine released speeds1500–3000; target has five1000–1400; the exact vocabularies are enumerated from the current audit CSV in Appendix B. Source backgrounds A,C,D,F versus target B,E,G. Published environmental shift is pre-mixed audio-only background identity at the same −4dB SNR, not varying SNR and not corresponding injected IMU environmental noise. The normal-training audio probe supports that distinction; it was not an anomaly model test.

Measured microphone duration is RA 7.499937s and BM 5.198938s, with IMU roughly 7.499930s/5.199911s. Sensor start spreads are about 0.036ms RA and 16.660ms BM; common intersection about 7.4999s/5.1823s, coverage0.99999/0.99343. After35ms gyro warm-up, useful duration is about 7.4649s/5.1473s, producing74/51 complete100ms windows. Discarded fractions are approximately 0.47%/1.01%. Co-clocked timestamp alignment is the repository interpretation; hardware clock/register/firmware evidence is not available to prove the acquisition mechanism.

Measured artifacts: gyro startup standard deviation12–47 times later values; BM accelerometer index0 rail values around±15.988832g in 1430/1593 files (89.8%), absent on RA. The initial37/40 sampled finding was superseded by the full scan. Primary preprocessing deterministically drops index0 in **all** BM accelerometer segments, independent of labels or observed clipping. BM IMU intervals are bimodal, approximately 2.7% at122.5µs and 97.3% at140.6µs. FIFO/firmware timing is a hypothesis, not established cause. Measured rates are consistent with documented IMU internal-frequency adjustment (roughly+40/+47 register values), but the actual registers are missing: do not claim that explanation is proven. Raw audit recorded no unreadable files, NaNs/Infs, constant channels, duplicate/nonmonotonic timestamps, gaps or orphan triplets; that is historical full-audit evidence, not a fresh recovery scan.

Accelerometer units are g and gyro mdps; microphone physical calibration/unit is UNKNOWN. BM audio magnitude reaches about±19.2 versus RA±1.65, with gain/scaling explanation unresolved.

RA anomaly vocabulary is `loosescrewsA`,116 per domain. BM has `belt`39/42 and `magnetic`39/36 source/target. Paper/record wording differs on loosened versus removed arm screws, unresolved. `_corr` appears on85 RA normal segments from 6 parents (3 source,3 target;77 corrN and 8 corr_N forms). It is an inert filename marker of unknown meaning, not a fault label or repair authorization. The earlier segment-level association test was inappropriate for siblings; the recording-level Fisher result was p≈0.59. A roughly 0.007% rate difference is consistent with session differences but does not prove a correction process. Canonical filenames preserve the marker; the processed-window schema does not provide a separate dedicated corr field.

BM has 1593 separate recordings: all 1437 normals occur10:20:21–16:40:50 and all 156 anomalies16:48:07–18:22:40 on23 April2024, a7.3min gap. Fault types also occupy separate later blocks. Fault versus acquisition-time/session effects are **not identifiable**. A future normal-only early/late drift diagnostic cannot deconfound or establish absence of anomaly-session drift.

## 4. Leakage history and final v3 roles

RA 2303 excerpts come from 101 continuous physical parents (median 16, maximum 48 segments).62 parents cross the official train/test split, covering1837 segments.212/232 normal test segments have a training sibling, versus 0/232 anomalies because training is normal-only. Timestamp checks confirm adjacent excerpts within a continuous run. This asymmetric normal-class familiarity creates a credible optimistic-bias mechanism, but neither the sign/magnitude of actual metric change nor inflation of published AUC has been demonstrated. A controlled same-baseline comparison is still needed; a raw Protocol-A/B difference also changes preprocessing/cohort and cannot by itself isolate leakage causally.

Protocol A preserves the pinned authors' release split and processing for historical comparability, with explicit leakage caveats. Protocol B is the primary grouped benchmark. Phase 2 v1 fixed parent leakage but left unequal S/T tests: RA 589 versus 481, BM 330 versus 312. Phase 2.1 moved adaptation-parent closure out of evaluation and froze identical S/T tests (RA 466/BM 312) and source validation. It mistakenly treated all 123 RA adaptation-parent excerpts as training examples. Phase 2.2 distinguished27 released adaptation examples from 96 siblings that must only be blocked. The v2 common-test cohort stayed unchanged.

| Final v3 role | RA segments / parents | BM segments / parents | Consumed by |
| --- | --- | --- | --- |
| source_training |1528 /44|1137 /1137|S and T train|
| source_validation |186 /6|126 /126|same source-normal calibration in S/T|
| target_adaptation |27 /18|18 /18|T train only; S unused|
| adaptation_sibling_blocked |96 /18|0 /0|neither fit, scaler, validation, calibration nor evaluation|
| common_evaluation |466 /33|312 /312|same test identities in S/T|

Blocked RA parents are the **same18** adaptation parents, not18 additional independent groups. S train totals1528/1137; T1555/1155. Test classes are RA 234 normal (214 source+20 target),232 anomaly (116+116); BM 156 normal/156 anomaly,78 per class per domain. Source validation RA 6 parents must never be described as the18 adaptation parents. Split seed 1995; current manifest is `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv`, SHA256 `68e5406d63e96cf5ee418ae58808f5c3e7790889ba1b8497420642d1f0d8409e`. Counts and parent disjointness were checked directly in recovery; its Git blob is unchanged from Phase 2.2 through current HEAD.

## 5. Actual preprocessing and safety implementation

Protocol A uses `configs/preprocessing/official_baseline_reproduction.json`: release Regime T,100ms right-inclusive timestamp slicing then pad/truncate at16k/7063/7063Hz (1600/706/706 samples),35ms gyro crop, no BM accelerometer correction, per-window per-channel standardization with population sigma+1e-5. Padded timestamps are NaN, not invented physical observations. Official strict pairwise AUC gives ties zero credit. Pinned upstream commit is `592bcbbed58893dc26bbab3240c5d7a3105a1a7f`. Processing and AUC compatibility helpers exist; a completed upstream autoencoder reproduction/training result does not. The original upstream window-level validation behavior is documented, not evidence of an implemented complete reproduction runner.

Protocol B uses `mechasight_few_shot_regime_s.json` / `_t.json`, version `phase2.2-v3`, with half-open100ms non-overlapping windows within the common physical-time intersection, startup transformations first and native-rate samples retained. No universal sensor upsampling or common fixed-size fusion tensor is built. Time features use native samples. Only frequency features are linearly interpolated to frozen near-native machine/sensor grids, intersection-only and no extrapolation. No anti-alias filter is applied at these near-native rates; a future downsampling choice needs separate justified filtering. Linear interpolation is not new physical information or a proven cure for irregular acquisition.

Implemented normalization options: `none`, deterministic `std_window`, and train-fitted per-channel `train_zscore` with recorded fitting IDs and non-train-fit rejection. Primary Phase 3.1 uses **no window-signal normalization**, then median each feature within each segment, followed by population feature-wise training-only zscore, epsilon 1e-5. Separate S/T training fits and separate machine fits; no validation/test statistics fit the model scaler.

98 features =7 channels×14: mean, standard deviation, RMS, variance, absolute peak, peak-to-peak, skewness, excess kurtosis, crest factor, sample energy; plus dominant non-DC frequency, one-sided Parseval spectral energy, centroid and spread. Configurable band-energy helper exists but is **not** in the98-feature primary representation. Features exclude labels, conditions and provenance from estimator input. Temporal representations currently mean ragged waveform storage/access, not a trained temporal network.

HDF5 `hdf5_ragged_v1` caches stream per-modality float32 values, float64 timestamps, offsets and lengths; readers open existing files read-only. Writers use exclusive creation, configuration-bound paths/attributes and identity validation. JSONL window manifests retain raw references, machine, segment/parent/window IDs, physical bounds, domain/condition/label audit metadata, counts and transformations, protocol/regime/split/data-pool/version/config identity. Processed v1/v2 caches remain historical and must not be consumed as v3. Current primary segment features are 8 train/validation parquet artifacts under `data/interim/phase31_primary/` and 4 common-test artifacts under `data/interim/phase32_common_test/`, each with metadata hashes. They are ignored local artifacts, not fitted estimator checkpoints.

Split/data guards check consuming roles, protocol/version/regime, exact frozen identities, uniqueness, normal-only training and parent isolation; non-consuming reserve/blocked roles cannot be materialized. The estimator-facing training object has features/reproducibility IDs but no anomaly labels. Threshold calibration verifies experiment identity, source domain, source_validation role, normal labels and direction. Result bundles and ledger use immutable identities/exclusive/atomic writes. These enforce the official workflow, not a security boundary against arbitrary scripts.

Important qualification: fixed frequency reference rates originate in the release-wide acquisition audit, before model-performance testing. They are not freshly estimated per training cohort. This differs from train-fitted model normalization and from prose suggesting all reference rates are train-derived. Also, the old R11 statement that only test metadata was inspected is not a faithful account of full signal-quality audits and later common-test smoke processing. Neither inconsistency is silently repaired here.

## 6. Phase 3.1 exact model freeze

28 configurations =2 machines×2 regimes×(1 statistical+5 IF+1 OCSVM). REAL TRAINING=YES; SOURCE-NORMAL VALIDATION=YES; COMMON TEST=NO **within Phase 3.1**. The ledger was0 before/after, as recorded in its phase report. No validation model ranking or hyperparameter search was performed according to the frozen selection rationale; validation was descriptive and used for q90 calibration. All experiment/model IDs and numeric thresholds are copied in Appendix C, with complete bundle configs/provenance in JSON.

| Family | Frozen implementation / parameters | Higher-is-more-anomalous score |
| --- | --- | --- |
| Statistical |RMS distance from training mean after feature zscore; no covariance fitting; deterministic canonical seed 0|sqrt(mean(z²))|
| Isolation Forest |sklearn1.9.0;200 estimators;max_samples auto;contamination auto;max_features1.0;bootstrap false;n_jobs1;random_state0,1,2,3,4|negative score_samples|
| One-Class SVM |RBF;gamma scale;nu0.05;tol0.001;shrinking true;cache_size512MB;max_iter−1;sklearn defaults degree3,coef0=0 (inactive for RBF);canonical seed 0|negative decision_function|

Each configuration's threshold is `numpy.quantile(source_validation_scores,0.90,method='higher')`, anomaly iff **score > threshold**. It is not an optimized F1 threshold or IF contamination default. Each seed has its own frozen threshold. No fitted estimator was serialized: Phase 3.2 first re-fit and required exact recorded validation scores, thresholds and identities before scoring test. Recovery verifies intact evidence of that reproduction but does not perform another fit.

Phase 3.1 generation commit: `78b536304d94d50afcc293348c5fa0c34e729f62`. Bundle `code_git_sha=3bf046dd8edb29d99f1d246b33f88674a5b53a2d` denotes the accepted base, **not** the final generation commit. Phase 3.1.1 explicitly resolves that semantic issue without rewriting28 bundles. Six-file source snapshot hash: `93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229`. It covers classical.py, phase31.py, data.py, pretest.py, preprocessing_primary_v1.json and baseline_ladder_v1.json; unmodified transitive code is anchored by the accepted base and complete generation tree. Hash construction uses Windows-native relative path strings; the identity is not a portable all-dependency environment hash.

## 7. Phase 3.1.1 reporting freeze

Commit `fefaa341df1d8f75afb4fcde8d68ffc1571bd783`; policy `p311-reporting-5e76ab7dabd5689c7b0ced68`; policy file SHA256 `a1f695d31a6042619bad3e4fa9103a62b32c41a699c54ce7d9ae35ce36b7da2c`. The historical policy status string PRETEST_FROZEN_NOT_YET_EXECUTED is retained as a freeze record, not the current project state. This phase accessed no common-test features/scores/metrics, ran no real bootstrap and left ledger 0. “Unopened” here means no model-performance test yet; it does not erase earlier acquisition audits or smoke access.

AUROC and Average Precision are primary; standardized McClish pAUROC at maxFPR 0.10 is secondary (not raw partial area). Confusion counts, FPR, recall, precision and F1 use the frozen thresholds. Separate complete per-machine, source-domain and target-domain slices; healthy-target counts/FP/FPR; no cross-machine pooling. IF primary family metric is arithmetic mean of all 5 seed metrics, **not** metric of averaged scores or best seed. Report every seed, population SD, minimum and maximum; deterministic models are single canonical runs.

10,000 class-stratified cluster-bootstrap draws, seed 1995,95% percentile intervals; RA physical parent with every child retained, BM one-to-one recording/segment. Same sampled units and multiplicities across seeds, regimes and compared models; calculate each IF seed's metric then mean inside each replicate. Class-only strata replace the earlier domain×class×operating-condition rule. Inference is conditional on this observed evaluation cohort and fixed seeds. Source/target slices are descriptive, not independently bootstrapped confirmatory conclusions. Contrasts are T−S and statistical−IF, statistical−OCSVM, IF−OCSVM within machine/regime. No p-values or unregistered statistical-significance claims. Marginal interval overlap is not the paired comparison rule.

## 8. Phase 3.2 first-test state and ledger

Starting HEAD `fefaa341df1d8f75afb4fcde8d68ffc1571bd783`; evaluator committed first at `99f6055156c314bbf26a847713d26792979fd050`; results committed at current `a576c9df8000dd1c0a5ad6fd73845f97facf98b2`. Batch `p32-test-batch-8f7d48324fd10df97ae29100`. Ledger0 before,28 entries added atomically before the first model scorer,28 after.28 records represent28 prespecified configurations/seeds in **one first-test batch**, not28 rounds of adaptive selection. Current entries and timestamps are copied in Appendix G.

RA common-test identity hash `23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da` (466 segments). BM `e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad` (312). Hashes sort unique segment IDs, newline-join without trailing newline, SHA256. S/T identities are equal per machine; differing feature artifact hashes reflect regime/provenance, not different cohort membership.

Persisted reproduction check states PASS_28_OF_28_EXACT. Recovery checks every score file's IDs, parents, domain and labels against the current frozen common-test manifest, intact config equality versus Phase 3.1, all 56 bundle manifests,155 Phase 3.2 artifact-manifest entries and 12 feature artifact hashes/metadata. Train/validation/blocked identities are absent from evaluation; v3 membership is Git-identical to Phase 2.2. Frozen code/configs/thresholds and reporting policy remain unchanged. Reports and source path support no best-seed selection, no post-test threshold fitting and no configuration retuning in this recorded run. This is repository evidence, not proof that no unrecorded activity ever occurred outside it.

All12 primary model-family results with AUROC/AP/pAUROC95%CIs, FPR/recall/precision/F1, all 5 IF seeds, source/target slices, healthy-target false positives,36 primary bootstrap interval rows,18 S/T delta rows and 36 model-family contrast rows are preserved below and in JSON. No result, bootstrap distribution or threshold was recalculated during recovery.

## 9. WHAT INFORMATION IS NOW POST-TEST?

The common test **HAS been opened** for performance evaluation. Known information includes every individual score and confusion count, all seed/family metrics and seed variability, domain/healthy-target behavior, absolute and paired uncertainty intervals, adaptation deltas and model-family differences, together with cohort prevalence/composition and timing/provenance. Anyone reading this handoff or the result artifacts is now exposed to those outcomes.

Do not change IF thresholds after seeing zero RA recall; OCSVM nu after BM results; features after target results; window length/stride/aggregation; selected seeds; adaptation allocation; preprocessing; or evaluation membership and call the resulting performance an untouched primary confirmation. Any outcome-inspired change needs explicit new version/provenance and **post-test exploratory/sensitivity** labeling, or a genuinely new independent confirmation cohort. Reusing saved scores for a new analysis can still be post-test analysis even without a new model-score ledger entry. Never delete/rebase away prior results or reset the ledger to make the test appear unused.

## 10. Supported interpretation, not optimization

**OBSERVATION:** no universal baseline winner. RA IF has the highest pooled AUROC point estimates (S0.9137,T0.9038), but every RA IF seed has zero frozen-threshold recall. Good ranking does not ensure useful q90 alarm behavior. Many paired IF-versus-OCSVM intervals include zero, so point ranks are not universal superiority evidence. RA T-OCSVM target AUROC0.7905/AP0.9519 versus S0.5793/0.8256; its healthy-target FPR falls0.60→0.30, still substantial. Target AP reflects116 anomalies versus only 20 normals.

**OBSERVATION:** BM T-OCSVM pooled AUROC0.8648 versus S0.7054, Δ0.1594 [0.1259,0.1964]; APΔ0.2262 [0.1937,0.2534]. BM IF S→T deteriorates in AUROC/AP/pAUROC; RA IF AUROC/AP also decrease. These are paired benchmark estimates, not proof of a general causal adaptation effect. Healthy-target BM false-positive rates remain high (roughly 0.79–1.00 for these rows), including T-OCSVM0.8974. Exact results and intervals are in Appendix D.

**SUPPORTED INTERPRETATION:** adaptation and method choice behave differently by machine and metric; ranking, low-FPR partial ranking and frozen-threshold operation must be reported separately. A benchmark result may expose both domain shift and calibration mismatch. Uncertainty can be wide with RA parent clusters and limited healthy-target observations.

**NOT YET JUSTIFIED:** isolated mechanical-fault detection on BM, general factory success, robust missing-sensor handling, fusion benefit, sensor attribution, deep-model advantage, best operational threshold, novel mechanism, or measured published-AUC inflation. BM class/session confounding is mandatory wherever BM is reported. Intervals do not include new-site/machine uncertainty, training-sample uncertainty or a distribution over random seed choices; no multiplicity-controlled testing procedure was added.

## 11. Repository health and discrepancies requiring honest preservation

Recovery full suite: `python -m pytest -q -p no:cacheprovider`, bytecode writes disabled; **264 passed,0 failed,0 skipped; no warnings emitted;56.33s**. These are current executed tests, not backfilled historical counts. The existing tests use synthetic/temporary scientific fixtures and read-only repository integrity checks; no common-test evaluator was run. Historical counts are only reported when phase artifacts support them; Phase 3.1/3.1.1 executed checkpoint totals are UNKNOWN here.

Integrity is verified by hashes and persisted metadata, not a fresh end-to-end scientific rerun. Current project dependencies match requirements.txt pins on Python3.14.4. `pip check` exits1 because the shared global Python has missing Google ADK/Cloud dependencies and starlette/websockets incompatibilities unrelated to the MechaSight pinned set. No installation or repair performed. Therefore “tests and frozen artifact integrity pass” is justified; “entire global environment dependency-clean” is not.

The artifact manifest omits itself by design:155 listed files /156 files in Phase 3.2.28 pretest bundles×5 hashed payloads and 28 result bundles×4 payloads also verify. Existing primary references, smoke paths and JSON parsing are checked in Appendix E/JSON; a directory inventory does not establish archival backup of ignored raw data. Legacy planned names such as `configs/splits/`, `src/data/splits.py`, `tests/test_splits.py` and `tests/test_no_leakage.py` are not the current implementation paths; actual split code is `src/preprocessing/splits.py` with phase-specific tests and manifests in reports. They are documentation drift, not missing required primary inputs. No external link reachability audit was performed.

Stale text remains intentionally untouched: README/research_definition/supervisor report say no models/results; preprocessing docs leave choices open that Phase 3.1 froze; risk-register bootstrap/seed/significance language conflicts with later scoped policy; some planned paths were never created. Old v1/v2 configs are historical and not safe primary entry points. The synopsis DOCX still contains the absolute “sufficient for the complete proposed scope” sentence, so its presence must not be interpreted as proof that every requested final correction was completed. Its PDF and renders exist, but approval/page-count/visual correctness were not re-audited or certified in this repository recovery.

One reporting convention needs independent review: `src/experiment/metrics.py` returns F1 null whenever precision/recall are undefined **or their sum is zero**. Thus some TP0/error cases produce null where a count-form F1 convention would give0. The recorded report also propagates undefined IF family precision when not all seed precisions are defined. Recovery preserves all first-test values; it does not recalculate or silently “fix” them. Any later reporting correction must retain the original record and make its convention change explicit.

## 12. Important artifact map

| Purpose | Existing authoritative location |
| --- | --- |
| Research definition / primary-source record |`docs/research_definition.md`; `docs/source_verification.md`|
| Risks / assumptions / protocol |`docs/leakage_risks.md`; `docs/assumptions_and_open_questions.md`; `docs/evaluation_protocol.md`|
| Full data audit / downloads / startup investigation |`reports/dataset_audit/`|
| Processing implementation/doc |`src/preprocessing/`; `src/features/`; `docs/preprocessing_phase2.md`|
| Phase 2 /2.1 /2.2 history |`reports/phase2/`; `reports/phase2_1/`; `reports/phase2_2/`|
| Current split and role proofs |`reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv`; `phase22_validation.json`; `blocked_sibling_isolation_proof.csv`|
| Safety / correction |`docs/experiment_safety_phase3.md`; `reports/phase3_0/`; `src/experiment/`|
| Model freeze |`docs/phase31_pretest_freeze.md`; `reports/phase3_1/`; `configs/phase3/preprocessing_primary_v1.json`; `baseline_ladder_v1.json`|
| Reporting/code-identity freeze |`docs/phase311_first_test_reporting_freeze.md`; `reports/phase3_1_1/`; `configs/phase3/first_test_reporting_policy_v1.json`|
| Primary preprocessing pair |`configs/preprocessing/mechasight_few_shot_regime_s.json`; `_t.json`|
| Train/validation feature artifacts (ignored) |`data/interim/phase31_primary/{machine}/{S,T}/{training,source_validation}.parquet` and metadata|
| Common-test feature artifacts (ignored) |`data/interim/phase32_common_test/{machine}/{S,T}/common_test.parquet` and metadata|
|28 immutable validation bundles |`reports/phase3_1/validation_only/{experiment_id}/`|
|28 test result bundles / score parquet |`reports/phase3_2/experiments/{experiment_id}/`|
| Primary/full first-test reports |`reports/phase3_2/primary_baselines.csv`; `phase32_first_test_report.md`; `.json`|
| Bootstrap / deltas / contrasts |`reports/phase3_2/bootstrap_confidence_intervals.csv`; `s_vs_t_adaptation_deltas.csv`; `model_family_paired_differences.csv`|
| All-seed/domain/healthy-target tables |remaining named CSVs in `reports/phase3_2/` (all embedded in recovery JSON)|
| Ledger |`reports/experiment_results/test_look_ledger.jsonl`|
| Supervisor report (untracked DOCX, ignored PDF) |`reports/supervisor/MechaSight_Project_Direction_and_Implementation_Report.docx`; `.pdf`|
| Synopsis (untracked DOCX, ignored PDF) |`reports/synopsis/MechaSight_MAIT_Synopsis_2026.docx`; `.pdf`|
| Prior document builders / render evidence |`tmp/build_synopsis.py`; `tmp/audit_synopsis.py`; `tmp/synopsis_a11y.json`; render/assets subdirectories|

## 13. Research governance and historical development plan

Research decisions are grounded in measured data, primary sources and inspected implementation rather than earlier planning assumptions. Simple methods may outperform advanced ones, and fusion remains a hypothesis. Reporting excludes invented metrics, hidden test tuning, favourable seed selection and selective corruption reporting. The experimental protocol preserves normal-only fitting, parent independence, train-fitted transforms and immutable raw files. Research-critical logic resides in tested modules and versioned configurations; notebooks support exploration. Experiment identities, hashes and test-look records provide traceability, with independent review at research checkpoints. A demonstration represents measured behaviour rather than simulated claims of success.

Previously planned before first performance test: Protocol-A reproduction with disclosure; genuine single-modality comparators; reconstruction models; justified embedding-distance comparator; temporal CNN-AE/1D CNN before considering TCN; equal versus reliability-aware fusion; prespecified noise/bias/gain/clipping/dropout/stuck/missing-sensor stress tests; attribution; practicality/latency; replay; optional separately validated live audio; final paper/report. XGBoost/LightGBM require a scientifically justified label-free objective, not a supervised classifier trained on test labels. Corruption severity grids and exact future model parameters are not recovered as frozen executable experiments.

At the time of this recovery, further work depended on review of the snapshot, separately agreed handling of local-only artifacts, resolution of reporting discrepancies and a defined experimental scope. Choices informed by the first common-test results require post-test labelling or independent confirmation. A previously planned model remains exploratory on an exposed cohort when its detailed design is chosen after those results. The current descriptive roadmap is maintained in the [README](../README.md#roadmap).

## 14. Context not recoverable from the repository

**CRITICAL before continuation:** any absent next-phase authorization; exact unfrozen future reliability/model/corruption design; an independently reserved confirmation dataset; informal approvals that are not in Git/artifacts. Do not invent these from remembered conversation. Current scientific checkpoint is recoverable; continued experimental authority is not implied.

**USEFUL BUT NON-CRITICAL:** lost review dialogue and rejected alternatives beyond commit notes; exact historical test executions where not logged; corr meaning, microphone calibration and acquisition firmware/registers; final faculty designations and institutional mapping approval; authoritative final approval of untracked document revisions. Names or dates present in an artifact do not establish independent verification or supervisor consent.

**COSMETIC / CHAT-ONLY:** conversational wording, presentation preferences and discarded aesthetic options not retained in files. They need not block scientific recovery and must not be fabricated.

## 15. Recovery deliverables and verification boundary

Only `docs/PROJECT_STATE_RECOVERY.md` and `reports/recovery/project_state_recovery.json` are added by this task. Existing tracked/untracked scientific and administrative files are preserved. No Git commit, branch change, environment repair, raw rewrite, experiment rerun or next-phase implementation is authorized or performed. Full evidence tables follow; JSON retains full-precision numeric values as originally stored and original CSV field strings for lossless audit traceability.

<!-- GENERATED EVIDENCE APPENDICES -->
## Appendix A — Complete Git and phase evidence

### Phase 0

Starting commit: `EMPTY REPOSITORY`. Ending commit: `5c23f8af9a7c1bbdfb21e68b81d470131ba669b5`.

**Purpose:** Research definition, literature verification, leakage register and per-machine hypotheses.

**Real data and training:** No real data or models; source literature/code inspected.

**Checkpoint tests:** UNKNOWN: no checkpoint test-run log found.

**Validation and test exposure:** No validation or common-test data access.

**Decisions discoveries corrections:** Marginal-CI decision rules and overstated novelty withdrawn; tree objectives deferred.

**Unresolved:** Dataset assumptions required real-data audit.

Exact introduced/changed paths (outputs included; historical files preserved):

- `.gitignore`
- `README.md`
- `app/.gitkeep`
- `configs/.gitkeep`
- `data/interim/.gitkeep`
- `data/processed/.gitkeep`
- `data/raw/.gitkeep`
- `docs/assumptions_and_open_questions.md`
- `docs/leakage_risks.md`
- `docs/research_definition.md`
- `docs/source_verification.md`
- `notebooks/.gitkeep`
- `reports/experiment_results/.gitkeep`
- `reports/figures/.gitkeep`
- `reports/tables/.gitkeep`
- `requirements.txt`
- `scripts/.gitkeep`
- `src/__init__.py`
- `src/data/__init__.py`
- `src/evaluation/__init__.py`
- `src/explainability/__init__.py`
- `src/features/__init__.py`
- `src/models/__init__.py`
- `src/preprocessing/__init__.py`
- `src/utils/__init__.py`
- `tests/.gitkeep`

### Phase 1

Starting commit: `5c23f8af9a7c1bbdfb21e68b81d470131ba669b5`. Ending commit: `99df4caea7af542fe38bf6d2706f9be54da64baa`.

**Purpose:** Download/checksum/extract and audit both machines.

**Real data and training:** Full released metadata and signal-quality audit; no anomaly model trained.

**Checkpoint tests:** UNKNOWN at Phase-1 end; Phase-2 report later identifies 77 pre-existing tests.

**Validation and test exposure:** Official test metadata and raw signal summaries accessed for audit; no model-performance test look. Normal-only audio probe.

**Decisions discoveries corrections:** RA parent overlap, BM session confound, rates/durations/startup artifacts discovered.

**Unresolved:** Nominal-rate wording, corr inference and full startup incidence needed correction.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `docs/assumptions_and_open_questions.md`
- `docs/leakage_risks.md`
- `docs/research_definition.md`
- `docs/source_verification.md`
- `pytest.ini`
- `reports/dataset_audit/audio_noise_probe.json`
- `reports/dataset_audit/audio_noise_probe_features.csv`
- `reports/dataset_audit/audit_findings.json`
- `reports/dataset_audit/channel_quality.csv`
- `reports/dataset_audit/condition_class_balance.csv`
- `reports/dataset_audit/count_comparison.csv`
- `reports/dataset_audit/dataset_summary.md`
- `reports/dataset_audit/download_provenance.json`
- `reports/dataset_audit/duration_summary.csv`
- `reports/dataset_audit/extraction_report.json`
- `reports/dataset_audit/label_condition_counts.csv`
- `reports/dataset_audit/quality_report.csv`
- `reports/dataset_audit/recording_times.csv`
- `reports/dataset_audit/sampling_rate_summary.csv`
- `reports/dataset_audit/segment_manifest.csv`
- `reports/dataset_audit/timestamp_alignment_summary.csv`
- `scripts/audio_noise_probe.py`
- `scripts/audit_imadds.py`
- `scripts/download_imadds.sh`
- `scripts/extract_imadds.py`
- `scripts/verify_download.py`
- `src/data/audit.py`
- `src/data/balance.py`
- `src/data/discovery.py`
- `src/data/metadata.py`
- `src/data/temporal.py`
- `src/data/timestamps.py`
- `tests/test_balance.py`
- `tests/test_metadata.py`
- `tests/test_temporal.py`
- `tests/test_timestamps.py`

### Phase 1.5

Starting commit: `99df4caea7af542fe38bf6d2706f9be54da64baa`. Ending commit: `9cd86101b08c150320f2f0694cb03d7d59b42942`.

**Purpose:** Correct scientific overclaims and freeze Protocol A/B distinction.

**Real data and training:** Real-data startup/corr investigation; no anomaly model.

**Checkpoint tests:** UNKNOWN: no checkpoint execution log found.

**Validation and test exposure:** Audit data, including released test artifact investigation; no performance evaluation.

**Decisions discoveries corrections:** Nominal is not effective rate; clock-register explanation unproven; 1430/1593 BM startup incidence; recording-level corr test; bias not measured.

**Unresolved:** Implement recording-disjoint preprocessing; BM confound remains unidentifiable.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `docs/assumptions_and_open_questions.md`
- `docs/evaluation_protocol.md`
- `docs/leakage_risks.md`
- `docs/research_definition.md`
- `reports/dataset_audit/accelerometer_artefact_scan.csv`
- `reports/dataset_audit/artefact_investigation.json`
- `reports/dataset_audit/dataset_summary.md`
- `scripts/investigate_artifacts.py`

### Phase 2

Starting commit: `9cd86101b08c150320f2f0694cb03d7d59b42942`. Ending commit: `1d0fc18b8f6b92da31f39d03cbab331883534948`.

**Purpose:** Implement aligned streaming preprocessing, features and v1 grouped splits.

**Real data and training:** Four train-segment smoke builds, 250 windows; no model trained.

**Checkpoint tests:** 104 passed, as reported in phase2_report.md.

**Validation and test exposure:** Training smoke only in this phase; metadata-derived test memberships.

**Decisions discoveries corrections:** Leakage-safe parents but unequal S/T cohorts remained: RA 589/481 and BM 330/312 test segments.

**Unresolved:** S/T comparisons invalid on unequal test cohorts; corrected next.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `configs/preprocessing/mechasight_regime_s.json`
- `configs/preprocessing/mechasight_regime_t.json`
- `configs/preprocessing/official_baseline_reproduction.json`
- `docs/assumptions_and_open_questions.md`
- `docs/evaluation_protocol.md`
- `docs/preprocessing_phase2.md`
- `docs/research_definition.md`
- `docs/source_verification.md`
- `reports/dataset_audit/dataset_summary.md`
- `reports/phase2/phase2_report.md`
- `reports/phase2/phase2_validation.json`
- `reports/phase2/processed_manifest_schema.json`
- `reports/phase2/protocol_b_candidate_statistics.csv`
- `reports/phase2/protocol_b_split_design.json`
- `reports/phase2/protocol_b_split_statistics.csv`
- `reports/phase2/protocol_b_split_v1.csv`
- `reports/phase2/upstream_baseline_provenance.json`
- `requirements.txt`
- `scripts/build_phase2_splits.py`
- `scripts/preprocess_imadds.py`
- `scripts/validate_phase2.py`
- `src/evaluation/auc.py`
- `src/features/extraction.py`
- `src/preprocessing/cache.py`
- `src/preprocessing/config.py`
- `src/preprocessing/manifest.py`
- `src/preprocessing/normalization.py`
- `src/preprocessing/pipeline.py`
- `src/preprocessing/signals.py`
- `src/preprocessing/splits.py`
- `tests/test_phase2_auc.py`
- `tests/test_phase2_cache_manifest_config.py`
- `tests/test_phase2_features_normalization.py`
- `tests/test_phase2_signals.py`
- `tests/test_phase2_splits.py`

### Phase 2.1

Starting commit: `1d0fc18b8f6b92da31f39d03cbab331883534948`. Ending commit: `74f6d2984b4efc34a89cdd582abcd35cc12059f6`.

**Purpose:** Freeze common evaluation and source-normal validation for S/T.

**Real data and training:** Four common-test smoke builds, 250 windows; no model trained.

**Checkpoint tests:** 122 passed, as reported in phase21_report.md.

**Validation and test exposure:** Common-test raw windows/features accessed in smoke checks; no model metrics.

**Decisions discoveries corrections:** Matched RA466/BM312 test; RA adaptation-parent closure123 incorrectly used as T training budget.

**Unresolved:** Separate released27 from96 blocked siblings.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `configs/evaluation/protocol_b_common_test_v2.json`
- `configs/preprocessing/mechasight_common_test_regime_s.json`
- `configs/preprocessing/mechasight_common_test_regime_t.json`
- `docs/assumptions_and_open_questions.md`
- `docs/evaluation_protocol.md`
- `docs/leakage_risks.md`
- `docs/preprocessing_phase2.md`
- `docs/research_definition.md`
- `reports/phase2_1/matched_test_identity_proof.csv`
- `reports/phase2_1/phase21_report.md`
- `reports/phase2_1/phase21_validation.json`
- `reports/phase2_1/processed_manifest_extension.json`
- `reports/phase2_1/protocol_b_common_test_v2.csv`
- `reports/phase2_1/protocol_b_common_test_v2_counts.csv`
- `reports/phase2_1/protocol_b_common_test_v2_design.json`
- `reports/phase2_1/target_adaptation_expansion.csv`
- `reports/phase2_1/target_adaptation_parent_groups.csv`
- `scripts/build_phase21_splits.py`
- `scripts/preprocess_imadds.py`
- `scripts/validate_phase21.py`
- `src/evaluation/policy.py`
- `src/preprocessing/pipeline.py`
- `src/preprocessing/splits.py`
- `tests/test_phase21_common_cohort.py`

### Phase 2.2

Starting commit: `74f6d2984b4efc34a89cdd582abcd35cc12059f6`. Ending commit: `92c6bb52734292f1646bdfa02650f65fd0997c6d`.

**Purpose:** Freeze released few-shot budget, v3 memberships and blocked-sibling isolation.

**Real data and training:** Six smoke builds,375 windows; no model trained.

**Checkpoint tests:** 141 passed;24 protocol assertions PASS in phase22 report/validation.

**Validation and test exposure:** Four common-test smoke builds and two released-adaptation smoke builds; no model metrics.

**Decisions discoveries corrections:** RA27 consumed,96 blocked; BM18 consumed,0 blocked. v2 test unchanged.

**Unresolved:** Safety infrastructure, primary representation and model/reporting freezes still required.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `configs/evaluation/protocol_b_few_shot_adaptation_v3.json`
- `configs/preprocessing/mechasight_few_shot_regime_s.json`
- `configs/preprocessing/mechasight_few_shot_regime_t.json`
- `docs/evaluation_protocol.md`
- `docs/leakage_risks.md`
- `docs/preprocessing_phase2.md`
- `reports/phase2_2/adaptation_budget.csv`
- `reports/phase2_2/blocked_sibling_isolation_proof.csv`
- `reports/phase2_2/phase22_report.md`
- `reports/phase2_2/phase22_validation.json`
- `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv`
- `reports/phase2_2/protocol_b_few_shot_adaptation_v3_counts.csv`
- `reports/phase2_2/protocol_b_few_shot_adaptation_v3_design.json`
- `reports/phase2_2/protocol_role_counts.csv`
- `scripts/build_phase22_splits.py`
- `scripts/preprocess_imadds.py`
- `scripts/validate_phase22.py`
- `src/preprocessing/pipeline.py`
- `src/preprocessing/splits.py`
- `tests/test_phase22_adaptation_budget.py`

### Phase 3.0

Starting commit: `92c6bb52734292f1646bdfa02650f65fd0997c6d`. Ending commit: `4c0be89203e228fb8093b0e387646c52c3f0cfe5`.

**Purpose:** Experiment identities, label-free estimator contracts, thresholds, ledger/gate and bundles.

**Real data and training:** Synthetic estimator/gate/bootstrap tests; real frozen-manifest preflight only; no real fit.

**Checkpoint tests:** 210 passed,0 failed/skipped/warnings,13.97s in phase30_report.md.

**Validation and test exposure:** No real validation scoring or common-test performance.

**Decisions discoveries corrections:** Phase2.2 acceptance recorded; guards implemented. Initial bootstrap overstratification still needed correction.

**Unresolved:** Resolve bootstrap strata and provenance parent counts.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `configs/phase3/baseline_ladder_v1.json`
- `configs/phase3/preprocessing_candidates_v1.json`
- `docs/evaluation_protocol.md`
- `docs/experiment_safety_phase3.md`
- `docs/leakage_risks.md`
- `docs/research_definition.md`
- `reports/phase2_2/phase22_report.md`
- `reports/phase3_0/phase30_report.md`
- `requirements.txt`
- `src/experiment/__init__.py`
- `src/experiment/aggregation.py`
- `src/experiment/bootstrap.py`
- `src/experiment/config.py`
- `src/experiment/constants.py`
- `src/experiment/data.py`
- `src/experiment/gate.py`
- `src/experiment/ledger.py`
- `src/experiment/metrics.py`
- `src/experiment/provenance.py`
- `src/experiment/scores.py`
- `src/experiment/store.py`
- `src/preprocessing/pipeline.py`
- `tests/test_phase30_data_contract.py`
- `tests/test_phase30_evaluation.py`
- `tests/test_phase30_experiment_contract.py`

### Phase 3.0.1

Starting commit: `4c0be89203e228fb8093b0e387646c52c3f0cfe5`. Ending commit: `3bf046dd8edb29d99f1d246b33f88674a5b53a2d`.

**Purpose:** Correct bootstrap/inference and scientific cohort provenance.

**Real data and training:** Synthetic tests and frozen metadata only; no real model fit.

**Checkpoint tests:** 222 passed,0 failed/skipped/warnings,13.88s in phase301_correction_report.md.

**Validation and test exposure:** No real validation scoring or test-performance look.

**Decisions discoveries corrections:** Class-only cluster strata; distinguish RA6 validation parents from18 adaptation parents; mandatory BM confound.

**Unresolved:** Pre-test model choices and fitting not yet performed.

Exact introduced/changed paths (outputs included; historical files preserved):

- `README.md`
- `configs/phase3/baseline_ladder_v1.json`
- `configs/phase3/preprocessing_candidates_v1.json`
- `docs/experiment_safety_phase3.md`
- `reports/phase3_0/phase301_correction_report.md`
- `reports/phase3_0/phase30_report.md`
- `src/experiment/__init__.py`
- `src/experiment/bootstrap.py`
- `src/experiment/config.py`
- `src/experiment/constants.py`
- `src/experiment/store.py`
- `tests/test_phase301_scientific_consistency.py`
- `tests/test_phase30_evaluation.py`
- `tests/test_phase30_experiment_contract.py`

### Phase 3.1

Starting commit: `3bf046dd8edb29d99f1d246b33f88674a5b53a2d`. Ending commit: `78b536304d94d50afcc293348c5fa0c34e729f62`.

**Purpose:** First real classical fits and source-normal q90 threshold freeze.

**Real data and training:** 28 real training configurations; validation summaries and thresholds; no saved fitted estimators.

**Checkpoint tests:** UNKNOWN executed checkpoint count: no test-run count in committed phase report.

**Validation and test exposure:** REAL TRAINING YES; SOURCE-NORMAL VALIDATION YES; COMMON TEST NO during this phase; ledger0.

**Decisions discoveries corrections:** 100ms/98-feature segment medians, train-only zscore; stat/IF/OCSVM frozen prospectively.

**Unresolved:** Clarify accepted-base code_git_sha versus generation commit; reporting policy must precede test.

Exact introduced/changed paths (outputs included; historical files preserved):

- `configs/phase3/baseline_ladder_v1.json`
- `configs/phase3/preprocessing_candidates_v1.json`
- `configs/phase3/preprocessing_primary_v1.json`
- `docs/phase31_pretest_freeze.md`
- `reports/phase3_1/phase31_pretest_freeze_report.json`
- `reports/phase3_1/phase31_pretest_freeze_report.md`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/config.json`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/provenance.json`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/threshold.json`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/validation_summary.json`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/config.json`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/provenance.json`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/threshold.json`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/validation_summary.json`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/config.json`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/provenance.json`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/threshold.json`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/validation_summary.json`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/config.json`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/provenance.json`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/threshold.json`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/validation_summary.json`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/config.json`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/provenance.json`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/threshold.json`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/validation_summary.json`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/config.json`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/provenance.json`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/threshold.json`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/validation_summary.json`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/config.json`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/provenance.json`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/threshold.json`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/validation_summary.json`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/config.json`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/provenance.json`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/threshold.json`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/validation_summary.json`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/config.json`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/provenance.json`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/threshold.json`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/validation_summary.json`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/config.json`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/provenance.json`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/threshold.json`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/validation_summary.json`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/config.json`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/provenance.json`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/threshold.json`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/validation_summary.json`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/config.json`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/provenance.json`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/threshold.json`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/validation_summary.json`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/config.json`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/provenance.json`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/threshold.json`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/validation_summary.json`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/config.json`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/provenance.json`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/threshold.json`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/validation_summary.json`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/config.json`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/provenance.json`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/threshold.json`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/validation_summary.json`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/config.json`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/provenance.json`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/threshold.json`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/validation_summary.json`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/config.json`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/provenance.json`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/threshold.json`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/validation_summary.json`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/config.json`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/provenance.json`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/threshold.json`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/validation_summary.json`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/config.json`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/provenance.json`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/threshold.json`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/validation_summary.json`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/config.json`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/provenance.json`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/threshold.json`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/validation_summary.json`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/config.json`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/provenance.json`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/threshold.json`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/validation_summary.json`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/config.json`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/provenance.json`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/threshold.json`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/validation_summary.json`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/config.json`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/provenance.json`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/threshold.json`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/validation_summary.json`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/config.json`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/provenance.json`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/threshold.json`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/validation_summary.json`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/config.json`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/provenance.json`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/threshold.json`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/validation_summary.json`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/config.json`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/provenance.json`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/threshold.json`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/validation_summary.json`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/config.json`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/provenance.json`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/threshold.json`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/validation_summary.json`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/bundle_manifest.json`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/config.json`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/provenance.json`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/threshold.json`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/validation_scores.csv`
- `reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/validation_summary.json`
- `requirements.txt`
- `scripts/prepare_phase31_features.py`
- `scripts/run_phase31_baselines.py`
- `src/experiment/constants.py`
- `src/experiment/data.py`
- `src/experiment/phase31.py`
- `src/experiment/pretest.py`
- `src/models/classical.py`
- `tests/test_phase31_baselines.py`

### Phase 3.1.1

Starting commit: `78b536304d94d50afcc293348c5fa0c34e729f62`. Ending commit: `fefaa341df1d8f75afb4fcde8d68ffc1571bd783`.

**Purpose:** Freeze first-test reporting and audit composite code provenance.

**Real data and training:** Existing28 bundles verified; no refit.

**Checkpoint tests:** UNKNOWN executed checkpoint count: no test-run count in committed phase report.

**Validation and test exposure:** No common-test metadata/features/scoring accessed in this phase; ledger remains0.

**Decisions discoveries corrections:** IF mean5; fixed paired bootstrap, metrics, domain/healthy-target/delta/comparison reporting; code provenance clarified.

**Unresolved:** Lock evaluator before authorized first performance test.

Exact introduced/changed paths (outputs included; historical files preserved):

- `configs/phase3/first_test_reporting_policy_v1.json`
- `docs/phase311_first_test_reporting_freeze.md`
- `docs/phase31_pretest_freeze.md`
- `reports/phase3_1_1/experiment_code_identity_audit.json`
- `reports/phase3_1_1/experiment_code_identity_audit.md`
- `reports/phase3_1_1/phase311_pretest_freeze_report.json`
- `src/evaluation/reporting.py`
- `tests/test_phase311_reporting_freeze.py`

### Phase 3.2

Starting commit: `fefaa341df1d8f75afb4fcde8d68ffc1571bd783`. Ending commit: `a576c9df8000dd1c0a5ad6fd73845f97facf98b2`.

**Purpose:** Lock evaluator, exactly reproduce pretest fits, then first common-test performance evaluation.

**Real data and training:** 28 train/validation refits recorded exact, followed by28 authorized test scorings; no retuning.

**Checkpoint tests:** Current recovery full suite:264 passed,0 failed,0 skipped,no warnings emitted; historical execution count not separately logged in result report.

**Validation and test exposure:** Common-test features/scores/metrics/bootstrap accessed; ledger0+28=28 in one batch.

**Decisions discoveries corrections:** Recorded per-machine, all-seed/domain/healthy-target/paired results; no universal winner.

**Unresolved:** Independent review; all result-inspired changes are now post-test. No next phase authorized.

Exact introduced/changed paths (outputs included; historical files preserved):

- `reports/experiment_results/test_look_ledger.jsonl`
- `reports/phase3_2/artifact_manifest.json`
- `reports/phase3_2/bootstrap_confidence_intervals.csv`
- `reports/phase3_2/common_test_cohort_verification.json`
- `reports/phase3_2/domain_slice_metrics_family.csv`
- `reports/phase3_2/domain_slice_metrics_individual.csv`
- `reports/phase3_2/evaluation_state.json`
- `reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/config.json`
- `reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/metrics.json`
- `reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/provenance.json`
- `reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/scores.parquet`
- `reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/config.json`
- `reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/metrics.json`
- `reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/provenance.json`
- `reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/scores.parquet`
- `reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/config.json`
- `reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/metrics.json`
- `reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/provenance.json`
- `reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/scores.parquet`
- `reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/config.json`
- `reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/metrics.json`
- `reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/provenance.json`
- `reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/scores.parquet`
- `reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/config.json`
- `reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/metrics.json`
- `reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/provenance.json`
- `reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/scores.parquet`
- `reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/config.json`
- `reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/metrics.json`
- `reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/provenance.json`
- `reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/scores.parquet`
- `reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/config.json`
- `reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/metrics.json`
- `reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/provenance.json`
- `reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/scores.parquet`
- `reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/config.json`
- `reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/metrics.json`
- `reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/provenance.json`
- `reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/scores.parquet`
- `reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/config.json`
- `reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/metrics.json`
- `reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/provenance.json`
- `reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/scores.parquet`
- `reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/config.json`
- `reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/metrics.json`
- `reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/provenance.json`
- `reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/scores.parquet`
- `reports/phase3_2/experiments/p3-649152f9076af141b8701f69/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-649152f9076af141b8701f69/config.json`
- `reports/phase3_2/experiments/p3-649152f9076af141b8701f69/metrics.json`
- `reports/phase3_2/experiments/p3-649152f9076af141b8701f69/provenance.json`
- `reports/phase3_2/experiments/p3-649152f9076af141b8701f69/scores.parquet`
- `reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/config.json`
- `reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/metrics.json`
- `reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/provenance.json`
- `reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/scores.parquet`
- `reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/config.json`
- `reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/metrics.json`
- `reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/provenance.json`
- `reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/scores.parquet`
- `reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/config.json`
- `reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/metrics.json`
- `reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/provenance.json`
- `reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/scores.parquet`
- `reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/config.json`
- `reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/metrics.json`
- `reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/provenance.json`
- `reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/scores.parquet`
- `reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/config.json`
- `reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/metrics.json`
- `reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/provenance.json`
- `reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/scores.parquet`
- `reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/config.json`
- `reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/metrics.json`
- `reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/provenance.json`
- `reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/scores.parquet`
- `reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/config.json`
- `reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/metrics.json`
- `reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/provenance.json`
- `reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/scores.parquet`
- `reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/config.json`
- `reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/metrics.json`
- `reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/provenance.json`
- `reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/scores.parquet`
- `reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/config.json`
- `reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/metrics.json`
- `reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/provenance.json`
- `reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/scores.parquet`
- `reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/config.json`
- `reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/metrics.json`
- `reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/provenance.json`
- `reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/scores.parquet`
- `reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/config.json`
- `reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/metrics.json`
- `reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/provenance.json`
- `reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/scores.parquet`
- `reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/config.json`
- `reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/metrics.json`
- `reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/provenance.json`
- `reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/scores.parquet`
- `reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/config.json`
- `reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/metrics.json`
- `reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/provenance.json`
- `reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/scores.parquet`
- `reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/config.json`
- `reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/metrics.json`
- `reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/provenance.json`
- `reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/scores.parquet`
- `reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/config.json`
- `reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/metrics.json`
- `reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/provenance.json`
- `reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/scores.parquet`
- `reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/config.json`
- `reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/metrics.json`
- `reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/provenance.json`
- `reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/scores.parquet`
- `reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/bundle_manifest.json`
- `reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/config.json`
- `reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/metrics.json`
- `reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/provenance.json`
- `reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/scores.parquet`
- `reports/phase3_2/healthy_target_false_positives_family.csv`
- `reports/phase3_2/healthy_target_false_positives_individual.csv`
- `reports/phase3_2/individual_experiment_metrics.csv`
- `reports/phase3_2/isolation_forest_seed_summaries.csv`
- `reports/phase3_2/model_family_paired_differences.csv`
- `reports/phase3_2/phase32_first_test_report.json`
- `reports/phase3_2/phase32_first_test_report.md`
- `reports/phase3_2/pretest_reproduction_check.json`
- `reports/phase3_2/primary_baselines.csv`
- `reports/phase3_2/s_vs_t_adaptation_deltas.csv`
- `scripts/run_phase32_first_test.py`
- `src/experiment/ledger.py`
- `src/experiment/phase32.py`
- `tests/test_phase32_evaluator.py`

### All commits

| sha | parent | date | subject |
| --- | --- | --- | --- |
| 5ba61c88c485614fd74b540d2764643c2f70b0fc | UNDEFINED | 2026-08-21T18:19:35+05:30 | Phase 0: research freeze - definition, source verification, leakage register |
| b25735464b338fd0554cf8f97b79471395547e80 | 5ba61c88c485614fd74b540d2764643c2f70b0fc | 2026-08-21T18:32:17+05:30 | Phase 0 review corrections: paired inference, deferred tree models, sourced novelty |
| 5c23f8af9a7c1bbdfb21e68b81d470131ba669b5 | b25735464b338fd0554cf8f97b79471395547e80 | 2026-08-21T18:54:02+05:30 | Phase 0 audit cleanup: purge marginal-CI language, freeze inference, split hypotheses |
| 904060ef1eac7a6be898899c189bdb527f5deea8 | 5c23f8af9a7c1bbdfb21e68b81d470131ba669b5 | 2026-08-21T19:29:26+05:30 | Phase 1: reproducible IMAD-DS dataset auditor |
| 1d5b3a9ca57611b80d696006882de2bdd3c7d9d8 | 904060ef1eac7a6be898899c189bdb527f5deea8 | 2026-08-21T20:02:55+05:30 | Phase 1: audit both IMAD-DS machines; find split leakage and rate errors |
| 99df4caea7af542fe38bf6d2706f9be54da64baa | 1d5b3a9ca57611b80d696006882de2bdd3c7d9d8 | 2026-08-21T20:04:45+05:30 | Phase 1: fold audit findings into the leakage register and README |
| 9cd86101b08c150320f2f0694cb03d7d59b42942 | 99df4caea7af542fe38bf6d2706f9be54da64baa | 2026-08-21T23:28:19+05:30 | Phase 1.5: correct overstated claims, freeze the evaluation protocol |
| 1d0fc18b8f6b92da31f39d03cbab331883534948 | 9cd86101b08c150320f2f0694cb03d7d59b42942 | 2026-08-22T00:15:05+05:30 | Phase 2: add leakage-safe preprocessing pipeline |
| 74f6d2984b4efc34a89cdd582abcd35cc12059f6 | 1d0fc18b8f6b92da31f39d03cbab331883534948 | 2026-08-22T01:31:36+05:30 | Phase 2.1: freeze matched S/T evaluation cohort |
| 92c6bb52734292f1646bdfa02650f65fd0997c6d | 74f6d2984b4efc34a89cdd582abcd35cc12059f6 | 2026-08-23T16:39:06+05:30 | Phase 2.2: separate consumed adaptation budget from blocked siblings |
| 4c0be89203e228fb8093b0e387646c52c3f0cfe5 | 92c6bb52734292f1646bdfa02650f65fd0997c6d | 2026-09-01T18:22:57+05:30 | Phase 3.0: freeze experiment safety and evaluation harness |
| 3bf046dd8edb29d99f1d246b33f88674a5b53a2d | 4c0be89203e228fb8093b0e387646c52c3f0cfe5 | 2026-09-01T20:22:04+05:30 | Phase 3.0.1: correct inference and cohort provenance |
| 78b536304d94d50afcc293348c5fa0c34e729f62 | 3bf046dd8edb29d99f1d246b33f88674a5b53a2d | 2026-09-03T10:17:48+05:30 | Phase 3.1: implement and freeze pre-test classical baselines |
| fefaa341df1d8f75afb4fcde8d68ffc1571bd783 | 78b536304d94d50afcc293348c5fa0c34e729f62 | 2026-09-03T10:32:44+05:30 | Phase 3.1.1: freeze first-test reporting policy |
| 99f6055156c314bbf26a847713d26792979fd050 | fefaa341df1d8f75afb4fcde8d68ffc1571bd783 | 2026-09-03T14:29:44+05:30 | Phase 3.2: lock common-test evaluator |
| a576c9df8000dd1c0a5ad6fd73845f97facf98b2 | 99f6055156c314bbf26a847713d26792979fd050 | 2026-09-03T14:38:42+05:30 | Phase 3.2: record first locked common-test results |

## Appendix B — Direct manifest counts and conditions

| machine | split | domain | anomaly_label | segments |
| --- | --- | --- | --- | --- |
| BrushlessMotor | test | source | belt | 39 |
| BrushlessMotor | test | source | magnetic | 39 |
| BrushlessMotor | test | source | normal | 78 |
| BrushlessMotor | test | target | belt | 42 |
| BrushlessMotor | test | target | magnetic | 36 |
| BrushlessMotor | test | target | normal | 78 |
| BrushlessMotor | train | source | normal | 1263 |
| BrushlessMotor | train | target | normal | 18 |
| RoboticArm | test | source | loosescrewsA | 116 |
| RoboticArm | test | source | normal | 116 |
| RoboticArm | test | target | loosescrewsA | 116 |
| RoboticArm | test | target | normal | 116 |
| RoboticArm | train | source | normal | 1812 |
| RoboticArm | train | target | normal | 27 |

| machine | domain | operating_conditions | background_conditions |
| --- | --- | --- | --- |
| BrushlessMotor | source | ["spd1500", "spd1600", "spd1700", "spd1800", "spd1900", "spd2000", "spd2400", "spd2800", "spd3000"] | ["BckgA", "BckgC", "BckgD", "BckgF"] |
| BrushlessMotor | target | ["spd1000", "spd1100", "spd1200", "spd1300", "spd1400"] | ["BckgB", "BckgE", "BckgG"] |
| RoboticArm | source | ["weight00", "weight10", "weight15", "weight20"] | ["BckgA", "BckgC", "BckgD", "BckgF"] |
| RoboticArm | target | ["weight25", "weight30", "weight35"] | ["BckgB", "BckgE", "BckgG"] |

| machine | regime | roles | parent_counts | train_segments | validation_segments | test_segments |
| --- | --- | --- | --- | --- | --- | --- |
| RoboticArm | S | {"adaptation_sibling_blocked": 96, "common_evaluation": 466, "source_training": 1528, "source_validation": 186, "target_adaptation": 27} | {"adaptation_sibling_blocked": 18, "common_evaluation": 33, "source_training": 44, "source_validation": 6, "target_adaptation": 18} | 1528 | 186 | 466 |
| RoboticArm | T | {"adaptation_sibling_blocked": 96, "common_evaluation": 466, "source_training": 1528, "source_validation": 186, "target_adaptation": 27} | {"adaptation_sibling_blocked": 18, "common_evaluation": 33, "source_training": 44, "source_validation": 6, "target_adaptation": 18} | 1555 | 186 | 466 |
| BrushlessMotor | S | {"common_evaluation": 312, "source_training": 1137, "source_validation": 126, "target_adaptation": 18, "adaptation_sibling_blocked": 0} | {"common_evaluation": 312, "source_training": 1137, "source_validation": 126, "target_adaptation": 18, "adaptation_sibling_blocked": 0} | 1137 | 126 | 312 |
| BrushlessMotor | T | {"common_evaluation": 312, "source_training": 1137, "source_validation": 126, "target_adaptation": 18, "adaptation_sibling_blocked": 0} | {"common_evaluation": 312, "source_training": 1137, "source_validation": 126, "target_adaptation": 18, "adaptation_sibling_blocked": 0} | 1155 | 126 | 312 |

## Appendix C — Every frozen model identity and exact q90 threshold

These are copied from the Phase-3.1 report, checked against intact bundles; no threshold was refitted. Full configs, threshold source IDs and provenance are embedded in the recovery JSON.

| machine | regime | family | seed | experiment_id | model_id | training_segments | validation_segments | feature_count | threshold_q90 | bundle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoboticArm | S | statistical_reference | 0 | p3-994e171b9c84299ee0506001 | p31-model-6e84ada532ccd25ea25b-4bbbbfa1 | 1528 | 186 | 98 | 1.991521141869127 | reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001 |
| RoboticArm | S | isolation_forest | 0 | p3-b7df07b4435147ce84abb2fb | p31-model-bcc23071948443594ad8-bd048884 | 1528 | 186 | 98 | 0.5768855112688169 | reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb |
| RoboticArm | S | isolation_forest | 1 | p3-d523942f6e3b110247ccd97a | p31-model-bcc23071948443594ad8-fbe47f6d | 1528 | 186 | 98 | 0.5819109785048906 | reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a |
| RoboticArm | S | isolation_forest | 2 | p3-d2e84d3ccdbb25446ed3b5c2 | p31-model-bcc23071948443594ad8-a61e75dc | 1528 | 186 | 98 | 0.5913373314434514 | reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2 |
| RoboticArm | S | isolation_forest | 3 | p3-61da2e22429837caf51172a3 | p31-model-bcc23071948443594ad8-4d687b64 | 1528 | 186 | 98 | 0.5823327632633399 | reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3 |
| RoboticArm | S | isolation_forest | 4 | p3-4bad66197b1b5b39bc0e1ce8 | p31-model-bcc23071948443594ad8-ef731a0a | 1528 | 186 | 98 | 0.5775036253711889 | reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8 |
| RoboticArm | S | one_class_svm | 0 | p3-c0b106088a660b45248207a8 | p31-model-856bdf04029b9ce21634-4bbbbfa1 | 1528 | 186 | 98 | 1.30732841470026 | reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8 |
| RoboticArm | T | statistical_reference | 0 | p3-00ef3593eef3eef458aac8ec | p31-model-6e84ada532ccd25ea25b-4bbbbfa1 | 1555 | 186 | 98 | 1.9787165942843699 | reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec |
| RoboticArm | T | isolation_forest | 0 | p3-f2e82658337725d643461f5a | p31-model-bcc23071948443594ad8-bd048884 | 1555 | 186 | 98 | 0.5662207915958274 | reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a |
| RoboticArm | T | isolation_forest | 1 | p3-4df2732b72997882e300c7ee | p31-model-bcc23071948443594ad8-fbe47f6d | 1555 | 186 | 98 | 0.5709038009785159 | reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee |
| RoboticArm | T | isolation_forest | 2 | p3-e0cb98b1d50e1b760ad00c33 | p31-model-bcc23071948443594ad8-a61e75dc | 1555 | 186 | 98 | 0.5923908784813829 | reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33 |
| RoboticArm | T | isolation_forest | 3 | p3-efee4a25a307b9556f1ef9e1 | p31-model-bcc23071948443594ad8-4d687b64 | 1555 | 186 | 98 | 0.5733348394350062 | reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1 |
| RoboticArm | T | isolation_forest | 4 | p3-b2c1eb73ec1f836d95401be0 | p31-model-bcc23071948443594ad8-ef731a0a | 1555 | 186 | 98 | 0.5786469310394343 | reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0 |
| RoboticArm | T | one_class_svm | 0 | p3-3dfb6a47e2b1ee1a4da8ed4f | p31-model-856bdf04029b9ce21634-4bbbbfa1 | 1555 | 186 | 98 | 1.0695964696815548 | reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f |
| BrushlessMotor | S | statistical_reference | 0 | p3-c3480b35881d464a75f3b359 | p31-model-6e84ada532ccd25ea25b-4bbbbfa1 | 1137 | 126 | 98 | 1.4321164988399824 | reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359 |
| BrushlessMotor | S | isolation_forest | 0 | p3-d75e7a94e44f1caf8a5a595a | p31-model-bcc23071948443594ad8-bd048884 | 1137 | 126 | 98 | 0.5184849328299442 | reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a |
| BrushlessMotor | S | isolation_forest | 1 | p3-611ddf562a1bccc1714e44b8 | p31-model-bcc23071948443594ad8-fbe47f6d | 1137 | 126 | 98 | 0.517708659275098 | reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8 |
| BrushlessMotor | S | isolation_forest | 2 | p3-296f1214c5d29239fd3f95a7 | p31-model-bcc23071948443594ad8-a61e75dc | 1137 | 126 | 98 | 0.5151139612807567 | reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7 |
| BrushlessMotor | S | isolation_forest | 3 | p3-92d37c41321bf4991c224a7b | p31-model-bcc23071948443594ad8-4d687b64 | 1137 | 126 | 98 | 0.5209415050902623 | reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b |
| BrushlessMotor | S | isolation_forest | 4 | p3-a164ab66f27ae30b8964b5d5 | p31-model-bcc23071948443594ad8-ef731a0a | 1137 | 126 | 98 | 0.5136590099992835 | reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5 |
| BrushlessMotor | S | one_class_svm | 0 | p3-c72840a71b6f3d36ed7a30d0 | p31-model-856bdf04029b9ce21634-4bbbbfa1 | 1137 | 126 | 98 | -0.2593140099053226 | reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0 |
| BrushlessMotor | T | statistical_reference | 0 | p3-f0f1a4a48cc815e2faf3d696 | p31-model-6e84ada532ccd25ea25b-4bbbbfa1 | 1155 | 126 | 98 | 1.4313577387177736 | reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696 |
| BrushlessMotor | T | isolation_forest | 0 | p3-a2b4c42e73f2ac6104e622c8 | p31-model-bcc23071948443594ad8-bd048884 | 1155 | 126 | 98 | 0.5143011075856394 | reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8 |
| BrushlessMotor | T | isolation_forest | 1 | p3-649152f9076af141b8701f69 | p31-model-bcc23071948443594ad8-fbe47f6d | 1155 | 126 | 98 | 0.5083941010963748 | reports/phase3_1/validation_only/p3-649152f9076af141b8701f69 |
| BrushlessMotor | T | isolation_forest | 2 | p3-56aaef3d39745f6f79059405 | p31-model-bcc23071948443594ad8-a61e75dc | 1155 | 126 | 98 | 0.5138554087984037 | reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405 |
| BrushlessMotor | T | isolation_forest | 3 | p3-93869e664dd39a5c050ce499 | p31-model-bcc23071948443594ad8-4d687b64 | 1155 | 126 | 98 | 0.5151069572443019 | reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499 |
| BrushlessMotor | T | isolation_forest | 4 | p3-5205733640c5094021b2a4b0 | p31-model-bcc23071948443594ad8-ef731a0a | 1155 | 126 | 98 | 0.5118176833685242 | reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0 |
| BrushlessMotor | T | one_class_svm | 0 | p3-3c27f8c1e723c409051319c8 | p31-model-856bdf04029b9ce21634-4bbbbfa1 | 1155 | 126 | 98 | -0.3267186962081494 | reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8 |

## Appendix D — ALL recorded Phase-3.2 results (not recomputed)

The following is a verbatim local-project report copy. UNDEFINED values and mean-five-seed conventions are preserved, including the F1 convention noted above. BrushlessMotor session confounding applies to every BM row.

### Recorded first-test report

Test evaluator commit: `99f6055156c314bbf26a847713d26792979fd050`. Reporting policy: `p311-reporting-5e76ab7dabd5689c7b0ced68`.

The 28 pre-registered configurations reproduced exactly before the test was opened. The ledger received exactly 28 entries atomically before the first model score.

## Table A — RoboticArm primary baselines

| Regime | Model | AUROC (95% CI) | AP (95% CI) | pAUROC@0.10 (95% CI) | FPR | Recall | Precision | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| S | statistical_reference | 0.8522 [0.6523, 0.9527] | 0.8091 [0.6171, 0.9425] | 0.6224 [0.4932, 0.8472] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| S | isolation_forest | 0.9137 [0.7736, 0.9759] | 0.8654 [0.7181, 0.9682] | 0.7126 [0.5108, 0.9052] | 0.0000 | 0.0000 | UNDEFINED | UNDEFINED |
| S | one_class_svm | 0.8985 [0.7831, 0.9589] | 0.8614 [0.7229, 0.9554] | 0.7019 [0.4950, 0.8798] | 0.0556 | 0.5560 | 0.9085 | 0.6898 |
| T | statistical_reference | 0.8457 [0.6419, 0.9508] | 0.8038 [0.6066, 0.9419] | 0.6185 [0.4930, 0.8440] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| T | isolation_forest | 0.9038 [0.7558, 0.9713] | 0.8528 [0.6988, 0.9631] | 0.7001 [0.5000, 0.8935] | 0.0017 | 0.0000 | UNDEFINED | UNDEFINED |
| T | one_class_svm | 0.8932 [0.7876, 0.9531] | 0.8933 [0.7666, 0.9574] | 0.7383 [0.5875, 0.8663] | 0.0256 | 0.4440 | 0.9450 | 0.6041 |

## Table B — BrushlessMotor primary baselines

| Regime | Model | AUROC (95% CI) | AP (95% CI) | pAUROC@0.10 (95% CI) | FPR | Recall | Precision | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| S | statistical_reference | 0.5988 [0.5337, 0.6614] | 0.5843 [0.5302, 0.6469] | 0.5085 [0.4906, 0.5457] | 0.4423 | 0.6474 | 0.5941 | 0.6196 |
| S | isolation_forest | 0.6549 [0.5917, 0.7149] | 0.6340 [0.5819, 0.6920] | 0.5458 [0.5224, 0.5756] | 0.5256 | 0.8423 | 0.6157 | 0.7113 |
| S | one_class_svm | 0.7054 [0.6429, 0.7642] | 0.6099 [0.5554, 0.6785] | 0.5010 [0.4863, 0.5358] | 0.5192 | 0.9551 | 0.6478 | 0.7720 |
| T | statistical_reference | 0.5933 [0.5278, 0.6558] | 0.6120 [0.5569, 0.6704] | 0.5436 [0.5174, 0.5787] | 0.4359 | 0.5705 | 0.5669 | 0.5687 |
| T | isolation_forest | 0.5961 [0.5267, 0.6637] | 0.5089 [0.4709, 0.5638] | 0.4812 [0.4755, 0.4929] | 0.4885 | 0.7808 | 0.6151 | 0.6878 |
| T | one_class_svm | 0.8648 [0.8208, 0.9046] | 0.8362 [0.7840, 0.8867] | 0.5969 [0.5624, 0.6938] | 0.4679 | 0.9487 | 0.6697 | 0.7851 |

## Domain-slice ranking metrics

| machine | regime | model_family | domain | sample_count | status | auroc | average_precision | standardized_pauroc_fpr_0_10 |
|---|---|---|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference | source | 330 | defined | 0.8107 | 0.6486 | 0.5610 |
| RoboticArm | S | statistical_reference | target | 136 | defined | 0.5677 | 0.8213 | 0.4737 |
| RoboticArm | S | isolation_forest | source | 330 | defined | 0.9078 | 0.8254 | 0.7050 |
| RoboticArm | S | isolation_forest | target | 136 | defined | 0.6747 | 0.8711 | 0.4805 |
| RoboticArm | S | one_class_svm | source | 330 | defined | 0.8781 | 0.8213 | 0.7406 |
| RoboticArm | S | one_class_svm | target | 136 | defined | 0.5793 | 0.8256 | 0.4737 |
| RoboticArm | T | statistical_reference | source | 330 | defined | 0.8021 | 0.6398 | 0.5580 |
| RoboticArm | T | statistical_reference | target | 136 | defined | 0.5776 | 0.8247 | 0.4737 |
| RoboticArm | T | isolation_forest | source | 330 | defined | 0.8926 | 0.8058 | 0.6920 |
| RoboticArm | T | isolation_forest | target | 136 | defined | 0.6451 | 0.8572 | 0.4760 |
| RoboticArm | T | one_class_svm | source | 330 | defined | 0.8481 | 0.7729 | 0.6949 |
| RoboticArm | T | one_class_svm | target | 136 | defined | 0.7905 | 0.9519 | 0.6030 |
| BrushlessMotor | S | statistical_reference | source | 156 | defined | 0.7240 | 0.7540 | 0.6194 |
| BrushlessMotor | S | statistical_reference | target | 156 | defined | 0.4752 | 0.5388 | 0.5360 |
| BrushlessMotor | S | isolation_forest | source | 156 | defined | 0.9237 | 0.9228 | 0.7877 |
| BrushlessMotor | S | isolation_forest | target | 156 | defined | 0.4591 | 0.5067 | 0.5137 |
| BrushlessMotor | S | one_class_svm | source | 156 | defined | 0.9775 | 0.9820 | 0.9472 |
| BrushlessMotor | S | one_class_svm | target | 156 | defined | 0.5828 | 0.5322 | 0.5000 |
| BrushlessMotor | T | statistical_reference | source | 156 | defined | 0.7130 | 0.7420 | 0.6169 |
| BrushlessMotor | T | statistical_reference | target | 156 | defined | 0.4599 | 0.5585 | 0.5647 |
| BrushlessMotor | T | isolation_forest | source | 156 | defined | 0.8991 | 0.8939 | 0.7230 |
| BrushlessMotor | T | isolation_forest | target | 156 | defined | 0.3310 | 0.3973 | 0.4806 |
| BrushlessMotor | T | one_class_svm | source | 156 | defined | 0.9814 | 0.9841 | 0.9462 |
| BrushlessMotor | T | one_class_svm | target | 156 | defined | 0.8034 | 0.7649 | 0.5867 |

## Table D — Healthy-target false positives

| machine | regime | model_family | healthy_target_sample_count | false_positive | false_positive_rate |
|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference | 20.0000 | 1.0000 | 0.0500 |
| RoboticArm | S | isolation_forest | 20.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | one_class_svm | 20.0000 | 12.0000 | 0.6000 |
| RoboticArm | T | statistical_reference | 20.0000 | 1.0000 | 0.0500 |
| RoboticArm | T | isolation_forest | 20.0000 | 0.4000 | 0.0200 |
| RoboticArm | T | one_class_svm | 20.0000 | 6.0000 | 0.3000 |
| BrushlessMotor | S | statistical_reference | 78.0000 | 63.0000 | 0.8077 |
| BrushlessMotor | S | isolation_forest | 78.0000 | 76.6000 | 0.9821 |
| BrushlessMotor | S | one_class_svm | 78.0000 | 78.0000 | 1.0000 |
| BrushlessMotor | T | statistical_reference | 78.0000 | 62.0000 | 0.7949 |
| BrushlessMotor | T | isolation_forest | 78.0000 | 70.8000 | 0.9077 |
| BrushlessMotor | T | one_class_svm | 78.0000 | 70.0000 | 0.8974 |

## Table C — S vs T deltas (T minus S)

| Machine | Model | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |
|---|---|---|---|---|
| RoboticArm | statistical_reference | -0.0064 [-0.0143, 0.0004] | -0.0053 [-0.0170, 0.0031] | -0.0039 [-0.0154, 0.0112] |
| RoboticArm | isolation_forest | -0.0099 [-0.0226, -0.0025] | -0.0126 [-0.0258, -0.0031] | -0.0124 [-0.0253, 0.0004] |
| RoboticArm | one_class_svm | -0.0053 [-0.0252, 0.0321] | 0.0319 [-0.0217, 0.0972] | 0.0364 [-0.0388, 0.1674] |
| BrushlessMotor | statistical_reference | -0.0055 [-0.0151, 0.0042] | 0.0277 [0.0081, 0.0485] | 0.0351 [0.0155, 0.0540] |
| BrushlessMotor | isolation_forest | -0.0588 [-0.0813, -0.0382] | -0.1251 [-0.1534, -0.0905] | -0.0646 [-0.0911, -0.0428] |
| BrushlessMotor | one_class_svm | 0.1594 [0.1259, 0.1964] | 0.2262 [0.1937, 0.2534] | 0.0959 [0.0704, 0.1626] |

## Table E — Paired model-family differences (first minus second)

| Machine | Regime | Comparison | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |
|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference_minus_isolation_forest | -0.0616 [-0.1365, -0.0166] | -0.0563 [-0.1352, 0.0021] | -0.0901 [-0.1452, -0.0025] |
| RoboticArm | S | statistical_reference_minus_one_class_svm | -0.0463 [-0.1842, 0.0336] | -0.0523 [-0.1708, 0.0258] | -0.0795 [-0.1720, 0.0361] |
| RoboticArm | S | isolation_forest_minus_one_class_svm | 0.0152 [-0.0583, 0.0726] | 0.0040 [-0.0637, 0.0717] | 0.0107 [-0.0653, 0.1085] |
| RoboticArm | T | statistical_reference_minus_isolation_forest | -0.0581 [-0.1318, -0.0144] | -0.0490 [-0.1303, 0.0087] | -0.0816 [-0.1400, 0.0033] |
| RoboticArm | T | statistical_reference_minus_one_class_svm | -0.0475 [-0.1866, 0.0360] | -0.0895 [-0.2021, 0.0176] | -0.1198 [-0.1827, 0.0363] |
| RoboticArm | T | isolation_forest_minus_one_class_svm | 0.0106 [-0.0709, 0.0704] | -0.0405 [-0.1155, 0.0484] | -0.0382 [-0.1429, 0.0886] |
| BrushlessMotor | S | statistical_reference_minus_isolation_forest | -0.0561 [-0.0850, -0.0267] | -0.0497 [-0.0932, -0.0028] | -0.0373 [-0.0687, 0.0036] |
| BrushlessMotor | S | statistical_reference_minus_one_class_svm | -0.1067 [-0.1328, -0.0827] | -0.0256 [-0.0529, -0.0005] | 0.0075 [-0.0030, 0.0220] |
| BrushlessMotor | S | isolation_forest_minus_one_class_svm | -0.0505 [-0.0820, -0.0191] | 0.0241 [-0.0309, 0.0714] | 0.0448 [0.0049, 0.0763] |
| BrushlessMotor | T | statistical_reference_minus_isolation_forest | -0.0028 [-0.0401, 0.0369] | 0.1031 [0.0519, 0.1461] | 0.0624 [0.0328, 0.0981] |
| BrushlessMotor | T | statistical_reference_minus_one_class_svm | -0.2715 [-0.3175, -0.2280] | -0.2242 [-0.2634, -0.1877] | -0.0533 [-0.1291, -0.0312] |
| BrushlessMotor | T | isolation_forest_minus_one_class_svm | -0.2687 [-0.3277, -0.2119] | -0.3273 [-0.3766, -0.2664] | -0.1157 [-0.2114, -0.0787] |

## Isolation-Forest individual-seed appendix

| machine | regime | metric | seed_0 | seed_1 | seed_2 | seed_3 | seed_4 | mean | population_std | minimum | maximum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RoboticArm | S | auroc | 0.9116 | 0.9239 | 0.9096 | 0.9167 | 0.9069 | 0.9137 | 0.0060 | 0.9069 | 0.9239 |
| RoboticArm | S | average_precision | 0.8633 | 0.8893 | 0.8566 | 0.8648 | 0.8531 | 0.8654 | 0.0127 | 0.8531 | 0.8893 |
| RoboticArm | S | standardized_pauroc_fpr_0_10 | 0.7178 | 0.7338 | 0.7089 | 0.7061 | 0.6962 | 0.7126 | 0.0127 | 0.6962 | 0.7338 |
| RoboticArm | S | true_negative | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 0.0000 | 234.0000 | 234.0000 |
| RoboticArm | S | false_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | false_negative | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 0.0000 | 232.0000 | 232.0000 |
| RoboticArm | S | true_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | false_positive_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | precision | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | S | f1 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | T | auroc | 0.9048 | 0.9061 | 0.8985 | 0.9037 | 0.9059 | 0.9038 | 0.0028 | 0.8985 | 0.9061 |
| RoboticArm | T | average_precision | 0.8610 | 0.8587 | 0.8460 | 0.8496 | 0.8488 | 0.8528 | 0.0059 | 0.8460 | 0.8610 |
| RoboticArm | T | standardized_pauroc_fpr_0_10 | 0.7087 | 0.7066 | 0.6859 | 0.6910 | 0.7084 | 0.7001 | 0.0097 | 0.6859 | 0.7087 |
| RoboticArm | T | true_negative | 232.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 233.6000 | 0.8000 | 232.0000 | 234.0000 |
| RoboticArm | T | false_positive | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.8000 | 0.0000 | 2.0000 |
| RoboticArm | T | false_negative | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 0.0000 | 232.0000 | 232.0000 |
| RoboticArm | T | true_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | T | false_positive_rate | 0.0085 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0017 | 0.0034 | 0.0000 | 0.0085 |
| RoboticArm | T | recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | T | precision | 0.0000 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | T | f1 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| BrushlessMotor | S | auroc | 0.6254 | 0.6620 | 0.6865 | 0.6450 | 0.6556 | 0.6549 | 0.0201 | 0.6254 | 0.6865 |
| BrushlessMotor | S | average_precision | 0.5934 | 0.6497 | 0.6870 | 0.6126 | 0.6275 | 0.6340 | 0.0323 | 0.5934 | 0.6870 |
| BrushlessMotor | S | standardized_pauroc_fpr_0_10 | 0.5209 | 0.5613 | 0.5749 | 0.5294 | 0.5426 | 0.5458 | 0.0199 | 0.5209 | 0.5749 |
| BrushlessMotor | S | true_negative | 75.0000 | 74.0000 | 74.0000 | 76.0000 | 71.0000 | 74.0000 | 1.6733 | 71.0000 | 76.0000 |
| BrushlessMotor | S | false_positive | 81.0000 | 82.0000 | 82.0000 | 80.0000 | 85.0000 | 82.0000 | 1.6733 | 80.0000 | 85.0000 |
| BrushlessMotor | S | false_negative | 27.0000 | 23.0000 | 27.0000 | 27.0000 | 19.0000 | 24.6000 | 3.2000 | 19.0000 | 27.0000 |
| BrushlessMotor | S | true_positive | 129.0000 | 133.0000 | 129.0000 | 129.0000 | 137.0000 | 131.4000 | 3.2000 | 129.0000 | 137.0000 |
| BrushlessMotor | S | false_positive_rate | 0.5192 | 0.5256 | 0.5256 | 0.5128 | 0.5449 | 0.5256 | 0.0107 | 0.5128 | 0.5449 |
| BrushlessMotor | S | recall | 0.8269 | 0.8526 | 0.8269 | 0.8269 | 0.8782 | 0.8423 | 0.0205 | 0.8269 | 0.8782 |
| BrushlessMotor | S | precision | 0.6143 | 0.6186 | 0.6114 | 0.6172 | 0.6171 | 0.6157 | 0.0026 | 0.6114 | 0.6186 |
| BrushlessMotor | S | f1 | 0.7049 | 0.7170 | 0.7030 | 0.7068 | 0.7249 | 0.7113 | 0.0083 | 0.7030 | 0.7249 |
| BrushlessMotor | T | auroc | 0.5912 | 0.6034 | 0.6072 | 0.5857 | 0.5930 | 0.5961 | 0.0080 | 0.5857 | 0.6072 |
| BrushlessMotor | T | average_precision | 0.4999 | 0.5082 | 0.5206 | 0.5078 | 0.5080 | 0.5089 | 0.0066 | 0.4999 | 0.5206 |
| BrushlessMotor | T | standardized_pauroc_fpr_0_10 | 0.4761 | 0.4791 | 0.4838 | 0.4873 | 0.4796 | 0.4812 | 0.0039 | 0.4761 | 0.4873 |
| BrushlessMotor | T | true_negative | 84.0000 | 76.0000 | 84.0000 | 79.0000 | 76.0000 | 79.8000 | 3.6000 | 76.0000 | 84.0000 |
| BrushlessMotor | T | false_positive | 72.0000 | 80.0000 | 72.0000 | 77.0000 | 80.0000 | 76.2000 | 3.6000 | 72.0000 | 80.0000 |
| BrushlessMotor | T | false_negative | 42.0000 | 26.0000 | 30.0000 | 39.0000 | 34.0000 | 34.2000 | 5.8103 | 26.0000 | 42.0000 |
| BrushlessMotor | T | true_positive | 114.0000 | 130.0000 | 126.0000 | 117.0000 | 122.0000 | 121.8000 | 5.8103 | 114.0000 | 130.0000 |
| BrushlessMotor | T | false_positive_rate | 0.4615 | 0.5128 | 0.4615 | 0.4936 | 0.5128 | 0.4885 | 0.0231 | 0.4615 | 0.5128 |
| BrushlessMotor | T | recall | 0.7308 | 0.8333 | 0.8077 | 0.7500 | 0.7821 | 0.7808 | 0.0372 | 0.7308 | 0.8333 |
| BrushlessMotor | T | precision | 0.6129 | 0.6190 | 0.6364 | 0.6031 | 0.6040 | 0.6151 | 0.0122 | 0.6031 | 0.6364 |
| BrushlessMotor | T | f1 | 0.6667 | 0.7104 | 0.7119 | 0.6686 | 0.6816 | 0.6878 | 0.0197 | 0.6667 | 0.7119 |

Isolation-Forest rows are arithmetic five-seed family summaries. Full individual-seed, domain, healthy-target, bootstrap, S-vs-T, and paired-comparison tables accompany this report.

No pooled cross-machine result or universal winner is reported.

BrushlessMotor limitation `brushless_motor_acquisition_time_class_confound_v1`: fault/class and acquisition session/time cannot be fully separated, so these results are not pure isolated mechanical-fault evidence.

RoboticArm uncertainty clusters by physical parent recording. Source-normal validation was 186 segments from six parents; Regime-T adaptation was 27 released segments from 18 parents, with 96 siblings blocked.

### Full recorded individual_experiment_metrics

| experiment_id | machine | regime | model_family | model_seed | threshold | scientific_limitation_id | metric_policy_version | auroc | average_precision | standardized_pauroc_fpr_0_10 | true_negative | false_positive | false_negative | true_positive | false_positive_rate | recall | precision | f1 | decision_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3-00ef3593eef3eef458aac8ec | RoboticArm | T | statistical_reference | 0 | 1.9787165942843699 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.8457117595048629 | 0.8037991437975067 | 0.6185451471295391 | 233 | 1 | 222 | 10 | 0.004273504273504274 | 0.04310344827586207 | 0.9090909090909091 | 0.0823045267489712 | anomaly if score > threshold |
| p3-296f1214c5d29239fd3f95a7 | BrushlessMotor | S | isolation_forest | 2 | 0.5151139612807567 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6865138067061145 | 0.6870350119905405 | 0.5748555313332641 | 74 | 82 | 27 | 129 | 0.5256410256410257 | 0.8269230769230769 | 0.6113744075829384 | 0.7029972752043596 | anomaly if score > threshold |
| p3-3c27f8c1e723c409051319c8 | BrushlessMotor | T | one_class_svm | 0 | -0.3267186962081494 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.8648093359631821 | 0.836199305848611 | 0.5968718640783418 | 83 | 73 | 8 | 148 | 0.46794871794871795 | 0.9487179487179487 | 0.669683257918552 | 0.7851458885941643 | anomaly if score > threshold |
| p3-3dfb6a47e2b1ee1a4da8ed4f | RoboticArm | T | one_class_svm | 0 | 1.0695964696815548 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.8931808134394341 | 0.8933362936710438 | 0.738296337661129 | 228 | 6 | 129 | 103 | 0.02564102564102564 | 0.44396551724137934 | 0.944954128440367 | 0.6041055718475073 | anomaly if score > threshold |
| p3-4bad66197b1b5b39bc0e1ce8 | RoboticArm | S | isolation_forest | 4 | 0.5775036253711889 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9069223401119952 | 0.85307971668659 | 0.696162377650581 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-4df2732b72997882e300c7ee | RoboticArm | T | isolation_forest | 1 | 0.5709038009785159 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9061118479221928 | 0.8586804053227699 | 0.7066134611506663 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-5205733640c5094021b2a4b0 | BrushlessMotor | T | isolation_forest | 4 | 0.5118176833685242 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.5929898093359631 | 0.5079739045979514 | 0.47961002110799683 | 76 | 80 | 34 | 122 | 0.5128205128205128 | 0.782051282051282 | 0.6039603960396039 | 0.6815642458100558 | anomaly if score > threshold |
| p3-56aaef3d39745f6f79059405 | BrushlessMotor | T | isolation_forest | 2 | 0.5138554087984037 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6071663379355687 | 0.5205918078789141 | 0.48380566801619435 | 84 | 72 | 30 | 126 | 0.46153846153846156 | 0.8076923076923077 | 0.6363636363636364 | 0.7118644067796609 | anomaly if score > threshold |
| p3-611ddf562a1bccc1714e44b8 | BrushlessMotor | S | isolation_forest | 1 | 0.517708659275098 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6619822485207101 | 0.6496945621940412 | 0.5612737464964186 | 74 | 82 | 23 | 133 | 0.5256410256410257 | 0.8525641025641025 | 0.6186046511627907 | 0.7169811320754716 | anomaly if score > threshold |
| p3-61da2e22429837caf51172a3 | RoboticArm | S | isolation_forest | 3 | 0.5823327632633399 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9166666666666666 | 0.8647794126802819 | 0.7060511579567841 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-649152f9076af141b8701f69 | BrushlessMotor | T | isolation_forest | 1 | 0.5083941010963748 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6033859303090072 | 0.5082253875869563 | 0.47909097200595174 | 76 | 80 | 26 | 130 | 0.5128205128205128 | 0.8333333333333334 | 0.6190476190476191 | 0.7103825136612022 | anomaly if score > threshold |
| p3-92d37c41321bf4991c224a7b | BrushlessMotor | S | isolation_forest | 3 | 0.5209415050902623 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6450115055884287 | 0.6125806585873036 | 0.5293522267206477 | 76 | 80 | 27 | 129 | 0.5128205128205128 | 0.8269230769230769 | 0.6172248803827751 | 0.7068493150684931 | anomaly if score > threshold |
| p3-93869e664dd39a5c050ce499 | BrushlessMotor | T | isolation_forest | 3 | 0.5151069572443019 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.5857166337935569 | 0.5078062389771378 | 0.48726599536316134 | 79 | 77 | 39 | 117 | 0.4935897435897436 | 0.75 | 0.6030927835051546 | 0.6685714285714285 | anomaly if score > threshold |
| p3-994e171b9c84299ee0506001 | RoboticArm | S | statistical_reference | 0 | 1.991521141869127 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.8521588564692013 | 0.8091232981386509 | 0.6224231001907953 | 233 | 1 | 222 | 10 | 0.004273504273504274 | 0.04310344827586207 | 0.9090909090909091 | 0.0823045267489712 | anomaly if score > threshold |
| p3-a164ab66f27ae30b8964b5d5 | BrushlessMotor | S | isolation_forest | 4 | 0.5136590099992835 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6555719921104537 | 0.627479922662441 | 0.5425879788227966 | 71 | 85 | 19 | 137 | 0.5448717948717948 | 0.8782051282051282 | 0.6171171171171171 | 0.7248677248677248 | anomaly if score > threshold |
| p3-a2b4c42e73f2ac6104e622c8 | BrushlessMotor | T | isolation_forest | 0 | 0.5143011075856394 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.5911817882971729 | 0.49991146982119833 | 0.4761064396691927 | 84 | 72 | 42 | 114 | 0.46153846153846156 | 0.7307692307692307 | 0.6129032258064516 | 0.6666666666666667 | anomaly if score > threshold |
| p3-b2c1eb73ec1f836d95401be0 | RoboticArm | T | isolation_forest | 4 | 0.5786469310394343 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9058908045977011 | 0.8488046376621883 | 0.7083779297935378 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-b7df07b4435147ce84abb2fb | RoboticArm | S | isolation_forest | 0 | 0.5768855112688169 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9115642499263188 | 0.8632576031699077 | 0.7178207454976965 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-c0b106088a660b45248207a8 | RoboticArm | S | one_class_svm | 0 | 1.30732841470026 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.8985042735042736 | 0.8614251513228788 | 0.70190174818124 | 221 | 13 | 103 | 129 | 0.05555555555555555 | 0.5560344827586207 | 0.9084507042253521 | 0.6898395721925134 | anomaly if score > threshold |
| p3-c3480b35881d464a75f3b359 | BrushlessMotor | S | statistical_reference | 0 | 1.4321164988399824 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.5987836949375411 | 0.5843020178851994 | 0.5085037544551715 | 87 | 69 | 55 | 101 | 0.4423076923076923 | 0.6474358974358975 | 0.5941176470588235 | 0.6196319018404908 | anomaly if score > threshold |
| p3-c72840a71b6f3d36ed7a30d0 | BrushlessMotor | S | one_class_svm | 0 | -0.2593140099053226 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.7054363905325444 | 0.6099496307306085 | 0.5009991695214367 | 75 | 81 | 7 | 149 | 0.5192307692307693 | 0.9551282051282052 | 0.6478260869565218 | 0.772020725388601 | anomaly if score > threshold |
| p3-d2e84d3ccdbb25446ed3b5c2 | RoboticArm | S | isolation_forest | 2 | 0.5913373314434514 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9096301208370174 | 0.8565750321926244 | 0.7089208432221137 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-d523942f6e3b110247ccd97a | RoboticArm | S | isolation_forest | 1 | 0.5819109785048906 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9238874152667257 | 0.8892663258055332 | 0.7338366916406844 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-d75e7a94e44f1caf8a5a595a | BrushlessMotor | S | isolation_forest | 0 | 0.5184849328299442 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.6254109138724524 | 0.5933633789575344 | 0.5209176788124157 | 75 | 81 | 27 | 129 | 0.5192307692307693 | 0.8269230769230769 | 0.6142857142857143 | 0.7049180327868853 | anomaly if score > threshold |
| p3-e0cb98b1d50e1b760ad00c33 | RoboticArm | T | isolation_forest | 2 | 0.5923908784813829 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.8985411140583555 | 0.8460080303839898 | 0.6858664122729458 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-efee4a25a307b9556f1ef9e1 | RoboticArm | T | isolation_forest | 3 | 0.5733348394350062 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9036803713527852 | 0.8495750304190979 | 0.690985310313804 | 234 | 0 | 232 | 0 | 0.0 | 0.0 |  |  | anomaly if score > threshold |
| p3-f0f1a4a48cc815e2faf3d696 | BrushlessMotor | T | statistical_reference | 0 | 1.4313577387177736 | brushless_motor_acquisition_time_class_confound_v1 | phase3.0-protocol-b-metrics-v1 | 0.5933185404339251 | 0.611993992862768 | 0.5435828229350497 | 88 | 68 | 67 | 89 | 0.4358974358974359 | 0.5705128205128205 | 0.5668789808917197 | 0.5686900958466453 | anomaly if score > threshold |
| p3-f2e82658337725d643461f5a | RoboticArm | T | isolation_forest | 0 | 0.5662207915958274 | robotic_arm_parent_clustered_inference_v1 | phase3.0-protocol-b-metrics-v1 | 0.9047855879752431 | 0.8610320382532922 | 0.7087269455690508 | 232 | 2 | 232 | 0 | 0.008547008547008548 | 0.0 | 0.0 |  | anomaly if score > threshold |

### Full recorded domain_slice_metrics_individual

| experiment_id | machine | regime | model_family | model_seed | domain | scientific_limitation_id | sample_count | status | metric_policy_version | auroc | average_precision | standardized_pauroc_fpr_0_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3-00ef3593eef3eef458aac8ec | RoboticArm | T | statistical_reference | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8020866902997099 | 0.6398307483945931 | 0.5580253405024 |
| p3-00ef3593eef3eef458aac8ec | RoboticArm | T | statistical_reference | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.5775862068965517 | 0.8246666331456067 | 0.47368421052631576 |
| p3-296f1214c5d29239fd3f95a7 | BrushlessMotor | S | isolation_forest | 2 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9171597633136095 | 0.915202457701684 | 0.7622755112633655 |
| p3-296f1214c5d29239fd3f95a7 | BrushlessMotor | S | isolation_forest | 2 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.5261341222879684 | 0.578503999292642 | 0.5394304301186892 |
| p3-3c27f8c1e723c409051319c8 | BrushlessMotor | T | one_class_svm | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9814266929651544 | 0.9841495666023946 | 0.9461919097546627 |
| p3-3c27f8c1e723c409051319c8 | BrushlessMotor | T | one_class_svm | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.8034188034188035 | 0.7649342009722172 | 0.586663898404789 |
| p3-3dfb6a47e2b1ee1a4da8ed4f | RoboticArm | T | one_class_svm | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8480905575249759 | 0.7728997027943376 | 0.6949047610970707 |
| p3-3dfb6a47e2b1ee1a4da8ed4f | RoboticArm | T | one_class_svm | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.7905172413793103 | 0.9519377790923941 | 0.602994555353902 |
| p3-4bad66197b1b5b39bc0e1ce8 | RoboticArm | S | isolation_forest | 4 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8975588140509184 | 0.8110618411825193 | 0.6865936190783113 |
| p3-4bad66197b1b5b39bc0e1ce8 | RoboticArm | S | isolation_forest | 4 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6478448275862069 | 0.8559321054626761 | 0.47368421052631576 |
| p3-4df2732b72997882e300c7ee | RoboticArm | T | isolation_forest | 1 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8983644859813085 | 0.8161466164699703 | 0.7004172532523704 |
| p3-4df2732b72997882e300c7ee | RoboticArm | T | isolation_forest | 1 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6400862068965517 | 0.8666928995498081 | 0.48502722323049 |
| p3-5205733640c5094021b2a4b0 | BrushlessMotor | T | isolation_forest | 4 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.893326758711374 | 0.8869470430936253 | 0.7105436174262085 |
| p3-5205733640c5094021b2a4b0 | BrushlessMotor | T | isolation_forest | 4 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.34072978303747536 | 0.4029463460913889 | 0.4811239143222949 |
| p3-56aaef3d39745f6f79059405 | BrushlessMotor | T | isolation_forest | 2 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9084483892176201 | 0.9024187281147074 | 0.7359770234264161 |
| p3-56aaef3d39745f6f79059405 | BrushlessMotor | T | isolation_forest | 2 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.3394148586456279 | 0.40840975312908806 | 0.4845842416692619 |
| p3-611ddf562a1bccc1714e44b8 | BrushlessMotor | S | isolation_forest | 1 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9339250493096647 | 0.9306069034452241 | 0.7882279663656182 |
| p3-611ddf562a1bccc1714e44b8 | BrushlessMotor | S | isolation_forest | 1 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.48471400394477315 | 0.5588567915070686 | 0.5496383957922419 |
| p3-61da2e22429837caf51172a3 | RoboticArm | S | isolation_forest | 3 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.9103689977441186 | 0.8220817851801289 | 0.7133504079244195 |
| p3-61da2e22429837caf51172a3 | RoboticArm | S | isolation_forest | 3 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6637931034482759 | 0.8645442761740624 | 0.48502722323049 |
| p3-649152f9076af141b8701f69 | BrushlessMotor | T | isolation_forest | 1 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.913050624589086 | 0.908593466664924 | 0.7489532509775425 |
| p3-649152f9076af141b8701f69 | BrushlessMotor | T | isolation_forest | 1 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.3454963839579225 | 0.3952495450482202 | 0.47368421052631576 |
| p3-92d37c41321bf4991c224a7b | BrushlessMotor | S | isolation_forest | 3 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9074621959237343 | 0.9079285515251974 | 0.7742136406104018 |
| p3-92d37c41321bf4991c224a7b | BrushlessMotor | S | isolation_forest | 3 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.4211045364891519 | 0.45491590951734695 | 0.4902937817917575 |
| p3-93869e664dd39a5c050ce499 | BrushlessMotor | T | isolation_forest | 3 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.8846153846153846 | 0.8809546659295877 | 0.7079483719159833 |
| p3-93869e664dd39a5c050ce499 | BrushlessMotor | T | isolation_forest | 3 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.3076923076923077 | 0.3957346046364442 | 0.49012076542440913 |
| p3-994e171b9c84299ee0506001 | RoboticArm | S | statistical_reference | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8107476635514018 | 0.6486112910780467 | 0.5609512017232898 |
| p3-994e171b9c84299ee0506001 | RoboticArm | S | statistical_reference | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.5676724137931034 | 0.8213177020951459 | 0.47368421052631576 |
| p3-a164ab66f27ae30b8964b5d5 | BrushlessMotor | S | isolation_forest | 4 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9419789612097303 | 0.9429037373927053 | 0.8318280909374027 |
| p3-a164ab66f27ae30b8964b5d5 | BrushlessMotor | S | isolation_forest | 4 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.49457593688362916 | 0.5283387092752255 | 0.5119208277103013 |
| p3-a2b4c42e73f2ac6104e622c8 | BrushlessMotor | T | isolation_forest | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.8962853385930308 | 0.8904881747376043 | 0.7115817156302987 |
| p3-a2b4c42e73f2ac6104e622c8 | BrushlessMotor | T | isolation_forest | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.321827744904668 | 0.38425277524713175 | 0.47368421052631576 |
| p3-b2c1eb73ec1f836d95401be0 | RoboticArm | T | isolation_forest | 4 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.897599097647438 | 0.8246219667851582 | 0.7179724205777092 |
| p3-b2c1eb73ec1f836d95401be0 | RoboticArm | T | isolation_forest | 4 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6211206896551724 | 0.842212505740099 | 0.47368421052631576 |
| p3-b7df07b4435147ce84abb2fb | RoboticArm | S | isolation_forest | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.9034402191427651 | 0.8226950564618217 | 0.7066082059806299 |
| p3-b7df07b4435147ce84abb2fb | RoboticArm | S | isolation_forest | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.690948275862069 | 0.8704380320702231 | 0.47368421052631576 |
| p3-c0b106088a660b45248207a8 | RoboticArm | S | one_class_svm | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8781018369320014 | 0.8212695592549735 | 0.7406160422002476 |
| p3-c0b106088a660b45248207a8 | RoboticArm | S | one_class_svm | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.5793103448275863 | 0.8256040066344714 | 0.47368421052631576 |
| p3-c3480b35881d464a75f3b359 | BrushlessMotor | S | statistical_reference | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.7240302432610125 | 0.7540417417372045 | 0.6193639918336274 |
| p3-c3480b35881d464a75f3b359 | BrushlessMotor | S | statistical_reference | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.47518080210387903 | 0.538776399899823 | 0.5359701027717222 |
| p3-c72840a71b6f3d36ed7a30d0 | BrushlessMotor | S | one_class_svm | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9774819197896121 | 0.9820344128539525 | 0.9472300079587529 |
| p3-c72840a71b6f3d36ed7a30d0 | BrushlessMotor | S | one_class_svm | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.5827580539119 | 0.5322141378044267 | 0.5 |
| p3-d2e84d3ccdbb25446ed3b5c2 | RoboticArm | S | isolation_forest | 2 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.9082742507251047 | 0.8318034174373502 | 0.7128415624946995 |
| p3-d2e84d3ccdbb25446ed3b5c2 | RoboticArm | S | isolation_forest | 2 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6474137931034483 | 0.8574125213868999 | 0.47368421052631576 |
| p3-d523942f6e3b110247ccd97a | RoboticArm | S | isolation_forest | 1 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.9192313889784081 | 0.839535493453447 | 0.7053784961921401 |
| p3-d523942f6e3b110247ccd97a | RoboticArm | S | isolation_forest | 1 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.7232758620689655 | 0.9070263803899399 | 0.49637023593466423 |
| p3-d75e7a94e44f1caf8a5a595a | BrushlessMotor | S | isolation_forest | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.9178172255095332 | 0.9171461751303933 | 0.7818263607737292 |
| p3-d75e7a94e44f1caf8a5a595a | BrushlessMotor | S | isolation_forest | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.36900065746219596 | 0.41298120881849465 | 0.47697152150593447 |
| p3-e0cb98b1d50e1b760ad00c33 | RoboticArm | T | isolation_forest | 2 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.891435707379955 | 0.8060971422427861 | 0.6964737011720407 |
| p3-e0cb98b1d50e1b760ad00c33 | RoboticArm | T | isolation_forest | 2 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6155172413793103 | 0.8499331586948585 | 0.47368421052631576 |
| p3-efee4a25a307b9556f1ef9e1 | RoboticArm | T | isolation_forest | 3 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8918385433451499 | 0.7960054199025044 | 0.6734060416914023 |
| p3-efee4a25a307b9556f1ef9e1 | RoboticArm | T | isolation_forest | 3 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6508620689655172 | 0.8535053703167905 | 0.47368421052631576 |
| p3-f0f1a4a48cc815e2faf3d696 | BrushlessMotor | T | statistical_reference | 0 | source | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.7130177514792899 | 0.7420275047749185 | 0.6169417626907505 |
| p3-f0f1a4a48cc815e2faf3d696 | BrushlessMotor | T | statistical_reference | 0 | target | brushless_motor_acquisition_time_class_confound_v1 | 156 | defined | phase3.0-protocol-b-metrics-v1 | 0.45989480604865224 | 0.5585191776352663 | 0.5646908197515484 |
| p3-f2e82658337725d643461f5a | RoboticArm | T | isolation_forest | 0 | source | robotic_arm_parent_clustered_inference_v1 | 330 | defined | phase3.0-protocol-b-metrics-v1 | 0.8838221076377699 | 0.7861626851485732 | 0.6717946978306224 |
| p3-f2e82658337725d643461f5a | RoboticArm | T | isolation_forest | 0 | target | robotic_arm_parent_clustered_inference_v1 | 136 | defined | phase3.0-protocol-b-metrics-v1 | 0.6978448275862069 | 0.8738303374225538 | 0.47368421052631576 |

### Full recorded healthy_target_false_positives_individual

| experiment_id | machine | regime | model_family | model_seed | scientific_limitation_id | healthy_target_sample_count | false_positive | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3-00ef3593eef3eef458aac8ec | RoboticArm | T | statistical_reference | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 1 | 0.05 |
| p3-296f1214c5d29239fd3f95a7 | BrushlessMotor | S | isolation_forest | 2 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 76 | 0.9743589743589743 |
| p3-3c27f8c1e723c409051319c8 | BrushlessMotor | T | one_class_svm | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 70 | 0.8974358974358975 |
| p3-3dfb6a47e2b1ee1a4da8ed4f | RoboticArm | T | one_class_svm | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 6 | 0.3 |
| p3-4bad66197b1b5b39bc0e1ce8 | RoboticArm | S | isolation_forest | 4 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-4df2732b72997882e300c7ee | RoboticArm | T | isolation_forest | 1 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-5205733640c5094021b2a4b0 | BrushlessMotor | T | isolation_forest | 4 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 74 | 0.9487179487179487 |
| p3-56aaef3d39745f6f79059405 | BrushlessMotor | T | isolation_forest | 2 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 67 | 0.8589743589743589 |
| p3-611ddf562a1bccc1714e44b8 | BrushlessMotor | S | isolation_forest | 1 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 76 | 0.9743589743589743 |
| p3-61da2e22429837caf51172a3 | RoboticArm | S | isolation_forest | 3 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-649152f9076af141b8701f69 | BrushlessMotor | T | isolation_forest | 1 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 74 | 0.9487179487179487 |
| p3-92d37c41321bf4991c224a7b | BrushlessMotor | S | isolation_forest | 3 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 76 | 0.9743589743589743 |
| p3-93869e664dd39a5c050ce499 | BrushlessMotor | T | isolation_forest | 3 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 71 | 0.9102564102564102 |
| p3-994e171b9c84299ee0506001 | RoboticArm | S | statistical_reference | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 1 | 0.05 |
| p3-a164ab66f27ae30b8964b5d5 | BrushlessMotor | S | isolation_forest | 4 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 78 | 1.0 |
| p3-a2b4c42e73f2ac6104e622c8 | BrushlessMotor | T | isolation_forest | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 68 | 0.8717948717948718 |
| p3-b2c1eb73ec1f836d95401be0 | RoboticArm | T | isolation_forest | 4 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-b7df07b4435147ce84abb2fb | RoboticArm | S | isolation_forest | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-c0b106088a660b45248207a8 | RoboticArm | S | one_class_svm | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 12 | 0.6 |
| p3-c3480b35881d464a75f3b359 | BrushlessMotor | S | statistical_reference | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 63 | 0.8076923076923077 |
| p3-c72840a71b6f3d36ed7a30d0 | BrushlessMotor | S | one_class_svm | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 78 | 1.0 |
| p3-d2e84d3ccdbb25446ed3b5c2 | RoboticArm | S | isolation_forest | 2 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-d523942f6e3b110247ccd97a | RoboticArm | S | isolation_forest | 1 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-d75e7a94e44f1caf8a5a595a | BrushlessMotor | S | isolation_forest | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 77 | 0.9871794871794872 |
| p3-e0cb98b1d50e1b760ad00c33 | RoboticArm | T | isolation_forest | 2 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-efee4a25a307b9556f1ef9e1 | RoboticArm | T | isolation_forest | 3 | robotic_arm_parent_clustered_inference_v1 | 20 | 0 | 0.0 |
| p3-f0f1a4a48cc815e2faf3d696 | BrushlessMotor | T | statistical_reference | 0 | brushless_motor_acquisition_time_class_confound_v1 | 78 | 62 | 0.7948717948717948 |
| p3-f2e82658337725d643461f5a | RoboticArm | T | isolation_forest | 0 | robotic_arm_parent_clustered_inference_v1 | 20 | 2 | 0.1 |

### Full recorded bootstrap_confidence_intervals

| machine | regime | model_family | metric | point_estimate | scientific_limitation_id | ci_lower | ci_upper | replicates_requested | valid_replicates | invalid_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BrushlessMotor | S | isolation_forest | auroc | 0.6548980933596319 | brushless_motor_acquisition_time_class_confound_v1 | 0.5916911160420776 | 0.7149328155818541 | 10000 | 10000 | 0 |
| BrushlessMotor | S | isolation_forest | average_precision | 0.6340307068783722 | brushless_motor_acquisition_time_class_confound_v1 | 0.5818898749269129 | 0.6919896871810248 | 10000 | 10000 | 0 |
| BrushlessMotor | S | isolation_forest | standardized_pauroc_fpr_0_10 | 0.5457974324371085 | brushless_motor_acquisition_time_class_confound_v1 | 0.5223705837572235 | 0.575573765528219 | 10000 | 10000 | 0 |
| BrushlessMotor | S | one_class_svm | auroc | 0.7054363905325444 | brushless_motor_acquisition_time_class_confound_v1 | 0.6428947855029586 | 0.7641975879355687 | 10000 | 10000 | 0 |
| BrushlessMotor | S | one_class_svm | average_precision | 0.6099496307306085 | brushless_motor_acquisition_time_class_confound_v1 | 0.5553928932356204 | 0.6785028362688469 | 10000 | 10000 | 0 |
| BrushlessMotor | S | one_class_svm | standardized_pauroc_fpr_0_10 | 0.5009991695214367 | brushless_motor_acquisition_time_class_confound_v1 | 0.4862700228832952 | 0.5357549136648327 | 10000 | 10000 | 0 |
| BrushlessMotor | S | statistical_reference | auroc | 0.5987836949375411 | brushless_motor_acquisition_time_class_confound_v1 | 0.533691855687048 | 0.6613669049967127 | 10000 | 10000 | 0 |
| BrushlessMotor | S | statistical_reference | average_precision | 0.5843020178851994 | brushless_motor_acquisition_time_class_confound_v1 | 0.530201328224624 | 0.6469497340449494 | 10000 | 10000 | 0 |
| BrushlessMotor | S | statistical_reference | standardized_pauroc_fpr_0_10 | 0.5085037544551715 | brushless_motor_acquisition_time_class_confound_v1 | 0.49055330634278 | 0.5456601006955258 | 10000 | 10000 | 0 |
| BrushlessMotor | T | isolation_forest | auroc | 0.5960880999342537 | brushless_motor_acquisition_time_class_confound_v1 | 0.5266845414201183 | 0.6637247287968442 | 10000 | 10000 | 0 |
| BrushlessMotor | T | isolation_forest | average_precision | 0.5089017617724315 | brushless_motor_acquisition_time_class_confound_v1 | 0.4709223114432409 | 0.5638199470414264 | 10000 | 10000 | 0 |
| BrushlessMotor | T | isolation_forest | standardized_pauroc_fpr_0_10 | 0.4811758192324994 | brushless_motor_acquisition_time_class_confound_v1 | 0.47548358074673863 | 0.492889243572442 | 10000 | 10000 | 0 |
| BrushlessMotor | T | one_class_svm | auroc | 0.8648093359631821 | brushless_motor_acquisition_time_class_confound_v1 | 0.8207552596975674 | 0.9045857988165681 | 10000 | 10000 | 0 |
| BrushlessMotor | T | one_class_svm | average_precision | 0.836199305848611 | brushless_motor_acquisition_time_class_confound_v1 | 0.7840244499960558 | 0.8866945883852688 | 10000 | 10000 | 0 |
| BrushlessMotor | T | one_class_svm | standardized_pauroc_fpr_0_10 | 0.5968718640783418 | brushless_motor_acquisition_time_class_confound_v1 | 0.56244160697602 | 0.6938064465898474 | 10000 | 10000 | 0 |
| BrushlessMotor | T | statistical_reference | auroc | 0.5933185404339251 | brushless_motor_acquisition_time_class_confound_v1 | 0.5277777777777778 | 0.6558195677186062 | 10000 | 10000 | 0 |
| BrushlessMotor | T | statistical_reference | average_precision | 0.611993992862768 | brushless_motor_acquisition_time_class_confound_v1 | 0.5569289164148028 | 0.670446059811293 | 10000 | 10000 | 0 |
| BrushlessMotor | T | statistical_reference | standardized_pauroc_fpr_0_10 | 0.5435828229350497 | brushless_motor_acquisition_time_class_confound_v1 | 0.5174130160213156 | 0.5787051455067649 | 10000 | 10000 | 0 |
| RoboticArm | S | isolation_forest | auroc | 0.9137341585617449 | robotic_arm_parent_clustered_inference_v1 | 0.7735908037071774 | 0.9759373276264254 | 10000 | 10000 | 0 |
| RoboticArm | S | isolation_forest | average_precision | 0.8653916181069874 | robotic_arm_parent_clustered_inference_v1 | 0.7181013709407593 | 0.9682055172145401 | 10000 | 10000 | 0 |
| RoboticArm | S | isolation_forest | standardized_pauroc_fpr_0_10 | 0.712558363193572 | robotic_arm_parent_clustered_inference_v1 | 0.5108080269840234 | 0.9051535552240989 | 10000 | 10000 | 0 |
| RoboticArm | S | one_class_svm | auroc | 0.8985042735042736 | robotic_arm_parent_clustered_inference_v1 | 0.7831023341890425 | 0.9589207721739396 | 10000 | 10000 | 0 |
| RoboticArm | S | one_class_svm | average_precision | 0.8614251513228788 | robotic_arm_parent_clustered_inference_v1 | 0.7229497787034096 | 0.955397354906035 | 10000 | 10000 | 0 |
| RoboticArm | S | one_class_svm | standardized_pauroc_fpr_0_10 | 0.70190174818124 | robotic_arm_parent_clustered_inference_v1 | 0.4950206061145515 | 0.8797645071820852 | 10000 | 10000 | 0 |
| RoboticArm | S | statistical_reference | auroc | 0.8521588564692013 | robotic_arm_parent_clustered_inference_v1 | 0.6523086971110653 | 0.952741495106612 | 10000 | 10000 | 0 |
| RoboticArm | S | statistical_reference | average_precision | 0.8091232981386509 | robotic_arm_parent_clustered_inference_v1 | 0.6170626030278072 | 0.9424862248034018 | 10000 | 10000 | 0 |
| RoboticArm | S | statistical_reference | standardized_pauroc_fpr_0_10 | 0.6224231001907953 | robotic_arm_parent_clustered_inference_v1 | 0.4932352936555061 | 0.847227465669866 | 10000 | 10000 | 0 |
| RoboticArm | T | isolation_forest | auroc | 0.9038019451812556 | robotic_arm_parent_clustered_inference_v1 | 0.7558137156865506 | 0.9713406124990702 | 10000 | 10000 | 0 |
| RoboticArm | T | isolation_forest | average_precision | 0.8528200284082675 | robotic_arm_parent_clustered_inference_v1 | 0.6987711964224718 | 0.9631012288842216 | 10000 | 10000 | 0 |
| RoboticArm | T | isolation_forest | standardized_pauroc_fpr_0_10 | 0.7001140118200009 | robotic_arm_parent_clustered_inference_v1 | 0.4999806449650225 | 0.8935316799366729 | 10000 | 10000 | 0 |
| RoboticArm | T | one_class_svm | auroc | 0.8931808134394341 | robotic_arm_parent_clustered_inference_v1 | 0.7875777190896831 | 0.9531280337136689 | 10000 | 10000 | 0 |
| RoboticArm | T | one_class_svm | average_precision | 0.8933362936710438 | robotic_arm_parent_clustered_inference_v1 | 0.766564994526753 | 0.9573543246492434 | 10000 | 10000 | 0 |
| RoboticArm | T | one_class_svm | standardized_pauroc_fpr_0_10 | 0.738296337661129 | robotic_arm_parent_clustered_inference_v1 | 0.5875442337289836 | 0.8662700057580166 | 10000 | 10000 | 0 |
| RoboticArm | T | statistical_reference | auroc | 0.8457117595048629 | robotic_arm_parent_clustered_inference_v1 | 0.641881270453726 | 0.9508346488677348 | 10000 | 10000 | 0 |
| RoboticArm | T | statistical_reference | average_precision | 0.8037991437975067 | robotic_arm_parent_clustered_inference_v1 | 0.6065938608877577 | 0.9418730082839175 | 10000 | 10000 | 0 |
| RoboticArm | T | statistical_reference | standardized_pauroc_fpr_0_10 | 0.6185451471295391 | robotic_arm_parent_clustered_inference_v1 | 0.493020647254117 | 0.8440320994249807 | 10000 | 10000 | 0 |

## Appendix E — Recovery integrity checks

| manifest | file_count | status | missing_or_mismatched |
| --- | --- | --- | --- |
| reports/phase3_2/artifact_manifest.json | 155 | PASS | [] |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/bundle_manifest.json | 5 | PASS | [] |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json | 4 | PASS | [] |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/bundle_manifest.json | 4 | PASS | [] |

| experiment_id | segment_count | cohort_hash | identity_and_metadata_match | config_unchanged | threshold_and_source_ids_unchanged |
| --- | --- | --- | --- | --- | --- |
| p3-994e171b9c84299ee0506001 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-b7df07b4435147ce84abb2fb | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-d523942f6e3b110247ccd97a | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-d2e84d3ccdbb25446ed3b5c2 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-61da2e22429837caf51172a3 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-4bad66197b1b5b39bc0e1ce8 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-c0b106088a660b45248207a8 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-00ef3593eef3eef458aac8ec | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-f2e82658337725d643461f5a | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-4df2732b72997882e300c7ee | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-e0cb98b1d50e1b760ad00c33 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-efee4a25a307b9556f1ef9e1 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-b2c1eb73ec1f836d95401be0 | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-3dfb6a47e2b1ee1a4da8ed4f | 466 | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da | True | True | True |
| p3-c3480b35881d464a75f3b359 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-d75e7a94e44f1caf8a5a595a | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-611ddf562a1bccc1714e44b8 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-296f1214c5d29239fd3f95a7 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-92d37c41321bf4991c224a7b | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-a164ab66f27ae30b8964b5d5 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-c72840a71b6f3d36ed7a30d0 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-f0f1a4a48cc815e2faf3d696 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-a2b4c42e73f2ac6104e622c8 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-649152f9076af141b8701f69 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-56aaef3d39745f6f79059405 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-93869e664dd39a5c050ce499 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-5205733640c5094021b2a4b0 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |
| p3-3c27f8c1e723c409051319c8 | 312 | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad | True | True | True |

### Feature artifact checks

| path | sha256_match | count_match | cohort_hash_match |
| --- | --- | --- | --- |
| data/interim/phase31_primary/BrushlessMotor/S/source_validation.parquet | True | True | True |
| data/interim/phase31_primary/BrushlessMotor/S/training.parquet | True | True | True |
| data/interim/phase31_primary/BrushlessMotor/T/source_validation.parquet | True | True | True |
| data/interim/phase31_primary/BrushlessMotor/T/training.parquet | True | True | True |
| data/interim/phase31_primary/RoboticArm/S/source_validation.parquet | True | True | True |
| data/interim/phase31_primary/RoboticArm/S/training.parquet | True | True | True |
| data/interim/phase31_primary/RoboticArm/T/source_validation.parquet | True | True | True |
| data/interim/phase31_primary/RoboticArm/T/training.parquet | True | True | True |
| data/interim/phase32_common_test/BrushlessMotor/S/common_test.parquet | True | True | True |
| data/interim/phase32_common_test/BrushlessMotor/T/common_test.parquet | True | True | True |
| data/interim/phase32_common_test/RoboticArm/S/common_test.parquet | True | True | True |
| data/interim/phase32_common_test/RoboticArm/T/common_test.parquet | True | True | True |

### Split checks

| machine | regime | unique_rows | parent_disjoint_consuming_splits | blocked_rows_not_consumed | cohort_hash |
| --- | --- | --- | --- | --- | --- |
| RoboticArm | S | True | True | True | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da |
| RoboticArm | T | True | True | True | 23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da |
| BrushlessMotor | S | True | True | True | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad |
| BrushlessMotor | T | True | True | True | e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad |

### Historical split identity

| commit | split_git_blob |
| --- | --- |
| 92c6bb52734292f1646bdfa02650f65fd0997c6d | f3eb4f331ac123e1b178d74ad48e89caced65e21 |
| 78b536304d94d50afcc293348c5fa0c34e729f62 | f3eb4f331ac123e1b178d74ad48e89caced65e21 |
| fefaa341df1d8f75afb4fcde8d68ffc1571bd783 | f3eb4f331ac123e1b178d74ad48e89caced65e21 |
| 99f6055156c314bbf26a847713d26792979fd050 | f3eb4f331ac123e1b178d74ad48e89caced65e21 |
| a576c9df8000dd1c0a5ad6fd73845f97facf98b2 | f3eb4f331ac123e1b178d74ad48e89caced65e21 |

Raw metadata references: 3,896 segments /11,688 sensor files verified present with file sizes matching the historical audit; no signal payload reread. All recorded Phase2/2.1/2.2 smoke-artifact references exist. Every config/report JSON parses. All28 result threshold values and source-ID lists exactly match their Phase3.1 freezes. No tracked or staged diff. Final additions: the two recovery files only, alongside unchanged pre-existing supervisor/synopsis/tmp directories.

### Environment

Python: `3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)]`; executable: `C:\Python314\python.exe`.

| package | version |
| --- | --- |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |
| h5py | 3.16.0 |
| matplotlib | 3.10.8 |
| pytest | 9.0.3 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

`pip check` fails due to unrelated global Google ADK/Cloud dependencies and starlette/websockets conflicts. No package changed. Project pins match installed versions.

## Appendix F — Complete useful local artifact inventory

Paths are repository-relative; root is the repository root. Git status distinguishes tracked files, pre-existing untracked files and ignored local products. Raw-data files are summarized separately (11,688 parquet and12 metadata CSV); no payload is duplicated here.

| path | bytes | git_status |
| --- | --- | --- |
| docs/assumptions_and_open_questions.md | 24330 | tracked |
| docs/evaluation_protocol.md | 28990 | tracked |
| docs/experiment_safety_phase3.md | 6616 | tracked |
| docs/leakage_risks.md | 28974 | tracked |
| docs/phase311_first_test_reporting_freeze.md | 3287 | tracked |
| docs/phase31_pretest_freeze.md | 6925 | tracked |
| docs/preprocessing_phase2.md | 9248 | tracked |
| docs/research_definition.md | 74741 | tracked |
| docs/source_verification.md | 23210 | tracked |
| configs/.gitkeep | 0 | tracked |
| configs/evaluation/protocol_b_common_test_v2.json | 2615 | tracked |
| configs/evaluation/protocol_b_few_shot_adaptation_v3.json | 6338 | tracked |
| configs/phase3/baseline_ladder_v1.json | 3098 | tracked |
| configs/phase3/first_test_reporting_policy_v1.json | 4159 | tracked |
| configs/phase3/preprocessing_candidates_v1.json | 1903 | tracked |
| configs/phase3/preprocessing_primary_v1.json | 1658 | tracked |
| configs/preprocessing/mechasight_common_test_regime_s.json | 932 | tracked |
| configs/preprocessing/mechasight_common_test_regime_t.json | 932 | tracked |
| configs/preprocessing/mechasight_few_shot_regime_s.json | 1098 | tracked |
| configs/preprocessing/mechasight_few_shot_regime_t.json | 1098 | tracked |
| configs/preprocessing/mechasight_regime_s.json | 918 | tracked |
| configs/preprocessing/mechasight_regime_t.json | 918 | tracked |
| configs/preprocessing/official_baseline_reproduction.json | 769 | tracked |
| reports/dataset_audit/accelerometer_artefact_scan.csv | 530551 | tracked |
| reports/dataset_audit/artefact_investigation.json | 6862 | tracked |
| reports/dataset_audit/audio_noise_probe.json | 28833 | tracked |
| reports/dataset_audit/audio_noise_probe_features.csv | 229214 | tracked |
| reports/dataset_audit/audit_findings.json | 31473 | tracked |
| reports/dataset_audit/channel_quality.csv | 3813558 | tracked |
| reports/dataset_audit/condition_class_balance.csv | 2559 | tracked |
| reports/dataset_audit/count_comparison.csv | 853 | tracked |
| reports/dataset_audit/dataset_summary.md | 27068 | tracked |
| reports/dataset_audit/download_provenance.json | 1277 | tracked |
| reports/dataset_audit/duration_summary.csv | 13924 | tracked |
| reports/dataset_audit/extraction_report.json | 401 | tracked |
| reports/dataset_audit/label_condition_counts.csv | 14406 | tracked |
| reports/dataset_audit/quality_report.csv | 1386 | tracked |
| reports/dataset_audit/recording_times.csv | 330540 | tracked |
| reports/dataset_audit/sampling_rate_summary.csv | 5148 | tracked |
| reports/dataset_audit/segment_manifest.csv | 4765195 | tracked |
| reports/dataset_audit/timestamp_alignment_summary.csv | 3101 | tracked |
| reports/experiment_results/.gitkeep | 0 | tracked |
| reports/experiment_results/test_look_ledger.jsonl | 21280 | tracked |
| reports/figures/.gitkeep | 0 | tracked |
| reports/phase2/phase2_report.md | 6822 | tracked |
| reports/phase2/phase2_validation.json | 8745 | tracked |
| reports/phase2/processed_manifest_schema.json | 912 | tracked |
| reports/phase2/protocol_b_candidate_statistics.csv | 9987 | tracked |
| reports/phase2/protocol_b_split_design.json | 846 | tracked |
| reports/phase2/protocol_b_split_statistics.csv | 36551 | tracked |
| reports/phase2/protocol_b_split_v1.csv | 1714318 | tracked |
| reports/phase2/upstream_baseline_provenance.json | 1511 | tracked |
| reports/phase2_1/matched_test_identity_proof.csv | 242 | tracked |
| reports/phase2_1/phase21_report.md | 6306 | tracked |
| reports/phase2_1/phase21_validation.json | 7521 | tracked |
| reports/phase2_1/processed_manifest_extension.json | 531 | tracked |
| reports/phase2_1/protocol_b_common_test_v2.csv | 1954987 | tracked |
| reports/phase2_1/protocol_b_common_test_v2_counts.csv | 1190 | tracked |
| reports/phase2_1/protocol_b_common_test_v2_design.json | 1714 | tracked |
| reports/phase2_1/target_adaptation_expansion.csv | 252 | tracked |
| reports/phase2_1/target_adaptation_parent_groups.csv | 1779 | tracked |
| reports/phase2_2/adaptation_budget.csv | 278 | tracked |
| reports/phase2_2/blocked_sibling_isolation_proof.csv | 283 | tracked |
| reports/phase2_2/phase22_report.md | 13916 | tracked |
| reports/phase2_2/phase22_validation.json | 15016 | tracked |
| reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv | 2019627 | tracked |
| reports/phase2_2/protocol_b_few_shot_adaptation_v3_counts.csv | 1328 | tracked |
| reports/phase2_2/protocol_b_few_shot_adaptation_v3_design.json | 4780 | tracked |
| reports/phase2_2/protocol_role_counts.csv | 562 | tracked |
| reports/phase3_0/phase301_correction_report.md | 5074 | tracked |
| reports/phase3_0/phase30_report.md | 7295 | tracked |
| reports/phase3_1/phase31_pretest_freeze_report.json | 19484 | tracked |
| reports/phase3_1/phase31_pretest_freeze_report.md | 3427 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/config.json | 3559 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/provenance.json | 3144 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/validation_scores.csv | 18204 | tracked |
| reports/phase3_1/validation_only/p3-00ef3593eef3eef458aac8ec/validation_summary.json | 821 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/validation_scores.csv | 12328 | tracked |
| reports/phase3_1/validation_only/p3-296f1214c5d29239fd3f95a7/validation_summary.json | 815 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/config.json | 2771 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/provenance.json | 2310 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/threshold.json | 4149 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/validation_scores.csv | 12390 | tracked |
| reports/phase3_1/validation_only/p3-3c27f8c1e723c409051319c8/validation_summary.json | 825 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/config.json | 3597 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/provenance.json | 3136 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/validation_scores.csv | 18368 | tracked |
| reports/phase3_1/validation_only/p3-3dfb6a47e2b1ee1a4da8ed4f/validation_summary.json | 820 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/validation_scores.csv | 18285 | tracked |
| reports/phase3_1/validation_only/p3-4bad66197b1b5b39bc0e1ce8/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/validation_scores.csv | 18286 | tracked |
| reports/phase3_1/validation_only/p3-4df2732b72997882e300c7ee/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/validation_scores.csv | 12328 | tracked |
| reports/phase3_1/validation_only/p3-5205733640c5094021b2a4b0/validation_summary.json | 818 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/validation_scores.csv | 12325 | tracked |
| reports/phase3_1/validation_only/p3-56aaef3d39745f6f79059405/validation_summary.json | 816 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/threshold.json | 4147 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/validation_scores.csv | 12324 | tracked |
| reports/phase3_1/validation_only/p3-611ddf562a1bccc1714e44b8/validation_summary.json | 815 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/validation_scores.csv | 18290 | tracked |
| reports/phase3_1/validation_only/p3-61da2e22429837caf51172a3/validation_summary.json | 822 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/validation_scores.csv | 12330 | tracked |
| reports/phase3_1/validation_only/p3-649152f9076af141b8701f69/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/validation_scores.csv | 12325 | tracked |
| reports/phase3_1/validation_only/p3-92d37c41321bf4991c224a7b/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/validation_scores.csv | 12333 | tracked |
| reports/phase3_1/validation_only/p3-93869e664dd39a5c050ce499/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/config.json | 3559 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/provenance.json | 3144 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/threshold.json | 6086 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/validation_scores.csv | 18207 | tracked |
| reports/phase3_1/validation_only/p3-994e171b9c84299ee0506001/validation_summary.json | 815 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/validation_scores.csv | 12327 | tracked |
| reports/phase3_1/validation_only/p3-a164ab66f27ae30b8964b5d5/validation_summary.json | 816 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/validation_scores.csv | 12335 | tracked |
| reports/phase3_1/validation_only/p3-a2b4c42e73f2ac6104e622c8/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/validation_scores.csv | 18296 | tracked |
| reports/phase3_1/validation_only/p3-b2c1eb73ec1f836d95401be0/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/validation_scores.csv | 18284 | tracked |
| reports/phase3_1/validation_only/p3-b7df07b4435147ce84abb2fb/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/config.json | 3597 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/provenance.json | 3136 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/threshold.json | 6085 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/validation_scores.csv | 18344 | tracked |
| reports/phase3_1/validation_only/p3-c0b106088a660b45248207a8/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/config.json | 2733 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/provenance.json | 2318 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/validation_scores.csv | 12277 | tracked |
| reports/phase3_1/validation_only/p3-c3480b35881d464a75f3b359/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/config.json | 2771 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/provenance.json | 2310 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/threshold.json | 4149 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/validation_scores.csv | 12387 | tracked |
| reports/phase3_1/validation_only/p3-c72840a71b6f3d36ed7a30d0/validation_summary.json | 826 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/validation_scores.csv | 18296 | tracked |
| reports/phase3_1/validation_only/p3-d2e84d3ccdbb25446ed3b5c2/validation_summary.json | 817 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/validation_scores.csv | 18294 | tracked |
| reports/phase3_1/validation_only/p3-d523942f6e3b110247ccd97a/validation_summary.json | 821 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/config.json | 2763 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/provenance.json | 2313 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/validation_scores.csv | 12330 | tracked |
| reports/phase3_1/validation_only/p3-d75e7a94e44f1caf8a5a595a/validation_summary.json | 819 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/validation_scores.csv | 18296 | tracked |
| reports/phase3_1/validation_only/p3-e0cb98b1d50e1b760ad00c33/validation_summary.json | 820 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/validation_scores.csv | 18306 | tracked |
| reports/phase3_1/validation_only/p3-efee4a25a307b9556f1ef9e1/validation_summary.json | 821 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/config.json | 2733 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/provenance.json | 2318 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/threshold.json | 4148 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/validation_scores.csv | 12273 | tracked |
| reports/phase3_1/validation_only/p3-f0f1a4a48cc815e2faf3d696/validation_summary.json | 822 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/bundle_manifest.json | 554 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/config.json | 3589 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/provenance.json | 3139 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/threshold.json | 6087 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/validation_scores.csv | 18301 | tracked |
| reports/phase3_1/validation_only/p3-f2e82658337725d643461f5a/validation_summary.json | 819 | tracked |
| reports/phase3_1_1/experiment_code_identity_audit.json | 3011 | tracked |
| reports/phase3_1_1/experiment_code_identity_audit.md | 1600 | tracked |
| reports/phase3_1_1/phase311_pretest_freeze_report.json | 1531 | tracked |
| reports/phase3_2/artifact_manifest.json | 20176 | tracked |
| reports/phase3_2/bootstrap_confidence_intervals.csv | 6172 | tracked |
| reports/phase3_2/common_test_cohort_verification.json | 2006 | tracked |
| reports/phase3_2/domain_slice_metrics_family.csv | 3838 | tracked |
| reports/phase3_2/domain_slice_metrics_individual.csv | 12250 | tracked |
| reports/phase3_2/evaluation_state.json | 459 | tracked |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/config.json | 3559 | tracked |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/metrics.json | 1074 | tracked |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-00ef3593eef3eef458aac8ec/scores.parquet | 15294 | tracked |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/metrics.json | 1086 | tracked |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-296f1214c5d29239fd3f95a7/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/config.json | 2771 | tracked |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/metrics.json | 1085 | tracked |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/provenance.json | 5561 | tracked |
| reports/phase3_2/experiments/p3-3c27f8c1e723c409051319c8/scores.parquet | 14881 | tracked |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/config.json | 3597 | tracked |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/metrics.json | 1072 | tracked |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-3dfb6a47e2b1ee1a4da8ed4f/scores.parquet | 15253 | tracked |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/metrics.json | 1011 | tracked |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-4bad66197b1b5b39bc0e1ce8/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/metrics.json | 1011 | tracked |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-4df2732b72997882e300c7ee/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/metrics.json | 1088 | tracked |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-5205733640c5094021b2a4b0/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/metrics.json | 1091 | tracked |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-56aaef3d39745f6f79059405/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/metrics.json | 1089 | tracked |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/provenance.json | 5559 | tracked |
| reports/phase3_2/experiments/p3-611ddf562a1bccc1714e44b8/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/metrics.json | 1011 | tracked |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-61da2e22429837caf51172a3/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/metrics.json | 1088 | tracked |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-649152f9076af141b8701f69/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/metrics.json | 1089 | tracked |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-92d37c41321bf4991c224a7b/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/metrics.json | 1076 | tracked |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-93869e664dd39a5c050ce499/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/config.json | 3559 | tracked |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/metrics.json | 1077 | tracked |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/provenance.json | 8319 | tracked |
| reports/phase3_2/experiments/p3-994e171b9c84299ee0506001/scores.parquet | 15294 | tracked |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/metrics.json | 1073 | tracked |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-a164ab66f27ae30b8964b5d5/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/metrics.json | 1091 | tracked |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-a2b4c42e73f2ac6104e622c8/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/metrics.json | 1012 | tracked |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-b2c1eb73ec1f836d95401be0/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/metrics.json | 1013 | tracked |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-b7df07b4435147ce84abb2fb/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/config.json | 3597 | tracked |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/metrics.json | 1075 | tracked |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/provenance.json | 8318 | tracked |
| reports/phase3_2/experiments/p3-c0b106088a660b45248207a8/scores.parquet | 15253 | tracked |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/config.json | 2733 | tracked |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/metrics.json | 1088 | tracked |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-c3480b35881d464a75f3b359/scores.parquet | 15022 | tracked |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/config.json | 2771 | tracked |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/metrics.json | 1053 | tracked |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/provenance.json | 5561 | tracked |
| reports/phase3_2/experiments/p3-c72840a71b6f3d36ed7a30d0/scores.parquet | 14745 | tracked |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/metrics.json | 1014 | tracked |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-d2e84d3ccdbb25446ed3b5c2/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/metrics.json | 1013 | tracked |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-d523942f6e3b110247ccd97a/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/config.json | 2763 | tracked |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/metrics.json | 1091 | tracked |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-d75e7a94e44f1caf8a5a595a/scores.parquet | 14997 | tracked |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/metrics.json | 1013 | tracked |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-e0cb98b1d50e1b760ad00c33/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/metrics.json | 1013 | tracked |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-efee4a25a307b9556f1ef9e1/scores.parquet | 15268 | tracked |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/config.json | 2733 | tracked |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/metrics.json | 1087 | tracked |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/provenance.json | 5560 | tracked |
| reports/phase3_2/experiments/p3-f0f1a4a48cc815e2faf3d696/scores.parquet | 15022 | tracked |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/bundle_manifest.json | 438 | tracked |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/config.json | 3589 | tracked |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/metrics.json | 1030 | tracked |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/provenance.json | 8320 | tracked |
| reports/phase3_2/experiments/p3-f2e82658337725d643461f5a/scores.parquet | 15268 | tracked |
| reports/phase3_2/healthy_target_false_positives_family.csv | 2077 | tracked |
| reports/phase3_2/healthy_target_false_positives_individual.csv | 3636 | tracked |
| reports/phase3_2/individual_experiment_metrics.csv | 8958 | tracked |
| reports/phase3_2/isolation_forest_seed_summaries.csv | 9701 | tracked |
| reports/phase3_2/model_family_paired_differences.csv | 8738 | tracked |
| reports/phase3_2/phase32_first_test_report.json | 101902 | tracked |
| reports/phase3_2/phase32_first_test_report.md | 15165 | tracked |
| reports/phase3_2/pretest_reproduction_check.json | 18370 | tracked |
| reports/phase3_2/primary_baselines.csv | 4401 | tracked |
| reports/phase3_2/s_vs_t_adaptation_deltas.csv | 3422 | tracked |
| reports/supervisor/MechaSight_Project_Direction_and_Implementation_Report.docx | 1809389 | untracked |
| reports/supervisor/MechaSight_Project_Direction_and_Implementation_Report.pdf | 2271766 | ignored |
| reports/synopsis/MechaSight_MAIT_Synopsis_2026.docx | 133487 | untracked |
| reports/synopsis/MechaSight_MAIT_Synopsis_2026.pdf | 470015 | ignored |
| reports/tables/.gitkeep | 0 | tracked |
| src/__init__.py | 0 | tracked |
| src/data/__init__.py | 0 | tracked |
| src/data/audit.py | 11054 | tracked |
| src/data/balance.py | 5620 | tracked |
| src/data/discovery.py | 6559 | tracked |
| src/data/metadata.py | 8914 | tracked |
| src/data/temporal.py | 10145 | tracked |
| src/data/timestamps.py | 12346 | tracked |
| src/evaluation/__init__.py | 0 | tracked |
| src/evaluation/auc.py | 620 | tracked |
| src/evaluation/policy.py | 5716 | tracked |
| src/evaluation/reporting.py | 6398 | tracked |
| src/experiment/__init__.py | 366 | tracked |
| src/experiment/aggregation.py | 1744 | tracked |
| src/experiment/bootstrap.py | 7984 | tracked |
| src/experiment/config.py | 7928 | tracked |
| src/experiment/constants.py | 3212 | tracked |
| src/experiment/data.py | 17936 | tracked |
| src/experiment/gate.py | 4288 | tracked |
| src/experiment/ledger.py | 4092 | tracked |
| src/experiment/metrics.py | 3011 | tracked |
| src/experiment/phase31.py | 18910 | tracked |
| src/experiment/phase32.py | 39384 | tracked |
| src/experiment/pretest.py | 5995 | tracked |
| src/experiment/provenance.py | 1029 | tracked |
| src/experiment/scores.py | 6104 | tracked |
| src/experiment/store.py | 4103 | tracked |
| src/explainability/__init__.py | 0 | tracked |
| src/features/__init__.py | 0 | tracked |
| src/features/extraction.py | 5582 | tracked |
| src/models/__init__.py | 0 | tracked |
| src/models/classical.py | 7474 | tracked |
| src/preprocessing/__init__.py | 0 | tracked |
| src/preprocessing/cache.py | 5479 | tracked |
| src/preprocessing/config.py | 4376 | tracked |
| src/preprocessing/manifest.py | 2833 | tracked |
| src/preprocessing/normalization.py | 2577 | tracked |
| src/preprocessing/pipeline.py | 12013 | tracked |
| src/preprocessing/signals.py | 9549 | tracked |
| src/preprocessing/splits.py | 24884 | tracked |
| src/utils/__init__.py | 0 | tracked |
| scripts/.gitkeep | 0 | tracked |
| scripts/audio_noise_probe.py | 9441 | tracked |
| scripts/audit_imadds.py | 20429 | tracked |
| scripts/build_phase21_splits.py | 5550 | tracked |
| scripts/build_phase22_splits.py | 6983 | tracked |
| scripts/build_phase2_splits.py | 4257 | tracked |
| scripts/download_imadds.sh | 584 | tracked |
| scripts/extract_imadds.py | 4373 | tracked |
| scripts/investigate_artifacts.py | 10043 | tracked |
| scripts/prepare_phase31_features.py | 1646 | tracked |
| scripts/preprocess_imadds.py | 4301 | tracked |
| scripts/run_phase31_baselines.py | 6904 | tracked |
| scripts/run_phase32_first_test.py | 30974 | tracked |
| scripts/validate_phase2.py | 5982 | tracked |
| scripts/validate_phase21.py | 5077 | tracked |
| scripts/validate_phase22.py | 11448 | tracked |
| scripts/verify_download.py | 4757 | tracked |
| tests/.gitkeep | 0 | tracked |
| tests/test_balance.py | 6947 | tracked |
| tests/test_metadata.py | 13384 | tracked |
| tests/test_phase21_common_cohort.py | 5628 | tracked |
| tests/test_phase22_adaptation_budget.py | 11867 | tracked |
| tests/test_phase2_auc.py | 389 | tracked |
| tests/test_phase2_cache_manifest_config.py | 3307 | tracked |
| tests/test_phase2_features_normalization.py | 2385 | tracked |
| tests/test_phase2_signals.py | 4547 | tracked |
| tests/test_phase2_splits.py | 3347 | tracked |
| tests/test_phase301_scientific_consistency.py | 9470 | tracked |
| tests/test_phase30_data_contract.py | 7129 | tracked |
| tests/test_phase30_evaluation.py | 8937 | tracked |
| tests/test_phase30_experiment_contract.py | 8517 | tracked |
| tests/test_phase311_reporting_freeze.py | 8372 | tracked |
| tests/test_phase31_baselines.py | 11043 | tracked |
| tests/test_phase32_evaluator.py | 11418 | tracked |
| tests/test_temporal.py | 8839 | tracked |
| tests/test_timestamps.py | 10079 | tracked |
| notebooks/.gitkeep | 0 | tracked |
| data/interim/.gitkeep | 0 | tracked |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1.h5 | 2575560 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__build_stats.json | 1130 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__features.jsonl | 268602 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__manifest.jsonl | 95965 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1.h5 | 3683488 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__build_stats.json | 1115 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__features.jsonl | 391003 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.1-v2-8688f5639d9fa7c1__manifest.jsonl | 135317 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a.h5 | 2575560 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__build_stats.json | 1130 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__features.jsonl | 268602 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__manifest.jsonl | 95965 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a.h5 | 3683488 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__build_stats.json | 1115 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__features.jsonl | 391003 | ignored |
| data/interim/phase21_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.1-v2-f497d27caee67b4a__manifest.jsonl | 135317 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae.h5 | 2575560 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__build_stats.json | 1174 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__features.jsonl | 268602 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/BrushlessMotor__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__manifest.jsonl | 95965 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae.h5 | 3683488 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__build_stats.json | 1159 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__features.jsonl | 391003 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_S/RoboticArm__mechasight_experimental_protocol__regime-S__test__phase2.2-v3-dcef3e9c758c24ae__manifest.jsonl | 135317 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d.h5 | 2575560 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__build_stats.json | 1174 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__features.jsonl | 268602 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__manifest.jsonl | 95965 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d.h5 | 2575764 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__build_stats.json | 1178 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__features.jsonl | 268758 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__manifest.jsonl | 96325 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d.h5 | 3683488 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__build_stats.json | 1159 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__features.jsonl | 391003 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__test__phase2.2-v3-0a0285c9afd6ad6d__manifest.jsonl | 135317 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d.h5 | 3683488 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__build_stats.json | 1163 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__features.jsonl | 390502 | ignored |
| data/interim/phase22_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2.2-v3-0a0285c9afd6ad6d__manifest.jsonl | 136594 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 2575212 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1099 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 246704 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 94326 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 3664224 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1084 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 358271 | ignored |
| data/interim/phase2_smoke/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 132514 | ignored |
| data/interim/phase2_smoke/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c.h5 | 3662448 | ignored |
| data/interim/phase2_smoke/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__build_stats.json | 1070 | ignored |
| data/interim/phase2_smoke/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__features.jsonl | 249506 | ignored |
| data/interim/phase2_smoke/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__manifest.jsonl | 140894 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 2575764 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1117 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 246675 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 94489 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 3679392 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1102 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 358243 | ignored |
| data/interim/phase2_smoke_final/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 132502 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c.h5 | 2552784 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__build_stats.json | 1103 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__features.jsonl | 172222 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__manifest.jsonl | 99290 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c.h5 | 3662448 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__build_stats.json | 1088 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__features.jsonl | 249641 | ignored |
| data/interim/phase2_smoke_final/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__manifest.jsonl | 140888 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 2575764 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1123 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 266922 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/BrushlessMotor__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 94489 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01.h5 | 3679392 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__build_stats.json | 1108 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__features.jsonl | 387399 | ignored |
| data/interim/phase2_smoke_release/cache/mechasight_experimental_protocol/regime_T/RoboticArm__mechasight_experimental_protocol__regime-T__train__phase2-v1-f0a93c55130baa01__manifest.jsonl | 132502 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c.h5 | 2552784 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__build_stats.json | 1109 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__features.jsonl | 192367 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/BrushlessMotor__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__manifest.jsonl | 99290 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c.h5 | 3662448 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__build_stats.json | 1094 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__features.jsonl | 278649 | ignored |
| data/interim/phase2_smoke_release/cache/official_baseline_reproduction/regime_T/RoboticArm__official_baseline_reproduction__regime-T__train__phase2-v1-f1aada899322ee9c__manifest.jsonl | 140888 | ignored |
| data/interim/phase31_primary/BrushlessMotor/S/source_validation.metadata.json | 883 | ignored |
| data/interim/phase31_primary/BrushlessMotor/S/source_validation.parquet | 179489 | ignored |
| data/interim/phase31_primary/BrushlessMotor/S/training.metadata.json | 871 | ignored |
| data/interim/phase31_primary/BrushlessMotor/S/training.parquet | 979879 | ignored |
| data/interim/phase31_primary/BrushlessMotor/T/source_validation.metadata.json | 883 | ignored |
| data/interim/phase31_primary/BrushlessMotor/T/source_validation.parquet | 179489 | ignored |
| data/interim/phase31_primary/BrushlessMotor/T/training.metadata.json | 871 | ignored |
| data/interim/phase31_primary/BrushlessMotor/T/training.parquet | 994701 | ignored |
| data/interim/phase31_primary/RoboticArm/S/source_validation.metadata.json | 880 | ignored |
| data/interim/phase31_primary/RoboticArm/S/source_validation.parquet | 223636 | ignored |
| data/interim/phase31_primary/RoboticArm/S/training.metadata.json | 868 | ignored |
| data/interim/phase31_primary/RoboticArm/S/training.parquet | 1256384 | ignored |
| data/interim/phase31_primary/RoboticArm/T/source_validation.metadata.json | 880 | ignored |
| data/interim/phase31_primary/RoboticArm/T/source_validation.parquet | 223636 | ignored |
| data/interim/phase31_primary/RoboticArm/T/training.metadata.json | 868 | ignored |
| data/interim/phase31_primary/RoboticArm/T/training.parquet | 1278303 | ignored |
| data/interim/phase32_common_test/BrushlessMotor/S/common_test.metadata.json | 827 | ignored |
| data/interim/phase32_common_test/BrushlessMotor/S/common_test.parquet | 330407 | ignored |
| data/interim/phase32_common_test/BrushlessMotor/T/common_test.metadata.json | 827 | ignored |
| data/interim/phase32_common_test/BrushlessMotor/T/common_test.parquet | 330407 | ignored |
| data/interim/phase32_common_test/RoboticArm/S/common_test.metadata.json | 823 | ignored |
| data/interim/phase32_common_test/RoboticArm/S/common_test.parquet | 441516 | ignored |
| data/interim/phase32_common_test/RoboticArm/T/common_test.metadata.json | 823 | ignored |
| data/interim/phase32_common_test/RoboticArm/T/common_test.parquet | 441516 | ignored |
| tmp/audit_synopsis.py | 3251 | untracked |
| tmp/build_synopsis.py | 34268 | untracked |
| tmp/page4_300.png | 594187 | untracked |
| tmp/synopsis_a11y.json | 2472 | untracked |
| tmp/synopsis_assets/methodology_flowchart.png | 69826 | untracked |
| tmp/synopsis_assets/source_0_X18.png | 25714 | untracked |
| tmp/synopsis_assets/source_1_X20.png | 11794 | untracked |
| tmp/synopsis_assets/source_2_X21.png | 19235 | untracked |
| tmp/synopsis_render_v1/page-1.png | 128721 | untracked |
| tmp/synopsis_render_v1/page-2.png | 239130 | untracked |
| tmp/synopsis_render_v1/page-3.png | 281147 | untracked |
| tmp/synopsis_render_v1/page-4.png | 355728 | untracked |
| tmp/synopsis_render_v1/page-5.png | 292112 | untracked |
| tmp/synopsis_render_v2/page-1.png | 127583 | untracked |
| tmp/synopsis_render_v2/page-2.png | 238674 | untracked |
| tmp/synopsis_render_v2/page-3.png | 281147 | untracked |
| tmp/synopsis_render_v2/page-4.png | 355728 | untracked |
| tmp/synopsis_render_v2/page-5.png | 290138 | untracked |
| tmp/synopsis_render_v3/page-1.png | 128618 | untracked |
| tmp/synopsis_render_v3/page-2.png | 239586 | untracked |
| tmp/synopsis_render_v3/page-3.png | 281147 | untracked |
| tmp/synopsis_render_v3/page-4.png | 355728 | untracked |
| tmp/synopsis_render_v3/page-5.png | 295531 | untracked |
| tmp/synopsis_render_v4/page-1.png | 128618 | untracked |
| tmp/synopsis_render_v4/page-2.png | 231880 | untracked |
| tmp/synopsis_render_v4/page-3.png | 281147 | untracked |
| tmp/synopsis_render_v4/page-4.png | 339304 | untracked |
| tmp/synopsis_render_v4/page-5.png | 295531 | untracked |
| tmp/synopsis_render_v5/page-1.png | 188990 | untracked |
| tmp/synopsis_render_v5/page-2.png | 323489 | untracked |
| tmp/synopsis_render_v5/page-3.png | 412449 | untracked |
| tmp/synopsis_render_v5/page-4.png | 458904 | untracked |
| tmp/synopsis_render_v5/page-5.png | 414844 | untracked |
| tmp/synopsis_render_v6/page-1.png | 188990 | untracked |
| tmp/synopsis_render_v6/page-2.png | 323432 | untracked |
| tmp/synopsis_render_v6/page-3.png | 412449 | untracked |
| tmp/synopsis_render_v6/page-4.png | 458904 | untracked |
| tmp/synopsis_render_v6/page-5.png | 414844 | untracked |
| tmp/synopsis_source_render/page-1.png | 195383 | untracked |
| tmp/synopsis_source_render/page-2.png | 233254 | untracked |
| tmp/synopsis_source_render/page-3.png | 144771 | untracked |

## Appendix G — Complete test-look ledger

No entry was added during recovery. One first-test batch,28 distinct configuration/seed records, not28 adaptive experiments.

```json
[
  {
    "experiment_id": "p3-00ef3593eef3eef458aac8ec",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-6e84ada532ccd25ea25b-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-296f1214c5d29239fd3f95a7",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-a61e75dc",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-3c27f8c1e723c409051319c8",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-856bdf04029b9ce21634-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-3dfb6a47e2b1ee1a4da8ed4f",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-856bdf04029b9ce21634-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-4bad66197b1b5b39bc0e1ce8",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-ef731a0a",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-4df2732b72997882e300c7ee",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-fbe47f6d",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-5205733640c5094021b2a4b0",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-ef731a0a",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-56aaef3d39745f6f79059405",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-a61e75dc",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-611ddf562a1bccc1714e44b8",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-fbe47f6d",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-61da2e22429837caf51172a3",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-4d687b64",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-649152f9076af141b8701f69",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-fbe47f6d",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-92d37c41321bf4991c224a7b",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-4d687b64",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-93869e664dd39a5c050ce499",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-4d687b64",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-994e171b9c84299ee0506001",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-6e84ada532ccd25ea25b-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-a164ab66f27ae30b8964b5d5",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-ef731a0a",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-a2b4c42e73f2ac6104e622c8",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-bd048884",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-b2c1eb73ec1f836d95401be0",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-ef731a0a",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-b7df07b4435147ce84abb2fb",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-bd048884",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-c0b106088a660b45248207a8",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-856bdf04029b9ce21634-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-c3480b35881d464a75f3b359",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-6e84ada532ccd25ea25b-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-c72840a71b6f3d36ed7a30d0",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-856bdf04029b9ce21634-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-d2e84d3ccdbb25446ed3b5c2",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-a61e75dc",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-d523942f6e3b110247ccd97a",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-fbe47f6d",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-d75e7a94e44f1caf8a5a595a",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-bd048884",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "S",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-e0cb98b1d50e1b760ad00c33",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-a61e75dc",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-efee4a25a307b9556f1ef9e1",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-4d687b64",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-f0f1a4a48cc815e2faf3d696",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "BrushlessMotor",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-6e84ada532ccd25ea25b-4bbbbfa1",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "e1b09c058261ae04f6574330dab17ca89127616bd4bd9fac4978bd19b821b3ad",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  },
  {
    "experiment_id": "p3-f2e82658337725d643461f5a",
    "git_sha": "99f6055156c314bbf26a847713d26792979fd050",
    "machine": "RoboticArm",
    "metrics_requested": [
      "auroc",
      "average_precision",
      "standardized_pauroc_fpr_0_10",
      "threshold_metrics"
    ],
    "model_config_identity": "p31-model-bcc23071948443594ad8-bd048884",
    "protocol": "mechasight_experimental_protocol",
    "reason": "first pre-registered Protocol-B baseline evaluation; batch=p32-test-batch-8f7d48324fd10df97ae29100; policy=p311-reporting-5e76ab7dabd5689c7b0ced68; slices=complete_per_machine,source_domain,target_domain,healthy_target",
    "regime": "T",
    "schema_version": "phase3.0-test-look-v1",
    "test_cohort_hash": "23c5b7e3f1b0e6f28d6159a8c893abb5de309a0b4b978c713f43ba4383a775da",
    "timestamp_utc": "2026-09-03T09:06:51.648233+00:00"
  }
]
```

The repository recovery was completed without starting a new experiment. Independent review
was pending when this historical snapshot was recorded; current status is linked at the top.
