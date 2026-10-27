# Source Verification Log — Phase 0

Every claim in `docs/research_definition.md` traces to a row here. Verification performed
**2026-08-21**. Where a claim could not be checked against a primary source, it is marked
unverified rather than quietly asserted.

Method: primary sources were fetched directly (Zenodo REST API, the DCASE proceedings PDF,
raw files from the authors' GitHub repository) and read in full. Search-engine summaries and
automatic page-summarisation were treated as pointers only, never as evidence — see the
"summariser discrepancy" note at the end, which is the reason for that rule.

---

## S1 — IMAD-DS paper (PRIMARY)

- **Citation:** Albertini D., Augusti F., Esmer K., Bernardini A., Sannino R.,
  *IMAD-DS: A Dataset for Industrial Multi-Sensor Anomaly Detection Under Domain Shift
  Conditions*, Proceedings of the Detection and Classification of Acoustic Scenes and Events
  (DCASE) 2024 Workshop, Tokyo, Japan, 23–25 October 2024.
- **URL:** https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Albertini_45.pdf
- **How verified:** PDF downloaded and text-extracted locally (`pdftotext -layout`), then read
  in full including Tables 1–4. Not summarised by a model.
- **Affiliations:** STMicroelectronics, Agrate Brianza, Italy (Albertini, Augusti, Sannino);
  Politecnico di Milano, DEIB (Esmer, Bernardini).
- **Claims sourced from it:** machines; anomaly mechanisms; sensor list and nominal rates;
  Table 1 domain-shift configuration; Table 2 segment counts; unsupervised normal-only protocol;
  audio simulation chain (ISM, pyroomacoustics, 10 × 7.5 × 4 m, T60 = 0.5 s, SNR −4 dB);
  audio-only environmental coupling assumption; baseline architecture and optimiser settings;
  median-over-windows segment scoring; AUC definition; Table 4 results; the "mistake domain
  changes for anomalies" explanation; the "exploring alternative multi-sensor methods is key"
  conclusion; the note that the same setup generated DCASE 2024 Task 2 audio.
- **Not committed to the repository** — third-party copyrighted PDF. `.gitignore` excludes
  `*.pdf` and `docs/papers/`. Re-download from the URL above to reproduce.
- **Internal inconsistency found:** the Table 1 caption reads "Letters A to F index 7 background
  noise recordings", but seven letters requires A–G, and the table body uses A,C,D,F | B,E,G.
  Treated as a caption typo; the table body is taken as authoritative.

## S2 — IMAD-DS dataset record on Zenodo (PRIMARY)

- **URL:** https://zenodo.org/records/12665499 · **API:** https://zenodo.org/api/records/12665499
- **How verified:** raw JSON pulled from the Zenodo REST API and parsed locally, so that field
  values (DOI, licence, file sizes, creator list, description) are read as data, not paraphrase.
- **Verified values:** DOI `10.5281/zenodo.12665499`; concept DOI `10.5281/zenodo.12636236`;
  publication date 2024-07-05; licence `cc-by-sa-4.0`; creators Augusti, Albertini, Esmer,
  Sannino, Bernardini; files `RoboticArm.7z` = 3,545,453,763 bytes and
  `BrushlessMotor.7z` = 1,577,018,050 bytes (two files only); description confirming
  STEVAL-STWINBX1, the three sensing modalities (from two physical devices — an IMP23ABSU
  microphone and an ISM330DHCX 6-axis IMU) at 16 kHz / 6.7 kHz / 6.7 kHz, anechoic-chamber recording,
  train/test folder split, per-sensor `.parquet` files and per-segment `.csv` metadata.
- **Discrepancy vs S1:** Zenodo says anomalies are created by "removing bolts at the nodes of the
  arm"; the paper says "loosening the screws at the arm's nodes". Recorded, unresolved.
- **Note:** the Zenodo `version` field is null in the API response; a secondary source described
  the record as "v2, modified 2024-08-28". Version numbering is therefore **unverified**.

## S3 — Authors' baseline code (PRIMARY)

- **URL:** https://github.com/augustif/IMAD-DS_baseline
- **Pinned Phase-2 snapshot:** commit
  `592bcbbed58893dc26bbab3240c5d7a3105a1a7f` on `main`, retrieved 2026-08-22.
  Protocol A is defined against this commit, not the moving branch head. Exact inspected-file
  SHA-256 values, dataset DOI/version status, and archive MD5s are recorded in
  `reports/phase2/upstream_baseline_provenance.json`.
- **How verified:** repository tree listed via the GitHub API; `README.md`, `utilities.py`,
  `model.py`, `preprocess.ipynb` and `baseline.ipynb` downloaded raw and read as source.
- **Actual repository contents:** `.gitignore`, `README.md`, `baseline.ipynb`,
  `download_dataset.ipynb`, `model.py`, `preprocess.ipynb`, `requirements.txt`, `utilities.py`.
- **Claims sourced from it:**
  - IMU sampling rate `7063` Hz with the comment *"Estimated sampling rate calculated by
    averaging time deltas across all files"* (`preprocess.ipynb`) — the basis for the
    nominal-vs-effective rate finding.
  - `GYROSCOPE_WARM_UP_TIME = 35 ms` discarded at the start of every segment.
  - Non-overlapping 100 ms timestamp-based windows over the sensors' common time interval;
    per-sensor native rates; zero-pad/truncate to fixed length.
  - The six `attributes_*.csv` filenames and the metadata column names
    (`segment_id`, `split_label`, `anomaly_label`, `domain_shift_op`, `domain_shift_env`).
  - `segment_id` derived by stripping the `imp23absu_mic_` filename prefix.
  - `split_label` value vocabulary (`Normal_Source_Test`, `Anomaly_Source_Test`,
    `Normal_Target_Test`, `Anomaly_Target_Test`) from `calculate_single_auc`.
  - `anomaly_label` tested as `== 'normal'` vs `!= 'normal'`, so it holds named anomaly types.
  - `get_masked_batch` zeroing other sensors — the basis for the finding that the paper's
    `S-x` rows are masked-input results, **not** independently trained single-sensor models.
  - Baseline hyperparameters actually run: `layer_dims [2048, 2048, 2048, 16]`, lr 1e-4,
    batch 1024, MSE, `valid_size 0.1`, seed 1995, `normalisation: 'std_window'`,
    segment aggregation `'median'`.
  - `calculate_single_auc` awards a pair one only when `normal_score < anomaly_score`; exact ties
    receive zero credit. Protocol A retains this strict pairwise implementation.
- **Defects found by reading the source** (recorded so they are not inherited):
  - `utilities.standardize()` returns `(data - mean) / std + 1e-5` — the epsilon is added after
    the division and therefore does not prevent division by zero, contradicting its docstring.
    **Scope limit:** this function is reached only by the `'std'` normalisation mode. The published
    `baseline.ipynb` sets `normalisation: 'std_window'`, which calls `standardize_window()`, where
    the epsilon is correctly in the denominator (`data /= std_ + 1e-5`). The defect is therefore
    **off the code path used for the published Table 4 results**, and must not be claimed to affect
    them without evidence. It is recorded because MechaSight intends to evaluate the train-fitted
    `'std'` path, which does contain it.
  - `baseline.ipynb` performs the train/validation split at **window** level via
    `sklearn.train_test_split`, so windows from one recording can appear on both sides.
  - `sensor_df[start:end]` uses pandas label-based datetime slicing, which is right-inclusive,
    so adjacent 100 ms windows can share a boundary sample.
  - Segment reading is wrapped in a bare `try/except` that prints and continues, so unreadable
    segments are silently skipped — a reason to independently count segments in Phase 1.
- **README caveat:** the `README.md` in this repository describes a *different, extended*
  framework (Hydra configs, `main.py`, `models/`, `datasets/`, MLflow, a `CNN_AE`) whose files
  do **not exist** in the repository. The README's config values (50 epochs, lr 0.001, batch 64)
  therefore describe that other framework, **not** the DCASE paper's baseline. The notebook
  values above are the ones that correspond to the published results.

## S4 — DCASE 2024 Challenge Task 2 description (SUPPORTING)

- **Citation:** *Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot
  Unsupervised Anomalous Sound Detection for Machine Condition Monitoring*, arXiv:2406.07250.
- **URL:** https://arxiv.org/abs/2406.07250
- **How verified:** HTML version fetched and queried. **Read via automatic summarisation, not
  full manual reading** — treat as supporting, not primary, and re-read before citing in the
  final report.
- **Claims sourced from it:** RoboticArm and BrushlessMotor are machine types in the DCASE 2024
  Task 2 *additional training / evaluation* split (not the development split); clips are
  single-channel, 6–10 s, 16 kHz; the first-shot setting means disjoint machine types between
  development and evaluation with one section per machine type; the official score is the
  harmonic mean of AUC and pAUC with **p = 0.1** — the source of MechaSight's pAUC@0.1 choice.

## S5 — Prior work on reliability/uncertainty-weighted fusion (PRIMARY, read in full)

Read to test the claim that reliability-aware fusion is novel, and to establish precisely how our
setting differs from the closest prior art. Two papers were fetched and read directly (not
summarised). Both confirm the mechanism is established; **neither covers our setting.**

### S5a — Chu G., Zeng P. (2025), PLoS ONE 20(11) — *the closest prior art*

*Real-time motor operating state recognition via multi-sensor fusion: A wavelet–neural–evidence
framework for industrial condition monitoring.*
DOI 10.1371/journal.pone.0335360 ·
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335360

- **How verified:** article HTML fetched, tags stripped locally, full text read.
- **Method:** vibration, acoustic and magnetic (Hall) signals acquired synchronously on an STM32
  platform; sliding-window segmentation (512 Hz, 1024-sample windows, 50% overlap); wavelet packet
  transform (db4, 3-level) energy-ratio features; a per-modality BP neural network per sensor;
  softmax outputs combined by **Dempster–Shafer evidence fusion with adaptive confidence
  weighting**, described as dynamically adjusting "sensor contribution weights based on
  instantaneous classification confidence, thereby mitigating the influence of noisy or partially
  missing sensor signals."
- **Primary result:** 92.8% overall accuracy on three motor operating states (shutdown, no-load,
  loaded), reported as beating all single-sensor baselines and simple averaging; sub-50 ms latency.

- **⚠️ Internal inconsistency in the published article — recorded, not resolved.**
  The article contradicts itself about whether fault-state data are present.
  - The Introduction states: *"Although fault-state data are not included in this version due to
    experimental equipment constraints, the proposed architecture is directly extendable to fault
    diagnosis scenarios."*
  - The experimental setup section states: *"In this study, fault states such as bearing wear or
    rotor imbalance were not artificially introduced,"* and elsewhere *"only three non-fault states
    are included in the current dataset."*
  - But **Section 4.4**, *"Experimental validation under fault conditions and evidence-conflict–based
    anomaly flagging"*, describes exactly such experiments: bearing wear emulated by controlled
    radial load asymmetry, rotor imbalance by a calibrated 5 g offset mass at 30 mm radius,
    recorded under no-load and loaded states, yielding *"an additional 1200 labeled samples
    (2 faults × 2 operating states × ∼300)."* The conclusion reports *"zero-shot evaluations under
    bearing-wear and rotor-imbalance faults maintained ≥90.9% accuracy, while the D–S conflict and
    entropy measures enabled anomaly flagging with ROC-AUC 0.982."*
  - Most likely a revision added Section 4.4 without updating the earlier text. **MechaSight must
    not cite this paper as containing no fault data.** An earlier draft of this project's
    documentation did exactly that, on the strength of the Introduction alone; that claim is
    withdrawn.

- **How our setting differs — using only the robust distinctions.** Note in particular that
  Section 4.4's zero-shot fault flagging via D–S conflict/entropy *is* a form of anomaly detection,
  so "unseen fault types" is **not** a distinguishing feature and must not be claimed as one.
  What remains robust:
  1. **The primary modelling task and protocol differ.** Their task is "naturally formulated as a
     supervised multi-class classification problem" over operating states, trained with state
     labels. MechaSight's protocol is normal-only one-class anomaly detection, with no supervised
     training target of any kind.
  2. **No operational/environmental domain-shift protocol.** There is no source/target split, no
     held-out operating condition, and no evaluation of generalisation to unseen conditions —
     which is the entire subject of MechaSight.
  3. **The reliability signal has a different origin.** Their confidence weighting derives from a
     *supervised* classifier's softmax outputs and the resulting D–S conflict. Under a normal-only
     protocol there are no class posteriors to read confidence from, so MechaSight must construct a
     reliability estimate by other means. This is a technical difference, not a framing one.
  4. **Different sensing regime.** Single-rate 512 Hz vibration plus acoustic and *magnetic*
     sensing, versus IMAD-DS's multi-rate 16 kHz microphone + ~7 kHz accelerometer + gyroscope.
     Multi-rate heterogeneity is a stated part of what IMAD-DS was built to pose.
  - **Honest note on closeness:** Section 4.4 makes this paper closer to MechaSight than the
    earlier draft of our documentation allowed. Confidence-weighted multi-sensor fusion producing
    an anomaly-flagging score on an industrial motor already exists in the literature. The
    distinction is the protocol and the domain-shift question, not the idea of flagging anomalies
    by fused evidence.

### S5b — Jeong S., Park J., Imani M. (2025), arXiv:2505.02393v2

*Uncertainty-Weighted Image-Event Multimodal Fusion for Video Anomaly Detection (IEF-VAD).*
https://arxiv.org/abs/2505.02393

- **How verified:** PDF downloaded, text-extracted locally, read directly.
- **Method:** models each modality's latent features together with a predictive variance, using a
  Student's t likelihood with a Laplace approximation, and fuses by **inverse-variance
  (precision) weighting**, plus Kalman-style sequential updates and iterative residual refinement.
- **Directly relevant motivation:** the paper names **modality dominance** as the problem —
  "multimodal attention mechanisms naturally gravitate toward the more expressive modality, this
  bias suppresses the complementary information that other inputs could provide" — and positions
  "uncertainty as an explicit control signal for fusion". That is the same failure mode MechaSight
  observes in the IMAD-DS Brushless Motor result, arrived at independently.
- **How our setting differs:**
  1. **Weakly supervised**, via multiple-instance learning over video-level anomaly labels —
     not normal-only training.
  2. **Image + synthetic event modalities** from video, not physical multi-rate machine sensors.
  3. **No operational/environmental domain-shift protocol**; benchmarks are UCF-Crime,
     XD-Violence, ShanghaiTech and MSAD.
- **Useful to us regardless:** inverse-variance weighting is a concrete, principled candidate
  mechanism for Phase 6, and it is available in a form that does not require class posteriors.

### Conclusion carried into research_definition.md §8 / D11

The mechanism — weighting modalities by an estimated reliability — is **established prior art**
and MechaSight claims no priority over it. S5a additionally shows that fused-evidence anomaly
flagging on an industrial motor already exists, so **"detecting unseen faults by
reliability-weighted fusion" is not itself a distinguishing claim** and must not be presented as one.

What remains open, and is therefore the contribution, is the combination of:
**(i) a normal-only (one-class) training protocol** with no supervised target from which to derive
a confidence signal; **(ii) multi-rate heterogeneous physical sensors**; and **(iii) evaluation
against a documented operational *and* environmental domain-shift protocol** with a defined
source/target split. Neither S5a nor S5b has any of these three. This must be stated as a
combination of setting conditions, not as invention of a mechanism.

**Still unread and required before the final report:** the DCASE Task 2 challenge-results papers
and at least one top-ranked system description, needed to substantiate the k-NN/Mahalanobis
embedding-distance comparator (research definition §11). Until then that comparator's
justification is marked `[UNVERIFIED]` and no claim about leaderboard rankings is made.

---

## S6 — The dataset itself (PRIMARY, Phase 1)

- **Obtained:** Zenodo record 12665499, both archives downloaded 2026-08-21 and verified against
  the official MD5 checksums published by the Zenodo API:
  - `BrushlessMotor.7z` — 1,577,018,050 B, `md5:d5f1a49b928b1b0b01c3d6de930cd277` — **VERIFIED**
  - `RoboticArm.7z` — 3,545,453,763 B, `md5:a64498b7dcc297946a7fb8366e38ba33` — **VERIFIED**
  - Provenance recorded in `reports/dataset_audit/download_provenance.json`; extraction refuses any
    archive that has not passed verification.
- **How audited:** `scripts/audit_imadds.py` over all 3,896 segments and 11,688 parquet files,
  plus `scripts/audio_noise_probe.py` on training-normal segments only. Outputs in
  `reports/dataset_audit/`. Reproducible from the command line.
- **Status of the dataset as a source:** this now **outranks S1–S3** for any factual claim about
  what the release contains. Where the paper, the Zenodo description or the reference code disagree
  with the files, the files win.

**Claims from S1–S3 that the data CONTRADICTS:**

| Claim | Source | Reality |
|---|---|---|
| IMU sampling rate 6.7 kHz | S1, S2 | 7063.294 Hz (RA), 7135.716/7135.331 Hz (BM) |
| IMU rate 7063 Hz applies to the dataset | S3 (`preprocess.ipynb`) | True for RoboticArm only; 1.03% wrong for BrushlessMotor, where the code silently truncates ~5 samples per 100 ms window |
| Segment length ~6–10 s (our inference from S4) | our Phase 0 inference | 5.199 s (BM), 7.500 s (RA) |
| One segment per recording (our Phase 0 assumption) | our Phase 0 assumption | True for BM; RoboticArm has median 16 and up to 48 excerpts per recording |

**Claims from S1–S3 that the data CONFIRMS:**
segment counts (3,896, cell by cell); Table 1 conditions exactly; normal-only training with
anomalies test-only; metadata column names and the six `attributes_*.csv` filenames; timestamp-based
alignment; the 35 ms gyroscope warm-up (std is 12–47× higher inside it); noise mixed into audio only;
audio pre-mixed in the release.

**Undocumented in every source, found only in the data:** the two machines' differing rates,
durations, alignment offsets and recording structure; 1 µs timestamp quantisation; BrushlessMotor's
bimodal IMU sampling; the accelerometer first-sample rail artefact; the `corr` marker on 85
RoboticArm ids; and the RoboticArm split's recording-level leakage.

## Verification status summary

| Claim | Status |
|---|---|
| Machines, anomaly mechanisms | VERIFIED (S1, S2) |
| Sensor parts, microphone 16 kHz | VERIFIED (S1, S2, S3) |
| IMU rate "6.7 kHz" | VERIFIED as **nominal**; CONTRADICTED as effective (S3 uses 7063 Hz) |
| Domain-shift parameter values (loads, speeds, noise letters) | VERIFIED (S1 Table 1) |
| SNR −4 dB, identical in source and target | VERIFIED (S1 Table 1) |
| Environmental noise applied to audio only | VERIFIED (S1 §3.1) |
| Segment counts and total 3896 | VERIFIED (S1 Table 2; total is our arithmetic on their table) |
| Normal-only training, anomalies test-only | VERIFIED (S1 §2) |
| Metadata columns and file names | VERIFIED (S3) |
| 100 ms non-overlapping timestamp-aligned windows, 35 ms gyro warm-up | VERIFIED (S3) |
| Published baseline AUCs (Table 4) | VERIFIED as *their* reported numbers; not independently reproduced |
| pAUC@0.1 convention | VERIFIED via S4 (summarised source) |
| RoboticArm/BrushlessMotor in DCASE 2024 Task 2 eval set | VERIFIED via S4 (summarised source) |
| Segment duration in seconds | **MEASURED (S6)** — 5.198938 s (BM), 7.499937 s (RA); the 6–10 s inference was wrong |
| Windows per segment (100 ms) | **MEASURED (S6)** — ~51 (BM), ~74 (RA) |
| `anomaly_label` value vocabulary | **MEASURED (S6)** — BM {normal, belt, magnetic}; RA {normal, loosescrewsA} |
| Whether released audio is pre-mixed with noise | **CONFIRMED (S6)** — pre-mixed; background label changes the mic spectrum but not the IMU |
| Actual file/segment counts in the release | **CONFIRMED (S6)** — 3,896 segments, 11,688 parquet, all readable |
| Zenodo record version number ("v2") | UNVERIFIED — API returns null |
| Reliability-aware fusion is not novel | VERIFIED (S5a, S5b — both read in full) |
| RoboticArm official split is recording-disjoint | **CONTRADICTED (S6)** — 62 of 101 recordings straddle train/test; 91.4% of normal test segments have a training sibling |
| BrushlessMotor classes overlap in recording time | **CONTRADICTED (S6)** — normals and anomalies are temporally disjoint (7.3 min gap) |
| Closest prior art (Chu & Zeng): primary task is supervised operating-state classification, no domain-shift protocol, confidence derived from supervised softmax | VERIFIED (S5a, full text) |
| Chu & Zeng contains no fault-state data | **CONTRADICTED** — the article is internally inconsistent; its §4.4 reports bearing-wear and rotor-imbalance experiments with ~1200 additional labelled samples. Earlier MechaSight claim withdrawn |
| k-NN/Mahalanobis embedding-distance justification | **UNVERIFIED** — no primary source read; must be substantiated before Phase 3 |
| `standardize()` epsilon defect affects the published Table 4 run | **CONTRADICTED** — that run uses `std_window`, whose epsilon is correctly placed |

---

## Summariser discrepancy — why raw sources were used

An automatic page-summarisation of the baseline repository's README returned a confident,
detailed description of a Hydra/MLflow pipeline with `conf/config.yaml`, `main.py`, a
`ModelFactory`, `IMADSDatasetTrain`/`IMADSDatasetTest` classes and HDF5 caching. Listing the
repository tree through the GitHub API showed those files do not exist there. The summary was
faithful to the README text — the README itself describes a different project — but taking it at
face value would have put non-existent components, and the wrong hyperparameters, into this
project's documentation as fact.

The rule adopted for the rest of MechaSight: **structural and numeric claims are taken only from
raw fetched text, raw API JSON, or source code read directly.** Summaries are used to find
sources, never to establish facts.
