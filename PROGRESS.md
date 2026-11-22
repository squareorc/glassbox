# GlassBox: Build Progress

> **Who this is for.** The project is built by one teammate and the final-year report is written by the other, so this file is the report writer's main source. It records the current state of the system and, for every stage:
> - what it does, and why it was designed that way
> - which alternatives were measured
> - the exact numbers, and how they were measured
> - what went wrong, and how it was fixed
>
> It's updated as work happens. `README.md` and `report_draft.md` get a full refresh on Day 7; until then, **trust this file over those two** for numbers and paths.

**Last updated:** 2026-11-22 (Day 5: end-to-end pipeline with audit traces, log PII scan, dashboard skeleton, §14)
**Branch:** `pipeline-build`
**Deadline:** 2026-11-28

---

## Contents
1. [Status at a glance](#1-status-at-a-glance)
2. [8-day plan](#2-8-day-plan)
3. [Stage: Datasets](#3-stage-datasets)
4. [Stage: QA evaluation set](#4-stage-qa-evaluation-set)
5. [Stage: PII redaction](#5-stage-pii-redaction)
6. [Stage: Retrieval](#6-stage-retrieval)
7. [Stage: Cited generation and claim verification](#7-stage-cited-generation-and-claim-verification)
8. [Robustness and overfitting checks](#8-robustness-and-overfitting-checks)
9. [Tests](#9-tests)
10. [How to run](#10-how-to-run)
11. [Key decisions log](#11-key-decisions-log)
12. [Known issues / open items](#12-known-issues--open-items)
13. [Daily log](#13-daily-log)
14. [Stage: End-to-end pipeline, audit trace and dashboard](#14-stage-end-to-end-pipeline-audit-trace-and-dashboard)

---

## 1. Status at a glance

| Stage | Status | Code |
|---|---|---|
| Synthetic datasets (250 + 2,500 notes) | ✅ Done | `glassbox/src/generate_*_dataset.py` |
| QA evaluation set v3 (390 + 160 questions) | ✅ Done | `glassbox/src/qa/build_qa_set.py` |
| PII redaction (incl. address recognizer, clinical-term guard) | ✅ Done | `glassbox/src/redaction/` |
| Question redaction | ✅ Done | `glassbox/src/redaction/query.py` |
| Retrieval (query analysis, two-hop selection) | ✅ Done | `glassbox/src/retrieval/` |
| Citation-based generation (with abstain guard) | ✅ Done | `glassbox/src/generation/` |
| Robustness / overfitting checks | ✅ Done (held-out results below) | `glassbox/src/robustness/` |
| Claim verification (NLI, 4 labels) | ✅ Done | `glassbox/src/verification/` |
| Redaction threshold sweep (privacy-utility curve) | ✅ Done (§5.1) | `glassbox/src/experiments/threshold_sweep.py` |
| End-to-end orchestrator with traces | ✅ Done (§14) | `glassbox/run.py`, `glassbox/src/pipeline.py` |
| Log / trace PII scan | ✅ Done (§14.3) | `glassbox/src/experiments/log_pii_scan.py` |
| Audit dashboard (Streamlit) | ⏳ Skeleton done (§14.4); experiments page Day 6 | `glassbox/src/dashboard/` |
| Experiments (privacy / utility / faithfulness) | ⬜ Day 6 | `glassbox/results/` |
| README + report refresh | ⬜ Day 7 | `README.md`, `report_draft.md` |

**Headline numbers so far** (realistic set, test split):

| | Value |
|---|---|
| Redaction F1 (typed, IoU ≥ 0.5) | **0.999** (Day 1: 0.893; original README: 0.838) |
| PII characters left visible after redaction | **0.2%** (Day 1: 15.4%) |
| PII values with any part visible in the retrieval index | **0.04%** (Day 2: 8.7%) |
| Answerable questions whose answer is in the retrieved context | **0.949**, 95% CI [0.922, 0.972] (Day 2: 0.840) |
| Context shown to the LLM per question | ~955 characters (whole-note retrieval: ~2,600) |
| Identity questions answerable from the redacted index | **0.000** (raw index with BM25: 1.000) |
| **Grounded answer accuracy** (answerable test questions, §7) | **0.858** [0.815, 0.899] (no-privacy baseline: 0.833) |
| Identity questions answered by the system (§7) | **0** of 84 (guard off: 84 of 84, all from other people's notes) |
| Corrupted claims caught by the verifier (§7.5) | **99.4%** [98.8, 99.9]; genuine claims wrongly flagged 1.2% (8.8% counting claims that cite incompletely) |
| Answers the verifier marks VERIFIED vs not | grounded accuracy **0.906** vs 0.492 |
| End-to-end time per question (§14) | **3.2 s** median when the model runs (redaction 4 ms, retrieval 6 ms, verification 37 ms; the rest is generation) |
| PII in the redacted system's logs and traces (§14.3) | **0** values of 17,603 searched (raw baseline traces: 3,143 hits) |

**How far these numbers generalise (§8). Read before quoting anything above.**

| | Template wording (above) | Held-out paraphrases (set B) |
|---|---|---|
| Retrieval evidence | 0.949 | **0.579** [0.512, 0.644] (before the robustness fixes: 0.324) |
| Grounded answer accuracy | 0.858 | **0.502** [0.435, 0.568] |
| Redaction, fully masked, held-out OOD notes | 0.999 F1 in-distribution | **0.819** (before fixes: 0.672) |

The headline numbers hold for questions and notes shaped like the generators' templates. On held-out wording and note formats the system is clearly weaker; §8 gives the causes.

---

## 2. 8-day plan

| Day | Date | Work | Status |
|---|---|---|---|
| 1 | Nov 20 | Cleanup, reproducible datasets, QA set v2, redaction fixes, test suite | ✅ |
| 2 | Nov 20 | Retrieval: chunking, ChromaDB + BM25 indexes (redacted and raw), access control, logging, leak audit; then query analysis, note grouping, fusion tuning | ✅ |
| — | Nov 20 | Independent audit of Days 1–2 (§12.1) | ✅ |
| 3a | Nov 20 | Audit fixes (§12.1): address recognizer, clinical-term guard, names in clinical context, question redaction, query-analysis ranges / "or" / invalid dates, low-information demotion, note-ordered and two-hop selection, QA set ×3, bootstrap CIs, `requirements.txt` | ✅ |
| 3 | Nov 21–22 | Generation: Qwen2.5-7B-Instruct (4-bit, MLX), cited JSON output, parsing and validation, abstain guard (§7). Then robustness and overfitting checks, general fixes, held-out evaluation (§8) | ✅ |
| 4 | Nov 22 | Verification: NLI verifier, 4 labels, corrupted-claim test set, model and margin chosen on dev (§7.4–7.5). Threshold sweep started overnight | ✅ (a day early) |
| 5 | Nov 22 | Orchestrator with trace JSON (question → redaction → retrieval → generation → verification); regenerate stale logs; log PII scan; dashboard skeleton (§14) | ✅ (two days early) |
| 6 | Nov 25 | Finish the dashboard; run the experiments and make the plots, with 95% confidence intervals; abstention and false-attribution rates as privacy metrics | ⬜ |
| 7 | Nov 26 | README and report refresh | ⬜ |
| 8 | Nov 27 | Polish, buffer, demo rehearsal, backup video | ⬜ |

---

## 3. Stage: Datasets

**What.** Two synthetic clinical-note corpora with injected fake PII. Every PII value's exact character position is recorded as ground truth.

| | Synthetic (baseline) | Realistic (main benchmark) |
|---|---|---|
| Generator | `generate_synthetic_dataset.py` | `generate_realistic_dataset.py` |
| Location | `glassbox/data/raw/synthetic/` | `glassbox/data/raw/realistic/` |
| Notes | 250 | 2,500 |
| Format | One short paragraph per note, 3 templates | EHR-style: 1,478 progress notes (3 templates incl. SOAP), 510 discharge summaries, 512 consultation notes |
| Conditions | 8 | 10 (diabetes, hypertension, asthma, depression, CKD stage 3, CAD, COPD, osteoarthritis, …) |
| PII entities | 838 | 18,041 |
| PII types | NAME, PATIENT_ID, PHONE_NUMBER, EMAIL, ADDRESS, DATE_OF_BIRTH | Same, plus provider, referring-physician and emergency-contact names; 6 phone formats; 4 patient-ID formats |

**Why synthetic.** Real clinical data (e.g. MIMIC) needs data-use approvals that don't fit the timeline. Synthetic data also gives *exact* ground truth, which makes the redaction evaluation more rigorous. The trade-off is lower external validity, which is stated as a limitation.

**Each raw note record contains:**
- `text`
- `entities`: the ground-truth PII spans
- `diagnosis`, `note_type`, `medications`
- `facts`: every value that went into the note. It contains PII and is used **only** to build the QA set; it never reaches retrieval or generation.
- `fields_in_text`: which values the chosen template actually rendered

**Fixes on Day 1:**
- **Reproducibility.** Dates used to be relative to the day the script ran, so regenerating gave different data. "Today" is now pinned to 2026-09-16, and regenerating produces byte-identical files (verified).
- **Discharge dates.** The admission date was computed from the run date, so it could fall after the discharge date. It's now 1–14 days before discharge.
- **Data layout.** The 250-note set had been overwritten by the 2,500-note set, so the 250-note numbers in the old README couldn't be reproduced. The two now live in separate folders.
- **Integrity.** 0 span errors across 18,879 ground-truth entities: every recorded span matches the text exactly.

---

## 4. Stage: QA evaluation set

**What.** Questions with gold answers and gold notes, used to score retrieval (Day 2), answer accuracy (Days 3 and 6) and the privacy-utility trade-off. Built automatically from the generators' `facts` by `glassbox/src/qa/build_qa_set.py`, saved as `qa_eval_set.jsonl` next to each dataset.

**Version 3 (Day 3a): three times larger.** The audit found that with 63 answerable test questions the 95% confidence interval on retrieval evidence was ±0.08, too wide to tell methods apart. The realistic set was tripled (60/40/30 → 180/120/90 = 390 questions) and the synthetic set doubled (40/25/15 → 80/50/30 = 160; it only has 34 possible aggregate questions).
- The builder shuffles candidates once and then takes the first *n*, so all 130 v2 questions are kept with the same IDs. They now sit mostly in the **dev** split.
- The new **test** split (273 questions) is almost entirely unseen: only 13 of its questions existed before, so nothing in it was inspected during development or the audit.
- All validity checks below were re-run on v3 and pass.

**Why it was rebuilt.**
- The old set had 8 questions, all of the form "What medications are prescribed for <diagnosis>?".
- Each had 100–336 relevant notes, which is impossible to retrieve in a top-5 system.
- Medications are never redacted, so the old set couldn't show any privacy cost.

**Three question types:**

| Type | Realistic n | Synthetic n | Example | Expected under redaction |
|---|---|---|---|---|
| `clinical_fact` | 180 | 80 | "What medication was prescribed for the Essential hypertension patient whose note is dated 2024-11-10?" | **Answerable.** The note is identified only by non-PII details. |
| `identity` | 120 | 50 | "What medication is John Wilkerson prescribed?" / "…for patient MRN0033092?" | **Should abstain.** The identifier is redacted. This *is* the measured privacy cost, and any confident answer is a privacy-induced hallucination for the verifier to catch. |
| `aggregate` | 90 | 30 | "Which medications were prescribed to Major depressive disorder patients who presented with depressed mood in December 2025?" | **Answerable.** Tests multi-note retrieval (2–4 gold notes each). |

- **Splits:** 30% **dev** (for tuning) and 70% **test** (for reporting), stratified by type. Realistic: dev 54 fact / 36 identity / 27 aggregate = 117; test 126 / 84 / 63 = 273. Synthetic: dev 48, test 112.
- **Scoring:** each answer item has `answer_keywords` (e.g. `["metformin", "500mg"]`). An item counts as correct if *all* its keywords appear in the generated answer. This is deterministic and model-free.
- **Validity checks:**
  - Every keyword verified present in its gold note (0 missing).
  - No gold answer is destroyed by redaction (v2 had 3, all because "Glipizide" was redacted as a name; fixed on Day 3a).
  - Every aggregate question lists *all* notes that match its diagnosis, symptom and period.
  - The diagnosis + date reference used by fact questions picks out exactly one note.
  - Names and IDs used in identity questions are unique across the whole corpus, including provider names.

---

## 5. Stage: PII redaction

**What.** Detects PII in each note and replaces it with a type tag (`[NAME]`, `[PATIENT_ID]`, …). It writes the sanitized notes and an audit log of every redaction (original value, tag, position, confidence). The same detector also redacts the user's **question** at query time (§6.6).

**Design.**
- **Base engine:** Microsoft Presidio (spaCy `en_core_web_lg` NER plus regex recognizers).
- **Custom recognizers on top**, because generic tools miss clinical formats:
  - **PATIENT_ID:** 1–2 letters + 6 digits, dash form (`P-123456`), MRN form (`MRN0012345`, score 1.0), and a purely numeric record number after an explicit label (`MRN: 00482913`, added Day 3a).
  - **DATE_OF_BIRTH:** only years 1936–2008, so visit dates aren't flagged. Covers ISO, US `MM/DD/YYYY` and written forms ("born March 20, 1985").
  - **PHONE_NUMBER:** two layers. Formatted numbers (parentheses, dots, `+1`/`001` prefixes, extensions) score 0.85. Raw 10-digit numbers score 0.40 and only pass the 0.5 threshold when a context word ("phone", "call", "contact", …) is nearby, which avoids false positives on medical codes.
  - **ADDRESS (added Day 3a).** spaCy split every address into fragments (a street name tagged as a person, a city as a place) and left house numbers and ZIP codes visible. The new recognizer matches the *structure* of a US postal address, so the whole address becomes one span:
    - street: house number + 1–5 capitalised words + optional unit (`Apt.`, `Suite`, `Unit`, `#` …) + `, City, ` + a **valid USPS state/territory code** + ZIP. The state + ZIP ending is what keeps it precise ("5 mg daily, Take PO" can't match).
    - military mail: `Unit 1234 Box 5678`, `PSC 1234, Box 5678`, `USNS/USS/USNV/USCGC <Name>`, then `APO/FPO/DPO AA/AE/AP` + ZIP.
    - a street line alone ("lives at 42 Oak Street") needs a common street-type word *and* a context word nearby ("address", "lives", "resides", "home"), because on its own it scores below the threshold.
    - None of these depend on the note templates' "Address:" labels or on a list of street names, so they generalise to free text (tested).
  - **Names in clinical context (added Day 3a).** spaCy missed some names in two standard positions: before a credential ("Lance Hebert, MD" on a signature line) and after a form label ("Referring Physician: Autumn Sharp", "Electronically signed by …", "Dr. …"). A pattern recognizer covers exactly these two positions.
  - The address and name patterns compile **case-sensitively**. Presidio compiles patterns case-insensitively by default, which made `[A-Z]` match lowercase words ("Dr. Patel to follow up" lost "to follow"). Found while testing; regression test added.
- **Clinical-term guard (added Day 3a, `clinical_terms.py`).** spaCy's general-purpose NER tagged clinical vocabulary as people or places. The guard drops a NAME detection, or a spaCy location, when the span is a measurement ("158/62", "BP 152/68") or made up only of clinical vocabulary:
  - ~230 common US generic drug names (roughly the top prescribed generics, not just the ones in the synthetic data),
  - clinical abbreviations (SpO2, BP, HR, PRN, QHS, …),
  - form labels ("Email", "Phone", "Patient", …).
  - It never filters the structural recognizers (IDs, phones, emails, DOBs, postal addresses).
- **Overlap resolution:** when detections overlap, the redactor keeps the best one. It ranks by confidence, then type specificity (pattern types like PATIENT_ID beat generic NER types like NAME), then length.
- Files: `detector.py`, `clinical_terms.py`, `redactor.py`, `query.py`, `evaluate.py`, `pipeline.py` in `glassbox/src/redaction/`.

**How it's measured.**
- **Typed precision/recall/F1:** a detection is a true positive if its type matches and its span overlaps the ground truth with IoU ≥ 0.5 (intersection over union of the character ranges).
- **Masking metrics:** these measure privacy directly.
  - Typed P/R also penalizes a span that was masked but got the wrong tag; from a privacy standpoint that span didn't leak.
  - *Masked recall* counts an entity as protected if any redaction covers it (IoU ≥ 0.5), whatever the type.
  - *Char leakage* is the share of ground-truth PII characters left visible.
- Metrics are computed on the spans actually redacted (after overlap resolution), i.e. on what downstream stages really see.
- **Independent check (audit, §12.1):** a plain string search of every ground-truth PII value in the redacted notes, outside the project's evaluator.

**Current results** (confidence 0.5, IoU 0.5). Source: `glassbox/results/redaction/{synthetic,realistic}_t0.50.json`

| Dataset | Precision | Recall | F1 | Masked recall | Char leakage |
|---|---|---|---|---|---|
| Synthetic (250 notes) | 0.999 | 0.998 | 0.998 | 0.998 | 0.4% |
| Realistic (2,500 notes) | 0.999 | 0.999 | **0.999** | 1.000 | 0.2% |

Realistic set, by type (Day 1 values in brackets):

| Entity | n | P | R | F1 | Char leakage |
|---|---|---|---|---|---|
| EMAIL | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| DATE_OF_BIRTH | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| PATIENT_ID | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| PHONE_NUMBER | 3,010 | 1.000 | 0.999 | 1.000 | 0.0% |
| NAME | 6,022 | 0.999 (0.841) | 0.997 (0.987) | 0.998 (0.908) | 0.9% (1.8%) |
| ADDRESS | 1,509 | 0.987 (0.000) | 1.000 (0.000) | 0.993 (0.000) | 0.0% (64.3%) |

- **Over-redaction:** 0 redactions now hit non-PII text, on both datasets (Day 2: 148, mostly "Glipizide", "SpO2", "Email" and BP readings).
- **Exact PII values still verbatim in the redacted notes:** 2 of 18,041 (two provider names) on the realistic set; 1 of 838 on the synthetic set (Day 2: 285, of which 234 addresses).
- **Why NAME precision rose too:** the audit showed 965 of the ~1,130 "wrong" NAME detections were fragments of addresses. A whole-address span replaces them.
- **Remaining "errors" are mostly scoring artefacts:** "Dr. William Warren" is gold, "William Warren" was masked (the title stays). The name is hidden; the span just doesn't match. 3 surnames that are also place names (Madison, Medina, Valencia) are tagged ADDRESS instead of NAME; still masked.
- **Read these numbers with care.** They are measured on synthetic notes whose addresses come from one generator (Faker), so the address recognizer is close to the data's format by construction. Generalisation is covered separately by hand-written tests on formats not in the data (§9), and by the stated limitations in §12.

**Problems found and fixed on Day 1.** Scaling to the 2,500-note set exposed these; they make good "engineering insight" material for the report.

| Problem | Symptom | Fix | Effect |
|---|---|---|---|
| Overlap resolver bug | Spans were processed left to right, so the *earliest* span always won. The documented "keep the higher-confidence span" never happened. | Rank all candidates globally by score, then type specificity, then length | Correct behaviour; regression test added |
| Tie between NER and ID regex | spaCy tags bare IDs like `pm010651` as PERSON at the same score, so 22 synthetic IDs got tagged `[NAME]` | Tie-break in favour of specific pattern types | PATIENT_ID recall 0.912 → 1.000 (synthetic) |
| NER spans crossing line breaks | e.g. `MRN0033092\nDate` tagged as one NAME, swallowing the ID and the next field label | Cut NER spans at the first newline | Fixed 98 realistic ID misses (with the next fix) |
| DEA licence false match | Presidio's DEA-number checksum recognizer matched `RN6659051` inside `MRN6659051` at score 1.0 | MRN pattern score raised to 1.0 (the explicit prefix is unambiguous) | PATIENT_ID recall → 1.000 (realistic) |
| UK NHS recognizer | Fired on 42 raw US phone numbers that happen to pass its checksum | Removed it (domain is US records) | PHONE recall 0.985 → 0.999 |
| "MD" read as Maryland | `Lindsey Roman, MD` → `[NAME], [ADDRESS]` | Drop clinical credentials tagged as locations, unless followed by a ZIP | Fewer ADDRESS false positives; "MD" stays readable |

Overall realistic F1 went from **0.838 (old README) to 0.893** on Day 1.

**Problems found and fixed on Day 3a** (from the audit, §12.1):

| Problem | Symptom | Fix | Effect (realistic) |
|---|---|---|---|
| Addresses fragmented by NER | 64% of address characters visible; 98% of addresses partly visible in the index; street names counted as NAME false positives | Structural US postal-address recognizer | ADDRESS F1 0.000 → 0.993; NAME precision 0.841 → 0.999 |
| Clinical terms redacted | "Glipizide" → `[NAME]` in 97 notes (3 gold answers lost), "SpO2" 26, "Email" 17, BP readings 9 | Clinical-term guard on NER detections | Over-redactions 148 → 0; 0 gold answers lost |
| Provider names missed | 51 names verbatim, mostly signatures ("Lance Hebert, MD") and "Referring Physician: …" | Name-before-credential and name-after-label patterns | Verbatim names 51 → 2 |
| Numeric MRNs missed | `MRN: 00482913` not redacted (not in the data; found by probing) | Labelled-number pattern | Tested |
| Case-insensitive patterns | Presidio's default flag let `[A-Z]` match lowercase words | Compile the new patterns case-sensitively | Tested |

Overall realistic F1: **0.893 → 0.999**.

### 5.1 Redaction threshold sweep (privacy-utility curve)

**What:** the detector's confidence threshold (a detection is redacted only if its score is at least the threshold) was varied from 0.3 to 0.8. For each value the whole downstream pipeline was re-run from scratch: redaction of all 2,500 notes, a fresh index (`redacted-t<thr>-section`), retrieval on the test split, and generation on all 390 questions (reported on test). Script: `glassbox/src/experiments/threshold_sweep.py`; results: `glassbox/results/experiments/threshold_sweep.json`, plus per-threshold redaction reports `glassbox/results/redaction/realistic_t0.{30..80}.json`. Each point took about 19–24 minutes (about 2 hours in total, run overnight Nov 22).

**Why:** 0.5 was used everywhere so far, and nobody had checked whether that choice mattered. The sweep shows what each side of the trade-off costs: privacy (PII leaked) against utility (clinical text kept, answers correct).

| Threshold | Precision | Recall | F1 | Missed PII spans | Over-redactions (clinical text) | Char leakage | Phones fully leaked | Retrieval evidence@auto | Grounded accuracy (test) [95% CI] | Identity answered |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.809 | 0.999 | 0.894 | 23 | **4,219** | 0.24% | 0% | 0.952 | 0.869 [0.827, 0.907] | 0 |
| 0.40 | 0.999 | 0.999 | 0.999 | 23 | 0 | 0.24% | 0% | 0.949 | 0.858 [0.815, 0.899] | 0 |
| **0.50 (chosen)** | 0.999 | 0.999 | 0.999 | 23 | 0 | 0.24% | 0% | 0.949 | 0.858 [0.815, 0.899] | 0 |
| 0.60 | 0.999 | 0.999 | 0.999 | 23 | 0 | 0.24% | 0% | 0.949 | 0.858 [0.815, 0.899] | 0 |
| 0.70 | 0.999 | 0.999 | 0.999 | 23 | 0 | 0.24% | 0% | 0.949 | 0.858 [0.815, 0.899] | 0 |
| 0.80 | 0.999 | 0.973 | 0.986 | **491** | 0 | **1.85%** | **15.6%** | 0.949 | 0.858 [0.815, 0.899] | 0 |

**How measured:** precision, recall and F1 come from span matching against the generator's ground-truth entities. Char leakage is the share of PII characters left visible. "Phones fully leaked" comes from the leak audit (`leak_audit` in the JSON): the share of gold phone numbers that appear verbatim in the redacted index. Retrieval and generation use the same metrics and test split as §6 and §7, with bootstrap 95% CIs.

**Findings:**
- **Flat region from 0.4 to 0.7.** All metrics are identical across this range, so 0.5 sits in the middle of a stable plateau and the results don't depend on its exact value. This is the justification for the operating point: it was fixed before the sweep, and the sweep confirms that nearby choices give the same results.
- **Below 0.4, over-redaction.** Several recognizers deliberately score 0.40–0.45 and only cross the threshold when a context word boosts them: bare 6–9 digit record numbers, bare 10-digit phones, street lines without a city or ZIP, written-out birth dates. At 0.3 they fire with no context at all, which gives 4,219 false redactions (PATIENT_ID F1 drops to 0.542) and destroys clinical numbers in the text.
- **Above 0.7, leakage.** The context-boosted bare 10-digit phone pattern scores 0.40 + 0.35 boost = 0.75, so at 0.8 it never fires: 15.6% of phone numbers (the unformatted ones) leak verbatim, missed spans go from 23 to 491, and char leakage rises about 8×. Formatted phones (score 0.85) are still caught.
- **0.3's higher accuracy (0.869) is not a real gain:** its CI overlaps 0.5's, and it comes from a setting that wipes out 4,219 pieces of clinical text.
- **Limitation:** utility is insensitive to the 0.8 leak. Questions never ask about phone numbers, and identity questions are refused at every threshold, so accuracy can't detect it. The privacy metrics have to decide the upper end. For the same reason the "flat" utility curve says more about this QA set than about every possible use.
- **Reproduction check:** the sweep re-ran 0.5 from scratch in its own folder and index. It reproduced the main results exactly (F1 0.999, evidence 0.949, grounded accuracy 0.858 [0.815, 0.899]; §5, §6, §7.2).

---

## 6. Stage: Retrieval

**Purpose.** Retrieval is how GlassBox minimizes data. Instead of giving the LLM the whole corpus, or whole notes as a typical RAG system does, it selects the few most relevant pieces of *sanitized* text for each question and logs exactly what was selected and why.

**Final pipeline:**

```
question → question redaction (names/IDs in the question → [NAME], …; only types are logged)
         → Query Analysis (filters + residual query + scope + ask/condition split)
         → candidate chunks satisfying the filters (redacted index only)
         → dense ranking + BM25 ranking → weighted reciprocal rank fusion (2 : 1)
         → low-information chunks (placeholder-only headers) moved to the back
         → note selection: notes ordered by their best chunk, ≤ 2 chunks each;
           multi-note questions: the 2nd chunk is that note's best section for the "ask"
         → top-k (5 or 10) → logged
```

Code is in `glassbox/src/retrieval/`: `chunker.py`, `bm25.py`, `index.py`, `query_analysis.py`, `retriever.py`, `evaluate.py`; question redaction in `glassbox/src/redaction/query.py`.

### 6.1 Chunking (`chunker.py`)
- **Default: one chunk per note section.**
  - The templates separate sections with blank lines (`SUBJECTIVE:`, `ASSESSMENT/PLAN:`, `S:`/`O:`/`A:`/`P:`, `HOSPITAL COURSE:`, …).
  - Short *unlabelled* blocks are merged into a neighbour: titles like "PROGRESS NOTE", sign-offs like "Thank you for this referral.".
  - Short labelled sections like "CHIEF COMPLAINT: headache" stay separate. A first version merged these too, which mis-named sections; it was caught by checking section counts per template.
- **Why sections:**
  1. Less data reaches the LLM.
  2. Each citation points at one short paragraph a human can check.
  3. The Day 4 NLI verifier needs short evidence passages.
- **Context line.** Each chunk starts with `[Progress note | 2024-11-10 | Essential hypertension]`, built only from non-PII metadata (note type, visit date, diagnosis), so a section read on its own still says which visit it belongs to.
- **Ablation: one chunk per whole note.** Compared in §6.8.
- **Result** (realistic redacted): 18,003 chunks, 7.2 per note. Every template's sections are recognised, and every character of every note is covered (both tested).
  - The synthetic notes are single paragraphs, so both strategies give one chunk per note there.
  - The raw index has 18,790 chunks: real names are longer than `[NAME]`, which pushes 781 sign-off lines over the merge threshold. The headline comparison isn't affected (§6.8).

### 6.2 Search methods (`bm25.py`, `retriever.py`)

| Method | How it works | Good at | Bad at |
|---|---|---|---|
| **Dense** | `BAAI/bge-small-en-v1.5` (384-dim) embeddings, cosine similarity, stored in ChromaDB. Run locally on the Apple GPU (MPS). Queries get bge's retrieval instruction prefix. | Meaning: "what medication" → the plan section | Exact strings: `2024-11-10` vs `2024-11-19` look almost identical |
| **BM25** | Classic keyword relevance (Okapi BM25, k1 = 1.5, b = 0.75), ~50 lines written in-house. Tokenizer keeps dates, IDs, BP readings and decimals as single tokens. | Exact dates, IDs, names, drug names | Synonyms: "heart rate" vs "HR" |
| **Hybrid** | Weighted reciprocal rank fusion: score = w_dense/(60 + rank_dense) + w_bm25/(60 + rank_bm25) (Cormack et al., 2009), over the top 50 candidates of each. Weights (2.0, 1.0) chosen on the dev split (§6.7). | Combines both | |

### 6.3 Query analysis (`query_analysis.py`)
This is the "Query Analysis" box from the README's architecture diagram. It reads the (already redacted) question and extracts:

| Output | Example | Used for |
|---|---|---|
| **Diagnosis filter** | "…Essential hypertension patient…" → `diagnosis = Essential hypertension`. Several are kept: "hypertension or diabetes" → either (Day 3a). | Only chunks from notes with that diagnosis are candidates. Exact names first, then a small alias table ("diabetes", "CKD", "COPD", …). |
| **Date filter** | ISO `2024-11-10` → that day. `03/15/2025` → that day. "December 2025" / "Sept 2025" / "May, 2025" → that month. "in 2024" → that year. **Ranges** (Day 3a): "between March 2025 and June 2025", "from March to June 2025", "2024-01-05 to 2024-02-10", "from 2024 to 2025". | Only chunks whose visit date falls in the range |
| **Invalid dates** | "2025-13-01", "2/30/2025" → reported as `invalid_date`, **no** date filter (Day 3a). Before, "from 2025-13-01" silently became "all of 2025". | Logged for the audit trail |
| **Residual query** | "What heart rate was recorded for the Asthma patient whose note is dated 2024-10-07?" → "What heart rate was recorded for the patient ?" | Ranking. Once the filters guarantee the constraints, repeating them in the ranking query only rewards chunks that restate the date or diagnosis (patient header, impression) over the chunk that answers. |
| **Scope** | Exact date → `single_note`. Month, year, range, several diagnoses or plural "patients" → `multi_note` | Chooses k: 5 for single-note questions, 10 for multi-note ones |
| **Ask / condition** (Day 3a) | Multi-note questions are split at the first relative clause: "Which medications were prescribed to patients **who** presented with fatigue?" → ask "Which medications were prescribed to patients", condition "presented with fatigue" | Two-hop selection (§6.4) |

- Filters use only non-PII metadata, so identity questions ("What is John Wilkerson prescribed?") get **no** filter (tested).
- If the filters match nothing, retrieval falls back to unfiltered search and logs `fallback: true`. Day 3 generation must abstain in that case, since the retrieved notes don't meet the question's constraints.
- The filters are applied inside ChromaDB (a `where` clause on diagnosis, `$eq` or `$in`, and an integer `visit_date_int`) and to BM25 candidates.
- Bug found while building it (Day 2): filtered BM25 dropped chunks sharing no word with the question, so "heart rate" never reached "HR 94". Now every filtered chunk is a candidate, with zero-score chunks last.
- Limitation: the QA set's questions are generated from templates, so the evaluation numbers only cover that phrasing. The free-text cases above (ranges, "or", commas, invalid dates) were found by probing and are covered by unit tests, not by the QA metrics.

### 6.4 Selection: which chunks fill the k slots
Three steps, each measured in §6.8.

1. **Low-information demotion** (Day 3a). After redaction, header, contact and signature sections ("PATIENT: [NAME] / MRN: [PATIENT_ID] / DOB: [DATE_OF_BIRTH]") are only placeholders and form labels. They can't answer anything, yet they match "patient" questions well, and they are what invites the misattribution risk in §12.1 #1. A chunk is *low-information* if it contains a placeholder and, after removing placeholders, form labels and numbers, fewer than 3 words remain. These chunks are moved behind all others, not dropped. This flags exactly the patient_info, contact and signature sections (2,500 / 2,499 / 1,001 of them) and no chunk of the raw index, so the baseline is unaffected.
2. **Note-grouped selection.** Each note contributes at most 2 chunks (`PER_NOTE`) before any note contributes more; leftover slots are filled in score order.
   - Day 2 version: the capped chunks stayed in score order, interleaving notes. A qualifying note's second chunk often fell below the cut-off.
   - Day 3a version: notes are ordered by their best chunk, and each note's chunks are placed together.
   - For a single-note question both versions equal plain ranking.
3. **Two-hop answer sections** (Day 3a). The audit found aggregate questions failed for one reason: their two hops live in different sections of each note. The symptom that *qualifies* a note is in SUBJECTIVE, while the medication that *answers* is in PLAN. Ranking the whole question finds every qualifying note (recall 1.0) but then picks another symptom-bearing section (chief complaint, HPI) as the note's second chunk.
   - Fix: notes are chosen by their best chunk for the whole question. Each chosen note's second slot goes to that note's best section for the **ask** alone, ranked with the same hybrid method.
   - Every returned chunk records `selected_by: "question" | "ask"` for the audit trail.
   - Remaining misses: the ask "Which medications were prescribed" sometimes ranks LAB ORDERS or REVIEW OF RECORDS above a plan line like "Start EPO 4000 units", which never uses the word "medication".

### 6.5 Indexes (`index.py`)
- 4 per dataset (8 total): {**redacted** (the system), **raw** (no-privacy baseline)} × {section, note}.
- Stored in `glassbox/data/indexes/<dataset>/<name>/` (gitignored; all 8 rebuild in about 90 s). Each holds:
  - `manifest.json`: provenance (source, threshold, chunking, model, counts, build time)
  - `chunks.jsonl`
  - `chroma/`
- The raw baseline reads only the note `text`. The loader drops `facts`, `entities` and `patient_name`.
- Integrity (audit): chunk file count = ChromaDB count = manifest count for all indexes, no duplicate IDs.

### 6.6 Access control, question redaction and logging
- The `Retriever` only opens an index whose manifest says `source: redacted`.
- Opening the raw baseline requires an explicit `baseline=True`; otherwise it raises `AccessDenied` (tested).
- At most 10 chunks per query (`MAX_K`).
- **Question redaction** (Day 3a, `redaction/query.py`). The audit found a name typed into the question went, unredacted, into the query embedding, the retrieval log and (from Day 3) the LLM prompt. Now the question is redacted first, with the same detector as the notes:
  - only the redacted question is embedded, ranked and logged;
  - the log records the PII *types* and offsets removed, never the values (tested: the name never appears in the log file);
  - `identifiers` (e.g. `["NAME"]`) tells Day 3 generation that the question is about a specific person, which triggers the abstain rule;
  - on by default for the redacted index, off for the raw baseline;
  - it changes no clinical question (dev evidence identical with and without it, §6.8).
- Not implemented: real role-based permissions. That's out of scope and should be stated as such.
- **Retrieval log.** With a log path set, every query appends one JSON line:
  - timestamp, question ID, the (redacted) question, the question-redaction record
  - index, source, baseline flag
  - the full query analysis (filters, matched phrases, residual query, scope, ask/condition, fallback)
  - method, fusion weights, per-note cap, demotion and two-hop flags, k, latency
  - for each returned chunk: rank, chunk/doc ID, section, character offsets, dense/BM25/fused scores and ranks, `low_information`, `selected_by`
- Evaluation runs write `glassbox/logs/retrieval/<dataset>/<index>.jsonl` (gitignored). This is stage 2 of the audit trail and feeds the dashboard.

### 6.7 How retrieval is measured (`evaluate.py`)

| Metric | Definition | Why |
|---|---|---|
| **recall** | Fraction of a question's gold notes with at least one chunk in the top-k | Did we find the right patients/visits? |
| **evidence** | Fraction of a question's gold answer items (e.g. each medication) present in top-k chunks *from gold notes* | Could the LLM actually answer from what it's shown? Stricter than recall: a note's symptom chunk counts for recall but not as evidence for its medication. |
| **context (chars)** | Characters of text in the top-k chunks | Data minimization: how much the LLM sees |
| MRR | Mean of 1/rank of the first gold-note chunk | Ranking quality |
| **95% CI** (Day 3a) | Percentile bootstrap over questions (5,000 resamples, seed 0) for answerable evidence | Says whether a difference is real |

- **@auto** means "at the k the system actually uses" (5 for single-note questions, 10 for multi-note; 5 when query analysis is off). Fixed k = 1, 3, 5, 10 are in the JSON too.
- **"Answerable"** means fact + aggregate questions. Identity questions are excluded because they're unanswerable under redaction by design; they're reported separately as the privacy result.
- **Splits.** Every design choice was made on the **dev** split (117 questions; 81 answerable). All tables below are on the **test** split (273 questions: 126 fact, 84 identity, 63 aggregate; 189 answerable).
- **Answer matching** (`qa/build_qa_set.py: answer_item_found`, reused by Day 3's answer scoring):
  - All keywords must appear.
  - Numbers must be standalone (HR "94" doesn't match inside "BP 157/94").
  - Follow-ups must read "<n> weeks".
- **Fusion weight tuning** (dev split, redacted section index, query analysis on, final selection; `results/retrieval/realistic_t0.50_fusion_tuning.json`):

| Dense weight | BM25 weight | Answerable evidence | Fact evidence | Aggregate evidence | Evidence@3 |
|---|---|---|---|---|---|
| 0.0 (BM25 only) | 1.0 | 0.801 | 0.926 | 0.552 | 0.635 |
| 0.25 | 1.0 | 0.889 | 0.963 | 0.741 | 0.653 |
| 0.5 | 1.0 | 0.939 | 1.000 | 0.818 | 0.748 |
| 1.0 (Day 2 choice) | 1.0 | 0.943 | 1.000 | 0.830 | 0.794 |
| 1.5 | 1.0 | 0.962 | 1.000 | 0.886 | 0.813 |
| **2.0 (chosen)** | **1.0** | **0.968** | 1.000 | 0.904 | **0.813** |
| 3.0 – 5.0 | 1.0 | 0.968 (plateau) | 1.000 | 0.904 | 0.786 |
| 1.0 (dense only) | 0.0 | 0.968 | 1.000 | 0.904 | 0.786 |

  - The plateau starts at 2.0, and 2.0 is also the best plateau point at k = 3, so it's the default.
  - Day 2's tuning (39 dev questions, old selection) had chosen (1.0, 1.0); with 3× the dev questions and the new selection, more dense weight is better.
  - Dense-only ties on dev, but hybrid is kept: whenever query analysis can't filter, BM25 is what matches exact names, IDs and drug strings (raw-index identity recall: BM25 1.000, dense 0.512).
  - Side effect: with dense weighted higher, the raw *hybrid* index finds fewer identity targets (0.845 at 1:1 → 0.345 when tuned on Day 3a; 0.548 with exact dense search, Day 5 §14.6: the approximate search had been hiding part of the baseline's identity matches). This only concerns the no-privacy baseline. For the privacy comparison, use the strongest attacker on raw data, BM25, which finds 100%.

### 6.8 Results (realistic set, test split, 273 questions)

**What each Day 3a change adds** (`results/retrieval/realistic_t0.50_selection_ablation.json`; redacted section index, hybrid 2:1, query analysis on; steps chosen on dev):

| Step | Dev answerable evidence | Test answerable evidence (95% CI) | Test fact evidence | Test aggregate evidence | Context (chars) |
|---|---|---|---|---|---|
| Day 2 selection | 0.832 | 0.840 [0.792, 0.885] | 0.984 | 0.553 | 955 |
| + question redaction | 0.832 | 0.840 [0.792, 0.885] | 0.984 | 0.553 | 955 |
| + low-information demotion | 0.847 | 0.854 [0.808, 0.896] | 1.000 | 0.561 | 966 |
| + note-ordered grouping | 0.919 | 0.919 [0.886, 0.949] | 1.000 | 0.757 | 972 |
| **+ two-hop answer sections (final)** | **0.968** | **0.949 [0.922, 0.972]** | **1.000** | **0.847** | **955** |

- Question redaction changes nothing for clinical questions, as intended: it's a privacy measure.
- The big gains are from selection: aggregate evidence 0.553 → 0.847 at the same context size.
- The final CI doesn't overlap the Day 2 CI, so the improvement is real, not noise.

**Effect of query analysis** (system configuration: redacted index, section chunks, hybrid, final selection):

| | Answerable recall | Answerable evidence | Fact evidence | Aggregate recall | Aggregate evidence | Context (chars) |
|---|---|---|---|---|---|---|
| No query analysis (k = 5) | 0.330 | 0.103 | 0.151 | 0.165 | 0.008 | 613 |
| **With query analysis** | **0.994** | **0.949** | **1.000** | **0.983** | **0.847** | 955 |

**Privacy vs. utility: redacted vs. raw index** (same configuration):

| Index | Answerable recall | Answerable evidence (95% CI) | Identity recall: hybrid | Identity recall: BM25 (strongest attacker) |
|---|---|---|---|---|
| Raw (no privacy) | 0.994 | 0.907 [0.870, 0.940] | 0.548 | **1.000** |
| **Redacted (GlassBox)** | **0.994** | **0.949** [0.922, 0.972] | **0.000** | **0.000** |

- **Redaction costs nothing measurable for clinical and aggregate questions, and completely blocks identity lookups.** That's the central privacy-utility result at the retrieval level.
- The redacted index scores *higher* than raw. Redaction doesn't add information: the difference comes from the low-information demotion, which can only recognise header sections once their values are placeholders. In the raw index those sections hold real names and compete for slots. The report should present this as "redaction costs nothing", not "redaction helps".
- One identity question (1/84) used to be found by BM25 on the redacted *whole-note* index: the redacted question "What symptom did Mrs. [NAME] report?" matched one of the few notes where the title "Mrs." survives in front of `[NAME]`. Since Day 3's single-name rule (a capitalised word after "did/is/for…" is a name), the question is redacted to "What symptom did [NAME]. [NAME] report?", the title is gone and the match no longer happens (identity recall 0.000 for every redacted configuration, re-measured Day 5). The underlying issue stays: titles left in the *notes* are quasi-identifiers (§12).

**Chunking: data minimization vs. evidence** (redacted index, query analysis on, hybrid):

| Chunking | Answerable evidence | Aggregate evidence | Context (chars) |
|---|---|---|---|
| **Section (default)** | 0.949 | 0.847 | **955** |
| Whole note | 1.000 | 1.000 | 2,598 (2.7× more) |

The gap was 0.840 vs 0.984 on Day 2; two-hop selection closed most of it. Whole notes still find every answer, but show the LLM 2.7× more text, including every other section of each note. This is a genuine trade-off for the report's discussion.

**Methods** (redacted, section, query analysis on, final selection):

| Method | Answerable evidence | Fact evidence | Aggregate evidence |
|---|---|---|---|
| Dense | 0.946 | 1.000 | 0.839 |
| BM25 | 0.812 | 0.944 | 0.548 |
| Hybrid 2:1 (chosen on dev) | 0.949 | 1.000 | 0.847 |

Dense and hybrid are statistically tied on test (CIs [0.919, 0.970] and [0.922, 0.972]); report them as tied. Hybrid was chosen on dev and handles exact strings when no filter applies.

<details>
<summary>Full table: all 24 configurations, realistic test split (@auto); regenerated Day 5 with the current code and exact dense search (§14.6)</summary>

| Index | Chunking | Method | Query analysis | Answerable recall | Answerable evidence | Context (chars) | Fact evidence | Aggregate recall | Aggregate evidence | Identity recall |
|---|---|---|---|---|---|---|---|---|---|---|
| redacted | section | dense | off | 0.154 | 0.045 | 609 | 0.064 | 0.161 | 0.008 | 0.000 |
| redacted | section | bm25 | off | 0.691 | 0.296 | 616 | 0.444 | 0.071 | 0.000 | 0.000 |
| redacted | section | hybrid | off | 0.330 | 0.103 | 613 | 0.151 | 0.165 | 0.008 | 0.000 |
| redacted | section | dense | on | 0.990 | 0.946 | 945 | 1.000 | 0.970 | 0.839 | 0.000 |
| redacted | section | bm25 | on | 1.000 | 0.812 | 968 | 0.944 | 1.000 | 0.548 | 0.000 |
| redacted | section | hybrid | on | 0.994 | 0.949 | 955 | 1.000 | 0.983 | 0.847 | 0.000 |
| redacted | note | dense | off | 0.143 | 0.156 | 3233 | 0.151 | 0.128 | 0.167 | 0.000 |
| redacted | note | bm25 | off | 0.684 | 0.689 | 3556 | 1.000 | 0.053 | 0.066 | 0.000 |
| redacted | note | hybrid | off | 0.302 | 0.318 | 3297 | 0.389 | 0.127 | 0.175 | 0.000 |
| redacted | note | dense | on | 0.994 | 0.996 | 2605 | 1.000 | 0.983 | 0.987 | 0.000 |
| redacted | note | bm25 | on | 1.000 | 1.000 | 2578 | 1.000 | 1.000 | 1.000 | 0.000 |
| redacted | note | hybrid | on | 1.000 | 1.000 | 2598 | 1.000 | 1.000 | 1.000 | 0.000 |
| raw | section | dense | off | 0.138 | 0.040 | 647 | 0.056 | 0.161 | 0.008 | 0.512 |
| raw | section | bm25 | off | 0.691 | 0.127 | 629 | 0.191 | 0.071 | 0.000 | 1.000 |
| raw | section | hybrid | off | 0.345 | 0.114 | 641 | 0.167 | 0.161 | 0.008 | 0.548 |
| raw | section | dense | on | 0.990 | 0.919 | 946 | 1.000 | 0.970 | 0.757 | 0.512 |
| raw | section | bm25 | on | 1.000 | 0.802 | 964 | 0.929 | 1.000 | 0.548 | 1.000 |
| raw | section | hybrid | on | 0.994 | 0.907 | 948 | 0.992 | 0.983 | 0.738 | 0.548 |
| raw | note | dense | off | 0.063 | 0.079 | 3958 | 0.048 | 0.094 | 0.140 | 0.429 |
| raw | note | bm25 | off | 0.695 | 0.703 | 3777 | 1.000 | 0.085 | 0.108 | 1.000 |
| raw | note | hybrid | off | 0.206 | 0.221 | 3861 | 0.254 | 0.111 | 0.156 | 0.512 |
| raw | note | dense | on | 0.998 | 0.998 | 2785 | 1.000 | 0.995 | 0.995 | 0.429 |
| raw | note | bm25 | on | 1.000 | 1.000 | 2756 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw | note | hybrid | on | 0.998 | 0.998 | 2774 | 1.000 | 0.995 | 0.995 | 0.512 |

</details>

**Synthetic set** (250 notes, 160 questions, test split, section index): with query analysis, every method reaches 1.000 recall and evidence on answerable questions, at about 770 characters of context; the selection steps make no difference there (single-paragraph notes). Redacted identity recall is 0.000 for hybrid (0.029 for BM25, one title match). Raw identity recall is 1.000 (BM25).

**Leak audit, realistic section indexes.** For every ground-truth PII value: is the full value, or any identifying part of it, visible in that note's indexed chunks (i.e. text an LLM could be shown)?

| PII type | n | Redacted: full value visible | Redacted: any part visible | Day 2: any part visible | Raw |
|---|---|---|---|---|---|
| ADDRESS | 1,509 | 0.0% | **0.0%** | 98.0% | 100% |
| NAME | 6,022 | 0.03% | 0.12% | 1.4% | 100% |
| PATIENT_ID / DOB / EMAIL / PHONE | 10,510 | 0.0% | 0.0% | 0.0% | 100% |
| **Overall** | 18,041 | 0.01% | **0.04%** | 8.7% | 100% |

### 6.9 The Day 2 morning baseline, for the "before" story
The first evaluation, with no query analysis, equal-weight fusion and k = 5, measured only note recall. Realistic test results (v2 QA set):
- Hybrid had answerable recall 0.645 and was *worse* than BM25 alone.
- Aggregate recall was 0.103 for BM25 and 0.127 for hybrid.
- Diagnosis: questions hinge on exact dates (which dense search can't distinguish), and on month names that neither method connects to ISO dates.

That analysis motivated §6.3 and §6.4. The full baseline numbers are in git history: commit `5c62519`, `results/retrieval/*.json`. The end-of-Day-2 numbers (answerable evidence 0.840) are in commit `6e4c656`.

---

## 7. Stage: Cited generation and claim verification

**What.** Stage 3 of the pipeline. A local LLM answers each question **only from the retrieved, redacted sections**, splits the answer into claims, and cites the section(s) each claim comes from. It refuses when it can't answer honestly. Code: `glassbox/src/generation/` (`prompt.py`, `parse.py`, `generator.py`, `evaluate.py`).

**Model.** Qwen2.5-7B-Instruct, 4-bit, run locally with MLX on the Apple GPU, so no data leaves the machine. Greedy decoding (temperature 0), so a re-run reproduces the same output. Measured: loads in 1.3 s, ~3.8 s per answer (3.0 s fact, 5.4 s aggregate), peak memory 4.5 GB.

### 7.1 Design
- **Abstain guard, before the model is called** (`generator.py: abstain_reason`):
  1. *identifier_in_question*: question redaction (§6.6) found a name, ID, phone, … and the index is the redacted one. The records can't say who `[NAME]` is, so any answer would attribute someone else's note to the person asked about.
  2. *retrieval_fallback*: the question's diagnosis/date constraints match no note, so the retrieved notes don't satisfy the question.
  - Why code, not a prompt instruction: the audit showed the model treats `[NAME]` as the person asked about (§12.1 #1). A rule can't be argued out of, costs nothing, and the log says exactly why the system refused. The guard-off experiment (§7.3) measures what happens without it.
  - The model can also refuse itself (`"insufficient": true`) when the sources don't contain the answer.
- **Prompt** (`prompt.py`): sources numbered `[C1] (Note 1) …`, each with its non-PII context line. The "Note n" label is an opaque per-prompt number that tells the model which sections come from the same visit (so it can connect a note's symptom to its own plan) without revealing any identifier. Rules: use only the sources; every claim cites; copy doses exactly; never guess bracketed values or assume `[NAME]` is someone in the question; for multi-patient questions check each note against every condition; say when the answer isn't there.
- **Output**: JSON `{"answer", "claims": [{"text", "citations"}], "insufficient"}`. JSON rather than inline markers gives the Day 4 verifier clean claim-by-claim input and makes malformed output detectable.
- **Parsing** (`parse.py`): strips code fences, takes the first complete JSON object, retries once with the error message if unusable. Citations are normalised (`C2`, `[C2]`, `c2`, `Source 2` → the chunk ID); citations to sources that weren't shown are dropped and recorded; claims left without a citation are flagged.
- **Logging**: one JSON line per question in `glassbox/logs/generation/`: the redacted question (never the original), chunk IDs shown, prompt hash and version, raw model output, parsed claims and citation problems, abstain reason, retries, tokens, latency. Stage 3 of the audit trail.

### 7.2 Prompt development (dev split only)

| Prompt | Change | Dev accuracy (plain) | All items correct | Drug precision |
|---|---|---|---|---|
| v1 | Basic rules | 0.890 | 0.790 | 0.957 |
| **v2 (chosen)** | + "Note n" labels, check each note against every condition, copy exact doses | **0.941** | **0.864** | 0.949 |
| v3 | v2 with claims before the answer in the JSON | 0.924 | 0.827 | 0.957 |

- v1's errors were almost all aggregate questions: incomplete drug lists, a drug from a note that didn't meet the conditions, drug names without doses.
- v3 tested whether writing claims first (note by note) improves the combined answer. It didn't, so v2 stays. All three runs are kept in `results/generation/realistic_t0.50_dev_v{1,2,3}.json`.
- The first v2 run showed identity abstention 0.972: question redaction had missed "Tiffany Price" (spaCy doesn't tag it; "Price" is also a noun). Fixed with a question-only name rule (§6.6); now all 160 identity questions in both datasets are caught.

### 7.3 Results (realistic test split, 273 questions; `results/generation/realistic_t0.50.json`)

**Metrics.** *Grounded accuracy* is the headline: the share of gold answer items that are in the answer **and** in a section the answer cites from a gold note. Plain accuracy (gold items anywhere in the answer) is shown for comparison; the robustness check (§8.4) showed it rewards listing everything. *Drug-list F1* balances finding the right drugs against including wrong ones.

| | Grounded accuracy (95% CI) | Plain accuracy | All items correct | Drug-list F1 | Identity questions answered |
|---|---|---|---|---|---|
| **GlassBox (redacted, guard on)** | **0.858** [0.815, 0.899] | 0.919 | 0.836 | 0.855 | **0 / 84** |
| Fact questions | 0.976 | 1.000 | 1.000 | 1.000 | |
| Aggregate questions | 0.622 | 0.757 | 0.508 | 0.795 | |
| No-privacy baseline (raw index) | 0.833 [0.788, 0.877] | 0.899 | 0.804 | 0.826 | 42 / 84 |
| Guard off (redacted, identity questions only) | | | | | **84 / 84** |

- **Privacy at answer level.** With the guard, the system refuses every identity question. With the guard off, the model answers **all 84**, every time from *other people's* notes (0% cite the person's own note); 18.5% of those answers happen to contain the person's real medication or diagnosis, by coincidence, because common drugs and diagnoses repeat. That is the misattribution the audit predicted (§12.1 #1), now measured.
- **No utility cost from redaction.** Redacted 0.858 vs raw 0.833 grounded accuracy (CIs overlap). As in retrieval, redacted isn't *better* because of redaction: part of the difference is the low-information demotion (§6.4).
- **On the raw baseline**, the model answered 42 of 84 identity questions; for 16 of the 84 (19%) the answer contained the person's true value cited from their own note. The raw hybrid index finds the named person's note 55% of the time (§6.7; 35% before the Day 5 exact-search fix, §14.6), and BM25 alone 100%.
- **Citations:** 99.5% of claims cite a valid source; JSON parsed first time for 100% of answers (raw: 99.5%); no answerable question was wrongly refused.
- **Dev → test:** 0.941 → 0.919 plain accuracy, so the prompt isn't overfitted to dev.
- **Weak spot: aggregate questions.** Grounded 0.622 against a retrieval ceiling of 0.847. The 7B model drops a qualifying patient's drug or includes one from a note that doesn't meet every condition; only 67% of its citations in aggregate answers come from gold notes.

### 7.4 Claim verification: design (Day 4)
**What.** Stage 4. Every claim of an answer is checked against the chunk(s) it cites by a separate model, a natural-language-inference (NLI) classifier, so verification doesn't depend on the LLM that wrote the claim. Code: `glassbox/src/verification/` (`verifier.py`, `corruptions.py`, `evaluate.py`).

- **NLI:** premise = a cited chunk (with its non-PII context line), hypothesis = the claim → P(entailment), P(neutral), P(contradiction). With several citations, each chunk alone and all of them together are tried, and the best-supporting premise is used.
- **Labels:** SUPPORTED (entailment), CONTRADICTED (contradiction), UNSUPPORTED (neutral, or the claim cites nothing), UNCERTAIN (top two probabilities closer than a margin). An answer is VERIFIED if every claim is SUPPORTED, CONTRADICTED if any claim is, otherwise PARTLY / NOT VERIFIED.
- **Model and margin chosen on the dev split** of the corrupted-claim set: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (balanced accuracy 0.967 vs 0.830 for `cross-encoder/nli-deberta-v3-base`). The UNCERTAIN margin is the smallest at which confident labels are ≥ 95% correct on dev. For the large model that is **0**: it's already 97.9% accurate with no margin, and accuracy stays flat (~0.978) up to a margin of 0.45 because its errors are *confident* (margins 0.7–1.0). A margin can't separate its errors from its correct answers, so with this model UNCERTAIN doesn't occur. It stays a label, used if another model or a stricter target gives a non-zero margin.
- **Corrupted-claim test set** (`data/verification/corrupted_claims.jsonl`, 1,289 items; dev 426, test 863), built from the system's own Day 3 answers so it has exact ground truth:
  - *genuine* claims (286): cite a gold note, contain a gold item found in the cited text, and every drug name and number in them appears in that text → expected SUPPORTED;
  - corruptions of each genuine claim → expected not SUPPORTED: drug swap, dose change, vital/follow-up number change, negation, symptom swap (ideal CONTRADICTED); an invented extra drug, or the evidence replaced by the same section of another note with the same diagnosis (ideal UNSUPPORTED).
  - Identical claim + evidence pairs are counted once.
- **Subject rewriting** (tried for the audit's misattribution finding): when the question named someone, a claim's generic subject is rewritten to "The person named in the question", which NLI correctly finds unsupported by any redacted note ("[NAME] is prescribed X" alone reads as entailed). See §7.5 for why this has little practical value.
- Speed: about 28 ms per claim (premise options included) on the Apple GPU when idle: 863 test items, re-measured after the sweep. The 57 ms in `verifier_eval.json` was measured while the threshold sweep shared the GPU.

### 7.5 Claim verification: results (test split; `results/verification/verifier_eval.json`)

| Claim type | n | Said SUPPORTED | Agrees with ideal label | Labels |
|---|---|---|---|---|
| Genuine | 193 | 0.912 | 0.912 | 176 S, 16 U, 1 C |
| Dose change | 95 | 0.000 | 0.958 | 91 C, 4 U |
| Drug swap | 100 | 0.030 | 0.900 | 90 C, 7 U, 3 S |
| Number change (HR, BP, weeks) | 61 | 0.000 | 1.000 | 61 C |
| Negation | 33 | 0.000 | 0.818 | 27 C, 6 U |
| Symptom swap | 21 | 0.000 | 0.905 | 19 C, 2 U |
| Invented extra drug | 193 | 0.000 | 0.985 | 190 U, 3 C |
| Wrong citation | 167 | 0.006 | 0.204 | 132 C, 34 U, 1 S |

- **Corrupted claims caught: 99.4%** [98.8%, 99.9%]; balanced accuracy 0.953.
- **Genuine claims wrongly flagged: 8.8% strict, 1.2% on clean claims.** 15 of the 17 flagged genuine claims restate the question's condition ("…prescribed to the COPD patients who presented with dyspnea in September 2024") while citing only the PLAN section, which doesn't show the symptom or month. The verifier is right that the citation doesn't support the whole claim; the generator cited incompletely (25 of 193 genuine test claims do this). Reported both ways; the ground truth was not relabelled.
- **Wrong citations** are caught (0.6% pass) but mostly labelled CONTRADICTED rather than the ideal UNSUPPORTED: another note's plan says "Continue Amlodipine", which NLI reads as conflicting with "prescribed Lisinopril". Either label correctly means "not supported".
- **The verdict tracks correctness on real answers** (test, 189 answerable): claims 211 SUPPORTED, 30 UNSUPPORTED, 1 CONTRADICTED. VERIFIED answers (165) have grounded accuracy **0.906**; NOT VERIFIED answers (21) **0.492**. So the label is a useful warning to the user.
- **Verification can't catch misattribution under redaction.** On the guard-off identity answers, 118 of 118 claims are SUPPORTED (Day 4 on older records: 93 of 94): the evidence does say what the claim says; it just isn't about the person asked about, and redaction removes the only thing that could show that. Subject rewriting lowers this to 78 of 118 (Day 4: 57 of 94; bare claims like "Acetaminophen 650mg PO Q6H PRN" stay entailed). And it can only trigger when question redaction found an identifier, which is exactly when the guard has already refused. **Conclusion for the report: identity protection must be enforced before generation (the guard); claim-level verification checks evidence, not identity.**

### 7.6 Not built yet
- **Experiments and dashboard completion (Day 6).** See §2. The orchestrator and dashboard skeleton are in §14.

---

## 8. Robustness and overfitting checks

**Why.** Every number above is measured on questions and notes produced by our own generators, and several components (query analysis, redaction patterns, the prompt) were developed while looking at that data. The question is whether the numbers measure the system or the templates. Each check below tries to make a number *harder* to earn.

### 8.1 Protocol
- **Development sets (set A)**, built 2026-11-21 *after* the system was frozen, used to find problems and develop fixes:
  - `paraphrases_a.jsonl`: all 189 answerable test questions reworded with templates that differ from the QA builder's (other sentence structures, 5 date formats, 4 month formats, diagnosis written canonically / as a listed alias / as a common abbreviation not in our table).
  - `ood_notes_a.json`: 240 free-text notes (3 narrative templates the generators never use, names from 8 Faker locales, addresses with full state names or no commas, other DOB/ID formats), exact PII spans recorded.
- **Held-out sets (set B)**, built and **committed before any fix** (commit `2f0bb6b`; Qwen half in `a0047eb`), never used while developing fixes, evaluated once:
  - `paraphrases_b.jsonl`: 60 test questions reworded by hand (Claude; typos, two-sentence questions, ordinal dates, shorthand) + the other 129 reworded by the local Qwen model, which has never seen the query-analysis code. Each checked for the original's facts (date, diagnosis, symptom); 1 hand-reviewed ("decreased ROM" = "reduced range of motion"). Qwen's accepted paraphrases were kept exactly as written, including ambiguous day-first dates.
  - `ood_notes_b.json`: 240 notes with 4 new templates (referral letter, nursing note, ED triage, phone message), 8 other locales, and PII formats in neither set A nor the data (PO boxes, multi-line addresses, `D.O.B. 3.4.1975`, `Acct #`, extensions).
- **Known limitation:** no human outside the project wrote set B. The hand-written half was written by someone who had seen set A's failures, and it does score much higher (§8.3), so **the Qwen half is the number to trust**.
- Before/after on set B: the same frozen data scored with the pre-fix code (a checkout of `2f0bb6b`) and the fixed code.
- Code: `glassbox/src/robustness/` (`paraphrase.py`, `manual_paraphrases.py`, `heldout_paraphrases.py`, `ood_notes.py`, `evaluate_paraphrases.py`). Results: `glassbox/results/robustness/`.

### 8.2 What held up

| Check | Result |
|---|---|
| Dev → test gap, retrieval evidence | 0.968 → 0.949: settings chosen on dev aren't overfitted to dev |
| Dev → test gap, generation | 0.941 → 0.919 (CIs overlap) |
| Leakage between splits | 13 / 273 test questions share a gold note with a dev question |
| New random dataset (generator seed 7, 2,500 new notes) | Redaction F1 0.999 again: the headline isn't the luck of one draw |
| Identity guard on varied phrasing (possessive, lowercase, sentence-initial, "Mr./Ms. Surname", bare/lowercase IDs, "MRN …") | 100% caught; first name alone 88% before the fix, 100% after |
| Clinical damage on out-of-distribution notes | No drug or symptom redacted in any set, before or after fixes |
| Answerable questions wrongly refused (template wording) | 0 / 189 |

### 8.3 Template overfitting in query analysis (the main finding)

| Retrieval evidence | Template wording | Set A (dev) | Set B (held out) |
|---|---|---|---|
| Before fixes | 0.949 | **0.450** [0.383, 0.519] | **0.324** [0.261, 0.388] |
| After fixes | 0.949 | 0.932 [0.900, 0.960] | **0.579** [0.512, 0.644] |
| After fixes, grounded answer accuracy | 0.858 | — | **0.502** [0.435, 0.568] |

Set B by author (after fixes):

| | Fact evidence | Fact grounded accuracy | Aggregate evidence | Aggregate grounded accuracy | Refused |
|---|---|---|---|---|---|
| Hand-written (60) | 1.000 | 0.950 | 0.583 | 0.463 | 7.5% / 15% |
| **Qwen-written (129)** | **0.384** | **0.349** | **0.579** | **0.421** | 27.9% / 16.7% |

- **Diagnosis (set A, before fixes).** Ranking was fine: when the filters came out right, paraphrased fact questions still scored 0.971. Query analysis wasn't: it understood ISO and US numeric dates only, shrank "23 Aug 2025" to the whole month, didn't know common abbreviations (HTN, MDD, OA, T2DM), and two-hop selection only triggered on a "who …" clause (aggregate evidence 1.000 with the split, 0.614 without). Question redaction turned "9/7/2026" into `[ADDRESS]` and "Emphysema" into `[NAME]`.
- **General fixes made** (developed on set A only): a date grammar for written, abbreviated, day-first, ordinal, `M/YYYY` and `YYYY-MM` forms, and unambiguous day-first numeric dates (`23/08/2025`); a standard clinical-abbreviation and synonym table, plurals included (it also contains the forms set A used); a condition/ask split that works without a relative clause ("patients with X", "complaining of X"); dates and disease words no longer tagged as PII in questions. On the original 550 QA questions (both datasets) query analysis output is unchanged, and the redacted notes are byte-identical, so all earlier results stay valid.
- **Why set B is still at 0.579** (Qwen half, 129 questions): "29-Sep-2024" (day-month-year with dashes) loses the day (42 questions); European day-first numeric dates are read in US order (16, e.g. "04/01/2026" meant 4 January); other unhandled formats ("24-9-2024", "8/Oct/2025"; 9); two-hop split misses "w/" (13 aggregates); Title-Case questions trigger the name rule ("Morning Stiffness" → `[NAME]`) and the system refuses (2).
- **What the report should claim:** on template wording, 0.949 evidence / 0.858 grounded accuracy; on unseen natural wording, 0.58 / 0.50. The gap is a stated limitation: rule-based query analysis covers the formats its author thought of. Natural follow-up work: a date parser library, trying both day/month orders when one matches no note, or LLM-based query parsing (slower, less auditable).
- Not fixed after seeing set B, deliberately: any fix now would make set B development data. A further round would need a new held-out set C.

### 8.4 Metric gaming
- Plain accuracy credited a gold item found *anywhere* in the answer. One aggregate answer scored 1.0 by listing four drugs from four wrong notes; the right drug was there by coincidence. Pasting all retrieved text would score 0.949, more than the model's 0.919, at much worse precision.
- Fix: **grounded accuracy** (headline) and **drug-list F1**, plus 3 test answers that score above what retrieval provided flagged as "right for the wrong reason". Test numbers: plain 0.919 → grounded 0.858; aggregate 0.757 → 0.622.

### 8.5 Redaction on out-of-distribution notes

Share of PII values fully masked (for names: every personal-name word hidden; titles and degrees such as "Dott." or "B.A." excluded):

| | Set A (dev): before | Set A: after | **Set B (held out): before** | **Set B: after** |
|---|---|---|---|---|
| Overall | 0.815 | 0.986 | **0.672** | **0.819** |
| Address | 0.40 | 1.00 | 0.29 | 0.29 |
| Date of birth | 0.46 | 1.00 | 0.32 | 1.00 |
| Patient ID | 0.57 | 1.00 | 0.03 | 0.70 |
| Phone | 1.00 | 1.00 | 0.84 | 0.84 |
| Email | 1.00 | 1.00 | 1.00 | 1.00 |
| Names | 0.92 | 0.97 | 0.88 | 0.89 |

- **Fixes** (developed on set A): addresses with full state names or no commas; DOBs with 1–2 digit day/month, `-`/`.` separators, `04-Mar-1975`, context-gated "March 5, 1975"; 6–9 digit IDs next to record-type words ("Record number", "Medical Record #", "Acct"); accented letters in the name patterns; "Pt" and dates no longer redacted. The template-data results are unchanged (redacted notes byte-identical).
- **Generalised on set B:** DOBs 0.32 → 1.00 (including formats set A never had), IDs 0.03 → 0.70.
- **Didn't:** PO boxes, multi-line addresses and addresses without a ZIP are still 0%; some phone formats (bare 10 digits without a context word, `###-###-#### ext ##`) 0.84.
- **Bias by name origin (set B, after fixes):** English (GB/IE) 0.97, Swedish 0.97, Danish 0.93, Dutch 0.87, Spanish (Spain) 0.73, Vietnamese 0.70, **Turkish 0.57**. spaCy's NER misses names from some cultures far more often. This belongs in the report's ethics discussion.
- **Correction to an earlier figure:** the first set A run reported German names at 0.50. That was a scoring artefact: Faker puts titles and degrees inside some names ("Univ.Prof. Leyla Ullrich"), and the strict metric counted a visible title as a leak. With the title-aware metric, pre-fix German names score 0.93. The metric was corrected before set B was scored, and set B uses it for both versions.
- Seed replication (same generator, new seed): F1 0.999. Together with set B this pins down what 0.999 means: stable within the generator's formats, not evidence for other formats.

---

## 9. Tests

`python -m pytest glassbox/tests` → **165 passed, 4 skipped, 1 xfailed** (Day 5, after the exact-search fix). The skipped tests load the real 7B model or the NLI model; with `GLASSBOX_SLOW=1`: **169 passed, 1 xfailed**.

| File | Covers |
|---|---|
| `tests/test_generalization.py` | Redaction on formats not in the data: dash IDs, MRN, numeric MRN after a label, US and written dates, 3 phone formats, full address. Day 3a: 5 address formats as single spans, street line only with context, case-sensitivity, clinical terms kept (drugs, SpO2, BP), field label "Email" kept, names before credentials / after labels / after "Dr.". Day 3 robustness: full state names and no-comma addresses, DOB variants, record numbers with/without context, accented names, "Pt"/dates/disease words kept. Plus regression tests for each Day 1 redaction fix. |
| `tests/test_retrieval.py` | Chunk coverage (no text lost), section recognition per template, context line, BM25 against a hand-computed score, tokenizer, access control, end-to-end retrieval with log format, k cap. Query analysis: filters for ISO, US, month and year dates and aliases; ranges, several diagnoses, "May, 2025", invalid dates; no filters for identity questions; residual query; Chroma/Python filter agreement incl. `$in`. Selection: note grouping (both versions), filtered BM25 keeping zero-score chunks, auto-k, low-information classifier, ask/condition split, two-hop picks each gold note's answer section. Question redaction: names removed, clinical questions unchanged, the name never reaches the log. Day 3 robustness: written / day-first / ordinal / `M/YYYY` / `YYYY-MM` dates, standard abbreviations, condition split without "who", original-template analysis unchanged. |
| `tests/test_pipeline.py` (Day 5) | Real question redaction + retrieval with a canned LLM reply: no PII from six PII-bearing questions (name, first name, MRN, address, phone, email) in the returned or saved trace; named person → guard refuses before the model; every trace section and timing present; raw index refused without baseline mode. With both models (`GLASSBOX_SLOW=1`): a QA question answered, grounded accuracy 1.0, VERIFIED, no gold answer copied into the trace. |
| `tests/test_dashboard.py` (Day 5) | Headless Streamlit (AppTest): both pages load; the trace view renders verified, guard-refused, model-insufficient, unparseable, raw-baseline and guard-off traces without errors; the guard refusal is explained; trace table helpers (broken files skipped, citation anchors, source labels). |
| `tests/test_verification.py` (Day 4) | Subject rewriting (incl. `[NAME]` as subject, bare claims), the label rule and UNCERTAIN margin, combined-evidence premises, genuine-claim detection, corruptions, the uncited-condition flag, the frozen set's consistency. With the model (`GLASSBOX_SLOW=1`): correct / changed-dose / uncited claims labelled SUPPORTED / CONTRADICTED / UNSUPPORTED, answer verdicts. |
| `tests/test_generation.py` (Day 3) | Parser (fences, prose around JSON, truncated or malformed JSON, citation normalisation, invalid citations, uncited claims), the abstain guard (identifier / fallback / raw baseline), the guard never calling the model, prompt construction (note labels, no internal IDs in the prompt), question-name rules (spaCy misses, first name alone, possessives, diagnoses and drugs left alone), grounded accuracy (right drug from the wrong note scores 0). End-to-end test with the real model (`GLASSBOX_SLOW=1`). |

- The full-address test was a strict `xfail` until Day 3a; it now passes.
- The one remaining strict `xfail` documents a known limitation: an all-lowercase address ("42 oak street, springfield, il 62704") is only partly caught, because the address patterns use capitalisation.
- The old README claimed 8/8 generalization cases, but the committed test file only had 3 of those cases.

---

## 10. How to run

Setup (Python 3.11, Apple-silicon Mac):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

From the repo root, with `.venv` active:

```bash
# Data
python -m glassbox.src.generate_synthetic_dataset                 # 250 notes + QA set
python -m glassbox.src.generate_realistic_dataset --size large    # 2,500 notes + QA set

# Redaction (writes data/processed/<dataset>/t0.50/ and results/redaction/)
python -m glassbox.src.redaction.pipeline --dataset realistic [--threshold 0.5]

# Retrieval
python -m glassbox.src.retrieval.index --dataset realistic --all  # build the 4 indexes
python -m glassbox.src.retrieval.evaluate --dataset realistic     # metrics + leak audit (~10 min)
python -m glassbox.src.retrieval.evaluate --dataset realistic --tune  # dev-split fusion weight grid
python -m glassbox.src.retrieval.evaluate --dataset realistic --ablate  # selection ablation, dev + test

# Rebuild only the QA set (e.g. after changing its size)
python -m glassbox.src.qa.build_qa_set --notes glassbox/data/raw/realistic/synthetic_notes.jsonl \
    --profile realistic --output glassbox/data/raw/realistic/qa_eval_set.jsonl

# Generation (~45 min for all three configurations; --split dev for prompt work)
python -m glassbox.src.generation.evaluate --dataset realistic
python -m glassbox.src.generation.evaluate --rescore           # re-score saved answers, no model calls

# Verification (corrupted-claim set from the saved generation answers, then evaluation, ~5 min)
python -m glassbox.src.verification.corruptions
python -m glassbox.src.verification.evaluate

# Redaction threshold sweep, 0.3-0.8 (~2.5 h; --skip-generation for redaction + retrieval only)
python -m glassbox.src.experiments.threshold_sweep

# Robustness (set A = development, set B = held out)
python -m glassbox.src.robustness.paraphrase                    # rebuild paraphrase set A
python glassbox/src/robustness/evaluate_paraphrases.py --set b --generate
python -m glassbox.src.robustness.ood_notes --set b              # OOD redaction (frozen notes)

# End-to-end pipeline (question in, verified cited answer + trace out; §14)
python -m glassbox.run "Which medications were prescribed to patients who presented with fatigue?"
python -m glassbox.run --qa-id realistic-cf-000          # one QA question, scored
python -m glassbox.run --qa-set                          # all 390 (~25 min); also --config raw / redacted_noguard
python -m glassbox.src.experiments.log_pii_scan          # PII scan of every log and trace (~4 s)
streamlit run glassbox/src/dashboard/app.py              # audit dashboard

# Tests
python -m pytest glassbox/tests
GLASSBOX_SLOW=1 python -m pytest glassbox/tests/test_generation.py   # + real-model test
```

---

## 11. Key decisions log

| Date | Decision | Why | Alternatives considered |
|---|---|---|---|
| Nov 20 | Pin generator "today" to 2026-09-16 | Byte-for-byte reproducible data | n/a |
| Nov 20 | Compute redaction metrics on the spans actually redacted | They describe what downstream stages see | Raw detector output (slightly different numbers) |
| Nov 20 | Add masking metrics alongside typed P/R | Privacy cares whether PII is hidden, not what it's labelled | Typed P/R only |
| Nov 20 | Drop `original_text` from redacted output | Retrieval indexes that file | Keep it "for debugging" (the old behaviour) |
| Nov 20 | Remove Presidio's UK NHS recognizer | US-only domain; it fired on phone numbers | Map UK_NHS to PHONE (rejected: wrong semantics) |
| Nov 20 | Include identity questions in the QA set | Turns the privacy cost into a measured result | Clinical questions only (would hide the trade-off) |
| Nov 20 | Section chunks with a non-PII context line | Precise citations, short NLI premises, 2.8× less text shown to the LLM | Whole-note chunks: more aggregate evidence (0.95 vs. 0.52) at 2.8× the context (§6.8) |
| Nov 20 | Hybrid dense + BM25, BM25 written in-house | Questions mix meaning with exact dates; no extra dependency | `rank-bm25` package |
| Nov 20 | Embedding model `bge-small-en-v1.5` | Strong small retrieval model; runs locally on MPS | `all-MiniLM-L6-v2` (downloaded, unused so far) |
| Nov 20 | Raw index needs explicit `baseline=True` | Makes "the LLM only sees sanitized data" enforced and auditable | Convention only |
| Nov 20 | k chosen by query analysis: 5 (single note) / 10 (multi-note), max 10 | Multi-note questions need ~2 chunks per note | Fixed k = 5 (too little for aggregates) |
| Nov 20 | Add a query analysis stage: diagnosis and date filters, residual query, scope | Fixes month-name and exact-date matching; answerable evidence 0.048 → 0.840 (§6.8) | Prepend month names to chunks (rejected: tailors data to question wording); query rewriting with the LLM (slower, less auditable) |
| Nov 20 | Rank on the residual query once filters apply | Filters already guarantee the constraints; repeating them rewards header chunks | Full question (dev evidence 0.52 vs. 0.80) |
| Nov 20 | Note-grouped selection, ≤ 2 chunks per note, then fill | Aggregate questions need a qualifying chunk and an answer chunk per note | Plain ranking (aggregate evidence 0.22 on dev); hard cap without fill (hurt fact questions) |
| Nov 20 | Fusion weights (1.0, 1.0), chosen on dev (superseded on Day 3a) | Plateau from 1.0 to 5.0; simplest point | BM25-only 0.722, dense-only 0.759 vs. hybrid 0.796 (dev answerable evidence) |
| Nov 20 (3a) | Structural US-address recognizer (valid state code + ZIP) | spaCy fragments addresses; structure generalises to free text | Template-label rule ("Address: …" to end of line): rejected, tailors to the generator; street-suffix word list: brittle |
| Nov 20 (3a) | Clinical-term guard with a general drug lexicon | Over-redaction destroyed clinical answers ("Glipizide") | Only the drugs in the synthetic data (rejected: overfits); lower the NER threshold (hurts name recall) |
| Nov 20 (3a) | Case-sensitive compilation for the new patterns | Capitalisation is the signal; Presidio's default IGNORECASE broke them | n/a |
| Nov 20 (3a) | Redact the user's question before retrieval | Names in questions reached embeddings and logs; the identifier flag drives the abstain rule | Redact only before the LLM prompt (logs would still hold names) |
| Nov 20 (3a) | Demote, not drop, placeholder-only chunks | They answer nothing and invite misattribution; demoting keeps them available if nothing else matches | Drop them (loses the ability to show the header in the dashboard) |
| Nov 20 (3a) | Two-hop selection via ask/condition split | Aggregate questions need a qualifying section and an answer section per note | Answer-type section prior ("medications" → plan): rejected, hard-codes section names; whole-note chunks (2.7× context) |
| Nov 20 (3a) | Fusion weights (2.0, 1.0) | Re-tuned on the 3× dev split with the final selection; plateau starts at 2.0 and it's best at k = 3 | (1.0, 1.0): 0.943 vs 0.968 dev |
| Nov 22 (4) | NLI verifier DeBERTa-v3-large (MNLI/FEVER/ANLI/LingNLI/WANLI) | Dev balanced accuracy 0.967 vs 0.830 for DeBERTa-v3-base; same speed | Base model; using the generating LLM to self-check (not independent) |
| Nov 22 (4) | UNCERTAIN = top-two probability margin, chosen on dev for >= 95% accuracy on confident labels -> 0 | Pre-stated rule; the large model's errors are confident, so a margin can't isolate them | A hand-picked margin |
| Nov 22 (5) | Exact dense search (NumPy cosine over all chunks) instead of ChromaDB's HNSW index | HNSW returned on average only 35–37 of the true top-50 neighbours on unfiltered questions (as few as 0) and differed between processes; exact search is reproducible and, at 18k chunks, faster | Keep HNSW and raise `ef_search` (still approximate, still not guaranteed reproducible) |
| Nov 22 (5) | Orchestrator only connects the existing stages; no new logic | A batch run must reproduce the evaluation numbers exactly, which proves the demo system *is* the evaluated system | A separate demo code path (could silently differ) |
| Nov 22 (5) | One trace file per question; no original question on redacted configs | The trace is the audit record and is safe to share; the question's identifiers are recorded as types/offsets only | Store the original for convenience (would make every trace sensitive) |
| Nov 22 (5) | Dashboard bound to localhost, telemetry off | It shows typed questions (possible PII); Streamlit's defaults expose it to the LAN and send usage data | Streamlit defaults |
| Nov 22 (5) | Dashboard reads traces, never recomputes them | What's audited is exactly what happened; works without loading models | Re-run the pipeline on demand for every view |
| Nov 22 (5) | Scrub the pre-fix name from the v1/v2 prompt-development logs instead of re-running them | Scores don't depend on it; a 25-min re-run would change nothing measured | Re-run prompt development; leave the name in place |
| Nov 22 (4) | Corrupted-claim set from the system's own answers | Exact ground truth on realistic claims | Hand-written claims (few, less realistic) |
| Nov 22 (4) | Report false alarms strict and on clean claims; flag, don't relabel | Most "false alarms" are incomplete citations by the generator; relabelling after seeing results would flatter the verifier | Relabel genuine claims |
| Nov 21 (3) | Abstain guard in code, before the model | The model treats `[NAME]` as the person asked about; guard off it answers 84/84 identity questions from other people's notes | Prompt instruction only (can be ignored by the model) |
| Nov 21 (3) | JSON output with per-claim citations | Clean input for the Day 4 verifier; malformed output is detectable and retried | Free text with inline [C1] markers |
| Nov 21 (3) | Greedy decoding (temperature 0) | Reproducible numbers | Sampling |
| Nov 21 (3) | Prompt v2 ("Note n" labels, note-by-note rule, exact doses) | Best on dev (0.941 vs 0.890 v1, 0.924 v3) | v3 claims-first JSON |
| Nov 21 (3) | Grounded accuracy + drug-list F1 as headline metrics | Plain accuracy rewarded listing everything (right drug from the wrong note scored 1.0) | Plain keyword accuracy |
| Nov 21 (3) | Robustness protocol: development set A, held-out set B committed before fixes, evaluated once | Measure generalisation honestly after looking at failures | Fix and re-measure on the same set (overfits to it) |
| Nov 21 (3) | No fixes after seeing set B | Keeps set B a held-out measurement; a further round would need a new set C | Fix the set B failures and report set B again |
| Nov 20 (3a) | Triple the QA set; report bootstrap 95% CIs | CI was ±0.08; method differences were noise | Keep 130 questions |
| Nov 20 | Add evidence and context-size metrics | Note recall alone hid that answer-bearing sections weren't retrieved; context size makes minimization measurable | Recall@k / MRR only |

---

## 12. Known issues / open items

- **Incomplete citations in multi-hop claims (§7.5).** 13% of genuine test claims restate the question's condition (symptom, month) but cite only the section with the drug. Fix candidate: a prompt rule to cite every section a claim relies on, tuned on dev (Day 8 if time).
- **The verifier is confidently wrong when it's wrong (§7.4).** Its margin can't flag its own errors, so UNCERTAIN doesn't occur with the chosen model. Report as a calibration limitation.
- **Generalisation to unseen wording (§8.3).** Held-out paraphrases: retrieval evidence 0.579, grounded accuracy 0.502, against 0.949 / 0.858 on template wording. Main causes: "29-Sep-2024"-style dates, European day-first numeric dates read in US order, other date formats, "w/" not recognised as a condition marker, Title-Case questions tripping the question-name rule (the system then refuses).
- **Redaction on unseen note formats (§8.5).** Held-out OOD notes: 0.819 fully masked. PO boxes, multi-line addresses and addresses without a ZIP are not caught; names of Turkish (0.57), Vietnamese (0.70) and Spanish (0.73) origin are missed more often than English ones (0.97).
- **Aggregate answers (§7.3).** Grounded accuracy 0.622 against a retrieval ceiling of 0.847: the 7B model drops a qualifying patient's drug or includes one from a note that doesn't meet every condition.
- **Brand-name drugs in questions** ("What is prescribed for Jardiance?") can trip the single-name rule; the system then refuses (privacy preferred over answering).
- **Misattribution under redaction (§12.1 #1).** Fixed for questions that name someone: the abstain guard refuses 84/84 (guard off: 84/84 answered from other people's notes). Day 4 showed verification can't be a second line of defence here (§7.5); protection depends on question redaction catching the identifier.
- **Titles are quasi-identifiers.** "Mrs. [NAME]" keeps the title, which reveals gender and let one redacted identity question match its note. Option: redact titles with the name; worth one sentence in the report either way.
- **Redaction generalisation limits** (for the report's limitations section; found by probing, not in the data): all-lowercase addresses are only partly caught; insurance member numbers, hospital names and birth years before 1936 are not redacted. Lowercase names, relatives' names, valid SSNs and numeric MRNs after a label are.
- **The redaction numbers are in-distribution.** 0.999 F1 is measured on synthetic notes from the same generator the recognizers were checked against. The hand-written generalisation tests (§9) are the evidence for other formats.
- **Remaining aggregate misses** (test aggregate evidence 0.847): the ask "Which medications were prescribed" sometimes ranks LAB ORDERS above a plan line that never says "medication".
- **Query analysis is template-evaluated.** The QA questions come from templates; free-text handling is unit-tested only.
- **The audit log stores raw PII values.** That's needed for the audit trail, but the log is itself a sensitive artifact. The report should say so. (Retrieval logs no longer contain PII from questions.)
- **Approximate search in earlier results (fixed Day 5, §14.6).** Everything committed before the Day 5 fix used ChromaDB's HNSW index for unfiltered dense search. All results were regenerated with exact search; only unfiltered (identity, no-query-analysis) numbers moved. Older documents (README, `report_draft.md`) still carry the old raw-baseline numbers (e.g. raw identity answered 48/84 instead of 42/84).
- **Git history still contains the one synthetic name** scrubbed from the v1/v2 prompt-development files on Day 5 (§14.3). The data is synthetic, so the fix is the scrub plus this note; rewriting history isn't worth the risk.
- **The docs are stale.** The README and `report_draft.md` show old numbers (0.838 / 0.869 / 0.858, 62.5%, 8/8) and old data paths. Refresh on Day 7.

### 12.1 Audit of Days 1–2 (Nov 20)

An independent check of everything built so far, using throwaway scripts rather than the project's own evaluators.

**Verified (all passed):**
- Test suite: 34 passed, 1 xfailed.
- Regenerating the realistic dataset gives **byte-identical** notes and QA set.
- QA gold: every gold answer is in its note; every "unique" fact question picks out exactly one note; every aggregate question lists *all* qualifying notes.
- Redaction, by direct string search of the redacted notes: 0 exact IDs, DOBs, phones or emails survive; names 0.8%, addresses 15.5% (matches §5 and §6.8).
- All 4 realistic indexes are internally consistent (chunk file = ChromaDB count = manifest count, no duplicate IDs).
- The headline retrieval numbers recompute exactly from `realistic_t0.50_per_question.json`.
- Days 3–4 stack runs on the laptop: Qwen loads in 1.3 s, answers in 2.6 s, peak memory 4.5 GB; both NLI models run on MPS.

**Findings and their status** (status column added on Day 3a):

| # | Severity | Finding | Evidence | Fix (day) |
|---|---|---|---|---|
| 1 | Critical | **Misattribution under redaction.** Asked "What is John Wilkerson's medication?" with a redacted note as context, Qwen answered with that note's drug, treating `[NAME]` as John. It phrased the claim "The patient is prescribed …", which both NLI models label *entailment* (Supported). "Identity recall 0.000" only shows retrieval can't find the person; without a guard the full system would still give a confident, verified, wrong answer. | Smoke test. With the name in the claim ("John Wilkerson is prescribed …") both models say *neutral* (Unsupported). | Abstain rule (Day 3); rewrite claims with the question's subject before verifying (Day 4); report abstention and false-attribution rates as privacy metrics (Day 6) **Status: ✅ guard fixed Day 3** (0/84 identity questions answered; §7.3). Day 4: claim verification can't catch misattribution under redaction (§7.5), so the guard is the protection. |
| 2 | Critical | **The question itself isn't redacted.** A name typed into the question reaches the LLM prompt, the query embedding and the retrieval log. | `retriever.py` uses the raw query | Redact the question at entry; detected identifiers also trigger the abstain rule in #1 (Day 3) **Status: ✅ fixed Day 3a** (§6.6). |
| 3 | Important | **Aggregate evidence 0.52 is a two-hop selection problem.** 6 of 21 aggregate test questions have zero evidence despite recall 1.0. The qualifying symptom is in *subjective*, the medication in *plan*; after the qualifying sections, the remaining slots go to *other* notes' plan sections. | e.g. ag-012, ag-020, ag-027 retrieve every gold note's subjective section, then non-gold plans | After selecting qualifying notes, add each one's answer section (plan / medications) (Day 3) **Status: ✅ fixed Day 3a**: two-hop selection, test aggregate evidence 0.553 → 0.847 (§6.4, §6.8). |
| 4 | Important | **NAME precision 0.84 is mostly an address problem.** Of ~1,130 NAME spans not matching gold exactly, 965 are address fragments and 92 are "Dr. X" where gold includes the title (name masked, scored as an error). Pure false positives: 148 (§12). | Audit log vs. gold spans | Day 5 address fix also raises NAME precision **Status: ✅ fixed Day 3a**: NAME precision 0.841 → 0.999, "Email" kept (§5). |
| 5 | Important | **Context spent on placeholders.** 18% of context slots go to `patient_info` / `contact` sections, mostly `[NAME]`-style tags after redaction; they also invite #1. | Per-question retrieved sections | Drop or rank them lower for non-identity questions (Day 3) **Status: ✅ fixed Day 3a**: low-information demotion (§6.4). |
| 6 | Important | **No `requirements.txt` or install steps**; matplotlib not installed (needed for plots). | Repo root | Day 5 **Status: ✅ fixed Day 3a**: `requirements.txt`, matplotlib 3.10.9 installed. |
| 7 | Important | **Small test split.** 63 answerable test questions; 95% bootstrap CI for evidence 0.840 is [0.757, 0.915]. Hybrid vs. dense differ on 8 questions (mean −0.008): not significant, report as "tied". | Bootstrap, 5,000 resamples | Enlarge QA set, CIs on plots (Day 6) **Status: ✅ fixed Day 3a**: QA set ×3, bootstrap CIs in every summary (§4, §6.7). |
| 8 | Minor | **Query analysis on free-typed questions** (the QA set is templated, so evaluation doesn't see these): "between March and June 2025" → March only; "hypertension or diabetes" → first diagnosis only; "May, 2025" not parsed; invalid "2025-13-01" becomes all of 2025; a date with no notes ("in 2019") falls back to unrelated notes. | Probe questions | Ranges and "or" (Day 5); pass `fallback` to the generator so it abstains (Day 3) **Status: ✅ fixed Day 3a** for ranges, "or", "May, 2025" and invalid dates (§6.3); fallback-abstain is Day 3. |
| 9 | Minor | **Redaction gaps outside the dataset** (limitations section): numeric-only MRNs (`00482913`), insurance numbers, hospital names, birth years before 1936 are not redacted. Lowercase names, relatives' names and valid SSNs are. | Probe sentences | Report as limitations **Status: partly fixed Day 3a**: numeric MRNs after a label; the rest are listed limitations. |
| 10 | Minor | Raw and redacted indexes chunk slightly differently (781 signature blocks merged in redacted, see §6.1). Headline unaffected (both 0.840). | Section counts | Note in report **Status: open** (documented, §6.1). |

---

## 13. Daily log

### Day 1: Nov 20
- Removed the clutter: empty `*mkdir` dirs, `-p`, nested `glassbox/glassbox`, `hello.txt`.
- Separated and regenerated the datasets deterministically; fixed the discharge dates.
- Built the QA eval set v2 (§4).
- Six redaction fixes (§5): realistic F1 0.838 → 0.893. Added the masking metrics.
- Test suite: 2 → 12 tests.
- Downloaded the embedding and NLI models.
- Commits `efe7049`, `429faeb`.

### Day 2: Nov 20
- Built the retrieval module: chunker, BM25, ChromaDB index builder, access-controlled and logged retriever, evaluation and leak audit (§6).
- Added `visit_date` (non-PII) to the redacted note metadata for the chunk context line.
- Built 8 indexes and evaluated all 24 (index × method) combinations on both datasets.
- Baseline findings (commit `5c62519`):
  - Clinical facts are fully retrievable after redaction.
  - Identity questions drop to 0 (the privacy result).
  - Aggregate retrieval is weak, and hybrid underperforms BM25.
- Approved and built the improvements:
  - Query analysis stage: filters, residual query, scope-based k.
  - Note-grouped selection.
  - Evidence and context-size metrics.
  - Dev-split fusion tuning.
  - Fixed filtered BM25 dropping zero-score chunks.
- Result (realistic test, redacted): answerable evidence 0.048 → **0.840**, recall 0.645 → **1.000**, identity 0.000.
- Found clinical over-redaction ("Glipizide" → `[NAME]`; §12).
- Tests: 12 → 35.

### Audit: Nov 20
- Independent audit of Days 1–2 (§12.1): reproducibility, QA gold, leakage and headline numbers all verified.
- Found the misattribution risk (critical, fixed in Days 3–4), unredacted questions, and the root cause of weak aggregate evidence.
- Moved the address and clinical-term redaction fixes from Day 7 to Day 5; plan in §2 updated.
- Pushed `pipeline-build` to GitHub.

### Day 3a: Nov 20 (audit fixes)
- Redaction: structural US-address recognizer, clinical-term guard, names before credentials / after labels, numeric MRNs, case-sensitive patterns. Realistic F1 0.893 → **0.999**; over-redactions 148 → 0; PII partly visible in the index 8.7% → **0.04%** (§5).
- Question redaction before retrieval; the log keeps types only (§6.6).
- Query analysis: date ranges, several diagnoses, "May, 2025", invalid dates, ask/condition split (§6.3).
- Selection: low-information demotion, note-ordered grouping, two-hop answer sections (§6.4).
- QA set v3: 390 + 160 questions; new test split almost entirely unseen (§4).
- Fusion weights re-tuned on dev: (2.0, 1.0) (§6.7).
- Bootstrap 95% CIs and a selection ablation in `evaluate.py`.
- Result (realistic test, redacted): answerable evidence 0.840 → **0.949** [0.922, 0.972]; aggregate 0.553 → 0.847; identity 0.000; context ~955 characters.
- `requirements.txt`; matplotlib installed.
- Tests: 35 → 71 (70 passed, 1 documented xfail).

### Day 3: Nov 21–22 (generation, robustness)
- Built cited generation with Qwen2.5-7B-Instruct (MLX): abstain guard, JSON claims with citations, parser with retry, audit log (§7).
- Prompt v1 → v2 → v3 compared on dev; v2 chosen (0.941 plain dev accuracy).
- Question-only name rule after "Tiffany Price" slipped past spaCy.
- Test results: grounded accuracy 0.858 (raw baseline 0.833); identity questions answered 0/84; guard off 84/84, all from other people's notes.
- Robustness and overfitting checks (§8): dev/test gaps, split leakage, seed replication, guard perturbations all fine. Found template overfitting in query analysis (paraphrases 0.949 → 0.450), redaction overfitting to the generator's formats, a gameable answer metric, and a scoring artefact in name-bias numbers (corrected).
- Froze held-out set B before any fix (commits `2f0bb6b`, `a0047eb`), made general fixes on set A, evaluated set B once: retrieval 0.324 → 0.579, grounded accuracy 0.502, OOD redaction 0.672 → 0.819.
- Original-template results unchanged by the fixes (query analysis identical on all 550 QA questions; redacted notes byte-identical).
- Tests: 71 → 128 (126 passed, 1 skipped slow-model test which passes with `GLASSBOX_SLOW=1`, 1 documented xfail).

### Day 4: Nov 22 (claim verification)
- Sanity check of Day 3: headline numbers recomputed independently from raw logs (match), generation deterministic (6/6 identical re-runs), held-out protocol verified from git history. Found stale pre-fix logs with one name and 4 identity questions whose retrieval changed (recall still 0): regenerate on Day 5.
- Started the full redaction threshold sweep (0.3–0.8, all 390 questions).
- Built the NLI verifier (4 labels), the corrupted-claim test set (1,289 items) and its evaluation (§7.4–7.5).
- Chose DeBERTa-v3-large on dev (0.967 vs 0.830); UNCERTAIN margin 0 by the pre-stated rule (errors are confident).
- Test: 99.4% of corrupted claims caught; 1.2% of clean genuine claims flagged (8.8% strict, mostly incomplete citations by the generator).
- VERIFIED answers: grounded accuracy 0.906 vs 0.492 for NOT VERIFIED.
- Finding: claim verification can't catch misattribution under redaction (93/94 guard-off claims SUPPORTED; 57/94 with subject rewriting); the guard is the protection.
- Fixed a subject-rewrite bug (`[NAME]` not matched) found by its unit test.
- Tests: 128 → 141 (137 passed, 3 slow skipped, all pass with `GLASSBOX_SLOW=1`; 1 documented xfail).
- Threshold sweep finished overnight (§5.1): plateau from 0.4 to 0.7, over-redaction at 0.3 (4,219), phone leakage at 0.8 (15.6% of phones). 0.5 is justified, and its point reproduces the main results exactly. Verifier speed re-measured on an idle GPU: 28 ms per claim.

### Day 5: Nov 22 (orchestrator, housekeeping, dashboard skeleton)
- Built the end-to-end pipeline (`glassbox/src/pipeline.py`) and CLI (`glassbox/run.py`): one call goes question → redaction → retrieval → guard → generation → verification and writes one JSON trace per question (§14.1).
- Batch runs over all 390 QA questions × 3 configurations reproduce the Day 3/4 evaluation exactly: test grounded accuracy 0.858 (raw 0.833); identity answered 0/84 (guard off 84/84, raw 42/84 after the exact-search fix, §14.6); all 390 redacted answers word-for-word identical (§14.2).
- End-to-end time: 3.2 s median per question when the model runs, about 98% of it generation.
- Regenerated the stale retrieval logs and results with the current code. The headline is unchanged (0.949); ablation and baseline rows moved slightly and the §6.8 full table was refreshed (§14.2).
- New PII scan of every log, trace and committed answers file (§14.3). Found one real leak: a synthetic patient's name in the Nov 21 prompt-development runs v1/v2 (from before the name-rule fix). Scrubbed; now 0 hits on the privacy side, 3,143 in raw-baseline traces (the positive control).
- Dashboard skeleton (§14.4): Ask page, stored-traces table, per-stage trace view with citation links and NLI badges.
- Re-measured guard-off verification on current traces: 78/118 claims still SUPPORTED with subject rewriting, 118/118 without (Day 4, older records: 57/94). Same conclusion.
- Tests: 141 → 160 (155 passed, 4 slow skipped, 1 documented xfail).
- Review fixes (§14.5): model preloading, question-redaction threshold matched to the evaluation, verdicts matched to claims by position, input validation, dashboard bound to localhost with telemetry off. Tests 160 → 161.
- **Found that ChromaDB's approximate (HNSW) dense search was the cause of differing raw-baseline runs and "stale" identity answers** (§14.6). Switched to exact search, then re-ran retrieval, generation, verification, both paraphrase sets, all pipeline traces, the threshold sweep and the PII scan (~4 h in the background). GlassBox's own numbers didn't change (every filtered question is identical); the raw baseline now finds named people more often (hybrid identity recall 0.333 → 0.548) and answers 42/84 identity questions, 16 with the person's true value.
- Determinism checked: generation and verification were run twice from scratch with exact search; all 900 stored answers identical, summaries identical apart from timings.
- Demo tooling (PLAN 6.3): Demo page in the dashboard (six one-click steps incl. "try to fool the verifier"), `python -m glassbox.verify` (re-runs 10 stored questions live: 10/10 identical in 34 s), `DEMO.md` script. Tests 161 → 170 (165 fast + 4 slow + 1 xfail).

---

## 14. Stage: End-to-end pipeline, audit trace and dashboard

Built on Day 5 (Nov 22). This turns the separate stages (§5–7) into one system: **a question goes in; a verified, cited answer and a complete audit trace come out.**

### 14.1 Orchestrator design (`glassbox/src/pipeline.py`, `glassbox/run.py`)

**What it does.** `GlassBox(dataset, config).answer(question)` runs, in order:

1. **Question redaction** (`redaction/query.py`). Names, IDs, phones etc. in the question become placeholders; `identifiers` records which *types* were found.
2. **Query analysis + retrieval** (`retrieval/`). Diagnosis/date filters, auto-k, hybrid ranking, two-hop selection.
3. **Abstain guard** (`generation/generator.py`). Refuses before the model if the question identifies a person or its conditions match no note.
4. **Cited generation** (Qwen2.5-7B, prompt v2). JSON claims with citations.
5. **Claim verification** (DeBERTa-v3-large NLI). One label per claim, and an answer verdict.
6. **Trace.** Everything above, plus per-stage timings, written as one JSON file.

The retriever, LLM and NLI model are loaded once and reused across questions.

**Configurations** (the same three as the evaluation, so results are comparable):

| Config | Index | Question redaction | Guard | Purpose |
|---|---|---|---|---|
| `redacted` | redacted | on | on | The GlassBox system |
| `raw` | raw (needs baseline access) | off | on (fallback only) | No-privacy baseline |
| `redacted_noguard` | redacted | on | **off** | What the model does on its own when asked about a person |

**Why it is "only glue".** The orchestrator contains no new logic: it calls the same functions, with the same settings, as the evaluation scripts. This was deliberate. If a batch run reproduces the evaluation numbers exactly (§14.2), then the system shown in the demo and dashboard *is* the system that was measured. A separate demo code path could silently drift. The one structural change is that question redaction is called by the orchestrator instead of inside the retriever, so it can be timed and recorded as its own stage. The result is identical (same function, same text), which the reproduction check confirms.

**The trace** (`glassbox/logs/traces/<dataset>/<config>/<qid or trace_id>.json`, about 9 KB each):

| Section | Contents |
|---|---|
| `trace_id`, `timestamp`, `config` | Index, source, thresholds, fusion weights, k rules, model names, prompt version, verifier margin |
| `question` | The **redacted** text, the types/offsets removed, `identifiers` |
| `query_analysis` | Filters, matched phrases, scope, residual and ranking query, ask/condition split, fallback flag |
| `retrieval` | Every chunk shown to the model: section, text (redacted), fused/dense/BM25 scores and ranks, `selected_by`, low-information flag; k and context size |
| `generation` | Guard decision and reason, answer, claims with citations (and dropped invalid ones), source map C1→chunk, raw model output, retries, parse errors, token counts |
| `verification` | Per claim: label, NLI probabilities, which chunk(s) were the premise, the hypothesis as checked (after subject rewriting); answer verdict |
| `timings_ms` | question_redaction, retrieval, generation, verification, total |
| `qa` (QA questions only) | qid, type, split, gold note ids and the scores. **Never** the question text or the gold answer. |

**Privacy rule.** On a redacted configuration the original question is never written: not in the trace, not in a log. `tests/test_pipeline.py` checks this with six PII-bearing questions (full name, first name alone, MRN, street address, phone, email): none of the values appear in the returned or the saved trace. The raw baseline has no question redaction by definition, so its traces contain the question as typed and are marked `baseline_mode: true`.

**Alternatives considered.** (a) Keep the original question in the trace for the dashboard's convenience. Rejected: every trace would become sensitive, and the dashboard can show the typed question from session memory instead. (b) Append to one big JSONL log per run (like the stage logs). Rejected for the dashboard use: one file per question is easy to open, link and diff.

**CLI.** `python -m glassbox.run "<question>"`, `--qa-id <id>`, `--qa-set [--split] [--types] [--config]`, `--batch <file>`, `--no-verify`. A QA-set batch also writes `glassbox/results/pipeline/<dataset>_<config>[_<split>|_<types>].json` with grounded accuracy (bootstrap 95% CI), identity questions answered, verdict counts, grounded accuracy per verdict, and median stage latencies.

### 14.2 Results: the pipeline reproduces the evaluation exactly

All 390 realistic QA questions × the three configurations (`results/pipeline/realistic_redacted.json`, `realistic_raw.json`, `realistic_redacted_noguard_identity.json`). Test split:

| | Redacted (GlassBox) | Raw (no privacy) | Guard off |
|---|---|---|---|
| Answerable grounded accuracy (95% CI) | **0.858** [0.815, 0.899] | 0.833 [0.788, 0.877] | n/a (identity only) |
| Clinical fact / aggregate grounded accuracy | 0.976 / 0.622 | 0.980 / 0.540 | |
| Identity questions answered | **0 / 84** | 42 / 84 (see §14.6) | 84 / 84 |
| Verdicts (273 questions) | VERIFIED 165, NOT_VERIFIED 21, PARTLY 2, CONTRADICTED 1, ABSTAINED 84 | VERIFIED 192, NOT_VERIFIED 33, PARTLY 2, CONTRADICTED 3, NO_CLAIMS 1, ABSTAINED 42 | VERIFIED 56, NOT_VERIFIED 27, PARTLY 1 |
| Grounded accuracy: VERIFIED vs NOT_VERIFIED | **0.906** vs 0.492 | 0.887 vs 0.492 | |
| Median time per question (model called) | **3.24 s** | 3.07 s | 3.58 s |

- **Reproduction check.** Grounded accuracy, CIs and identity counts are identical to the Day 3/4 evaluation (§7.3, §7.5) on both splits (dev 0.896 too). Per question: all 390 redacted answers are word-for-word identical to the evaluation's, and no question's score differs in any configuration. Greedy decoding makes this possible; it also confirms that moving question redaction into the orchestrator changed nothing.
- **Where the time goes** (redacted, test, median): question redaction 4 ms, retrieval (incl. query analysis) 6 ms, verification 37 ms; the rest of the 3.24 s is generation. Privacy (redaction) and verifiability (NLI) together cost **under 2%** of the response time; the 7B model is about 98%. (Retrieval was 29 ms before the switch to exact dense search, §14.6: a NumPy matrix product over 18,003 embeddings is faster than a ChromaDB query.) Questions the guard refuses never call the model and finish in well under a second. Measured on an M-series Mac with nothing else running on the GPU; the first question of a session also pays ~15 s model loading, excluded by the median.
- **Guard-off guard check, re-measured.** On the current guard-off traces, 78 of 118 test claims are still SUPPORTED by the NLI verifier with subject rewriting on, and 118 of 118 without it (Day 4, older records: 57 of 94). The Day 4 conclusion stands: verification checks evidence, not identity; the guard is the protection.
- **Stale records found, then explained.** The first Day 5 batch found 49 guard-off/raw identity answers whose wording or retrieved chunks differed from the committed `results/generation/realistic_t0.50_answers.jsonl`, and raw-baseline retrieval that differed between runs. The cause was not old records but approximate dense search (§14.6). After switching to exact search and re-running everything, all 900 pipeline traces (390 redacted, 390 raw, 120 guard-off) match the evaluation records exactly.

**Housekeeping: retrieval results regenerated.** `retrieval.evaluate` was re-run on both datasets so logs and committed results match the current code (they dated from Nov 20, before the Day 3 robustness fixes). The headline configuration (redacted, section, hybrid, query analysis on) is unchanged: answerable evidence 0.949, identity recall 0.000, leak audit 0.04%. What moved: the no-query-analysis ablation (evidence 0.077 → 0.096), raw-index hybrid identity recall (0.345 → 0.333 section, 0.381 → 0.405 note), and the redacted whole-note BM25 identity recall (0.012 → 0.000, because the question-name rule now also removes the "Mrs." title, §6.8). The §6.8 tables were refreshed from the new results. Synthetic: only raw-baseline rows moved (8 questions).

### 14.3 Log and trace PII scan (`glassbox/src/experiments/log_pii_scan.py`)

**Question.** Does any identifying value reach a file the redacted system writes? The audit trail is only trustworthy if it doesn't itself leak.

**Method.** Every log, every trace, and the committed generation answers (which are public on GitHub) are scanned, record by record, with three checks:
1. **Full values.** All 17,603 ground-truth PII values of 5+ characters in the realistic notes (836 synthetic), matched as word sequences, ignoring case and punctuation (so "(555) 867-5309" also matches "555 867 5309").
2. **Question text.** The original wording of every identity question.
3. **Name parts.** For an identity question's record, the first or last name of the person it asks about, as a capitalised word (catches a half-redacted "[NAME] Price"). Titles (Mr/Mrs/Ms/Dr) are excluded: they're kept by design and listed as a quasi-identifier in §12.

Raw-baseline files are scanned separately as the **positive control**: they must contain PII, which shows the scan can find it. The scan stores counts and qids, never the values found. It runs in about 4 s.

**Results** (`results/experiments/log_pii_scan.json`):

| Group | Records | Full values | Question text | Name parts |
|---|---|---|---|---|
| Redacted system: retrieval logs, generation logs (incl. sweep, paraphrases), traces, committed answers | 9,630 (realistic) + 1,920 (synthetic) | **0** | **0** | **0** |
| Raw baseline: retrieval logs, raw traces (positive control) | 5,460 (realistic) + 1,920 (synthetic) | 7,952 + 1,800 | 1,680 + 600 | 900 + 360 |

**Problem found and fixed.** The first run found one real leak: the Nov 21 prompt-development runs v1 and v2 (`logs/generation/realistic/redacted_v{1,2}.jsonl` and the committed `results/generation/realistic_t0.50_dev_v{1,2}_answers.jsonl`) contained the full name from QA question `realistic-id-019` ("What symptom did … report?"). Those runs predate the question-only name rule (§7, "Tiffany Price"), which was added *because* of this question. The name was replaced by `[NAME]` in the four files (3 occurrences each); no score depends on it. Re-running prompt development (25 min of GPU) would have changed nothing measured. The name remains in git history (synthetic data; noted in §12). The first scan also produced three false positives (title "Mrs" counted as part of a name), which led to the title exclusion above.

**What the leak-audit examples are.** `results/retrieval/*_t0.50.json` lists example values in its `leak_audit` section on purpose: it records *which* PII leaked into the index (§6.8). That file is a measurement, not a system log, and isn't part of this scan.

### 14.4 Audit dashboard skeleton (`glassbox/src/dashboard/app.py`)

Run with `streamlit run glassbox/src/dashboard/app.py`. The sidebar picks the dataset and configuration.

- **Ask page.** Type a question, or pick a QA-set question by type; the pipeline runs (models loaded once and cached) and the trace is shown and saved.
- **Stored traces page.** A table of every saved trace (id, type, split, redacted question, verdict, abstain reason, claims, grounded accuracy, chunks, time) with filters by verdict, type and split and summary metrics; selecting a row opens its audit view. It only reads trace files: what's shown is exactly what happened, and no model needs to load.
- **Trace view (the audit), one panel per stage:**
  1. Question: as typed (Ask page only, from session memory, never stored) next to the redacted text every later stage used; removed identifier types as badges; the guard decision with a plain-language reason.
  2. Query analysis: filters as chips, scope, k, fallback, ranking query, two-hop ask/condition.
  3. Retrieval: one card per chunk with rank, its `C` label, section, note, `selected_by`, *cited* and *low-information* badges, scores and ranks, and the redacted text.
  4–5. Answer and verification: the answer verdict badge; each claim with links that jump to the cited chunk cards and its NLI label (green SUPPORTED, red CONTRADICTED, orange UNSUPPORTED, grey UNCERTAIN), probabilities, premise chunks and the hypothesis as checked; dropped invalid citations; the raw model output with token counts.
  Plus per-stage timings and the full trace JSON.
- **Privacy.** Only what the trace holds is shown: redacted text and placeholders. Raw-baseline traces carry a warning banner.
- **Tested** headlessly with Streamlit's AppTest (`tests/test_dashboard.py`): both pages load and six kinds of trace render without errors. Still to do (Day 6, PLAN 6.2): the experiments page with figures, and a manual check in the browser.
- **Network exposure (found and fixed, Day 5 review).** Started as a real server, Streamlit listened on every network interface by default (it printed a LAN and a public "External URL") and sends usage statistics to its developer. For a page that shows the question as typed, that's a privacy hole: anyone on the same Wi-Fi could open it. `.streamlit/config.toml` now binds to `localhost` and disables telemetry; checked: the LAN address refuses connections.

### 14.5 Review fixes (Day 5, after the first commit)

A review of the Day 5 code found and fixed:

| Problem | Effect | Fix |
|---|---|---|
| Models loaded lazily on the first question | A one-off question (CLI or dashboard) reported ~15 s extra "generation" time | `GlassBox(preload=True)` loads the LLM, NLI model, embedder and redaction pipeline up front; the CLI and dashboard use it. Batch medians were unaffected. |
| Question redaction used the *index's* threshold | At thresholds other than 0.5 the pipeline would redact questions differently from the evaluated retriever (which always uses 0.5), breaking the "only glue" guarantee | Fixed to the same 0.5 (`QUESTION_REDACTION_THRESHOLD`), recorded in the trace config |
| Dashboard and CLI matched verdicts to claims by claim text | Two identical claims with different citations would show the same label | Matched by position (the verifier returns one verdict per claim, in order); regression test added |
| `run.py` with an empty selection crashed after loading models | Confusing error after ~15 s | Inputs validated before any model loads; clear message |
| Dashboard reachable from the network; telemetry on | See §14.4 | `.streamlit/config.toml` |
| Raw-baseline trace table had no warning | Un-redacted questions shown without context | Banner on the table page too |

| Dense search used ChromaDB's approximate HNSW index | Missed true neighbours on unfiltered questions and gave different results in different processes | Exact search (§14.6) |

### 14.6 Approximate dense search: finding and fix (Day 5)

**Symptom.** Two things didn't add up during the Day 5 reproduction check (§14.2). (1) On the raw baseline, the same identity question retrieved different chunks in the batch run than in the stored evaluation, and sometimes in two runs of the same code. (2) 49 guard-off/raw identity answers differed from the committed records. Greedy decoding can't explain either, so the variation had to come from retrieval.

**Cause.** ChromaDB answers dense queries with an HNSW graph, an *approximate* nearest-neighbour index. Two properties matter here:
- **Recall.** On questions with metadata filters, Chroma searches the small filtered set exhaustively, so results are exact. Unfiltered questions, which include every identity question and the no-query-analysis ablation, walk the graph and can miss true neighbours.
- **Reproducibility.** The graph walk isn't guaranteed to return the same list in a different process (index loading and multi-threaded construction), so ties and near-misses change between runs.

**Measurement** (`dense_backend="hnsw"` vs `"exact"` on the same index, top-50 dense results for all 120 identity questions; the redacted index gets the redacted question):

| Index (chunks) | Mean overlap with the true top-50 | Worst question | Questions with all 50 found | Same top-1 chunk |
|---|---|---|---|---|
| raw, section (18,790) | 36.6 / 50 | 1 / 50 | 4 / 120 | 77 / 120 |
| redacted t0.50, section (18,003) | 35.1 / 50 | 0 / 50 | 25 / 120 | 98 / 120 |

So on an average unfiltered question, about 27–30% of the chunks HNSW returned as the "top 50" weren't in the true top 50.

**Fix.** `Retriever` now loads all chunk embeddings from the Chroma collection once (in chunk order) and ranks by exact cosine similarity with a NumPy matrix product. Ties are broken by chunk position, so the order is fully deterministic. Filtered questions rank only the chunks that pass the filters (the same set Chroma searched exhaustively before). Cost: about 28 MB of memory per index, and retrieval became *faster* (median 29 ms → 6 ms per question). Chroma still stores the index, and `dense_backend="hnsw"` keeps the old behaviour for comparison. Three new tests: exact search equals a brute-force ranking, two separately loaded retrievers return identical rankings, and filtered results equal Chroma's own.

**Alternatives.** Raising HNSW's `ef_search` would raise recall but still wouldn't guarantee exact or reproducible results. A FAISS flat index would be equivalent to the fix but adds a dependency for 18k vectors.

**Re-run** (everything that depends on retrieval, ~4 h in the background): retrieval evaluation (both datasets), generation (all three configurations), verification, paraphrase sets A and B, the three pipeline batches, the threshold sweep and the log PII scan. Not re-run: the Nov 21 prompt-development runs (v1–v3, dev split). They record how prompt v2 was chosen; the comparison was between prompts on the same retrieval, and the score that decided it (grounded accuracy on answerable dev questions) comes from filtered questions, whose retrieval is exact before and after.

**What changed and what didn't:**

| Result | Before (HNSW) | After (exact) |
|---|---|---|
| GlassBox headline: answerable evidence (redacted, hybrid, QA) | 0.949 | **0.949** (unchanged) |
| GlassBox: grounded accuracy / identity answered | 0.858 / 0 of 84 | **0.858 / 0 of 84** (unchanged) |
| Questions with filters whose retrieval changed | n/a | **0 of 270** |
| Raw hybrid identity recall@auto, section / note | 0.333 / 0.405 | **0.548 / 0.512** |
| Raw dense identity recall@auto, section / note | 0.298 / 0.333 | **0.512 / 0.429** |
| Redacted no-query-analysis ablation: answerable evidence | 0.096 | 0.103 |
| Raw baseline: identity questions answered (test) | 48 of 84 (mixed old records) | **42 of 84**; 16 of 84 (19%) with the person's true value cited from their own note |
| Guard off: identity answered / claims SUPPORTED with and without subject rewriting | 84 of 84 / 74 of 112, n/a | 84 of 84 / 78 of 118, 118 of 118 |
| Paraphrase sets A and B (all metrics) | §8 | identical |
| Threshold sweep: privacy, retrieval evidence, grounded accuracy at every threshold | §5.1 | identical (only the context size of the refused identity questions moved, e.g. 708 → 643 characters at 0.5) |
| Synthetic dataset | | only raw-section hybrid @1 rows moved (e.g. identity recall@1 0.514 → 0.486) |

**Checks after the re-run:**
- **Determinism.** Generation and verification were run twice from scratch with exact search. All 900 stored answers were identical (answer, chunks, citations, refusal, score), and the summaries were identical apart from timings.
- **Pipeline = evaluation.** All 900 pipeline traces (390 redacted, 390 raw, 120 guard-off) match the evaluation records exactly, the raw baseline included.
- **Live check.** `python -m glassbox.verify` re-ran 10 stored questions through the full pipeline: 10/10 identical, 34 s.
- **PII scan.** 0 hits in 9,630 realistic privacy-side records; the raw positive control still finds 7,952 values.
- **Tests.** 165 passed, 4 skipped, 1 xfailed (fast); 169 passed, 1 xfailed with `GLASSBOX_SLOW=1`.

**Why it matters for the report.** (1) The privacy result was never affected: GlassBox's redacted index already gave 0 identity recall, and its answerable questions are filtered (exact). (2) The no-privacy baseline was *under*-estimated: with exact search it finds and leaks more, so the contrast with GlassBox is stronger than first reported. (3) It's a methods lesson worth one paragraph: an approximate index silently made a baseline weaker and non-reproducible, and the end-to-end reproduction check (§14.2) is what exposed it.

### 14.7 Demo tooling (Day 5, PLAN 6.3)

The full experiments take hours (390 questions × 3 configurations ≈ 1 h; threshold sweep ≈ 2 h). A live demo can't run them, so the demo has three parts that together show the system works:

1. **Demo page** (dashboard, first page; `glassbox/src/dashboard/demo_script.py`). A "Load models" button (~15 s), then six steps, each one button and ~3 s: (1) a clinical fact, verified; (2) an aggregate question with two-hop retrieval; (3) a named person: GlassBox refuses before the model runs, while the raw baseline next to it gives the person's medication (checked against the note: Losartan 50mg); (4) conditions that match no note: refused by rule; (5) "try to fool the verifier": edit a verified claim and re-check it (CONTRADICTED / UNSUPPORTED); (6) the audience's own question. Each step has talking points and a full-trace toggle. The questions were picked from the stored evaluation because they behave the same every time (greedy decoding); all four scripted steps were re-run after the exact-search fix and behave as described.
2. **Stored traces page.** All 390 evaluation questions, already run, searchable and auditable without loading any model.
3. **`python -m glassbox.verify`** (~1 min). Re-runs a fixed, stratified sample (seed 0: 4 clinical fact, 3 aggregate, 3 identity) through the whole pipeline and compares retrieved chunks, answer text, citations, refusal reason and score with the committed evaluation file. It prints questions in redacted form only and exits with code 1 on any difference. This proves live that the stored numbers come from the code as it is. Result: 10/10 identical.

`DEMO.md` is the 15-minute script: pre-meeting checklist, what to say at each step, likely questions with answers, and fallbacks if something fails.
