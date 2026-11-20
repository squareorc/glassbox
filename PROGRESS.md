# GlassBox: Build Progress

> **Who this is for.** The project is built by one teammate and the final-year report is written by the other, so this file is the report writer's main source. It records the current state of the system and, for every stage:
> - what it does, and why it was designed that way
> - which alternatives were measured
> - the exact numbers, and how they were measured
> - what went wrong, and how it was fixed
>
> It's updated as work happens. `README.md` and `report_draft.md` get a full refresh on Day 7; until then, **trust this file over those two** for numbers and paths.

**Last updated:** 2026-11-20 (Day 3a: audit fixes complete; generation next)
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
7. [Stages not built yet](#7-stages-not-built-yet)
8. [Tests](#8-tests)
9. [How to run](#9-how-to-run)
10. [Key decisions log](#10-key-decisions-log)
11. [Known issues / open items](#11-known-issues--open-items)
12. [Daily log](#12-daily-log)

---

## 1. Status at a glance

| Stage | Status | Code |
|---|---|---|
| Synthetic datasets (250 + 2,500 notes) | ✅ Done | `glassbox/src/generate_*_dataset.py` |
| QA evaluation set v3 (390 + 160 questions) | ✅ Done | `glassbox/src/qa/build_qa_set.py` |
| PII redaction (incl. address recognizer, clinical-term guard) | ✅ Done | `glassbox/src/redaction/` |
| Question redaction | ✅ Done | `glassbox/src/redaction/query.py` |
| Retrieval (query analysis, two-hop selection) | ✅ Done | `glassbox/src/retrieval/` |
| Citation-based generation (with abstain rule) | ⬜ Day 3 | `glassbox/src/generation/` |
| Claim verification (NLI, claims rewritten with the question's subject) | ⬜ Day 4 | `glassbox/src/verification/` |
| End-to-end orchestrator with traces | ⬜ Day 5 | (planned `glassbox/run.py`) |
| Audit dashboard (Streamlit) | ⬜ Days 5–6 | `glassbox/src/dashboard/` |
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

---

## 2. 8-day plan

| Day | Date | Work | Status |
|---|---|---|---|
| 1 | Nov 20 | Cleanup, reproducible datasets, QA set v2, redaction fixes, test suite | ✅ |
| 2 | Nov 20 | Retrieval: chunking, ChromaDB + BM25 indexes (redacted and raw), access control, logging, leak audit; then query analysis, note grouping, fusion tuning | ✅ |
| — | Nov 20 | Independent audit of Days 1–2 (§11.1) | ✅ |
| 3a | Nov 20 | Audit fixes (§11.1): address recognizer, clinical-term guard, names in clinical context, question redaction, query-analysis ranges / "or" / invalid dates, low-information demotion, note-ordered and two-hop selection, QA set ×3, bootstrap CIs, `requirements.txt` | ✅ |
| 3 | Nov 22 | Generation: Qwen2.5-7B-Instruct (4-bit, MLX), cited JSON output, parsing and validation. **Abstain** when the question names a person (`identifiers` from question redaction) or retrieval fell back to unfiltered search | ⬜ |
| 4 | Nov 23 | Verification: NLI verifier, 4 labels, corrupted-claim test set. **Rewrite each claim with the question's subject before verifying** (§11.1 #1) | ⬜ |
| 5 | Nov 24 | Orchestrator with trace JSON; start the dashboard; threshold sweep overnight. (The redaction fixes planned for this day were done on Day 3a.) | ⬜ |
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
- **Independent check (audit, §11.1):** a plain string search of every ground-truth PII value in the redacted notes, outside the project's evaluator.

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
- **Read these numbers with care.** They are measured on synthetic notes whose addresses come from one generator (Faker), so the address recognizer is close to the data's format by construction. Generalisation is covered separately by hand-written tests on formats not in the data (§8), and by the stated limitations in §11.

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

**Problems found and fixed on Day 3a** (from the audit, §11.1):

| Problem | Symptom | Fix | Effect (realistic) |
|---|---|---|---|
| Addresses fragmented by NER | 64% of address characters visible; 98% of addresses partly visible in the index; street names counted as NAME false positives | Structural US postal-address recognizer | ADDRESS F1 0.000 → 0.993; NAME precision 0.841 → 0.999 |
| Clinical terms redacted | "Glipizide" → `[NAME]` in 97 notes (3 gold answers lost), "SpO2" 26, "Email" 17, BP readings 9 | Clinical-term guard on NER detections | Over-redactions 148 → 0; 0 gold answers lost |
| Provider names missed | 51 names verbatim, mostly signatures ("Lance Hebert, MD") and "Referring Physician: …" | Name-before-credential and name-after-label patterns | Verbatim names 51 → 2 |
| Numeric MRNs missed | `MRN: 00482913` not redacted (not in the data; found by probing) | Labelled-number pattern | Tested |
| Case-insensitive patterns | Presidio's default flag let `[A-Z]` match lowercase words | Compile the new patterns case-sensitively | Tested |

Overall realistic F1: **0.893 → 0.999**.

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

1. **Low-information demotion** (Day 3a). After redaction, header, contact and signature sections ("PATIENT: [NAME] / MRN: [PATIENT_ID] / DOB: [DATE_OF_BIRTH]") are only placeholders and form labels. They can't answer anything, yet they match "patient" questions well, and they are what invites the misattribution risk in §11.1 #1. A chunk is *low-information* if it contains a placeholder and, after removing placeholders, form labels and numbers, fewer than 3 words remain. These chunks are moved behind all others, not dropped. This flags exactly the patient_info, contact and signature sections (2,500 / 2,499 / 1,001 of them) and no chunk of the raw index, so the baseline is unaffected.
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
  - Dense-only ties on dev, but hybrid is kept: whenever query analysis can't filter, BM25 is what matches exact names, IDs and drug strings (raw-index identity recall: BM25 1.000, dense 0.309).
  - Side effect: with dense weighted higher, the raw *hybrid* index finds fewer identity targets (0.845 at 1:1 → 0.345). This only concerns the no-privacy baseline. For the privacy comparison, use the strongest attacker on raw data, BM25, which finds 100%.

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
| No query analysis (k = 5) | 0.314 | 0.077 | 0.111 | 0.163 | 0.008 | 618 |
| **With query analysis** | **0.994** | **0.949** | **1.000** | **0.983** | **0.847** | 955 |

**Privacy vs. utility: redacted vs. raw index** (same configuration):

| Index | Answerable recall | Answerable evidence (95% CI) | Identity recall: hybrid | Identity recall: BM25 (strongest attacker) |
|---|---|---|---|---|
| Raw (no privacy) | 0.994 | 0.907 [0.870, 0.940] | 0.345 | **1.000** |
| **Redacted (GlassBox)** | **0.994** | **0.949** [0.922, 0.972] | **0.000** | **0.000** |

- **Redaction costs nothing measurable for clinical and aggregate questions, and completely blocks identity lookups.** That's the central privacy-utility result at the retrieval level.
- The redacted index scores *higher* than raw. Redaction doesn't add information: the difference comes from the low-information demotion, which can only recognise header sections once their values are placeholders. In the raw index those sections hold real names and compete for slots. The report should present this as "redaction costs nothing", not "redaction helps".
- One identity question (1/84) is found by BM25 on the redacted *whole-note* index: the redacted question "What symptom did Mrs. [NAME] report?" matched one of the few notes where the title "Mrs." survives in front of `[NAME]`. Titles are quasi-identifiers (§11).

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
<summary>Full table: all 24 configurations, realistic test split (@auto)</summary>

| Index | Chunking | Method | Query analysis | Answerable recall | Answerable evidence | Context (chars) | Fact evidence | Aggregate recall | Aggregate evidence | Identity recall |
|---|---|---|---|---|---|---|---|---|---|---|
| redacted | section | dense | off | 0.134 | 0.034 | 613 | 0.048 | 0.164 | 0.008 | 0.000 |
| redacted | section | bm25 | off | 0.691 | 0.296 | 616 | 0.444 | 0.071 | 0.000 | 0.000 |
| redacted | section | hybrid | off | 0.314 | 0.077 | 618 | 0.111 | 0.163 | 0.008 | 0.000 |
| redacted | section | dense | on | 0.990 | 0.946 | 945 | 1.000 | 0.970 | 0.839 | 0.000 |
| redacted | section | bm25 | on | 1.000 | 0.812 | 968 | 0.944 | 1.000 | 0.548 | 0.000 |
| redacted | section | hybrid | on | 0.994 | 0.949 | 955 | 1.000 | 0.983 | 0.847 | 0.000 |
| redacted | note | dense | off | 0.143 | 0.156 | 3233 | 0.151 | 0.128 | 0.167 | 0.000 |
| redacted | note | bm25 | off | 0.684 | 0.689 | 3556 | 1.000 | 0.053 | 0.066 | 0.012 |
| redacted | note | hybrid | off | 0.286 | 0.302 | 3299 | 0.365 | 0.127 | 0.175 | 0.000 |
| redacted | note | dense | on | 0.994 | 0.996 | 2605 | 1.000 | 0.983 | 0.987 | 0.000 |
| redacted | note | bm25 | on | 1.000 | 1.000 | 2578 | 1.000 | 1.000 | 1.000 | 0.012 |
| redacted | note | hybrid | on | 1.000 | 1.000 | 2598 | 1.000 | 1.000 | 1.000 | 0.000 |
| raw | section | dense | off | 0.133 | 0.034 | 648 | 0.048 | 0.161 | 0.008 | 0.309 |
| raw | section | bm25 | off | 0.691 | 0.127 | 629 | 0.191 | 0.071 | 0.000 | 1.000 |
| raw | section | hybrid | off | 0.324 | 0.108 | 642 | 0.159 | 0.161 | 0.008 | 0.345 |
| raw | section | dense | on | 0.990 | 0.919 | 946 | 1.000 | 0.970 | 0.757 | 0.309 |
| raw | section | bm25 | on | 1.000 | 0.802 | 964 | 0.929 | 1.000 | 0.548 | 1.000 |
| raw | section | hybrid | on | 0.994 | 0.907 | 948 | 0.992 | 0.983 | 0.738 | 0.345 |
| raw | note | dense | off | 0.063 | 0.079 | 3958 | 0.048 | 0.094 | 0.140 | 0.309 |
| raw | note | bm25 | off | 0.695 | 0.703 | 3777 | 1.000 | 0.085 | 0.108 | 1.000 |
| raw | note | hybrid | off | 0.201 | 0.216 | 3862 | 0.246 | 0.111 | 0.156 | 0.381 |
| raw | note | dense | on | 0.998 | 0.998 | 2785 | 1.000 | 0.995 | 0.995 | 0.309 |
| raw | note | bm25 | on | 1.000 | 1.000 | 2756 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw | note | hybrid | on | 0.998 | 0.998 | 2774 | 1.000 | 0.995 | 0.995 | 0.381 |

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

## 7. Stages not built yet

- **Generation (Day 3).** Qwen2.5-7B-Instruct 4-bit via MLX (already downloaded, runs locally so no data leaves the machine). The prompt gives numbered chunks `[C1]…[Ck]` and requires JSON output `{"answer", "claims": [{"text", "citations"}]}`. Output is parsed, retried once on bad JSON, and has its citations validated (the model writes both `C1` and `[C1]`, so citations are normalised).
  - **Abstain rule (from the audit).** If question redaction found an identifier, or retrieval fell back to unfiltered search, the system must answer "can't be determined from the de-identified records" instead of answering from whatever notes were retrieved. In the audit smoke test, Qwen answered "What is John Wilkerson's medication?" from a redacted note, treating `[NAME]` as John.
  - Speed measured in the audit: model loads in 1.3 s, answers in ~2.6 s, peak memory 4.5 GB. A 390-question run takes ~17 minutes.
- **Verification (Day 4).** An NLI model checks each claim against its cited chunk and assigns **Supported**, **Contradicted**, **Unsupported** or **Uncertain**.
  - Candidate models, already downloaded: `cross-encoder/nli-deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  - Evaluated on deliberately corrupted claims (swapped drug or dose, invented symptom, wrong citation), so verifier accuracy has exact ground truth.
  - **Claims are rewritten with the question's subject before verifying** (from the audit). Both NLI models label "The patient is prescribed X" as *entailment* against a redacted note, but "John Wilkerson is prescribed X" as *neutral*. Without the rewrite, a misattributed answer would be marked Supported.
- **Orchestrator, dashboard, experiments (Days 5–6).** See §2.

---

## 8. Tests

`python -m pytest glassbox/tests` → **70 passed, 1 xfailed**.

| File | Covers |
|---|---|
| `tests/test_generalization.py` | Redaction on formats not in the data: dash IDs, MRN, numeric MRN after a label, US and written dates, 3 phone formats, full address. Day 3a: 5 address formats as single spans, street line only with context, case-sensitivity, clinical terms kept (drugs, SpO2, BP), field label "Email" kept, names before credentials / after labels / after "Dr.". Plus regression tests for each Day 1 redaction fix. |
| `tests/test_retrieval.py` | Chunk coverage (no text lost), section recognition per template, context line, BM25 against a hand-computed score, tokenizer, access control, end-to-end retrieval with log format, k cap. Query analysis: filters for ISO, US, month and year dates and aliases; ranges, several diagnoses, "May, 2025", invalid dates; no filters for identity questions; residual query; Chroma/Python filter agreement incl. `$in`. Selection: note grouping (both versions), filtered BM25 keeping zero-score chunks, auto-k, low-information classifier, ask/condition split, two-hop picks each gold note's answer section. Question redaction: names removed, clinical questions unchanged, the name never reaches the log. |

- The full-address test was a strict `xfail` until Day 3a; it now passes.
- The one remaining strict `xfail` documents a known limitation: an all-lowercase address ("42 oak street, springfield, il 62704") is only partly caught, because the address patterns use capitalisation.
- The old README claimed 8/8 generalization cases, but the committed test file only had 3 of those cases.

---

## 9. How to run

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

# Tests
python -m pytest glassbox/tests
```

---

## 10. Key decisions log

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
| Nov 20 (3a) | Triple the QA set; report bootstrap 95% CIs | CI was ±0.08; method differences were noise | Keep 130 questions |
| Nov 20 | Add evidence and context-size metrics | Note recall alone hid that answer-bearing sections weren't retrieved; context size makes minimization measurable | Recall@k / MRR only |

---

## 11. Known issues / open items

- **Misattribution under redaction (critical; Days 3–4).** See §11.1 #1. Retrieval can't find a named person in the redacted index, but an LLM given redacted notes may still attribute one of them to that person, and a claim like "The patient is prescribed X" verifies as Supported. Fixes are planned: the abstain rule (Day 3), claims rewritten with the question's subject (Day 4), and abstention / false-attribution rates as privacy metrics (Day 6).
- **Titles are quasi-identifiers.** "Mrs. [NAME]" keeps the title, which reveals gender and let one redacted identity question match its note. Option: redact titles with the name; worth one sentence in the report either way.
- **Redaction generalisation limits** (for the report's limitations section; found by probing, not in the data): all-lowercase addresses are only partly caught; insurance member numbers, hospital names and birth years before 1936 are not redacted. Lowercase names, relatives' names, valid SSNs and numeric MRNs after a label are.
- **The redaction numbers are in-distribution.** 0.999 F1 is measured on synthetic notes from the same generator the recognizers were checked against. The hand-written generalisation tests (§8) are the evidence for other formats.
- **Remaining aggregate misses** (test aggregate evidence 0.847): the ask "Which medications were prescribed" sometimes ranks LAB ORDERS above a plan line that never says "medication".
- **Query analysis is template-evaluated.** The QA questions come from templates; free-text handling is unit-tested only.
- **The audit log stores raw PII values.** That's needed for the audit trail, but the log is itself a sensitive artifact. The report should say so. (Retrieval logs no longer contain PII from questions.)
- **The docs are stale.** The README and `report_draft.md` show old numbers (0.838 / 0.869 / 0.858, 62.5%, 8/8) and old data paths. Refresh on Day 7.

### 11.1 Audit of Days 1–2 (Nov 20)

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
| 1 | Critical | **Misattribution under redaction.** Asked "What is John Wilkerson's medication?" with a redacted note as context, Qwen answered with that note's drug, treating `[NAME]` as John. It phrased the claim "The patient is prescribed …", which both NLI models label *entailment* (Supported). "Identity recall 0.000" only shows retrieval can't find the person; without a guard the full system would still give a confident, verified, wrong answer. | Smoke test. With the name in the claim ("John Wilkerson is prescribed …") both models say *neutral* (Unsupported). | Abstain rule (Day 3); rewrite claims with the question's subject before verifying (Day 4); report abstention and false-attribution rates as privacy metrics (Day 6) **Status: planned, Days 3–4** (needs generation/verification). |
| 2 | Critical | **The question itself isn't redacted.** A name typed into the question reaches the LLM prompt, the query embedding and the retrieval log. | `retriever.py` uses the raw query | Redact the question at entry; detected identifiers also trigger the abstain rule in #1 (Day 3) **Status: ✅ fixed Day 3a** (§6.6). |
| 3 | Important | **Aggregate evidence 0.52 is a two-hop selection problem.** 6 of 21 aggregate test questions have zero evidence despite recall 1.0. The qualifying symptom is in *subjective*, the medication in *plan*; after the qualifying sections, the remaining slots go to *other* notes' plan sections. | e.g. ag-012, ag-020, ag-027 retrieve every gold note's subjective section, then non-gold plans | After selecting qualifying notes, add each one's answer section (plan / medications) (Day 3) **Status: ✅ fixed Day 3a**: two-hop selection, test aggregate evidence 0.553 → 0.847 (§6.4, §6.8). |
| 4 | Important | **NAME precision 0.84 is mostly an address problem.** Of ~1,130 NAME spans not matching gold exactly, 965 are address fragments and 92 are "Dr. X" where gold includes the title (name masked, scored as an error). Pure false positives: 148 (§11). | Audit log vs. gold spans | Day 5 address fix also raises NAME precision **Status: ✅ fixed Day 3a**: NAME precision 0.841 → 0.999, "Email" kept (§5). |
| 5 | Important | **Context spent on placeholders.** 18% of context slots go to `patient_info` / `contact` sections, mostly `[NAME]`-style tags after redaction; they also invite #1. | Per-question retrieved sections | Drop or rank them lower for non-identity questions (Day 3) **Status: ✅ fixed Day 3a**: low-information demotion (§6.4). |
| 6 | Important | **No `requirements.txt` or install steps**; matplotlib not installed (needed for plots). | Repo root | Day 5 **Status: ✅ fixed Day 3a**: `requirements.txt`, matplotlib 3.10.9 installed. |
| 7 | Important | **Small test split.** 63 answerable test questions; 95% bootstrap CI for evidence 0.840 is [0.757, 0.915]. Hybrid vs. dense differ on 8 questions (mean −0.008): not significant, report as "tied". | Bootstrap, 5,000 resamples | Enlarge QA set, CIs on plots (Day 6) **Status: ✅ fixed Day 3a**: QA set ×3, bootstrap CIs in every summary (§4, §6.7). |
| 8 | Minor | **Query analysis on free-typed questions** (the QA set is templated, so evaluation doesn't see these): "between March and June 2025" → March only; "hypertension or diabetes" → first diagnosis only; "May, 2025" not parsed; invalid "2025-13-01" becomes all of 2025; a date with no notes ("in 2019") falls back to unrelated notes. | Probe questions | Ranges and "or" (Day 5); pass `fallback` to the generator so it abstains (Day 3) **Status: ✅ fixed Day 3a** for ranges, "or", "May, 2025" and invalid dates (§6.3); fallback-abstain is Day 3. |
| 9 | Minor | **Redaction gaps outside the dataset** (limitations section): numeric-only MRNs (`00482913`), insurance numbers, hospital names, birth years before 1936 are not redacted. Lowercase names, relatives' names and valid SSNs are. | Probe sentences | Report as limitations **Status: partly fixed Day 3a**: numeric MRNs after a label; the rest are listed limitations. |
| 10 | Minor | Raw and redacted indexes chunk slightly differently (781 signature blocks merged in redacted, see §6.1). Headline unaffected (both 0.840). | Section counts | Note in report **Status: open** (documented, §6.1). |

---

## 12. Daily log

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
- Found clinical over-redaction ("Glipizide" → `[NAME]`; §11).
- Tests: 12 → 35.

### Audit: Nov 20
- Independent audit of Days 1–2 (§11.1): reproducibility, QA gold, leakage and headline numbers all verified.
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

