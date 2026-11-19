# GlassBox: Build Progress

> **Who this is for.** The project is built by one teammate and the final-year report is written by the other, so this file is the report writer's main source. It records the current state of the system and, for every stage:
> - what it does, and why it was designed that way
> - which alternatives were measured
> - the exact numbers, and how they were measured
> - what went wrong, and how it was fixed
>
> It's updated as work happens. `README.md` and `report_draft.md` get a full refresh on Day 7; until then, **trust this file over those two** for numbers and paths.

**Last updated:** 2026-11-20 (Day 2 of 8, retrieval complete)
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
| QA evaluation set v2 | ✅ Done | `glassbox/src/qa/build_qa_set.py` |
| PII redaction | ✅ Done (address detection still open, Day 7) | `glassbox/src/redaction/` |
| Retrieval (incl. query analysis) | ✅ Done | `glassbox/src/retrieval/` |
| Citation-based generation | ⬜ Day 3 | `glassbox/src/generation/` |
| Claim verification (NLI) | ⬜ Day 4 | `glassbox/src/verification/` |
| End-to-end orchestrator with traces | ⬜ Day 5 | (planned `glassbox/run.py`) |
| Audit dashboard (Streamlit) | ⬜ Days 5–6 | `glassbox/src/dashboard/` |
| Experiments (privacy / utility / faithfulness) | ⬜ Day 6 | `glassbox/results/` |
| README + report refresh | ⬜ Day 7 | `README.md`, `report_draft.md` |

---

## 2. 8-day plan

| Day | Date | Work | Status |
|---|---|---|---|
| 1 | Nov 20 | Cleanup, reproducible datasets, QA set v2, redaction fixes, test suite | ✅ |
| 2 | Nov 20 | Retrieval: chunking, ChromaDB + BM25 indexes (redacted and raw), access control, logging, leak audit; then query analysis, note grouping, fusion tuning | ✅ |
| 3 | Nov 22 | Generation: Qwen2.5-7B-Instruct (4-bit, MLX), cited JSON output, parsing and validation | ⬜ |
| 4 | Nov 23 | Verification: NLI verifier, 4 labels, corrupted-claim test set | ⬜ |
| 5 | Nov 24 | Orchestrator with trace JSON; start the dashboard; threshold sweep overnight | ⬜ |
| 6 | Nov 25 | Finish the dashboard; run the experiments and make the plots | ⬜ |
| 7 | Nov 26 | Address-detection fix; README and report refresh | ⬜ |
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

**Why it was rebuilt.**
- The old set had 8 questions, all of the form "What medications are prescribed for <diagnosis>?".
- Each had 100–336 relevant notes, which is impossible to retrieve in a top-5 system.
- Medications are never redacted, so the old set couldn't show any privacy cost.

**Three question types:**

| Type | Realistic n | Synthetic n | Example | Expected under redaction |
|---|---|---|---|---|
| `clinical_fact` | 60 | 40 | "What medication was prescribed for the Essential hypertension patient whose note is dated 2024-11-10?" | **Answerable.** The note is identified only by non-PII details. |
| `identity` | 40 | 25 | "What medication is John Wilkerson prescribed?" / "…for patient MRN0033092?" | **Should abstain.** The identifier is redacted. This *is* the measured privacy cost, and any confident answer is a privacy-induced hallucination for the verifier to catch. |
| `aggregate` | 30 | 15 | "Which medications were prescribed to Major depressive disorder patients who presented with depressed mood in December 2025?" | **Answerable.** Tests multi-note retrieval (2–4 gold notes each). |

- **Clinical-fact answer fields** (realistic set): medication 13, symptom 13, heart rate 10, follow-up weeks 9, blood pressure 8, lab orders 7.
- **Splits:** 30% **dev** (for tuning) and 70% **test** (for reporting), stratified by type. Realistic test split: 42 fact, 28 identity, 21 aggregate = 91 questions.
- **Scoring:** each answer item has `answer_keywords` (e.g. `["metformin", "500mg"]`). An item counts as correct if *all* its keywords appear in the generated answer. This is deterministic and model-free.
- **Validity checks:**
  - Every keyword verified present in its gold note (0 missing).
  - The diagnosis + date reference used by fact questions picks out exactly one note.
  - Names and IDs used in identity questions are unique across the whole corpus, including provider names.

---

## 5. Stage: PII redaction

**What.** Detects PII in each note and replaces it with a type tag (`[NAME]`, `[PATIENT_ID]`, …). It writes the sanitized notes and an audit log of every redaction (original value, tag, position, confidence).

**Design.**
- **Base engine:** Microsoft Presidio (spaCy `en_core_web_lg` NER plus regex recognizers).
- **Custom recognizers on top**, because generic tools miss clinical formats:
  - **PATIENT_ID:** 1–2 letters + 6 digits, dash form (`P-123456`), and MRN form (`MRN0012345`, score 1.0).
  - **DATE_OF_BIRTH:** only years 1936–2008, so visit dates aren't flagged. Covers ISO, US `MM/DD/YYYY` and written forms ("born March 20, 1985").
  - **PHONE_NUMBER:** two layers. Formatted numbers (parentheses, dots, `+1`/`001` prefixes, extensions) score 0.85. Raw 10-digit numbers score 0.40 and only pass the 0.5 threshold when a context word ("phone", "call", "contact", …) is nearby, which avoids false positives on medical codes.
- **Overlap resolution:** when detections overlap, the redactor keeps the best one. It ranks by confidence, then type specificity (pattern types like PATIENT_ID beat generic NER types like NAME), then length.
- Files: `detector.py`, `redactor.py`, `evaluate.py`, `pipeline.py` in `glassbox/src/redaction/`.

**How it's measured.**
- **Typed precision/recall/F1:** a detection is a true positive if its type matches and its span overlaps the ground truth with IoU ≥ 0.5 (intersection over union of the character ranges).
- **Masking metrics (new on Day 1):** these measure privacy directly.
  - Typed P/R also penalizes a span that was masked but got the wrong tag; from a privacy standpoint that span didn't leak.
  - *Masked recall* counts an entity as protected if any redaction covers it (IoU ≥ 0.5), whatever the type.
  - *Char leakage* is the share of ground-truth PII characters left visible.
- Metrics are computed on the spans actually redacted (after overlap resolution), i.e. on what downstream stages really see.

**Current results** (confidence 0.5, IoU 0.5). Source: `glassbox/results/redaction/{synthetic,realistic}_t0.50.json`

| Dataset | Precision | Recall | F1 | Masked recall | Char leakage |
|---|---|---|---|---|---|
| Synthetic (250 notes) | 0.864 | 0.897 | 0.881 | 0.900 | 18.2% |
| Realistic (2,500 notes) | 0.876 | 0.912 | **0.893** | 0.913 | 15.4% |

Realistic set, by type:

| Entity | n | P | R | F1 | Char leakage |
|---|---|---|---|---|---|
| EMAIL | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| DATE_OF_BIRTH | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| PATIENT_ID | 2,500 | 1.000 | 1.000 | 1.000 | 0.0% |
| PHONE_NUMBER | 3,010 | 1.000 | 0.999 | 0.999 | 0.0% |
| NAME | 6,022 | 0.841 | 0.987 | 0.908 | 1.8% |
| ADDRESS | 1,509 | 0.000 | 0.000 | 0.000 | **64.3%** |

**Problems found and fixed on Day 1.** Scaling to the 2,500-note set exposed these; they make good "engineering insight" material for the report.

| Problem | Symptom | Fix | Effect |
|---|---|---|---|
| Overlap resolver bug | Spans were processed left to right, so the *earliest* span always won. The documented "keep the higher-confidence span" never happened. | Rank all candidates globally by score, then type specificity, then length | Correct behaviour; regression test added |
| Tie between NER and ID regex | spaCy tags bare IDs like `pm010651` as PERSON at the same score, so 22 synthetic IDs got tagged `[NAME]` | Tie-break in favour of specific pattern types | PATIENT_ID recall 0.912 → 1.000 (synthetic) |
| NER spans crossing line breaks | e.g. `MRN0033092\nDate` tagged as one NAME, swallowing the ID and the next field label | Cut NER spans at the first newline | Fixed 98 realistic ID misses (with the next fix) |
| DEA licence false match | Presidio's DEA-number checksum recognizer matched `RN6659051` inside `MRN6659051` at score 1.0 | MRN pattern score raised to 1.0 (the explicit prefix is unambiguous) | PATIENT_ID recall → 1.000 (realistic) |
| UK NHS recognizer | Fired on 42 raw US phone numbers that happen to pass its checksum | Removed it (domain is US records) | PHONE recall 0.985 → 0.999 |
| "MD" read as Maryland | `Lindsey Roman, MD` → `[NAME], [ADDRESS]` | Drop clinical credentials tagged as locations, unless followed by a ZIP | Fewer ADDRESS false positives; "MD" stays readable |

Overall realistic F1 went from **0.838 (old README) to 0.893**. The old 250-note figure was 0.869; re-measured on the regenerated 250-note set it's 0.881.

---

## 6. Stage: Retrieval

**Purpose.** Retrieval is how GlassBox minimizes data. Instead of giving the LLM the whole corpus, or whole notes as a typical RAG system does, it selects the few most relevant pieces of *sanitized* text for each question and logs exactly what was selected and why.

**Final pipeline:**

```
question → Query Analysis (filters + residual query + scope)
         → candidate chunks satisfying the filters (redacted index only)
         → dense ranking + BM25 ranking → reciprocal rank fusion
         → note-grouped selection (≤ 2 chunks per note first) → top-k (5 or 10)
         → logged
```

Code is in `glassbox/src/retrieval/`: `chunker.py`, `bm25.py`, `index.py`, `query_analysis.py`, `retriever.py`, `evaluate.py`.

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
- **Result** (realistic redacted): 18,009 chunks, 7.2 per note, median body length 91 characters. Every template's sections are recognised, and every character of every note is covered (both tested).
  - The synthetic notes are single paragraphs, so both strategies give one chunk per note there.
  - The raw index has 18,790 chunks: real names are longer than `[NAME]`, which pushes some sign-off lines over the merge threshold.

### 6.2 Search methods (`bm25.py`, `retriever.py`)

| Method | How it works | Good at | Bad at |
|---|---|---|---|
| **Dense** | `BAAI/bge-small-en-v1.5` (384-dim) embeddings, cosine similarity, stored in ChromaDB. Run locally on the Apple GPU (MPS). Queries get bge's retrieval instruction prefix. | Meaning: "what medication" → the plan section | Exact strings: `2024-11-10` vs `2024-11-19` look almost identical |
| **BM25** | Classic keyword relevance (Okapi BM25, k1 = 1.5, b = 0.75), ~50 lines written in-house. Tokenizer keeps dates, IDs, BP readings and decimals as single tokens. | Exact dates, IDs, names, drug names | Synonyms: "heart rate" vs "HR" |
| **Hybrid** | Weighted reciprocal rank fusion: score = w_dense/(60 + rank_dense) + w_bm25/(60 + rank_bm25) (Cormack et al., 2009), over the top 50 candidates of each. Weights chosen on the dev split (§6.7). | Combines both | |

### 6.3 Query analysis (`query_analysis.py`), added on Day 2 after the baseline
This is the "Query Analysis" box from the README's architecture diagram. It reads the question and extracts:

| Output | Example | Used for |
|---|---|---|
| **Diagnosis filter** | "…Essential hypertension patient…" → `diagnosis = Essential hypertension` | Only chunks from notes with that diagnosis are candidates. Exact names first, then a small alias table ("diabetes", "CKD", "COPD", …). |
| **Date filter** | ISO `2024-11-10` → that day. `03/15/2025` → that day. "December 2025" / "Sept 2025" → that month. "in 2024" → that year. | Only chunks whose visit date falls in the range |
| **Residual query** | "What heart rate was recorded for the Asthma patient whose note is dated 2024-10-07?" → "What heart rate was recorded for the patient ?" | Ranking. Once the filters guarantee the constraints, repeating them in the ranking query only rewards chunks that restate the date or diagnosis (patient header, impression) over the chunk that answers. |
| **Scope** | Exact date → `single_note`. Month, year or plural "patients" → `multi_note` | Chooses k: 5 for single-note questions, 10 for multi-note ones |

- Filters use only non-PII metadata, so identity questions ("What is John Wilkerson prescribed?") get **no** filter and behave as before (tested).
- If the filters match nothing, retrieval falls back to unfiltered search and logs the fallback.
- The filters are applied inside ChromaDB (a `where` clause on diagnosis and an integer `visit_date_int`) and to BM25 candidates.
- Bug found while building it: filtered BM25 dropped chunks sharing no word with the question, so "heart rate" never reached "HR 94". Now every filtered chunk is a candidate, with zero-score chunks last.

### 6.4 Note-grouped selection
- **The problem.** Aggregate questions need *two* chunks per qualifying note: one showing why the note qualifies (the symptom) and one with the answer (the medication). Ranking alone filled the top-k with near-identical medication sections from *non-qualifying* notes of the same diagnosis and month.
- **The fix.** Each note contributes at most 2 chunks (`PER_NOTE`) before any note contributes more; leftover slots are filled in score order. For a single-note question this is identical to plain ranking.
- **Effect on the dev split** (prototype): aggregate evidence 0.22 → 0.56. A naive cap without the fill step hurt fact questions (0.94 → 0.78); that version was rejected.

### 6.5 Indexes (`index.py`)
- 4 per dataset (8 total): {**redacted** (the system), **raw** (no-privacy baseline)} × {section, note}.
- Stored in `glassbox/data/indexes/<dataset>/<name>/` (gitignored; all 8 rebuild in about 90 s). Each holds:
  - `manifest.json`: provenance (source, threshold, chunking, model, counts, build time)
  - `chunks.jsonl`
  - `chroma/`
- The raw baseline reads only the note `text`. The loader drops `facts`, `entities` and `patient_name`.

### 6.6 Access control and logging
- The `Retriever` only opens an index whose manifest says `source: redacted`.
- Opening the raw baseline requires an explicit `baseline=True`; otherwise it raises `AccessDenied` (tested).
- At most 10 chunks per query (`MAX_K`).
- Not implemented: real role-based permissions. That's out of scope and should be stated as such.
- **Retrieval log.** With a log path set, every query appends one JSON line:
  - timestamp, question ID, question
  - index, source, baseline flag
  - the full query analysis (filters, matched phrases, residual query, scope, fallback)
  - method, fusion weights, per-note cap, k, latency
  - for each returned chunk: rank, chunk/doc ID, section, character offsets, dense/BM25/fused scores and ranks
- Evaluation runs write `glassbox/logs/retrieval/<dataset>/<index>.jsonl` (gitignored). This is stage 2 of the audit trail and feeds the dashboard.

### 6.7 How retrieval is measured (`evaluate.py`)

| Metric | Definition | Why |
|---|---|---|
| **recall** | Fraction of a question's gold notes with at least one chunk in the top-k | Did we find the right patients/visits? |
| **evidence** | Fraction of a question's gold answer items (e.g. each medication) present in top-k chunks *from gold notes* | Could the LLM actually answer from what it's shown? Stricter than recall: a note's symptom chunk counts for recall but not as evidence for its medication. |
| **context (chars)** | Characters of text in the top-k chunks | Data minimization: how much the LLM sees |
| MRR | Mean of 1/rank of the first gold-note chunk | Ranking quality |

- **@auto** means "at the k the system actually uses" (5 for single-note questions, 10 for multi-note; 5 when query analysis is off). Fixed k = 1, 3, 5, 10 are in the JSON too.
- **"Answerable"** means fact + aggregate questions. Identity questions are excluded because they're unanswerable under redaction by design; they're reported separately as the privacy result.
- **Splits.** Every design choice was made on the **dev** split (39 questions; 27 answerable). All tables below are on the **test** split (91 questions: 42 fact, 28 identity, 21 aggregate).
- **Answer matching** (`qa/build_qa_set.py: answer_item_found`, reused by Day 3's answer scoring):
  - All keywords must appear.
  - Numbers must be standalone (HR "94" doesn't match inside "BP 157/94").
  - Follow-ups must read "<n> weeks".
- **Fusion weight tuning** (dev split, redacted section index, query analysis on; `results/retrieval/realistic_t0.50_fusion_tuning.json`):

| Dense weight | BM25 weight | Answerable evidence | Fact evidence | Aggregate evidence |
|---|---|---|---|---|
| 0.0 (BM25 only) | 1.0 | 0.722 | 0.889 | 0.389 |
| 0.25 | 1.0 | 0.759 | 0.889 | 0.500 |
| 0.5 | 1.0 | 0.778 | 0.944 | 0.444 |
| **1.0 (chosen)** | **1.0** | **0.796** | 0.944 | 0.500 |
| 1.5 – 5.0 | 1.0 | 0.796 (plateau) | 0.944 | 0.500 |
| 1.0 (dense only) | 0.0 | 0.759 | 0.944 | 0.389 |

  Hybrid beats both single methods on dev. The scores are flat from 1.0 to 5.0, so the simplest point, equal weights, was chosen.
  - Note: an earlier tuning run picked 0.25, but it ran before the filtered-BM25 bug fix (§6.3). It was re-run after the fix, and the weights were re-chosen on dev, not on test.

### 6.8 Results (realistic set, test split)

**Effect of query analysis.** This is the main Day 2 result; system configuration = redacted index, section chunks, hybrid.

| | Answerable recall | Answerable evidence | Fact evidence | Aggregate recall | Aggregate evidence | Context (chars) |
|---|---|---|---|---|---|---|
| Baseline (no query analysis, k = 5) | 0.645 | 0.048 | 0.071 | 0.127 | 0.000 | 579 |
| **+ query analysis, grouping, auto-k** | **1.000** | **0.840** | **1.000** | **1.000** | **0.520** | 940 |

**Privacy vs. utility: redacted vs. raw index** (same configuration):

| Index | Answerable recall | Answerable evidence | Identity recall |
|---|---|---|---|
| Raw (no privacy) | 1.000 | 0.840 | 0.857 |
| **Redacted (GlassBox)** | **1.000** | **0.840** | **0.000** |

- **Redaction costs nothing measurable for clinical and aggregate questions, and completely blocks identity lookups.** That's the central privacy-utility result at the retrieval level.
- Caveat: 1 gold answer (cf-010) is unanswerable after redaction because spaCy redacts the drug "Glipizide" as a name (§11); it isn't in the test split.

**Chunking: data minimization vs. evidence** (redacted index, query analysis on, hybrid):

| Chunking | Answerable evidence | Aggregate evidence | Context (chars) |
|---|---|---|---|
| **Section (default)** | 0.840 | 0.520 | **940** |
| Whole note | 0.984 | 0.952 | 2,630 (2.8× more) |

Whole-note retrieval finds more evidence for aggregate questions, but shows the LLM 2.8× more text, including every other section of each note. Section chunking keeps the minimization principle; its remaining weakness is aggregate questions (§11). This is a genuine trade-off for the report's discussion.

**Methods** (redacted, section, query analysis on):

| Method | Answerable evidence | Fact evidence | Aggregate evidence |
|---|---|---|---|
| Dense | 0.848 | 1.000 | 0.544 |
| BM25 | 0.798 | 0.929 | 0.536 |
| Hybrid (chosen on dev) | 0.840 | 1.000 | 0.520 |

On test, dense and hybrid are effectively tied: 0.008 apart, which is one aggregate item out of 21 questions. Hybrid was chosen on dev, and it's also much better whenever query analysis can't apply. Without filters, BM25 handles exact names and IDs: raw-index identity recall is 0.857 for hybrid vs. 0.250 for dense.

<details>
<summary>Full table: all 24 configurations, realistic test split (@auto)</summary>

| Index | Chunking | Method | Query analysis | Answerable recall | Answerable evidence | Context (chars) | Fact evidence | Aggregate recall | Aggregate evidence | Identity recall |
|---|---|---|---|---|---|---|---|---|---|---|
| redacted | section | dense | off | 0.286 | 0.032 | 587 | 0.048 | 0.191 | 0.000 | 0.000 |
| redacted | section | bm25 | off | 0.701 | 0.064 | 614 | 0.095 | 0.103 | 0.000 | 0.000 |
| redacted | section | hybrid | off | 0.645 | 0.048 | 579 | 0.071 | 0.127 | 0.000 | 0.000 |
| redacted | section | dense | on | 1.000 | 0.848 | 938 | 1.000 | 1.000 | 0.544 | 0.000 |
| redacted | section | bm25 | on | 1.000 | 0.798 | 946 | 0.929 | 1.000 | 0.536 | 0.000 |
| redacted | section | hybrid | on | 1.000 | 0.840 | 940 | 1.000 | 1.000 | 0.520 | 0.000 |
| redacted | note | dense | off | 0.149 | 0.157 | 3394 | 0.167 | 0.115 | 0.139 | 0.000 |
| redacted | note | bm25 | off | 0.689 | 0.693 | 3610 | 1.000 | 0.068 | 0.079 | 0.000 |
| redacted | note | hybrid | off | 0.368 | 0.373 | 3426 | 0.500 | 0.103 | 0.119 | 0.000 |
| redacted | note | dense | on | 0.996 | 0.976 | 2645 | 1.000 | 0.988 | 0.929 | 0.000 |
| redacted | note | bm25 | on | 1.000 | 0.984 | 2630 | 1.000 | 1.000 | 0.952 | 0.000 |
| redacted | note | hybrid | on | 1.000 | 0.984 | 2630 | 1.000 | 1.000 | 0.952 | 0.000 |
| raw | section | dense | off | 0.294 | 0.016 | 627 | 0.024 | 0.214 | 0.000 | 0.250 |
| raw | section | bm25 | off | 0.693 | 0.111 | 618 | 0.167 | 0.079 | 0.000 | 1.000 |
| raw | section | hybrid | off | 0.622 | 0.079 | 617 | 0.119 | 0.151 | 0.000 | 0.857 |
| raw | section | dense | on | 1.000 | 0.856 | 954 | 1.000 | 1.000 | 0.568 | 0.250 |
| raw | section | bm25 | on | 1.000 | 0.798 | 947 | 0.929 | 1.000 | 0.536 | 1.000 |
| raw | section | hybrid | on | 1.000 | 0.840 | 945 | 1.000 | 1.000 | 0.520 | 0.857 |
| raw | note | dense | off | 0.064 | 0.068 | 4009 | 0.071 | 0.048 | 0.059 | 0.321 |
| raw | note | bm25 | off | 0.689 | 0.693 | 3738 | 1.000 | 0.068 | 0.079 | 1.000 |
| raw | note | hybrid | off | 0.333 | 0.341 | 3895 | 0.452 | 0.095 | 0.119 | 0.786 |
| raw | note | dense | on | 0.996 | 0.992 | 2728 | 1.000 | 0.988 | 0.976 | 0.321 |
| raw | note | bm25 | on | 1.000 | 1.000 | 2732 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw | note | hybrid | on | 1.000 | 1.000 | 2722 | 1.000 | 1.000 | 1.000 | 0.786 |

</details>

**Synthetic set** (250 notes, test split, section index): with query analysis, every method reaches 1.000 recall and evidence on answerable questions, at about 840 characters of context. Redacted identity recall is 0.059: one name that wasn't fully redacted was found by BM25, an actual leak the evaluation caught. Raw identity recall is 1.000 (BM25).

**Leak audit, realistic section indexes.** For every ground-truth PII value: is the full value, or any identifying part of it, visible in that note's indexed chunks (i.e. text an LLM could be shown)?

| PII type | n | Redacted: full value visible | Redacted: any part visible | Raw |
|---|---|---|---|---|
| ADDRESS | 1,509 | 15.5% | **98.0%** | 100% |
| NAME | 6,022 | 0.9% | 1.4% | 100% |
| PATIENT_ID / DOB / EMAIL / PHONE | 10,510 | 0.0% | 0.0% | 100% |
| **Overall** | 18,041 | 1.6% | 8.7% | 100% |

- Address parts (street, city, ZIP) are almost always visible. That's the main privacy gap, and it gets fixed on Day 7.
- The visible names are mostly provider names in signatures (e.g. "Lance Hebert") and names with titles ("Dr. …") that spaCy only partly tags.

### 6.9 The Day 2 morning baseline, for the "before" story
The first evaluation, with no query analysis, equal-weight fusion and k = 5, measured only note recall. Realistic test results:
- Hybrid had answerable recall 0.645 and was *worse* than BM25 alone.
- Aggregate recall was 0.103 for BM25 and 0.127 for hybrid.
- Diagnosis: questions hinge on exact dates (which dense search can't distinguish), and on month names that neither method connects to ISO dates.

That analysis motivated §6.3 and §6.4. The full baseline numbers are in git history: commit `5c62519`, `results/retrieval/*.json`.

---

## 7. Stages not built yet

- **Generation (Day 3).** Qwen2.5-7B-Instruct 4-bit via MLX (already downloaded, runs locally so no data leaves the machine). The prompt gives numbered chunks `[C1]…[Ck]` and requires JSON output `{"answer", "claims": [{"text", "citations"}]}`. Output is parsed, retried once on bad JSON, and has its citations validated.
- **Verification (Day 4).** An NLI model checks each claim against its cited chunk and assigns **Supported**, **Contradicted**, **Unsupported** or **Uncertain**.
  - Candidate models, already downloaded: `cross-encoder/nli-deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  - Evaluated on deliberately corrupted claims (swapped drug or dose, invented symptom, wrong citation), so verifier accuracy has exact ground truth.
- **Orchestrator, dashboard, experiments (Days 5–6).** See §2.

---

## 8. Tests

`python -m pytest glassbox/tests` → **34 passed, 1 xfailed**.

| File | Covers |
|---|---|
| `tests/test_generalization.py` | Redaction on formats not in the training data: dash IDs, MRN, US and written dates, 3 phone formats, full address (xfail). Plus regression tests for each Day 1 redaction fix. |
| `tests/test_retrieval.py` | Chunk coverage (no text lost), section recognition per template, context line, BM25 against a hand-computed score, tokenizer, access control, end-to-end retrieval with log format, k cap. Query analysis: filters for ISO, US, month and year dates and aliases; no filters for identity questions; residual query; Chroma/Python filter agreement. Note grouping, filtered BM25 keeping zero-score chunks, auto-k. |

- Generalization count: **7/8**. The address case is a strict `xfail`; it'll flag once Day 7's fix makes it pass.
- The old README claimed 8/8, but the committed test file only had 3 of those cases.

---

## 9. How to run

From the repo root, with `.venv` active:

```bash
# Data
python -m glassbox.src.generate_synthetic_dataset                 # 250 notes + QA set
python -m glassbox.src.generate_realistic_dataset --size large    # 2,500 notes + QA set

# Redaction (writes data/processed/<dataset>/t0.50/ and results/redaction/)
python -m glassbox.src.redaction.pipeline --dataset realistic [--threshold 0.5]

# Retrieval
python -m glassbox.src.retrieval.index --dataset realistic --all  # build the 4 indexes
python -m glassbox.src.retrieval.evaluate --dataset realistic     # metrics + leak audit (~3 min)
python -m glassbox.src.retrieval.evaluate --dataset realistic --tune  # dev-split fusion weight grid

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
| Nov 20 | Fusion weights (1.0, 1.0), chosen on dev | Plateau from 1.0 to 5.0; simplest point | BM25-only 0.722, dense-only 0.759 vs. hybrid 0.796 (dev answerable evidence) |
| Nov 20 | Add evidence and context-size metrics | Note recall alone hid that answer-bearing sections weren't retrieved; context size makes minimization measurable | Recall@k / MRR only |

---

## 11. Known issues / open items

- **ADDRESS detection (Day 7).**
  - spaCy splits addresses into fragments, so 64% of address characters stay visible, and 98% of addresses have some part visible in the index.
  - The fix is a custom address recognizer.
  - This is also the main source of NAME false positives (street names tagged as people).
- **Provider names in signatures.** About 1% of names (mostly "Dr./Mr. …" forms and signature lines) are only partly redacted.
- **Clinical over-redaction (found Day 2).** 148 redactions hit non-PII text.
  - Mainly the drug **"Glipizide"** tagged as a person's name (97 notes), "SpO2" (26) and a few BP readings (9).
  - Glipizide becomes `[NAME]`, which makes 3 QA gold answers unanswerable after redaction (1 dev, 2 test).
  - Fix candidate: a clinical-term guard (drug lexicon, vitals patterns) on NER detections. This should be done **with the address fix, before the Day 6 experiments**, so the final numbers include it.
- **Aggregate questions on section chunks.** Evidence is 0.520: within each qualifying note, the medication section doesn't always make that note's top 2. Whole-note chunks reach 0.952 but expose 2.8× more text. Possible future work: an answer-type-aware section prior (e.g. "medications" → plan, recommendations and discharge-medications sections).
- **The audit log stores raw PII values.** That's needed for the audit trail, but the log is itself a sensitive artifact. The report should say so.
- **The docs are stale.** The README and `report_draft.md` show old numbers (0.838 / 0.869 / 0.858, 62.5%, 8/8) and old data paths. Refresh on Day 7.

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
