# GlassBox: Build Progress

> **Who this is for.** The project is built by one teammate and the final-year report is written by the other, so this file is the report writer's main source. It records the current state of the system and, for every stage:
> - what it does, and why it was designed that way
> - which alternatives were measured
> - the exact numbers, and how they were measured
> - what went wrong, and how it was fixed
>
> It's updated as work happens. `README.md` and `report_draft.md` get a full refresh on Day 7; until then, **trust this file over those two** for numbers and paths.

**Last updated:** 2026-11-20 (Day 2 of 8)
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
| Retrieval | 🟡 Built and evaluated; improvement pending a decision (§6.8) | `glassbox/src/retrieval/` |
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
| 2 | Nov 20–21 | Retrieval: chunking, ChromaDB + BM25 indexes (redacted and raw), access control, logging, recall@k / MRR, leak audit | 🟡 Baseline done |
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

### 6.1 Chunking (`retrieval/chunker.py`)
- **Default: one chunk per note section.**
  - The templates separate sections with blank lines (`SUBJECTIVE:`, `ASSESSMENT/PLAN:`, `S:`/`O:`/`A:`/`P:`, `HOSPITAL COURSE:`, …).
  - Short unlabelled blocks are merged into a neighbour: titles like "PROGRESS NOTE", sign-offs like "Thank you for this referral.".
- **Why sections:**
  1. Less data reaches the LLM.
  2. Each citation points at one short paragraph a human can check.
  3. The Day 4 NLI verifier needs short evidence passages.
- **Context line.** Each chunk starts with `[Progress note | 2024-11-10 | Essential hypertension]`, built only from non-PII metadata (note type, visit date, diagnosis), so a section read on its own still says which visit it belongs to.
- **Ablation: one chunk per whole note.**
- **Result** (realistic redacted): 18,009 chunks, 7.2 per note, median body length 91 characters. Every template's sections are recognised, and every character of every note is covered (both tested).
  - The synthetic notes are single paragraphs, so both strategies give one chunk per note and the synthetic section/note results are identical.
  - The raw index has 18,790 chunks: real names are longer than `[NAME]`, which pushes some sign-off lines over the merge threshold.

### 6.2 Search methods (`retrieval/bm25.py`, `retrieval/retriever.py`)

| Method | How it works | Good at | Bad at |
|---|---|---|---|
| **Dense** | `BAAI/bge-small-en-v1.5` (384-dim) embeddings, cosine similarity, stored in ChromaDB. Run locally on the Apple GPU (MPS). Queries get bge's retrieval instruction prefix. | Paraphrase and meaning | Exact strings: `2024-11-10` vs `2024-11-19` look almost identical |
| **BM25** | Classic keyword relevance (Okapi BM25, k1 = 1.5, b = 0.75), ~50 lines written in-house. Tokenizer keeps dates, IDs, BP readings and decimals as single tokens. | Exact dates, IDs, names, drug names | Synonyms and rephrasing |
| **Hybrid** | Reciprocal rank fusion: score = Σ 1/(60 + rank) over the two rankings (Cormack et al., 2009). Top 50 candidates from each. | Intended to combine both | See §6.7 |

### 6.3 Indexes (`retrieval/index.py`)
- 4 per dataset (8 total): {**redacted** (the system), **raw** (no-privacy baseline)} × {section, note}.
- Stored in `glassbox/data/indexes/<dataset>/<name>/` (gitignored; rebuild in about 90 s for all 8). Each holds:
  - `manifest.json`: provenance (source, threshold, chunking, model, counts, build time)
  - `chunks.jsonl`
  - `chroma/`
- The raw baseline reads only the note `text`. The loader drops `facts`, `entities` and `patient_name`.

### 6.4 Access control
The README's architecture promises "access-controlled retrieval". This is the implemented version:
- The `Retriever` only opens an index whose manifest says `source: redacted`.
- Opening the raw baseline requires an explicit `baseline=True`; otherwise it raises `AccessDenied` (tested).
- At most 10 chunks per query (`MAX_K`); the default is 5.
- Every log record states the index, its source and whether baseline mode was on, so the logs show that the system answered from sanitized data.
- Not implemented: real role-based permissions. That's out of scope and should be stated as such.

### 6.5 Retrieval logging
- With a log path set, every query appends one JSON line: timestamp, question ID, question, index, source, baseline flag, method, k, latency, and for each returned chunk its rank, chunk/doc ID, section, character offsets, and dense/BM25/fused scores and ranks.
- Evaluation runs write `glassbox/logs/retrieval/<dataset>/<index>.jsonl` (gitignored).
- This is stage 2 of the audit trail and feeds the dashboard.

### 6.6 How retrieval is measured (`retrieval/evaluate.py`)
- **Note level:** a retrieved chunk counts for the note it came from.
- **recall@k:** the fraction of a question's gold notes found among the top-k chunks, averaged over questions (k = 1, 3, 5, 10).
- **MRR (mean reciprocal rank):** the average of 1/rank of the first chunk from a gold note (0 if none in the top 10).
- Reported on the **test split**, per index × chunking × method × question type.
- Files:
  - `glassbox/results/retrieval/<dataset>_t0.50.json` (summary)
  - `…_per_question.json` (every question's retrieved notes)
- **Leak audit:** for every ground-truth PII value, it checks whether the value (*full leak*) or any identifying part of it, such as a first or last name, street, city or ZIP (*partial leak*, an upper bound), is visible in that note's indexed chunks. This measures leakage at the exact text the LLM can be shown.

### 6.7 Baseline results (Day 2), realistic set, test split (91 questions: 42 fact, 28 identity, 21 aggregate)

| Index | Method | R@1 | R@5 | R@10 | MRR | Fact R@5 | Identity R@5 | Aggregate R@5 |
|---|---|---|---|---|---|---|---|---|
| redacted, section | dense | 0.124 | 0.214 | 0.252 | 0.205 | 0.357 | 0.000 | 0.214 |
| redacted, section | **bm25** | 0.440 | 0.485 | 0.511 | 0.476 | **1.000** | 0.000 | 0.103 |
| redacted, section | hybrid | 0.300 | 0.447 | 0.499 | 0.393 | 0.905 | 0.000 | 0.127 |
| redacted, note | dense | 0.041 | 0.103 | 0.156 | 0.100 | 0.167 | 0.000 | 0.115 |
| redacted, note | bm25 | 0.465 | 0.477 | 0.496 | 0.492 | 1.000 | 0.000 | 0.068 |
| redacted, note | hybrid | 0.182 | 0.255 | 0.293 | 0.246 | 0.500 | 0.000 | 0.103 |
| raw, section | dense | 0.179 | 0.302 | 0.365 | 0.272 | 0.333 | 0.321 | 0.214 |
| raw, section | **bm25** | 0.747 | 0.787 | 0.813 | 0.782 | 1.000 | **1.000** | 0.079 |
| raw, section | hybrid | 0.509 | 0.694 | 0.763 | 0.619 | 0.857 | 0.857 | 0.151 |
| raw, note | dense | 0.137 | 0.165 | 0.213 | 0.172 | 0.071 | 0.393 | 0.048 |
| raw, note | bm25 | 0.773 | 0.785 | 0.812 | 0.801 | 1.000 | 1.000 | 0.068 |
| raw, note | hybrid | 0.346 | 0.483 | 0.566 | 0.426 | 0.452 | 0.821 | 0.095 |

Synthetic set, test split (56 questions), section index:

| Index | Method | R@5 | MRR | Fact | Identity | Aggregate |
|---|---|---|---|---|---|---|
| redacted | dense | 0.372 | 0.282 | 0.500 | 0.000 | 0.621 |
| redacted | bm25 | 0.632 | 0.625 | 1.000 | 0.059 | 0.583 |
| redacted | hybrid | 0.597 | 0.523 | 0.893 | 0.059 | 0.674 |
| raw | bm25 | 0.918 | 0.934 | 1.000 | 1.000 | 0.583 |

**What the numbers say:**
1. **Redaction costs nothing for clinical questions.** Clinical-fact recall@5 is 1.000 on both the redacted and raw indexes (BM25).
2. **Identity questions collapse under redaction, by design.** Recall@5 goes from 1.000 (raw) to 0.000 (redacted). This is the privacy guarantee showing up as a number.
   - The redacted *overall* figures are low mainly because identity questions are 31% of the test set.
   - The one synthetic identity hit (0.059) came from a name that wasn't fully redacted (see the leak audit). That's an actual leak caught by the evaluation.
3. **Dense search is weak here (R@5 ≈ 0.2).** Fact questions hinge on an exact date, which embeddings can't distinguish.
4. **Hybrid is worse than BM25 alone.** Equal-weight fusion lets the weak dense ranking push correct BM25 results down.
5. **Aggregate questions are badly under-retrieved by every method (R@5 ≈ 0.1).** They say "in December 2025" while the notes say `2025-12-03`; neither method connects the two.
6. **Section chunking ≥ whole-note chunking** for every method except BM25 on fact questions (tied). Hybrid sections 0.447 vs. notes 0.255. This supports the chunking choice.

**Leak audit, realistic section indexes** (what an LLM could be shown):

| PII type | n | Redacted: full value visible | Redacted: any part visible | Raw |
|---|---|---|---|---|
| ADDRESS | 1,509 | 15.5% | **98.0%** | 100% |
| NAME | 6,022 | 0.9% | 1.4% | 100% |
| PATIENT_ID / DOB / EMAIL / PHONE | 10,510 | 0.0% | 0.0% | 100% |
| **Overall** | 18,041 | 1.6% | 8.7% | 100% |

- Address parts (street, city, ZIP) are almost always visible. That's the main privacy gap, and it gets fixed on Day 7.
- The visible names are mostly provider names in signatures, e.g. "Lance Hebert", "Kimberly Little", and names with titles like "Dr. …" that spaCy only partly tags.

### 6.8 Pending decision: improving retrieval
Proposed next step, awaiting approval: add the **Query Analysis** stage from the README's architecture diagram.
- Parse dates, months/years and the diagnosis out of the question.
- Apply them as metadata filters (ChromaDB `where`, and the same filter on BM25 candidates).
- Re-tune the fusion on the **dev** split only (e.g. weighted RRF, or BM25 alone if dense still doesn't help).
- Re-report on test.
- Expected to fix the aggregate questions, and to make dense search useful once the date filter does the exact-match work.

---

## 7. Stages not built yet

- **Generation (Day 3).** Qwen2.5-7B-Instruct 4-bit via MLX (already downloaded, runs locally so no data leaves the machine). The prompt gives numbered chunks `[C1]…[Ck]` and requires JSON output `{"answer", "claims": [{"text", "citations"}]}`. Output is parsed, retried once on bad JSON, and has its citations validated.
- **Verification (Day 4).** An NLI model checks each claim against its cited chunk and assigns **Supported**, **Contradicted**, **Unsupported** or **Uncertain**.
  - Candidate models, already downloaded: `cross-encoder/nli-deberta-v3-base` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  - Evaluated on deliberately corrupted claims (swapped drug or dose, invented symptom, wrong citation), so verifier accuracy has exact ground truth.
- **Orchestrator, dashboard, experiments (Days 5–6).** See §2.

---

## 8. Tests

`python -m pytest glassbox/tests` → **22 passed, 1 xfailed**.

| File | Covers |
|---|---|
| `tests/test_generalization.py` | Redaction on formats not in the training data: dash IDs, MRN, US and written dates, 3 phone formats, full address (xfail). Plus regression tests for each Day 1 redaction fix. |
| `tests/test_retrieval.py` | Chunk coverage (no text lost), section recognition per template, context line, BM25 against a hand-computed score, tokenizer, access control, end-to-end retrieval with log format, k cap. |

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
python -m glassbox.src.retrieval.evaluate --dataset realistic     # metrics + leak audit

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
| Nov 20 | Section chunks with a non-PII context line | Precise citations, short NLI premises, less exposure | Whole-note chunks: measured, worse (§6.7) |
| Nov 20 | Hybrid dense + BM25, BM25 written in-house | Questions mix meaning with exact dates; no extra dependency | `rank-bm25` package |
| Nov 20 | Embedding model `bge-small-en-v1.5` | Strong small retrieval model; runs locally on MPS | `all-MiniLM-L6-v2` (downloaded, unused so far) |
| Nov 20 | Raw index needs explicit `baseline=True` | Makes "the LLM only sees sanitized data" enforced and auditable | Convention only |
| Nov 20 | Default k = 5, max 10 | Covers aggregate questions (2–4 notes) while limiting exposure | n/a |

---

## 11. Known issues / open items

- **ADDRESS detection (Day 7).**
  - spaCy splits addresses into fragments, so 64% of address characters stay visible, and 98% of addresses have some part visible in the index.
  - The fix is a custom address recognizer.
  - This is also the main source of NAME false positives (street names tagged as people).
- **Provider names in signatures.** About 1% of names (mostly "Dr./Mr. …" forms and signature lines) are only partly redacted.
- **Retrieval for aggregate questions is weak.** See the proposal in §6.8.
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
- Main findings:
  - Clinical facts are fully retrievable after redaction.
  - Identity questions drop to 0 (the privacy result).
  - Aggregate retrieval is weak, and hybrid underperforms BM25.
  - Proposed fix in §6.8, awaiting approval.
- Tests: 12 → 23.
