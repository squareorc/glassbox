# GlassBox: Build Progress

The project's current state, updated as work happens. The README and `report_draft.md` get a full refresh on Day 7. Until then, this file is the source of truth for current numbers, paths and status.

**Last updated:** 2026-11-20 (Day 1 of 8)
**Branch:** `pipeline-build`
**Deadline:** 2026-11-28

---

## Status at a glance

| Stage | Status | Location |
|---|---|---|
| Synthetic datasets (250 + 2,500 notes) | ✅ Done | `glassbox/src/generate_*_dataset.py` → `glassbox/data/raw/{synthetic,realistic}/` |
| QA eval set v2 | ✅ Done | `glassbox/src/qa/build_qa_set.py` |
| PII redaction | ✅ Done (address detection still open) | `glassbox/src/redaction/` |
| Retrieval | ⏳ Next (Day 2) | `glassbox/src/retrieval/` |
| Citation-based generation | ⬜ Not started | `glassbox/src/generation/` |
| Claim verification (NLI) | ⬜ Not started | `glassbox/src/verification/` |
| End-to-end orchestrator + traces | ⬜ Not started | (planned: `glassbox/run.py`) |
| Audit dashboard (Streamlit) | ⬜ Not started | `glassbox/src/dashboard/` |
| Experiments (privacy / utility / faithfulness) | ⬜ Not started | (planned: `glassbox/results/`) |
| README + report refresh | ⬜ Day 7 | `README.md`, `report_draft.md` |

---

## 8-day plan

| Day | Date | Work | Status |
|---|---|---|---|
| 1 | Nov 20 | Cleanup, reproducible datasets, QA set v2, redaction fixes, full test suite | ✅ Done |
| 2 | Nov 21 | Retrieval: section chunking, two ChromaDB indexes (redacted and raw), logging, recall@k / MRR | ⏳ |
| 3 | Nov 22 | Generation: Qwen2.5-7B (MLX), cited JSON output, parsing and validation | ⬜ |
| 4 | Nov 23 | Verification: NLI verifier, 4 labels, corrupted-claim test set | ⬜ |
| 5 | Nov 24 | Orchestrator with trace JSON; start the dashboard; threshold sweep overnight | ⬜ |
| 6 | Nov 25 | Finish the dashboard; run the experiments and make the plots | ⬜ |
| 7 | Nov 26 | Address-detection fix; README and report refresh with final numbers | ⬜ |
| 8 | Nov 27 | Polish, buffer, demo rehearsal, backup video | ⬜ |

---

## Current results

### Redaction (confidence 0.5, IoU 0.5)
Source files: `glassbox/results/redaction/{synthetic,realistic}_t0.50.json`

| Dataset | Precision | Recall | F1 | Masked recall | PII char leakage |
|---|---|---|---|---|---|
| Synthetic (250 notes, 838 entities) | 0.864 | 0.897 | 0.881 | 0.900 | 18.2% |
| Realistic (2,500 notes, 18,041 entities) | 0.876 | 0.912 | **0.893** | 0.913 | 15.4% |

Per entity type, realistic set:

| Entity | P | R | F1 | Char leakage |
|---|---|---|---|---|
| EMAIL | 1.000 | 1.000 | 1.000 | 0.0% |
| DATE_OF_BIRTH | 1.000 | 1.000 | 1.000 | 0.0% |
| PATIENT_ID | 1.000 | 1.000 | 1.000 | 0.0% |
| PHONE_NUMBER | 1.000 | 0.999 | 0.999 | 0.0% |
| NAME | 0.841 | 0.987 | 0.908 | 1.8% |
| ADDRESS | 0.000 | 0.000 | 0.000 | **64.3%** |

- *Masked recall* counts a PII entity as protected if any redaction span covers it (IoU ≥ 0.5), whatever its label.
- *Char leakage* is the share of ground-truth PII characters left visible.
- Neither existed before Day 1. They measure privacy directly, whereas typed P/R also penalizes a correctly masked span that got the wrong label.

### Tests
`pytest glassbox/tests`: **11 passed, 1 xfailed**. The xfail is full-span address detection, marked strict so it flags once the Day 7 fix makes it pass. Honest generalization count: **7/8**.

### Retrieval / generation / verification
Not built yet.

---

## Data and QA set

| Path | Contents |
|---|---|
| `glassbox/data/raw/synthetic/` | 250 simple notes, 8 conditions; `qa_eval_set.jsonl` (80 questions) |
| `glassbox/data/raw/realistic/` | 2,500 EHR-style notes (1,478 progress, 510 discharge, 512 consultation), 10 conditions; `qa_eval_set.jsonl` (130 questions) |
| `glassbox/data/processed/<dataset>/t<threshold>/` | Redacted notes and audit log. Gitignored and regenerable |
| `glassbox/results/` | Committed metrics JSON |

Raw notes include `facts`, the structured ground truth used to build the QA set. That field contains PII and must never reach retrieval or generation; those stages read only `data/processed/`.

QA question types, realistic set (30% dev / 70% test):

| Type | n | Description | Expected under redaction |
|---|---|---|---|
| `clinical_fact` | 60 | One note, referred to by diagnosis and date; asks for medication, symptom, HR, BP, labs or follow-up | Still answerable |
| `identity` | 40 | One note, referred to by the patient's name or ID | The system should abstain (this is the measured privacy cost) |
| `aggregate` | 30 | 2–4 notes sharing diagnosis, symptom and month | Still answerable; tests multi-document retrieval |

Scoring uses `answer_keywords`: every keyword of an answer item must appear in the generated answer. It's deterministic and needs no model.

---

## How to run (from repo root, venv active)

```bash
python -m glassbox.src.generate_synthetic_dataset            # 250 notes + QA set
python -m glassbox.src.generate_realistic_dataset --size large  # 2,500 notes + QA set
python -m glassbox.src.redaction.pipeline --dataset realistic [--threshold 0.5]
python -m pytest glassbox/tests
```

---

## Key decisions

| Date | Decision | Why |
|---|---|---|
| Nov 20 | Pinned generator "today" to 2026-09-16 | Regeneration was date-dependent; it's now byte-for-byte reproducible |
| Nov 20 | Redaction metrics are computed on the spans actually redacted (after overlap resolution) | They describe what downstream stages really see |
| Nov 20 | Redacted output no longer includes `original_text` | Retrieval indexes that file; raw text stays in `data/raw/` |
| Nov 20 | Removed Presidio's UK NHS recognizer | US-only domain; it fired on raw 10-digit phone numbers |
| Nov 20 | Identity questions included in the QA set | Turns the privacy cost into a measured result (abstain rate) |
| Nov 20 | Embedding: `bge-small-en-v1.5` (fallback `all-MiniLM-L6-v2`). NLI: `nli-deberta-v3-base` vs. `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, chosen on Day 4 | All local; already downloaded |

---

## Known issues / open items

- **ADDRESS detection (Day 7).** spaCy splits addresses into fragments, so 64% of address characters leak. The fix is a custom address recognizer.
- **NAME false positives.** 1,122 on the realistic set, mostly street names and towns inside addresses. This should largely go away with the address fix.
- **The audit log stores raw PII values.** That's needed for the audit trail, but it's a sensitive artifact. The report needs to say so.
- **Docs are stale.** The README and `report_draft.md` still show the pre-Day-1 numbers (0.838 / 0.869 / 0.858, 62.5%, 8/8) and the old data paths. Refresh on Day 7.

---

## Log

### Day 1: Nov 20
- Removed the clutter: empty `*mkdir` dirs, `-p`, nested `glassbox/glassbox`, `hello.txt`.
- Separated and regenerated the datasets deterministically. Fixed discharge notes whose admission date fell after discharge.
- Built the QA eval set v2. Every gold answer verified present in its gold note; 0 PII span errors across 18,879 entities.
- Redaction fixes:
  - Overlap resolver now ranks spans globally instead of letting the earliest span win.
  - NER spans cut at newlines.
  - Credential filter for "MD".
  - MRN pattern scores 1.0.
  - NHS recognizer removed.
- Realistic F1: 0.838 → 0.893.
- Added type-agnostic masking metrics.
- Test suite: 2 → 12 tests.
- Downloaded the embedding and NLI models.
- Commit `efe7049` on `pipeline-build`.
