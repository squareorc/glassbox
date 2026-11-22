# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GlassBox: a privacy-preserving, verifiably explainable question-answering pipeline over synthetic clinical notes (final-year project). Everything runs locally on an Apple-silicon Mac (MLX for the LLM, MPS for embeddings/NLI).

```
question → question redaction → query analysis → access-controlled retrieval
         → cited generation (local Qwen, abstain guard) → NLI claim verification → audit logs
```

**`PLAN.md` is the blueprint**: every phase from start to finish, with tasks, "done when" criteria and status. Read it first to know what to build next, and update a task's status when you finish it.
**`PROGRESS.md` is the record**: design decisions, alternatives, measured numbers and problems for everything built. `README.md` and `report_draft.md` are stale until their planned refresh (PLAN.md Phase 7); don't trust their numbers or paths.

## Working agreement (from the project owner)

- Before editing, explain the plan in bullets and wait for approval when the user asks for review.
- Work on branch `pipeline-build`; commit at the end of each stage; **push only when asked**.
- Keep `PROGRESS.md` current and detailed after every stage (what / why / alternatives / numbers / how measured / problems / fixes). A teammate writes the final report from it on GitHub.
- Results must be "feasible, not flashy": tune only on the **dev** split, report on **test** with bootstrap 95% CIs, and state limitations.

## Commands

Run everything from the repo root with the venv active (`source .venv/bin/activate`), as modules (`python -m glassbox.src...`). Ad-hoc scripts that import the package need `PYTHONPATH=.`.

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_lg   # setup (Python 3.11)

python -m pytest glassbox/tests                                   # fast suite (~30 s)
python -m pytest glassbox/tests/test_retrieval.py::test_auto_k_follows_query_scope   # single test
GLASSBOX_SLOW=1 python -m pytest glassbox/tests                   # + tests that load the 7B LLM / NLI model

# Pipeline stages (each writes its outputs; later stages read earlier outputs)
python -m glassbox.src.generate_realistic_dataset --size large    # 2,500 notes + QA set (deterministic)
python -m glassbox.src.redaction.pipeline --dataset realistic [--threshold 0.5]
python -m glassbox.src.retrieval.index --dataset realistic --all  # rebuild indexes after any redaction change
python -m glassbox.src.retrieval.evaluate --dataset realistic [--tune | --ablate]
python -m glassbox.src.generation.evaluate --dataset realistic [--split dev] [--rescore]
python -m glassbox.src.verification.corruptions && python -m glassbox.src.verification.evaluate
python -m glassbox.src.experiments.threshold_sweep               # ~2.5 h
python -m glassbox.run "question" | --qa-id ID | --qa-set [--config raw|redacted_noguard]   # end-to-end, writes traces
python -m glassbox.src.experiments.log_pii_scan                  # PII scan of all logs/traces (~4 s)
streamlit run glassbox/src/dashboard/app.py                      # audit dashboard (reads traces)
python glassbox/src/robustness/evaluate_paraphrases.py --set b [--generate]
python -m glassbox.src.robustness.ood_notes --set b
```

Generation is ~4 s/question (a full 390-question run is ~20–30 min per configuration), so run long jobs in the background. Don't run two GPU-heavy jobs at once if latency numbers matter. `--rescore` re-scores saved generation logs without calling the model.

## Architecture (the parts that span files)

- **Data** (`glassbox/data/raw/{synthetic,realistic}/`): generator output `synthetic_notes.jsonl` (with ground-truth PII `entities`, and `facts`, which contain PII and are used **only** to build the QA set) plus `qa_eval_set.jsonl` (`glassbox/src/qa/build_qa_set.py`). Generators pin "today" to 2026-09-16, so regeneration is byte-identical. QA types: `clinical_fact`, `aggregate` (multi-note), `identity` (must be unanswerable under redaction). Splits: 30% dev / 70% test.
- **Redaction** (`src/redaction/`): Presidio + spaCy, plus custom pattern recognizers in `detector.py` (patient IDs, DOB year range 1936–2008, phones with context gating, structural US addresses, names before credentials / after labels). `clinical_terms.py` guards NER against redacting drugs, vitals, labels and disease words. Overlaps are resolved globally by score, then type priority (`redactor.py`). `query.py` redacts the *user question* too (extra question-only name rules); its `identifiers` field drives the abstain guard. Outputs go to `data/processed/<ds>/t<thr>/` (gitignored). The audit log holds raw PII by design.
  - Presidio compiles patterns case-insensitively by default; capitalisation-based patterns pass `global_regex_flags=CASE_SENSITIVE`. Presidio uses the `regex` module, so `\p{Lu}` works.
- **Retrieval** (`src/retrieval/`): section chunking with a non-PII context line (`chunker.py`), ChromaDB dense index (bge-small) + in-house BM25, weighted RRF (2:1 dense:BM25). `query_analysis.py` turns diagnosis/date phrases into metadata filters, a residual ranking query, a scope (auto k = 5 or 10) and an ask/condition split. `retriever.py` adds low-information demotion (placeholder-only chunks), note-grouped and two-hop selection, access control (the raw index needs `baseline=True`, else `AccessDenied`), and JSONL logs. Index name pattern: `redacted-t0.50-section`, `raw-note`, …
- **Generation** (`src/generation/`): Qwen2.5-7B-Instruct 4-bit via mlx-lm, greedy. `abstain_reason` refuses *before* the model call when the question names a person on the redacted index, or when retrieval fell back to unfiltered search. Prompt v2 (`prompt.py`) numbers sources `[Cn] (Note k)`; the reply is JSON `{answer, claims[{text, citations}], insufficient}`, parsed and validated in `parse.py`. Headline metric: **grounded accuracy** (gold item in the answer *and* in a cited gold-note chunk), not plain keyword accuracy.
- **Verification** (`src/verification/`): DeBERTa-v3-large NLI checks each claim against its cited chunk(s). Labels SUPPORTED / CONTRADICTED / UNSUPPORTED / UNCERTAIN (margin chosen on dev, currently 0). `corruptions.py` builds a frozen corrupted-claim test set from saved generation answers.
- **Robustness** (`src/robustness/`): paraphrase and out-of-distribution-note sets. **Set A** = development; **set B** = held out, committed before fixes (commits `2f0bb6b`, `a0047eb`). Never tune on set B; a new fix round needs a new held-out set.
- **Pipeline** (`src/pipeline.py`, `run.py`): `GlassBox(dataset, config).answer(q)` chains the stages with no logic of its own (a QA-set batch must reproduce `generation.evaluate` exactly) and writes one trace JSON per question to `logs/traces/<ds>/<config>/`. Traces on redacted configs never contain the original question. The dashboard (`src/dashboard/`) only reads traces.
- **Results** (`glassbox/results/`, committed) vs regenerable outputs (`data/processed`, `data/indexes`, `logs`, gitignored). Shared helpers: `retrieval.evaluate.bootstrap_ci`, `qa.build_qa_set.answer_item_found` (answer matching used by retrieval, generation and verification).
