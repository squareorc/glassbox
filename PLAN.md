# GlassBox: Project Plan (blueprint)

The plan from start to finish, kept current. Status: ✅ done · ⏳ in progress · ⬜ to do · ◻ optional.
Numbers and design rationale live in `PROGRESS.md`. This file says **what** gets built, in what order, and **when it counts as done**.

**Deadline:** 2026-11-28 · **Branch:** `pipeline-build` · **Last updated:** 2026-11-22 (Day 5)

---

## 0. Goal and deliverables

**Thesis.** An LLM question-answering system over sensitive records doesn't have to be trusted. Privacy and verifiability can both be enforced and *measured* by an auditable pipeline: redact before anything sees the data, retrieve only what's needed, cite every claim, verify every claim independently, and log every step.

**Deliverables**
1. A working end-to-end system (one command / API: question in, verified cited answer and audit trace out).
2. A Streamlit audit dashboard showing every stage for any question, plus an experiments overview.
3. Experiments with plots and 95% CIs: privacy, utility, faithfulness, robustness, privacy-utility trade-off.
4. A refreshed `README.md` and `report_draft.md`; `PROGRESS.md` as the report writer's source.
5. A demo (rehearsed) and a backup video.

**Scope limits (stated in the report):** synthetic data only (no real patient records); redaction rather than formal guarantees (no differential privacy); access control is index-level, not role-based; local 7B model.

---

## 1. Evaluation rules (apply to every stage)

- Design choices are made on the **dev** split (30%); results are reported on the **test** split (70%) with **bootstrap 95% CIs**.
- Robustness: **set A** = development, **set B** = held out, committed before fixes and evaluated once. A new round of fixes needs a **new** held-out set (set C).
- Headline answer metric is **grounded accuracy** (gold item in the answer *and* in a cited gold-note chunk), with drug-list F1; plain keyword accuracy is secondary.
- Report template-wording results and held-out-wording results side by side.
- Every finding is written to `PROGRESS.md` (what, why, alternatives, numbers, how measured, problems, fixes) and each stage ends with a commit.

---

## Phase 1: Foundations (Day 1, Nov 20) ✅

| | Task | Done when | Status |
|---|---|---|---|
| 1.1 | Clean the repo (empty dirs, stray files), `.gitignore` for regenerable outputs | Tree contains only source, data, results, docs | ✅ |
| 1.2 | Reproducible generators: pin "today" (2026-09-16), separate synthetic (250) and realistic (2,500) sets, fix discharge dates | Regeneration is byte-identical | ✅ |
| 1.3 | Record `facts` and `fields_in_text` per note (QA-only, never indexed) | Every QA answer traceable to a note field | ✅ |
| 1.4 | QA set v2: clinical-fact / identity / aggregate questions, keyword answers, dev/test split, validity checks | Every gold answer verified in its note; unique references | ✅ |
| 1.5 | Redaction fixes found at scale (overlap resolver, NER/ID ties, newline-crossing spans, DEA false match, NHS recognizer, "MD" as Maryland) | Regression test per fix | ✅ |
| 1.6 | Masking metrics (masked recall, char leakage) next to typed P/R/F1 | In results JSON | ✅ |
| 1.7 | `PROGRESS.md` created | Detailed per-stage record | ✅ |

## Phase 2: Retrieval (Day 2, Nov 20) ✅

| | Task | Done when | Status |
|---|---|---|---|
| 2.1 | Section chunker with a non-PII context line; whole-note ablation | All text covered; sections recognised per template | ✅ |
| 2.2 | Indexes: ChromaDB (bge-small, MPS) + in-house BM25; redacted and raw × section and note | 8 indexes with manifests | ✅ |
| 2.3 | Access control (raw index needs `baseline=True`), `MAX_K`, JSONL retrieval logs | Tested | ✅ |
| 2.4 | Evaluation: recall, evidence, context size, MRR; leak audit | 24 configurations × 2 datasets | ✅ |
| 2.5 | Query analysis (diagnosis/date filters, residual query, scope → auto-k) | Answerable evidence 0.048 → 0.840 | ✅ |
| 2.6 | Note-grouped selection; dev-split fusion tuning | Weights chosen on dev | ✅ |

## Phase 3a: Audit and fixes (Nov 20) ✅

| | Task | Done when | Status |
|---|---|---|---|
| 3a.1 | Independent audit of Days 1–2 (`PROGRESS.md` §12.1) | Findings ranked with evidence | ✅ |
| 3a.2 | Structural address recognizer, clinical-term guard, names in clinical context, numeric MRNs, case-sensitive patterns | Realistic F1 0.893 → 0.999; 0 over-redactions | ✅ |
| 3a.3 | Question redaction before retrieval (types only in logs) | Name never reaches logs (tested) | ✅ |
| 3a.4 | Query analysis: ranges, several diagnoses, invalid dates | Tested | ✅ |
| 3a.5 | Low-information demotion, note-ordered grouping, two-hop answer sections; fusion re-tuned (2:1) | Test evidence 0.840 → 0.949 | ✅ |
| 3a.6 | QA set ×3 (390 / 160 questions), bootstrap CIs, selection ablation, `requirements.txt` | CIs in every summary | ✅ |

## Phase 3: Cited generation + robustness (Nov 21–22) ✅

| | Task | Done when | Status |
|---|---|---|---|
| 3.1 | Generator: Qwen2.5-7B-Instruct 4-bit (MLX), greedy, JSON claims with citations, parser with one retry, citation validation, audit log | Parses 100% first try on test | ✅ |
| 3.2 | Abstain guard before the model (identifier in question / retrieval fallback) | 0/84 identity questions answered | ✅ |
| 3.3 | Prompt v1–v3 compared on dev; v2 chosen | Documented | ✅ |
| 3.4 | Evaluation: redacted / raw / guard-off; grounded accuracy, drug-list F1, `--rescore` | Test grounded accuracy 0.858 | ✅ |
| 3.5 | Robustness checks: dev/test gap, leakage, seed replication, guard perturbations, paraphrases, OOD notes, metric gaming | `PROGRESS.md` §8 | ✅ |
| 3.6 | Held-out set B frozen before fixes; general fixes on set A; set B evaluated once (before/after) | Retrieval 0.324 → 0.579, grounded 0.502; OOD redaction 0.672 → 0.819 | ✅ |

## Phase 4: Claim verification (Nov 22) ✅

| | Task | Done when | Status |
|---|---|---|---|
| 4.1 | NLI verifier: premise = cited chunk(s), hypothesis = claim; best of single and combined premises | Module + tests | ✅ |
| 4.2 | Labels SUPPORTED / CONTRADICTED / UNSUPPORTED / UNCERTAIN; answer verdict (VERIFIED / PARTLY / NOT / CONTRADICTED / ABSTAINED) | Tested | ✅ |
| 4.3 | Corrupted-claim set from the system's own claims (7 corruption types) | 1,289 items, frozen | ✅ |
| 4.4 | Model (DeBERTa base vs large) and UNCERTAIN margin chosen on dev; test report | 99.4% corruptions caught; 1.2% clean false alarms | ✅ |
| 4.5 | Verifier on real answers and guard-off answers | VERIFIED 0.906 vs NOT 0.492 grounded accuracy; misattribution finding | ✅ |

---

## Phase 5: Integration (Nov 22, two days early) ✅

### 5.1 Redaction threshold sweep ✅
- Thresholds 0.3–0.8; for each: redaction P/R/F1, masking, over-redactions, leak audit, retrieval evidence, grounded accuracy, identity answered. Script: `glassbox/src/experiments/threshold_sweep.py` → `results/experiments/threshold_sweep.json`.
- **Done when:** all 6 points saved; the chosen operating point (0.5) is justified from the curve in `PROGRESS.md`.
- Afterwards: confirm the sweep's t=0.50 point reproduces the main results (each threshold has its own `processed/t<thr>/` folder and index, so the 0.50 outputs are regenerated, not replaced by another threshold), and re-measure verifier speed on a free GPU.
- **Result:** plateau from 0.4 to 0.7; 0.3 over-redacts (4,219); 0.8 leaks 15.6% of phones. 0.5 reproduces the main results exactly; verifier 28 ms per claim on an idle GPU. See PROGRESS §5.1.

### 5.2 Orchestrator ✅ (`glassbox/run.py` + `glassbox/src/pipeline.py`)
- One function `answer(question, dataset="realistic", baseline=False) -> trace` and a CLI (`python -m glassbox.run "question"` and `--qa-id`, `--batch file`).
- Stages: question redaction → query analysis → retrieval → abstain guard → generation → verification. Models are loaded once and reused.
- **Trace JSON** (one file per question in `glassbox/logs/traces/`), schema:
  - `trace_id`, timestamp, config (index, thresholds, model names, prompt version, fusion weights)
  - `question`: redacted text + redaction record (types/offsets only, **never** the original)
  - `query_analysis`: filters, matched phrases, scope, residual query, ask/condition, fallback
  - `retrieval`: chunks (id, section, text, scores, ranks, `selected_by`, `low_information`), k, latency
  - `generation`: abstained + reason, answer, claims with citations, citation problems, raw output, retries, tokens, latency
  - `verification`: per-claim label, probabilities, premise chunks; answer verdict
  - `timings` per stage
- **Done when:** a trace for every QA question can be produced in batch; no original question text or PII value appears in any trace (tested); end-to-end test with real models (`GLASSBOX_SLOW=1`).
- **Result:** 390 questions × 3 configs; reproduces the evaluation exactly (test grounded 0.858, identity 0/84, all 390 answers identical); 3.2 s median per question. PROGRESS §14.1–14.2.

### 5.3 Housekeeping ✅
- Regenerate stale logs: re-run `retrieval.evaluate` (fixes the pre-fix "Tiffany Price" logs and the 4 identity questions whose retrieval changed).
- Make the per-question retrieval results match current code.
- **Done when:** a PII scan of all redacted-system logs finds 0 identity names/IDs.
- **Result:** `experiments/log_pii_scan.py`; found and scrubbed one pre-fix name in the v1/v2 prompt-dev files; now 0 in all privacy-side logs, traces and committed answers (raw baseline as positive control). Headline retrieval unchanged (0.949). PROGRESS §14.2–14.3.

### 5.4 Dashboard skeleton ✅ (`glassbox/src/dashboard/app.py`, Streamlit)
- **Ask page:** a question box (or pick a QA question) → runs the orchestrator → shows the trace.
- **Trace view (the audit), one panel per stage:**
  1. Question: original (shown only to the local user) vs redacted, identifiers found, guard decision.
  2. Query analysis: filters as chips, residual query, scope/k, ask/condition.
  3. Retrieval: chunk cards with section, scores, ranks, `selected_by`, low-information flag; context size.
  4. Answer: claims with citation links to the chunk cards; abstention message and reason.
  5. Verification: label badge per claim with probabilities; answer verdict.
- Reads traces only (no recomputation for stored traces).
- **Done when:** the app launches locally and renders any stored trace.
- **Result:** Ask + Stored traces pages, per-stage trace view; headless AppTest renders 6 trace kinds. Browser check on Day 6. PROGRESS §14.4.

### 5.5 Review fixes ✅
- Exact dense search instead of ChromaDB HNSW (approximate search missed true neighbours on unfiltered questions and differed between runs; filtered questions unaffected). Re-run: retrieval, generation, verification, paraphrase sets A/B, pipeline traces, threshold sweep.
- Model preloading, question-redaction threshold matched to the evaluation, verdicts matched by position, input validation, dashboard on localhost with telemetry off.
- **Done when:** all re-runs finished, every changed number updated in `PROGRESS.md` with before/after, one commit.
- **Result:** all re-runs done; GlassBox numbers unchanged (0/270 filtered questions changed), raw baseline identity recall 0.333 → 0.548, raw answered 42/84 (16 true values); two runs from scratch identical; 900/900 traces match the evaluation. PROGRESS §14.6.

## Phase 6: Experiments, figures, dashboard (Nov 25) ⏳

### 6.1 Experiment set ✅ (all on realistic test unless noted; CIs everywhere)
| # | Experiment | Source | Figure / table |
|---|---|---|---|
| E1 | Redaction quality per type; masking; OOD sets A/B before/after; name-origin bias | `results/redaction`, `results/robustness` | Bar chart per type; OOD before/after table; bias bar chart |
| E2 | Leakage: PII visible in the index, redacted vs raw | retrieval leak audit | Table |
| E3 | Utility: redacted vs raw (retrieval evidence, grounded accuracy) | retrieval + generation results | Paired bars with CIs |
| E4 | Privacy-utility trade-off: threshold sweep | `threshold_sweep.json` | Line plot: leakage & over-redaction vs grounded accuracy across thresholds |
| E5 | Identity protection: guard on / off / raw baseline | generation results | Table + bar (answered, true value leaked) |
| E6 | Retrieval design ablation (Day 2 → final selection) | `*_selection_ablation.json` | Step chart |
| E7 | Faithfulness: verifier on corrupted claims, per type; on real answers | `verifier_eval.json` | Detection-rate bars; verdict vs grounded accuracy |
| E8 | Robustness: template vs set A vs set B wording | `results/robustness` | Grouped bars |
| E9 | Synthetic (250-note) replication of headline numbers | run generation + verification on synthetic | Table |
- Script: `glassbox/src/experiments/figures.py` → `glassbox/results/figures/*.png` and a `summary.json` of every headline number.
- **Done when:** every figure regenerates from committed results with one command; numbers match `PROGRESS.md`.
- **Result (Day 5 evening):** E1–E8 as 8 figures + `tables.md` + `summary.json`; a test regenerates them and checks headline numbers. E9 run: synthetic GlassBox 0.942 vs raw 0.916, identity 0 vs 25/35. PROGRESS §15, §15.1.

### 6.3 Demo tooling ⏳ (built and checked; owner's browser dry run left)
- **Demo page** in the dashboard (`glassbox/src/dashboard/demo_script.py`): 4 one-click live questions (fact, aggregate, named person GlassBox vs raw side by side, no-matching-record), "try to fool the verifier" (edit a claim, re-verify live), free question box; talking points; explicit "Load models" step. ✅ built, headless-tested
- **`python -m glassbox.verify`**: re-runs a fixed stratified sample of 10 stored evaluation questions live and checks chunks / answer / citations / refusal / score are identical; `--tests` adds the test suite. ✅ real-model run: 10/10 identical in 34 s
- **`DEMO.md`**: pre-meeting checklist, 15-minute script, likely questions with answers, failure fallbacks. ✅ numbers re-checked after the re-runs
- **Done when:** a full dry run of DEMO.md works end to end in the browser (owner), `glassbox.verify` prints 10/10.

### 6.2 Dashboard completion ✅
- **Experiments page:** headline numbers and figures from `results/figures`.
- **Batch view:** a table of stored traces (filter by verdict / abstention / question type).
- Privacy check: the dashboard never displays original PII from notes (only redacted text and placeholders).
- **Done when:** the demo flow (ask → trace → experiments) works end to end without errors.
- **Result:** Experiments page (metrics + figures + tables); stored traces open on the test split; live traces kept in `logs/traces/live/`; headless real-model dry run of every button with no errors. PROGRESS §14.4.

## Phase 7: Documentation (Nov 26) ⬜

| | Task | Done when |
|---|---|---|
| 7.1 ✅ | `README.md` rewrite: purpose, architecture diagram, setup, quick start (`run.py`, dashboard), results summary with figures, limitations, reproduction commands | All numbers from `summary.json`; old numbers removed |
| 7.2 ✅ | `report_draft.md` refresh: methods per stage, experiments E1–E9, limitations (held-out gap, name bias, verification vs misattribution, synthetic data) | Consistent with `PROGRESS.md` |
| 7.3 | `PROGRESS.md` final pass: status, final numbers, daily log | Groupmate can write the report from it |
| 7.4 ✅ | Remove or archive stale docs (`REALISTIC_DATASET_EVALUATION.md`, `demo_dashboard.html`) after checking with the owner | Owner approved |

## Phase 8: Polish and buffer (Nov 27) ⬜

| | Task | Status |
|---|---|---|
| 8.1 | Full clean-run check: fresh venv from `requirements.txt`, regenerate data → redaction → indexes → evaluations → figures; tests pass | ✅ Nov 23 (PROGRESS §15.2) |
| 8.2 | Demo rehearsal following `DEMO.md` (Demo page + `glassbox.verify`); backup screen recording (owner) | ⬜ |
| 8.3 | Merge `pipeline-build` into `main` (with the owner's approval) | ✅ Nov 22 (fast-forward) |
| 8.4 | ◻ Generation: prompt rule to cite every section a claim relies on (fixes incomplete citations, §7.5), tuned on dev | ◻ |
| 8.5 | ◻ Robustness round 2: dash/day-first date formats, day/month ambiguity (try both readings when one matches no note), "w/" as condition marker, Title-Case question guard; evaluated on a **new** held-out set C | ◻ |
| 8.6 | ◻ Redaction: PO boxes, multi-line addresses; name-origin bias mitigation | ◻ |

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Long GPU runs collide (generation, NLI, sweep) | Run in the background one at a time; measure latency only on a free GPU |
| Overfitting to our own data | Dev/test discipline, held-out sets committed before fixes, report template vs held-out side by side |
| Dashboard scope creep | Read-only over traces; no new analysis in the UI |
| Stale numbers in docs | All docs read from `results/` / `summary.json`; final consistency pass in Phase 7 |
| Machine/time limits for re-runs | `--rescore` and saved logs avoid re-generation when only metrics change |
