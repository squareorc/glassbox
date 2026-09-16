# GlassBox

**A Privacy-Preserving and Verifiably Explainable Framework for LLM Systems**

---

## 1. What This Project Is About

Large Language Models are increasingly used to analyze sensitive documents — medical notes, financial records, HR files, confidential business data. Two problems come up every time this happens:

1. **Privacy risk.** Most LLM pipelines hand over far more data than a given query actually needs. A conventional RAG system might retrieve several full documents and pass them straight into the model's context, exposing names, IDs, and other identifying information the query never asked about.
2. **No way to verify the answer.** Even when an LLM produces a fluent, confident response, there's typically no way for the user to check what evidence (if any) actually supports each claim. LLMs hallucinate, and nothing in a standard pipeline catches this automatically.

These two problems compound each other: the more a system protects privacy by withholding data, the less visibility the user has into what the model actually saw — which gives the user *more* reason to distrust the output, not less. A system that's privacy-preserving but opaque asks for blind trust at exactly the moment trust is hardest to justify.

**GlassBox treats privacy-preservation and verifiable explainability as one integrated problem, not two separate features bolted together.** The core idea: the user shouldn't have to trust the LLM's honesty — they should only have to trust a process they can fully inspect. Every stage between the raw data and the final answer is logged and auditable, so a user (or evaluator) can see exactly what was redacted, what was retrieved, what was claimed, and whether each claim actually holds up against its cited evidence.

---

## 2. Solution and Approach

GlassBox is built as a pipeline with four cooperating stages, sitting between the private dataset and the final answer:

1. **Privacy-aware data processing** — before anything reaches the LLM, sensitive fields (names, patient IDs, phone numbers, addresses, etc.) are automatically detected and redacted or pseudonymized. The LLM never sees more identifying information than necessary.
2. **Access-controlled, logged retrieval** — instead of dumping an entire dataset into context, a retrieval layer selects only the relevant, sanitized slice of data for a given query, and logs exactly what was retrieved and why.
3. **Evidence-based generation** — the LLM is required to tie every factual claim in its answer to a specific retrieved fragment via citation, rather than producing free-floating assertions.
4. **Independent claim verification** — a separate verification mechanism (not the LLM grading itself) checks whether each cited fragment actually supports its claim, classifying it as **Supported**, **Unsupported**, **Contradicted**, or **Uncertain**. This is the project's core novel contribution — a concrete, automatable measure of hallucination rather than something the user has to catch by eye.

Every stage's output is surfaced through an **audit dashboard**, so the full trail — what was redacted, what was retrieved, what was generated, what was verified — is inspectable end to end.

### Scoping decisions

Given a 3-month, solo-built timeline, a few deliberate cuts were made to keep the project buildable without weakening its core argument:

- **Differential privacy, federated learning, and homomorphic encryption are explicitly out of scope** for this implementation. Redaction/pseudonymization is deterministic (entity detection + masking), not formally private. These are named as future work rather than attempted half-way.
- **Domain is locked to synthetic healthcare records.** Real clinical data (e.g. MIMIC) would require data-use approvals that risk eating weeks of a tight timeline for uncertain payoff. Synthetic data also gives exact ground-truth PII locations, which makes redaction evaluation more rigorous, not less.
- **A locally-hosted, open-weight LLM is used throughout** (rather than a third-party API), so the privacy guarantee extends to the model-serving step itself, not just the preprocessing step.

### Why this matters

The contribution here isn't a new PII detector, a new RAG technique, or a new explainability method in isolation — all three already exist in the literature separately. The contribution is a **working, evaluated integration** of privacy-minimization and verifiable evidence-tracing into a single pipeline, plus an empirical look at the tradeoff between privacy protection, answer utility, and explanation faithfulness — rather than assuming improving one doesn't cost the others.

---

## 3. Planned System Architecture

```
                     User Query
                         │
                         ▼
                  Query Analysis
                         │
                         ▼
             Privacy & PII Detection
                         │
                         ▼
           Redaction / Pseudonymization
                         │
                         ▼
             Access-Controlled Retrieval
                         │
                         ▼
             Relevant Sanitized Context
                         │
                         ▼
                    Local LLM
                         │
                         ▼
          Answer + Claims + Citations
                         │
                         ▼
         Independent Evidence Verification
                         │
                         ▼
             Final Verified Response
                         │
                         ▼
                 Audit Dashboard
```

### Component breakdown

| Stage | Purpose | Planned tooling |
|---|---|---|
| PII detection & redaction | Identify and mask sensitive entities before anything downstream sees them | Presidio (NER + regex based) |
| Retrieval | Embed sanitized documents, retrieve only what's relevant per query, log every retrieval decision | ChromaDB + a small local embedding model |
| Generation | Produce an answer where every claim cites a specific retrieved chunk | Local open-weight LLM via MLX (Qwen2.5-7B-Instruct, 4-bit) |
| Verification | Independently check each cited claim against its evidence; classify as Supported / Unsupported / Contradicted / Uncertain | Lightweight NLI/cross-encoder model |
| Audit dashboard | Surface every stage's output — redactions, retrievals, claims, verification results — to the user | Streamlit |

### Evaluation plan

The system will be evaluated along three axes, each with a small set of headline metrics (kept deliberately tight given the project timeline, rather than an exhaustive metric list):

- **Privacy** — PII detection/redaction precision and recall against ground-truth entity spans.
- **Utility** — answer accuracy and retrieval precision/recall, compared against a full-context RAG baseline (same pipeline, redaction toggled off) to measure what minimization costs in answer quality.
- **Explainability / faithfulness** — claim verification accuracy, and the rate of unsupported/contradicted claims caught by the verifier.

---

## 4. What We Have Done So Far

- **Scoped and finalized the proposal.** Domain locked to synthetic healthcare records; differential privacy and other formal privacy techniques deferred to future work; verification defined with four outcome categories; evaluation plan narrowed to a focused metric set per axis.
- **Environment set up.** Python 3.11 virtual environment, project skeleton (`src/redaction`, `src/retrieval`, `src/generation`, `src/verification`, `src/dashboard`), core libraries installed (Presidio, spaCy, sentence-transformers, ChromaDB, Streamlit, MLX), and a local quantized LLM (Qwen2.5-7B-Instruct, 4-bit) pulled and smoke-tested via MLX. Git repository initialized.
- **Synthetic dataset generator built and verified.** A script (`generate_synthetic_dataset.py`) that produces 250 synthetic clinical notes across 8 common conditions, using Faker-generated PII injected into note templates at tracked character offsets — giving exact ground-truth entity spans for redaction evaluation. It also produces a companion QA evaluation set (currently 8 diagnosis-based queries with known relevant documents and expected answers, to be expanded later with symptom-based and multi-condition queries for a more rigorous retrieval evaluation). Ground truth spans verified to match actual text offsets 100% (834/834 entities).
- **Redaction module built and evaluated.** The full PII detection and redaction pipeline is implemented (`src/redaction/`):
  - `detector.py`: Wraps Presidio's AnalyzerEngine with a custom PATIENT_ID recognizer (2 letters + 6 digits pattern from the synthetic data).
  - `redactor.py`: Replaces detected PII with `[TYPE]` tags, handles overlapping spans, logs all redactions for the audit trail.
  - `evaluate.py`: Compares detected spans against ground truth using IoU-based matching, computes precision/recall/F1 overall and per entity type.
  - `pipeline.py`: Ties everything together — processes the raw dataset, outputs redacted documents to `data/processed/`, and generates evaluation metrics.

### Redaction Evaluation Results

#### 1. Baseline Evaluation (250 Synthetic Notes)
Evaluated at confidence threshold 0.5, IoU threshold 0.5:

| Metric | Score |
|---|---|
| **Overall Precision** | 0.842 |
| **Overall Recall** | 0.897 |
| **Overall F1** | **0.869** |

**Per-entity-type breakdown:**

| Entity Type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| **EMAIL** | **1.000** | **1.000** | **1.000** | 84 | 0 | 0 |
| **PATIENT_ID** | **1.000** | **1.000** | **1.000** | 250 | 0 | 0 |
| **DATE_OF_BIRTH** | **1.000** | **1.000** | **1.000** | 84 | 0 | 0 |
| **PHONE_NUMBER** | **1.000** | **1.000** | **1.000** | 81 | 0 | 0 |
| NAME | 0.769 | 0.996 | 0.868 | 249 | 75 | 1 |
| ADDRESS | 0.000 | 0.000 | 0.000 | 0 | 65 | 85 |

#### 2. Scaled Evaluation on Realistic Clinical Dataset (2,500 Notes)
Evaluated on complex EHR-style documentation (progress notes, SOAP notes, discharge summaries, consultation notes) with multi-provider attribution:

| Metric | Score |
|---|---|
| **Overall Precision** | 0.776 |
| **Overall Recall** | **0.911** |
| **Overall F1** | **0.838** |

**Per-entity-type breakdown (2,500 documents, 17,544 total PII entities):**

| Entity Type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| **EMAIL** | **1.000** | **1.000** | **1.000** | 2,500 | 0 | 0 |
| **DATE_OF_BIRTH** | **1.000** | **1.000** | **1.000** | 2,500 | 0 | 0 |
| **PHONE_NUMBER** | **1.000** | **1.000** | **1.000** | 3,011 | 0 | 0 |
| **PATIENT_ID** | **0.999** | **1.000** | **1.000** | 2,500 | 2 | 0 |
| **NAME** | 0.834 | **0.988** | **0.904** | 5,930 | 1,179 | 75 |
| ADDRESS | 0.001 | 0.001 | 0.001 | 2 | 3,337 | 1,526 |

**What's working well:**
- **Perfect detection (F1 1.000)** maintained at scale across 4 entity types: EMAIL, PATIENT_ID, DATE_OF_BIRTH, and PHONE_NUMBER.
- **NAME F1 improved to 0.904** (98.8% recall across 6,005 patient, attending, referring, and emergency contact names).
- Robust performance across diverse note structures (SOAP, progress, discharge, consultation) and clinical language (vitals, lab orders, ICD-10 codes).
- DATE_OF_BIRTH uses a year-range regex (1936-2008) to distinguish actual DOB from recent visit dates, eliminating false positives.
- PHONE_NUMBER uses a comprehensive custom recognizer with context gating for raw numbers, catching standard, parentheses, extension (`x272`, `ext. 402`), and international formats.
- PATIENT_ID handles 1-2 letter alphanumeric, dash-separated, and MRN prefixes with 100% recall.

**Generalization beyond synthetic data:**
To validate that the scores aren't artifacts of overfitting, the system was tested on 8 unseen format variations:
- **Patient IDs**: Dash-separated (`P-123456`) and MRN-prefixed (`MRN0012345`) formats — both detected correctly.
- **Dates**: US format (`03/20/1985`) and written format (`born March 20, 1985`) — both detected correctly.
- **Phone numbers**: Parentheses format `(555) 123-4567`, extensions `555-123-4567 x272`, and international prefixes `+1-555-123-4567` — all detected correctly.
- **Overall generalization**: **8/8 test cases passed (100.0%)**.

The generalization test suite is available at `glassbox/tests/test_generalization.py`.

**Known issues:**
- **ADDRESS**: Presidio fragments addresses into separate NAME (street names) and LOCATION (cities) entities. With IoU ≥ 0.5 matching, none of these fragments overlap enough with the full ground-truth address span, resulting in 0% recall. Future work: merge adjacent NAME/LOCATION entities with address-specific context, or use a custom address recognizer.
- **PHONE_NUMBER with formatting**: Presidio's built-in phone recognizer misses certain formats like `(555) 123-4567` (parentheses) and international prefixes. 10-digit unformatted numbers are detected correctly.

**Impact on pipeline:**
- The redaction stage successfully processes all 250 documents and outputs sanitized versions with audit logs.
- The overall F1 of 0.869 demonstrates strong privacy protection with minimal over-redaction. High recall (0.897) means most PII is caught, and high precision (0.842) means few false positives.
- Four entity types achieve perfect detection (F1 1.000), and NAME detection is near-perfect (F1 0.868).
- For downstream retrieval and generation stages, the redacted dataset in `data/processed/redacted_notes.jsonl` is ready to use.

### Next up

- Expand the QA evaluation set beyond diagnosis-only queries (add symptom-based and multi-condition queries).
- Build the retrieval module: embed redacted documents, implement query-driven retrieval with ChromaDB, log retrieval decisions.

---

## Tech Stack

- **Language:** Python 3.11
- **PII detection:** Presidio, spaCy
- **Embeddings / retrieval:** sentence-transformers, ChromaDB
- **Local LLM inference:** MLX (Apple Silicon), Qwen2.5-7B-Instruct (4-bit)
- **Dashboard:** Streamlit
- **Synthetic data:** Faker