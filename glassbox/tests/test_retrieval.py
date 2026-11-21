"""
Tests for the retrieval module: chunking, BM25, access control, end-to-end.

End-to-end tests use the built indexes and skip if they haven't been built:
    python -m glassbox.src.retrieval.index --dataset realistic --all
"""

import json
import math
from pathlib import Path

import pytest

from glassbox.src.retrieval.bm25 import BM25, tokenize
from glassbox.src.retrieval.chunker import chunk_document
from glassbox.src.retrieval.index import index_dir
from glassbox.src.retrieval.retriever import AccessDenied, Retriever

REDACTED_NOTES = Path("glassbox/data/processed/realistic/t0.50/redacted_notes.jsonl")


def _load_redacted_notes():
    if not REDACTED_NOTES.exists():
        pytest.skip("run the redaction pipeline first")
    with REDACTED_NOTES.open() as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- chunking

def test_chunks_cover_every_character():
    """No note text is lost: every non-whitespace character lies in some chunk."""
    for doc in _load_redacted_notes()[:300]:
        covered = [False] * len(doc["text"])
        for c in chunk_document(doc):
            assert doc["text"][c["start"]:c["end"]] == c["body"]
            for i in range(c["start"], c["end"]):
                covered[i] = True
        assert all(covered[i] or ch.isspace() for i, ch in enumerate(doc["text"]))


@pytest.mark.parametrize("note_type,expected", [
    ("progress", {"patient_info", "contact"}),
    ("discharge", {"patient_info", "principal_diagnosis", "hospital_course", "discharge_medications",
                   "discharge_instructions", "contact"}),
    ("consultation", {"patient_info", "reason_for_consultation", "history", "examination",
                      "impression", "recommendations", "contact"}),
])
def test_every_template_section_recognised(note_type, expected):
    docs = [d for d in _load_redacted_notes() if d["note_type"] == note_type]
    for doc in docs[:50]:
        sections = {c["section"] for c in chunk_document(doc)}
        assert expected <= sections
        assert "note" not in sections  # nothing left unnamed


def test_chunk_context_line_has_no_pii_fields():
    doc = {"doc_id": "doc_x", "text": "PLAN: Continue Amlodipine 5mg PO daily.",
           "diagnosis": "Essential hypertension", "note_type": "progress", "visit_date": "2024-11-10"}
    (chunk,) = chunk_document(doc)
    assert chunk["text"].split("\n")[0] == "[Progress note | 2024-11-10 | Essential hypertension]"
    assert chunk["chunk_id"] == "doc_x#plan"


def test_note_strategy_gives_one_chunk_per_note():
    doc = _load_redacted_notes()[0]
    assert len(chunk_document(doc, "note")) == 1


# ---------------------------------------------------------------- BM25

def test_bm25_matches_hand_computed_score():
    bm25 = BM25(["xray yoga", "xray zinc zinc"], k1=1.5, b=0.75)
    # 'zinc' appears in doc 1 only: df=1, N=2 -> idf = ln(1 + 1.5/1.5) = ln 2
    # tf=2, |d|=3, avgdl=2.5 -> norm = 1.5 * (0.25 + 0.75 * 3/2.5) = 1.725
    expected = math.log(2) * 2 * 2.5 / (2 + 1.725)
    scores = bm25.scores("zinc")
    assert set(scores) == {1}
    assert scores[1] == pytest.approx(expected)


def test_tokenizer_keeps_dates_ids_and_vitals_whole():
    toks = tokenize("Seen 2024-11-10, MRN0033092, id f-819600, BP 157/94, temp 98.5")
    for t in ("2024-11-10", "mrn0033092", "f-819600", "157/94", "98.5"):
        assert t in toks


# ---------------------------------------------------------------- access control

def test_raw_index_requires_baseline_flag(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "raw-section", "source": "raw"}))
    with pytest.raises(AccessDenied):
        Retriever(tmp_path)


# ---------------------------------------------------------------- end to end

def _index(variant):
    d = index_dir("realistic", variant, "section")
    if not (d / "manifest.json").exists():
        pytest.skip("build the realistic indexes first")
    return d


def test_redacted_retrieval_finds_clinical_fact_note_and_logs(tmp_path):
    log = tmp_path / "retrieval.jsonl"
    retriever = Retriever(_index("redacted"), log_path=log)
    with open("glassbox/data/raw/realistic/qa_eval_set.jsonl") as f:
        q = next(r for r in map(json.loads, f) if r["type"] == "clinical_fact")
    results = retriever.retrieve(q["question"], k=5, method="bm25", qid=q["qid"])
    assert q["relevant_doc_ids"][0] in {r["doc_id"] for r in results}

    record = json.loads(log.read_text().splitlines()[0])
    assert record["qid"] == q["qid"] and record["source"] == "redacted"
    assert len(record["results"]) == 5 and "text" not in record["results"][0]


def test_k_is_capped():
    retriever = Retriever(_index("redacted"))
    with pytest.raises(ValueError):
        retriever.retrieve("asthma", k=50)


# ---------------------------------------------------------------- query analysis

from glassbox.src.retrieval.query_analysis import analyze_query, chroma_where, matches  # noqa: E402

DX = ["Essential hypertension", "Major depressive disorder", "Type 2 diabetes mellitus",
      "Chronic kidney disease stage 3", "Asthma"]


@pytest.mark.parametrize("question,filters,scope", [
    ("What medication was prescribed for the Essential hypertension patient whose note is dated 2024-11-10?",
     {"diagnosis": "Essential hypertension", "date_from": 20241110, "date_to": 20241110}, "single_note"),
    ("Which medications were prescribed to Major depressive disorder patients who presented with "
     "depressed mood in December 2025?",
     {"diagnosis": "Major depressive disorder", "date_from": 20251201, "date_to": 20251231}, "multi_note"),
    ("Which medications were prescribed to Asthma patients who presented with wheezing in 2024?",
     {"diagnosis": "Asthma", "date_from": 20240101, "date_to": 20241231}, "multi_note"),
    ("any diabetes patients seen on 03/15/2025?",
     {"diagnosis": "Type 2 diabetes mellitus", "date_from": 20250315, "date_to": 20250315}, "single_note"),
    ("ckd notes from Sept 2025",
     {"diagnosis": "Chronic kidney disease stage 3", "date_from": 20250901, "date_to": 20250930}, "multi_note"),
])
def test_query_analysis_extracts_filters(question, filters, scope):
    a = analyze_query(question, DX)
    assert a["filters"] == filters
    assert a["scope"] == scope


@pytest.mark.parametrize("question", [
    "What medication is John Wilkerson prescribed?",
    "What is the diagnosis recorded for patient MRN0033092?",
])
def test_identity_questions_get_no_filters(question):
    a = analyze_query(question, DX)
    assert a["filters"] == {} and a["residual_query"] == question


@pytest.mark.parametrize("question,filters", [
    ("Asthma patients in May, 2025", {"diagnosis": "Asthma", "date_from": 20250501, "date_to": 20250531}),
    ("Asthma notes between March 2025 and June 2025", {"diagnosis": "Asthma", "date_from": 20250301, "date_to": 20250630}),
    ("Asthma notes from March to June 2025", {"diagnosis": "Asthma", "date_from": 20250301, "date_to": 20250630}),
    ("asthma notes from 2024 to 2025", {"diagnosis": "Asthma", "date_from": 20240101, "date_to": 20251231}),
    ("asthma notes 2024-01-05 to 2024-02-10", {"diagnosis": "Asthma", "date_from": 20240105, "date_to": 20240210}),
    ("patients with hypertension or diabetes in 2025",
     {"diagnosis": ["Essential hypertension", "Type 2 diabetes mellitus"], "date_from": 20250101, "date_to": 20251231}),
])
def test_query_analysis_ranges_and_multiple_diagnoses(question, filters):
    a = analyze_query(question, DX)
    assert a["filters"] == filters and a["scope"] == "multi_note"


@pytest.mark.parametrize("question", ["notes from 2025-13-01 for asthma", "BP on 2/30/2025?"])
def test_invalid_date_gives_no_date_filter(question):
    """A malformed date is reported, never silently widened (2025-13-01 used to become all of 2025)."""
    a = analyze_query(question, DX)
    assert "date_from" not in a["filters"]
    assert [e["type"] for e in a["extracted"]].count("invalid_date") == 1


def test_multiple_diagnoses_filter():
    f = {"diagnosis": ["Asthma", "COPD"]}
    assert matches({"diagnosis": "COPD", "visit_date": "2024-06-01"}, f)
    assert not matches({"diagnosis": "Osteoarthritis", "visit_date": "2024-06-01"}, f)
    assert chroma_where(f) == {"diagnosis": {"$in": ["Asthma", "COPD"]}}


def test_residual_query_drops_constraint_phrases():
    a = analyze_query("What heart rate was recorded for the Asthma patient whose note is dated 2024-10-07?", DX)
    assert a["residual_query"] == "What heart rate was recorded for the patient ?"


def test_filter_matching_and_chroma_where_agree():
    f = {"diagnosis": "Asthma", "date_from": 20240101, "date_to": 20241231}
    assert matches({"diagnosis": "Asthma", "visit_date": "2024-06-01"}, f)
    assert not matches({"diagnosis": "Asthma", "visit_date": "2025-01-01"}, f)
    assert not matches({"diagnosis": "COPD", "visit_date": "2024-06-01"}, f)
    assert chroma_where(f) == {"$and": [{"diagnosis": {"$eq": "Asthma"}},
                                        {"visit_date_int": {"$gte": 20240101}},
                                        {"visit_date_int": {"$lte": 20241231}}]}
    assert chroma_where({}) is None


# ---------------------------------------------------------------- selection

def test_note_grouping_caps_chunks_per_note_then_fills():
    r = Retriever.__new__(Retriever)  # grouping only needs self.chunks
    r.chunks = [{"doc_id": d} for d in ["a", "a", "a", "b", "b", "c"]]
    ranked = [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
    order = [i for i, _ in r._group_by_note(ranked, per_note=2)]
    assert order == [0, 1, 3, 4, 5, 2]          # a,a,b,b,c first; third 'a' chunk last
    assert [i for i, _ in r._group_by_note(ranked, None)] == [0, 1, 2, 3, 4, 5]


def test_filtered_bm25_keeps_chunks_without_query_words():
    """After filtering to one note, 'heart rate' must still reach the 'HR 94' chunk."""
    retriever = Retriever(_index("redacted"))
    with open("glassbox/data/raw/realistic/qa_eval_set.jsonl") as f:
        q = next(r for r in map(json.loads, f) if r["type"] == "clinical_fact" and r["answer_field"] == "hr")
    results = retriever.retrieve(q["question"], k=10, method="bm25")
    assert {r["doc_id"] for r in results} == set(q["relevant_doc_ids"])
    assert any(q["answer"][0] in r["text"] for r in results)


def test_auto_k_follows_query_scope():
    retriever = Retriever(_index("redacted"))
    single = retriever.retrieve("What medication was prescribed for the Asthma patient whose note is dated 2024-10-07?")
    multi = retriever.retrieve("Which medications were prescribed to Asthma patients who presented with wheezing in October 2024?")
    assert len(single) == 5 and len(multi) == 10


# ---------------------------------------------------------------- Day 3 audit fixes

from glassbox.src.redaction.query import redact_query  # noqa: E402
from glassbox.src.retrieval.query_analysis import split_condition  # noqa: E402
from glassbox.src.retrieval.retriever import is_low_information  # noqa: E402


def test_question_is_redacted_and_flags_identifiers():
    r = redact_query("What medication is John Wilkerson prescribed?")
    assert r["text"] == "What medication is [NAME] prescribed?"
    assert r["identifiers"] == ["NAME"]
    assert "John" not in str(r["redacted"])  # only types and offsets, never values


def test_clinical_question_is_not_changed_by_redaction():
    q = "What medication was prescribed for the Type 2 diabetes mellitus patient whose note is dated 2024-11-10?"
    r = redact_query(q)
    assert r["text"] == q and r["identifiers"] == []


def test_retrieval_log_never_contains_the_name(tmp_path):
    log = tmp_path / "r.jsonl"
    retriever = Retriever(_index("redacted"), log_path=log)
    retriever.retrieve("What medication is John Wilkerson prescribed?")
    assert "Wilkerson" not in log.read_text()
    assert retriever.last_query_redaction["identifiers"] == ["NAME"]


@pytest.mark.parametrize("body,low", [
    ("PATIENT: [NAME]\nMRN: [PATIENT_ID]\nDOB: [DATE_OF_BIRTH]", True),
    ("Contact: [PHONE_NUMBER] (cell), [EMAIL]\n\n- [NAME], MD", True),
    ("PLAN: Continue Glipizide 5mg PO daily. Follow up in 8 weeks.", False),
    ("Patient: Allison Hill (ID: Bc321819)", False),   # raw text: no placeholders
])
def test_low_information_chunks(body, low):
    assert is_low_information(body) is low


def test_split_condition():
    assert split_condition("Which medications were prescribed to patients who presented with fatigue ?") == {
        "ask": "Which medications were prescribed to patients", "condition": "presented with fatigue"}
    assert split_condition("What medication was prescribed for the patient ?") is None


def test_note_ordered_grouping_keeps_each_notes_chunks_together():
    r = Retriever.__new__(Retriever)
    r.chunks = [{"doc_id": d} for d in ["a", "b", "a", "b", "c"]]
    ranked = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)]
    assert [i for i, _ in r._group_by_note(ranked, 2)] == [0, 2, 1, 3, 4]
    assert [i for i, _ in r._group_by_note(ranked, 2, note_ordered=False)] == [0, 1, 2, 3, 4]


def test_two_hop_adds_each_qualifying_notes_answer_section():
    retriever = Retriever(_index("redacted"))
    qa = [json.loads(l) for l in open("glassbox/data/raw/realistic/qa_eval_set.jsonl")]
    q = next(q for q in qa if q["type"] == "aggregate")
    results = retriever.retrieve(q["question"])
    assert retriever.last_analysis["ask"]
    by_note = {}
    for r in results:
        by_note.setdefault(r["doc_id"], []).append(r["selected_by"])
    for doc in q["relevant_doc_ids"]:
        assert by_note.get(doc) == ["question", "ask"]


# ---------------------------------------------------------------- robustness fixes (paraphrase set A)

@pytest.mark.parametrize("question,date_from,date_to", [
    ("Heart rate at the September 29, 2024 visit?", 20240929, 20240929),
    ("Labs requested at the 23 Aug 2025 visit?", 20250823, 20250823),
    ("seen on the 16th of June 2025?", 20250616, 20250616),
    ("BP on Nov. 16, 2025", 20251116, 20251116),
    ("visit 23/08/2025", 20250823, 20250823),          # first number > 12 can only be the day
    ("patients in 5/2025", 20250501, 20250531),
    ("patients in 2026-01", 20260101, 20260131),
])
def test_more_date_formats(question, date_from, date_to):
    f = analyze_query(question, DX)["filters"]
    assert (f.get("date_from"), f.get("date_to")) == (date_from, date_to)


@pytest.mark.parametrize("question,dx", [
    ("HTN visit on 2025-09-29", "Essential hypertension"),
    ("type 2 diabetics in May 2025", "Type 2 diabetes mellitus"),
    ("MDD patient seen 2025-01-04", "Major depressive disorder"),
    ("CKD3 visit", "Chronic kidney disease stage 3"),
    ("asthmatics in 2024", "Asthma"),
])
def test_standard_abbreviations(question, dx):
    assert analyze_query(question, DX)["filters"]["diagnosis"] == dx


@pytest.mark.parametrize("residual,ask,condition", [
    ("What drugs were the patients with dizziness given?", "What drugs were the patients given", "with dizziness"),
    ("Medications prescribed to patients complaining of dizziness?", "Medications prescribed to patients",
     "complaining of dizziness"),
])
def test_condition_split_without_relative_clause(residual, ask, condition):
    assert split_condition(residual) == {"ask": ask, "condition": condition}


def test_original_template_analysis_unchanged():
    a = analyze_query("Which medications were prescribed to Asthma patients who presented with wheezing in "
                      "October 2024?", DX)
    assert a["ask"] == "Which medications were prescribed to patients" and a["condition"] == "presented with wheezing"
