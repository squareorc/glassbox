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
