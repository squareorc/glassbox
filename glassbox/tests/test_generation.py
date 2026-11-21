"""
Tests for cited generation: output parsing, the abstain guard, prompt
construction, and question redaction's name rule.

The end-to-end test loads the real model (~4.5 GB) and only runs with
GLASSBOX_SLOW=1:
    GLASSBOX_SLOW=1 python -m pytest glassbox/tests/test_generation.py
"""

import json
import os

import pytest

from glassbox.src.generation.generator import ABSTAIN_ANSWERS, Generator, abstain_reason
from glassbox.src.generation.parse import ParseError, extract_json, normalise_citation, parse_reply
from glassbox.src.generation.prompt import build_messages, format_sources

ID_MAP = {"C1": "doc_1#plan", "C2": "doc_1#subjective", "C3": "doc_2#plan"}


# ---------------------------------------------------------------- parsing

def test_extracts_json_from_fences_and_prose():
    text = 'Sure, here it is:\n```json\n{"answer": "x {y}", "claims": []}\n```\nHope that helps.'
    assert extract_json(text) == {"answer": "x {y}", "claims": []}


@pytest.mark.parametrize("text,msg", [
    ("no json here", "no JSON object"),
    ('{"answer": "cut off', "not closed"),
    ('{"answer": 1, "claims": []}', '"answer"'),
    ('{"answer": "a", "claims": "b"}', '"claims"'),
])
def test_unusable_replies_raise(text, msg):
    with pytest.raises(ParseError, match=msg):
        parse_reply(text, ID_MAP)


@pytest.mark.parametrize("raw,label", [("C2", "C2"), ("[C2]", "C2"), ("c2", "C2"), ("C 2", "C2"),
                                       ("Source 2", "C2"), (2, "C2"), ("doc_1", None), (None, None)])
def test_citation_normalisation(raw, label):
    assert normalise_citation(raw) == label


def test_citations_mapped_validated_and_flagged():
    reply = json.dumps({"answer": "a", "insufficient": False, "claims": [
        {"text": "Takes X 5mg.", "citations": ["[C1]", "C1", "C9"]},
        {"text": "Has Y.", "citations": []},
        "Bare string claim.",
        {"text": "  ", "citations": ["C1"]},
    ]})
    out = parse_reply(reply, ID_MAP)
    assert [c["text"] for c in out["claims"]] == ["Takes X 5mg.", "Has Y.", "Bare string claim."]
    first, second, third = out["claims"]
    assert first["citations"] == ["doc_1#plan"] and first["invalid_citations"] == ["C9"] and not first["uncited"]
    assert second["uncited"] and third["uncited"]


# ---------------------------------------------------------------- guard

@pytest.mark.parametrize("analysis,redaction,source,reason", [
    ({"fallback": False}, {"identifiers": ["NAME"]}, "redacted", "identifier_in_question"),
    ({"fallback": False}, {"identifiers": ["NAME"]}, "raw", None),        # baseline: no privacy
    ({"fallback": True}, {"identifiers": []}, "redacted", "retrieval_fallback"),
    ({"fallback": False}, {"identifiers": []}, "redacted", None),
    ({"fallback": False}, None, "redacted", None),
])
def test_abstain_guard(analysis, redaction, source, reason):
    assert abstain_reason(analysis, redaction, source) == reason


def test_guard_abstains_without_calling_the_model(tmp_path):
    gen = Generator(model_id="not-a-real-model", log_path=tmp_path / "gen.jsonl")
    rec = gen.answer("What is [NAME] prescribed?", [], {"fallback": False}, {"identifiers": ["NAME"]}, "redacted")
    assert rec["abstained"] and not rec["model_called"]
    assert rec["answer"] == ABSTAIN_ANSWERS["identifier_in_question"]
    logged = json.loads((tmp_path / "gen.jsonl").read_text())
    assert logged["abstain_reason"] == "identifier_in_question"


# ---------------------------------------------------------------- prompt

def test_sources_numbered_with_note_labels():
    chunks = [{"chunk_id": "doc_7#subjective", "doc_id": "doc_7", "text": "[Progress note | 2025-01-02 | Asthma]\nS: wheezing"},
              {"chunk_id": "doc_3#plan", "doc_id": "doc_3", "text": "[Progress note | 2025-01-09 | Asthma]\nP: Montelukast"},
              {"chunk_id": "doc_7#plan", "doc_id": "doc_7", "text": "[Progress note | 2025-01-02 | Asthma]\nP: Albuterol"}]
    block, id_map = format_sources(chunks)
    assert id_map == {"C1": "doc_7#subjective", "C2": "doc_3#plan", "C3": "doc_7#plan"}
    assert "[C1] (Note 1)" in block and "[C2] (Note 2)" in block and "[C3] (Note 1)" in block
    assert "doc_7" not in block  # internal IDs never reach the prompt


def test_messages_contain_question_and_rules():
    messages, _ = build_messages("What was prescribed?", [])
    assert messages[0]["role"] == "system" and "[NAME]" in messages[0]["content"]
    assert messages[1]["content"].endswith("Question: What was prescribed?")


# ---------------------------------------------------------------- question name rule

from glassbox.src.redaction.query import redact_query  # noqa: E402


@pytest.mark.parametrize("question,expected", [
    ("What symptom did Tiffany Price report?", "What symptom did [NAME] report?"),   # spaCy misses this one
    ("Any notes for Tiffany Price from May 2025?", "Any notes for [NAME] from May 2025?"),
    ("Which Coronary Artery Disease patients were seen in March 2025?",
     "Which Coronary Artery Disease patients were seen in March 2025?"),
    ("Was Metformin Extended Release started?", "Was Metformin Extended Release started?"),
])
def test_question_name_rule(question, expected):
    assert redact_query(question)["text"] == expected


# ---------------------------------------------------------------- end to end (slow)

@pytest.mark.skipif(os.environ.get("GLASSBOX_SLOW") != "1", reason="loads the 7B model; set GLASSBOX_SLOW=1")
def test_model_answers_with_valid_citations():
    chunks = [{"chunk_id": "doc_1#plan", "doc_id": "doc_1",
               "text": "[Progress note | 2024-11-10 | Essential hypertension]\nP: Continue Amlodipine 5mg PO daily."}]
    rec = Generator().answer("What medication was prescribed for the patient?", chunks,
                             {"fallback": False}, {"identifiers": []}, "redacted")
    assert rec["ok"] and not rec["abstained"]
    assert "amlodipine 5mg" in rec["answer"].lower()
    assert all(c["citations"] == ["doc_1#plan"] for c in rec["claims"])


@pytest.mark.parametrize("question,expected", [
    ("What is Jessica taking?", "What is [NAME] taking?"),
    ("What is Jessica's diagnosis?", "What is [NAME]'s diagnosis?"),
    ("What was prescribed for Asthma?", "What was prescribed for Asthma?"),
    ("What is Glipizide used for?", "What is Glipizide used for?"),
])
def test_single_first_name_rule(question, expected):
    assert redact_query(question)["text"] == expected


def test_grounded_accuracy_needs_a_cited_gold_note():
    from glassbox.src.generation.evaluate import score
    q = {"answer_field": "medication", "answer": ["Metformin 500mg PO BID"], "answer_keywords": [["metformin", "500mg"]],
         "relevant_doc_ids": ["doc_1"]}
    chunks = [{"chunk_id": "doc_1#plan", "text": "P: Metformin 500mg PO BID"},
              {"chunk_id": "doc_2#plan", "text": "P: Metformin 500mg PO BID"}]
    def rec(cites):
        return {"answer": "Metformin 500mg PO BID", "abstained": False, "abstain_reason": None, "ok": True,
                "model_called": True, "errors": [],
                "claims": [{"text": "Metformin 500mg PO BID.", "citations": cites, "invalid_citations": [], "uncited": False}]}
    assert score(rec(["doc_1#plan"]), q, chunks)["grounded_accuracy"] == 1.0
    wrong = score(rec(["doc_2#plan"]), q, chunks)          # right drug, wrong patient's note
    assert wrong["accuracy"] == 1.0 and wrong["grounded_accuracy"] == 0.0
