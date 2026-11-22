"""
Dashboard tests: both pages load, and the trace view renders every kind of
trace (verified, abstained by the guard, model-insufficient, unparsed, raw
baseline) without errors. Uses Streamlit's headless AppTest; no models load.
"""

import copy
import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from glassbox.src.dashboard.traces import chunk_anchor, load_rows, source_label, summary_row  # noqa: E402

CHUNKS = [
    {"rank": 1, "chunk_id": "doc_0001#plan", "doc_id": "doc_0001", "section": "plan",
     "text": "[Progress note | 2024-11-10 | Essential hypertension]\nP: Amlodipine 5mg PO daily.", "start": 0, "end": 10,
     "score": 0.05, "dense_score": 0.8, "bm25_score": 3.1, "dense_rank": 1, "bm25_rank": 2,
     "low_information": False, "selected_by": "question"},
    {"rank": 2, "chunk_id": "doc_0001#header", "doc_id": "doc_0001", "section": "header",
     "text": "PATIENT: [NAME] MRN: [PATIENT_ID]", "start": 0, "end": 10, "score": 0.01, "dense_score": None,
     "bm25_score": 1.0, "dense_rank": None, "bm25_rank": 9, "low_information": True, "selected_by": "question"},
]

VERIFIED = {
    "trace_version": 1, "trace_id": "abc123", "timestamp": "2026-11-22T00:00:00Z", "qid": "t-1",
    "config": {"name": "redacted", "index": "redacted-t0.50-section", "baseline_mode": False,
               "fusion_weights": [2.0, 1.0], "generator_model": "qwen", "prompt_version": "v2", "verifier_model": "nli"},
    "question": {"text": "What was prescribed?", "redacted": [], "identifiers": [], "redaction_applied": True},
    "query_analysis": {"filters": {"diagnosis": "Essential hypertension"}, "scope": "single_note", "fallback": False,
                       "ranking_query": "What was prescribed?", "extracted": []},
    "retrieval": {"k": 2, "context_chars": 100, "chunks": CHUNKS},
    "generation": {"guard_decision": "pass", "abstained": False, "abstain_reason": None, "model_called": True, "ok": True,
                   "answer": "Amlodipine 5mg PO daily.", "source_map": {"C1": "doc_0001#plan", "C2": "doc_0001#header"},
                   "claims": [{"text": "Amlodipine 5mg PO daily is prescribed.", "citations": ["doc_0001#plan"],
                               "invalid_citations": ["C9"], "uncited": False}],
                   "attempts": 1, "errors": [], "raw_outputs": ["{}"], "prompt_tokens": 10, "generation_tokens": 5},
    "verification": {"verdict": "VERIFIED", "claims": [
        {"claim": "Amlodipine 5mg PO daily is prescribed.", "checked_as": "Amlodipine 5mg PO daily is prescribed.",
         "label": "SUPPORTED", "probabilities": {"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01},
         "premise_chunks": ["doc_0001#plan"], "margin": 0.0}]},
    "timings_ms": {"question_redaction": 5.0, "retrieval": 40.0, "generation": 3000.0, "verification": 30.0, "total": 3075.0},
    "qa": {"qid": "t-1", "type": "clinical_fact", "split": "test", "answer_field": "medication",
           "gold_doc_ids": ["doc_0001"], "scores": {"grounded_accuracy": 1.0}},
}


def variants():
    guard = copy.deepcopy(VERIFIED)
    guard["question"].update(text="What is [NAME] taking?", redacted=[{"type": "NAME", "start": 8, "end": 15}],
                             identifiers=["NAME"])
    guard["generation"].update(guard_decision="abstain", abstained=True, abstain_reason="identifier_in_question",
                               model_called=False, claims=[], answer="Can't be determined.", source_map=None)
    guard["verification"] = {"verdict": "ABSTAINED", "claims": []}
    insufficient = copy.deepcopy(VERIFIED)
    insufficient["generation"].update(abstained=True, abstain_reason="model_insufficient", claims=[])
    insufficient["verification"] = {"verdict": "ABSTAINED", "claims": []}
    unparsed = copy.deepcopy(VERIFIED)
    unparsed["generation"].update(ok=False, claims=[], answer="", errors=["no JSON object found"], attempts=2)
    unparsed["verification"] = {"verdict": "NO_CLAIMS", "claims": []}
    raw = copy.deepcopy(VERIFIED)
    raw["config"].update(name="raw", baseline_mode=True)
    del raw["qa"]
    noguard = copy.deepcopy(guard)
    noguard["generation"]["guard_decision"] = "disabled"
    return {"verified": VERIFIED, "guard": guard, "insufficient": insufficient, "unparsed": unparsed,
            "raw": raw, "noguard": noguard}


RENDER = """
import json, sys
sys.path.insert(0, ".")
from glassbox.src.dashboard.app import show_trace
show_trace(json.loads({payload!r}), original={original!r})
"""


@pytest.mark.parametrize("name", list(variants()))
def test_trace_view_renders(name):
    trace = variants()[name]
    at = AppTest.from_string(RENDER.format(payload=json.dumps(trace), original="What is Jane taking?"),
                             default_timeout=60).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert trace["verification"]["verdict"] in text


def test_guard_refusal_is_explained():
    at = AppTest.from_string(RENDER.format(payload=json.dumps(variants()["guard"]), original=None),
                             default_timeout=60).run()
    assert any("refused" in e.value for e in at.error)


def test_both_pages_load():
    at = AppTest.from_file("../src/dashboard/app.py", default_timeout=60).run()   # relative to this file
    assert not at.exception
    at.sidebar.radio[0].set_value("Stored traces").run()
    assert not at.exception


def test_helpers(tmp_path):
    d = tmp_path / "realistic" / "redacted"
    d.mkdir(parents=True)
    (d / "t-1.json").write_text(json.dumps(VERIFIED))
    (d / "broken.json").write_text("{")
    rows = load_rows("realistic/redacted", tmp_path)
    assert len(rows) == 1 and rows[0]["verdict"] == "VERIFIED" and rows[0]["grounded_accuracy"] == 1.0
    assert summary_row(variants()["raw"], d / "x.json")["type"] == "ad hoc"
    assert chunk_anchor("doc_0001#plan") == "chunk-doc-0001-plan"
    assert source_label(VERIFIED) == {"doc_0001#plan": "C1", "doc_0001#header": "C2"}
