"""
Tests for the end-to-end pipeline (glassbox/src/pipeline.py) and its trace.

The fast tests use the real question redaction, query analysis and retrieval
(the realistic redacted index must be built) with the LLM replaced by a canned
reply and verification off. The end-to-end test loads both real models and
only runs with GLASSBOX_SLOW=1.
"""

import json
import os

import pytest

from glassbox.src.pipeline import GlassBox
from glassbox.src.retrieval.index import index_dir

pytestmark = pytest.mark.skipif(not (index_dir("realistic", "redacted", "section") / "manifest.json").exists(),
                                reason="realistic redacted index not built")

# Questions carrying PII, and the PII values that must not appear in any trace
PII_QUESTIONS = [
    ("What medication is Tiffany Price taking?", ["Tiffany", "Price"]),
    ("What symptom did Jessica report?", ["Jessica"]),
    ("What was prescribed to the patient with MRN MRN-482913?", ["482913"]),
    ("Who is the patient at 412 Oak Street, Springfield, IL 62704?", ["412 Oak", "62704"]),
    ("What is the diagnosis for the patient reachable at (555) 867-5309?", ["867-5309"]),
    ("What medication does robert.miller@example.com take?", ["robert.miller", "example.com"]),
]


@pytest.fixture(scope="module")
def gb(tmp_path_factory):
    g = GlassBox("realistic", "redacted", verify=False, trace_dir=tmp_path_factory.mktemp("traces"))
    canned = json.dumps({"answer": "Amlodipine 5mg PO daily.", "insufficient": False,
                         "claims": [{"text": "Amlodipine 5mg PO daily is prescribed.", "citations": ["C1"]}]})
    g.generator._complete = lambda messages: {"text": canned, "prompt_tokens": 1, "generation_tokens": 1}
    return g


@pytest.mark.parametrize("question,secrets", PII_QUESTIONS)
def test_no_question_pii_in_trace(gb, question, secrets):
    trace = gb.answer(question, qid="pii-test")
    saved = (gb.trace_dir / "pii-test.json").read_text()
    for s in secrets:
        assert s.lower() not in json.dumps(trace).lower(), s
        assert s.lower() not in saved.lower(), s
    assert question not in saved
    assert trace["question"]["redaction_applied"] and trace["question"]["redacted"]


def test_named_person_triggers_guard_before_model(gb):
    trace = gb.answer("What medication is Tiffany Price taking?", save=False)
    g = trace["generation"]
    assert trace["question"]["identifiers"] == ["NAME"]
    assert g["abstained"] and g["abstain_reason"] == "identifier_in_question"
    assert not g["model_called"] and g["guard_decision"] == "abstain"
    assert trace["verification"]["verdict"] == "NOT_RUN"


def test_trace_has_every_stage(gb):
    trace = gb.answer("What medication was prescribed for the Essential hypertension patient whose note "
                      "is dated 2024-11-10?", save=False)
    for key in ("trace_id", "timestamp", "config", "question", "query_analysis", "retrieval", "generation",
                "verification", "timings_ms"):
        assert key in trace
    assert trace["config"]["index_source"] == "redacted" and trace["config"]["prompt_version"] == "v2"
    assert trace["query_analysis"]["filters"]
    assert 1 <= trace["retrieval"]["k"] <= 10
    c = trace["retrieval"]["chunks"][0]
    assert {"chunk_id", "section", "text", "score", "dense_rank", "bm25_rank", "selected_by", "low_information"} <= set(c)
    g = trace["generation"]
    assert g["model_called"] and not g["abstained"] and g["guard_decision"] == "pass"
    assert g["claims"][0]["citations"] == [c["chunk_id"]]
    assert set(trace["timings_ms"]) == {"question_redaction", "retrieval", "generation", "verification", "total"}


def test_raw_index_needs_baseline_config():
    from glassbox.src.retrieval.retriever import AccessDenied, Retriever
    with pytest.raises(AccessDenied):
        Retriever(index_dir("realistic", "raw", "section"))


@pytest.mark.skipif(os.environ.get("GLASSBOX_SLOW") != "1", reason="loads the 7B model and NLI model; set GLASSBOX_SLOW=1")
def test_end_to_end_with_real_models(tmp_path):
    from glassbox.run import run_qa
    from glassbox.src.qa.build_qa_set import load_qa_set
    from glassbox.src.retrieval.index import DATA_ROOT
    q = next(q for q in load_qa_set(DATA_ROOT / "raw" / "realistic" / "qa_eval_set.jsonl")
             if q["qid"] == "realistic-cf-000")
    trace = run_qa(GlassBox("realistic", "redacted", trace_dir=tmp_path), q)
    assert trace["generation"]["ok"] and not trace["generation"]["abstained"]
    assert trace["qa"]["scores"]["grounded_accuracy"] == 1.0
    assert trace["verification"]["verdict"] == "VERIFIED"
    saved = json.loads((tmp_path / "realistic-cf-000.json").read_text())
    assert "answer" not in saved["qa"] and "question" not in saved["qa"]   # no gold answer / question copy
