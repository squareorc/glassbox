"""
Tests for claim verification. Tests that load the NLI model (~1.7 GB) run only
with GLASSBOX_SLOW=1.
"""

import json
import os
import random

import pytest

from glassbox.src.verification.corruptions import asserts_uncited_condition, corruptions, genuine
from glassbox.src.verification.verifier import (CONTRADICTED, QUESTION_SUBJECT, SUPPORTED, UNCERTAIN, UNSUPPORTED,
                                                Verifier, rewrite_claim)

Q_MED = {"question": "What medication was prescribed for the Asthma patient whose note is dated 2024-10-07?",
         "answer_field": "medication", "answer": ["Montelukast 10mg PO QHS"], "answer_keywords": [["montelukast", "10mg"]]}
EVIDENCE = "[Progress note | 2024-10-07 | Asthma]\nPLAN: Start Montelukast 10mg PO QHS. Follow up in 6 weeks."


@pytest.mark.parametrize("claim,expected", [
    ("The patient is prescribed X.", f"{QUESTION_SUBJECT} is prescribed X."),
    ("[NAME] is prescribed X.", f"{QUESTION_SUBJECT} is prescribed X."),
    ("They reported cough.", f"{QUESTION_SUBJECT} reported cough."),
    ("Acetaminophen 650mg PO Q6H PRN", f"{QUESTION_SUBJECT}: Acetaminophen 650mg PO Q6H PRN"),
])
def test_rewrite_with_identifier(claim, expected):
    assert rewrite_claim(claim, ["NAME"]) == expected


def test_no_rewrite_without_identifier():
    assert rewrite_claim("The patient is prescribed X.", []) == "The patient is prescribed X."


def test_label_rule_and_margin():
    v = Verifier.__new__(Verifier)
    v.margin = 0.0
    assert v.label({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}) == SUPPORTED
    assert v.label({"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}) == CONTRADICTED
    assert v.label({"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1}) == UNSUPPORTED
    assert v.label({"entailment": 0.45, "neutral": 0.40, "contradiction": 0.15}, margin=0.1) == UNCERTAIN


def test_premise_options_include_combined_evidence():
    assert Verifier.premises(["a"]) == ["a"]
    assert Verifier.premises(["a", "b"]) == ["a", "b", "a\n\nb"]


def test_genuine_claim_detection():
    assert genuine("The patient was started on Montelukast 10mg PO QHS.", [EVIDENCE], Q_MED)
    assert not genuine("The patient was started on Montelukast 20mg PO QHS.", [EVIDENCE], Q_MED)   # number not in evidence
    assert not genuine("Montelukast 10mg PO QHS and Warfarin.", [EVIDENCE], Q_MED)                # drug not in evidence


def test_corruptions_change_the_claim():
    made = corruptions("The patient was started on Montelukast 10mg PO QHS.", EVIDENCE, Q_MED, random.Random(0))
    kinds = {k: t for k, t, _ in made}
    assert {"drug_swap", "dose_change", "negation", "unsupported_addition"} <= set(kinds)
    assert "Montelukast" not in kinds["drug_swap"] and "20mg" in kinds["dose_change"]
    assert "was not started" in kinds["negation"]


def test_uncited_condition_flag():
    q = {"question": "Which medications were prescribed to COPD patients who presented with dyspnea in September 2024?"}
    plan = "[Progress note | 2024-09-19 | COPD]\nPLAN: Prescribed Tiotropium 18mcg inhaler daily"
    assert asserts_uncited_condition("Tiotropium was prescribed to COPD patients who presented with dyspnea.", plan, q)
    assert not asserts_uncited_condition("Tiotropium 18mcg inhaler daily was prescribed.", plan, q)


def test_corrupted_set_is_frozen_and_consistent():
    path = "glassbox/data/verification/corrupted_claims.jsonl"
    if not os.path.exists(path):
        pytest.skip("build the corrupted-claim set first")
    items = [json.loads(l) for l in open(path)]
    assert all((it["kind"] == "genuine") == (it["expected"] == SUPPORTED) for it in items)
    assert {it["split"] for it in items} == {"dev", "test"}


# ---------------------------------------------------------------- with the model (slow)

slow = pytest.mark.skipif(os.environ.get("GLASSBOX_SLOW") != "1", reason="loads the NLI model; set GLASSBOX_SLOW=1")


@slow
def test_verifier_labels_real_cases():
    v = Verifier()
    claims = [{"text": "The patient was started on Montelukast 10mg PO QHS.", "citations": ["c1"]},
              {"text": "The patient was started on Montelukast 20mg PO QHS.", "citations": ["c1"]},
              {"text": "The patient was started on Warfarin.", "citations": []}]
    out = v.verify_claims(claims, {"c1": EVIDENCE})
    assert [c["label"] for c in out] == [SUPPORTED, CONTRADICTED, UNSUPPORTED]


@slow
def test_answer_verdicts():
    v = Verifier()
    ok = {"abstained": False, "claims": [{"text": "Follow-up is in 6 weeks.", "citations": ["c1"]}]}
    assert v.verify_answer(ok, {"c1": EVIDENCE})["verdict"] == "VERIFIED"
    assert v.verify_answer({"abstained": True, "claims": []}, {})["verdict"] == "ABSTAINED"
