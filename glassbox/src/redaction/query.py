"""
Redaction of the user's QUESTION, before it reaches any other stage.

Notes are redacted offline, but a question typed at query time can contain
PII too ("What is John Wilkerson prescribed?"). Without this step that name
would go into the query embedding, the retrieval log and the LLM prompt.

The same detector and redactor as for notes are used. The returned record
holds the redacted text and the TYPES and offsets of what was removed, never
the original values, so it is safe to log. `identifiers` tells later stages
that the question is about a specific person; under redaction the system
can't know who [NAME] is in any note, so generation must abstain rather than
attribute a redacted note to that person.
"""

import calendar
import re
from functools import lru_cache
from typing import Any, Dict, List

from .clinical_terms import CLINICAL_TERMS
from .detector import PIIDetector
from .redactor import PIIRedactor

# PII types that identify a specific individual
IDENTIFYING_TYPES = {"NAME", "PATIENT_ID", "DATE_OF_BIRTH", "PHONE_NUMBER", "EMAIL", "ADDRESS", "US_SSN"}


# Question-only name rule. spaCy needs context to tag a PERSON, and a short
# question gives little: "What symptom did Tiffany Price report?" is missed
# ("Price" is also a common noun). In a question, a run of 2+ capitalised
# words that isn't at the start of the sentence is almost always a name.
# Excluded: clinical vocabulary, disease words ("Coronary Artery Disease")
# and month/day names. Not used on notes, whose headings and line starts
# would make it misfire.
_CAP_RUN = re.compile(r"\b[A-Z][a-z'\-]+(?:[ ][A-Z][a-z'\-]+)+\b")
_NOT_NAME = (
    CLINICAL_TERMS
    | {m.lower() for m in calendar.month_name if m} | {d.lower() for d in calendar.day_name}
    | {"disease", "disorder", "syndrome", "failure", "hypertension", "diabetes", "mellitus", "asthma",
       "kidney", "renal", "heart", "cardiac", "artery", "coronary", "chronic", "acute", "major", "minor",
       "type", "stage", "essential", "pulmonary", "obstructive", "osteoarthritis", "arthritis",
       "depression", "depressive", "anxiety", "cancer", "infection", "pneumonia", "progress", "note",
       "discharge", "summary", "consultation", "emergency", "department", "hospital", "clinic"}
)


# A single capitalised word is a name when it is possessive ("Jessica's") or follows a
# verb/preposition that takes a person ("What is Jessica taking?", "did Jessica report").
# Robustness fix: first names alone slipped past the guard 12% of the time. Trade-off:
# an unknown brand-name drug in that position ("for Jardiance?") is also redacted, so the
# system abstains on that question; privacy is preferred.
_SINGLE_NAME = re.compile(r"\b(?:(?<=\bis )|(?<=\bdid )|(?<=\bdoes )|(?<=\bwas )|(?<=\bhas )|(?<=\bfor )"
                          r"|(?<=\babout )|(?<=\bof ))[A-Z][a-z'\-]+\b(?! patients?\b)|\b[A-Z][a-z\-]+(?='s\b)")


def _question_name_spans(question: str) -> List[Dict[str, Any]]:
    spans = []
    for m in _SINGLE_NAME.finditer(question):
        if m.group(0).lower().rstrip("'") not in _NOT_NAME:
            spans.append({"type": "NAME", "start": m.start(), "end": m.end(), "score": 0.55})
    for m in _CAP_RUN.finditer(question):
        before = question[:m.start()].rstrip()
        if not before or before[-1] in ".?!:":      # sentence start: capitalised anyway
            continue
        if any(w.lower() in _NOT_NAME for w in m.group(0).split()):
            continue
        spans.append({"type": "NAME", "start": m.start(), "end": m.end(), "score": 0.6})
    return spans


@lru_cache(maxsize=4)
def _pipeline(threshold: float):
    return PIIDetector(confidence_threshold=threshold), PIIRedactor()


def redact_query(query: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Returns {"text": redacted question,
             "redacted": [{"type", "start", "end"}],   # offsets in the ORIGINAL question
             "identifiers": sorted identifying types found}
    """
    detector, redactor = _pipeline(threshold)
    text, log = redactor.redact(query, detector.detect(query) + _question_name_spans(query))
    spans = [{"type": r["type"], "start": r["start"], "end": r["end"]} for r in log]
    return {
        "text": text,
        "redacted": spans,
        "identifiers": sorted({s["type"] for s in spans} & IDENTIFYING_TYPES),
    }
