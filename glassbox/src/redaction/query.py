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

from functools import lru_cache
from typing import Any, Dict

from .detector import PIIDetector
from .redactor import PIIRedactor

# PII types that identify a specific individual
IDENTIFYING_TYPES = {"NAME", "PATIENT_ID", "DATE_OF_BIRTH", "PHONE_NUMBER", "EMAIL", "ADDRESS", "US_SSN"}


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
    text, log = redactor.redact(query, detector.detect(query))
    spans = [{"type": r["type"], "start": r["start"], "end": r["end"]} for r in log]
    return {
        "text": text,
        "redacted": spans,
        "identifiers": sorted({s["type"] for s in spans} & IDENTIFYING_TYPES),
    }
