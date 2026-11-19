"""
Query analysis: extract structured constraints from a question.

This is the "Query Analysis" stage of the GlassBox architecture. It pulls out
the parts of a question that refer to non-identifying note metadata and turns
them into retrieval filters:

    "...the Essential hypertension patient whose note is dated 2024-11-10?"
        -> diagnosis = "Essential hypertension", date in [2024-11-10, 2024-11-10]
    "...Major depressive disorder patients ... in December 2025?"
        -> diagnosis = "Major depressive disorder", date in [2025-12-01, 2025-12-31]

Why: dense embeddings can't do exact date matching or date arithmetic
("December 2025" vs "2025-12-03"), and BM25 can't connect a month name to
ISO dates. Applying these constraints as metadata filters before ranking
leaves each search method to do what it is good at: finding the right
section among the notes that satisfy the constraints.

Filters only use metadata that is NOT PII (visit date, diagnosis), so
identity questions ("What is John Wilkerson prescribed?") get no filter and
behave exactly as before.

Everything extracted is returned with its matched text so it can be logged
and shown in the audit dashboard.
"""

import calendar
import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_US_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_YEAR = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{4})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:in|during|from)\s+((?:19|20)\d{2})\b", re.IGNORECASE)

# Common short forms -> a word that identifies the canonical diagnosis.
# Applied only if exactly one known diagnosis contains the target word.
DIAGNOSIS_ALIASES = {
    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "diabetes": "diabetes",
    "t2dm": "diabetes",
    "depression": "depressive",
    "ckd": "kidney",
    "kidney disease": "kidney",
    "cad": "coronary",
    "coronary disease": "coronary",
    "copd": "copd",
    "gerd": "gastroesophageal",
    "reflux": "gastroesophageal",
    "anxiety": "anxiety",
    "hypothyroidism": "hypothyroidism",
    "osteoarthritis": "osteoarthritis",
    "asthma": "asthma",
}


def date_to_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def iso_to_int(iso: str) -> Optional[int]:
    try:
        return date_to_int(date.fromisoformat(iso))
    except (TypeError, ValueError):
        return None


def _find_diagnosis(query: str, diagnoses: Iterable[str]) -> Optional[Dict[str, Any]]:
    q = query.lower()
    # 1. Exact diagnosis names, longest first
    for dx in sorted(diagnoses, key=len, reverse=True):
        if re.search(r"\b" + re.escape(dx.lower()) + r"\b", q):
            return {"value": dx, "matched": dx, "via": "exact"}
    # 2. Aliases that point unambiguously to one diagnosis
    for alias, target in sorted(DIAGNOSIS_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            hits = [dx for dx in diagnoses if target in dx.lower()]
            if len(hits) == 1:
                return {"value": hits[0], "matched": alias, "via": "alias"}
    return None


def _find_date_range(query: str) -> Optional[Dict[str, Any]]:
    m = _ISO_DATE.search(query)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return {"from": date_to_int(d), "to": date_to_int(d), "matched": m.group(0), "granularity": "day"}
        except ValueError:
            pass
    m = _US_DATE.search(query)
    if m:
        try:
            d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            return {"from": date_to_int(d), "to": date_to_int(d), "matched": m.group(0), "granularity": "day"}
        except ValueError:
            pass
    m = _MONTH_YEAR.search(query)
    if m:
        month, year = _MONTHS[m.group(1).lower()], int(m.group(2))
        last = calendar.monthrange(year, month)[1]
        return {"from": date_to_int(date(year, month, 1)), "to": date_to_int(date(year, month, last)),
                "matched": m.group(0), "granularity": "month"}
    m = _YEAR.search(query)
    if m:
        year = int(m.group(1))
        return {"from": year * 10000 + 101, "to": year * 10000 + 1231, "matched": m.group(0), "granularity": "year"}
    return None


def analyze_query(query: str, diagnoses: Iterable[str]) -> Dict[str, Any]:
    """
    Extract retrieval filters from a question.

    Args:
        query: The user's question.
        diagnoses: Diagnosis names present in the index.

    Returns:
        {"filters": {"diagnosis"?, "date_from"?, "date_to"?},
         "extracted": [{"type", "matched", "value", ...}, ...]}
    """
    diagnoses = list(diagnoses)
    filters: Dict[str, Any] = {}
    extracted: List[Dict[str, Any]] = []

    dx = _find_diagnosis(query, diagnoses)
    if dx:
        filters["diagnosis"] = dx["value"]
        extracted.append({"type": "diagnosis", **dx})

    dr = _find_date_range(query)
    if dr:
        filters["date_from"], filters["date_to"] = dr["from"], dr["to"]
        extracted.append({"type": "date", "value": [dr["from"], dr["to"]], **{k: dr[k] for k in ("matched", "granularity")}})

    # Scope: does the question target one note or several? An exact date
    # (with or without a diagnosis) pins one visit; a month/year range or a
    # plural "patients" question can span several notes and needs more context.
    if dr and dr["granularity"] == "day":
        scope = "single_note"
    elif dr or re.search(r"\bpatients\b", query, re.IGNORECASE):
        scope = "multi_note"
    else:
        scope = "single_note"

    return {"filters": filters, "extracted": extracted, "scope": scope,
            "residual_query": residual_query(query, extracted)}


def residual_query(query: str, extracted: List[Dict[str, Any]]) -> str:
    """
    The question with the filter phrases removed, used for RANKING once the
    filters are applied. Every candidate already satisfies the constraints,
    so repeating them in the ranking query only rewards chunks that restate
    the date/diagnosis (patient header, impression) over the chunk that
    answers what is actually asked ("What medication was prescribed ...?").
    """
    out = query
    for e in extracted:
        out = re.sub(re.escape(e["matched"]), " ", out, flags=re.IGNORECASE)
    # Drop connective words left dangling by the removal ("dated", "in", "on")
    out = re.sub(r"\b(?:whose note is dated|dated|seen on|in|on|during|from)\s*(?=[?.!,]|$)", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def matches(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Does a chunk satisfy the filters? (Python-side twin of chroma_where.)"""
    if "diagnosis" in filters and chunk["diagnosis"] != filters["diagnosis"]:
        return False
    if "date_from" in filters:
        d = iso_to_int(chunk.get("visit_date", ""))
        if d is None or not filters["date_from"] <= d <= filters["date_to"]:
            return False
    return True


def chroma_where(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate filters into a ChromaDB `where` clause."""
    conds = []
    if "diagnosis" in filters:
        conds.append({"diagnosis": {"$eq": filters["diagnosis"]}})
    if "date_from" in filters:
        conds.append({"visit_date_int": {"$gte": filters["date_from"]}})
        conds.append({"visit_date_int": {"$lte": filters["date_to"]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
