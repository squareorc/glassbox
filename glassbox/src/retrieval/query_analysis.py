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
_MONTH_NAME = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_MONTH_YEAR = re.compile(r"\b(" + _MONTH_NAME[3:-1] + r")\.?,?\s+(\d{4})\b", re.IGNORECASE)
# "between March 2025 and June 2025", "from 2025-03-01 to 2025-06-30", "March to June 2025"
_DATE_EXPR = (r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|" + _MONTH_NAME + r"\.?(?:,?\s+\d{4})?|(?:19|20)\d{2})")
_RANGE = re.compile(r"\b(?:between|from)\s+(" + _DATE_EXPR + r")\s+(?:and|to|until|through)\s+(" + _DATE_EXPR + r")"
                    r"|\b(" + _DATE_EXPR + r")\s*(?:-|to|through|until)\s*(" + _DATE_EXPR + r")(?=\W|$)", re.IGNORECASE)
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


def _find_diagnoses(query: str, diagnoses: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Every diagnosis mentioned in the question ("hypertension or diabetes"
    gives two). Exact names are matched first, longest first; then aliases
    that point unambiguously to one diagnosis. Matched text is masked out as
    it's found, so "Type 2 diabetes mellitus" isn't matched again by the
    alias "diabetes".
    """
    q = query.lower()
    found: List[Dict[str, Any]] = []

    def take(m, dx, via):
        nonlocal q
        found.append({"value": dx, "matched": query[m.start():m.end()], "via": via, "pos": m.start()})
        q = q[:m.start()] + " " * (m.end() - m.start()) + q[m.end():]

    for dx in sorted(diagnoses, key=len, reverse=True):
        for m in list(re.finditer(r"\b" + re.escape(dx.lower()) + r"\b", q)):
            take(m, dx, "exact")
    for alias, target in sorted(DIAGNOSIS_ALIASES.items(), key=lambda x: -len(x[0])):
        hits = [dx for dx in diagnoses if target in dx.lower()]
        if len(hits) != 1:
            continue
        for m in list(re.finditer(r"\b" + re.escape(alias) + r"\b", q)):
            take(m, hits[0], "alias")
    found.sort(key=lambda f: f.pop("pos"))
    unique, seen = [], set()
    for f in found:
        if f["value"] not in seen:
            seen.add(f["value"])
            unique.append(f)
    return unique


class InvalidDate(ValueError):
    pass


def _single_date(text: str, default_year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Parse one date expression into a range. Returns None if there is no date
    expression; raises InvalidDate for one that looks like a date but isn't
    (2025-13-01), so the caller doesn't silently fall back to a coarser match.
    """
    m = _ISO_DATE.search(text)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            raise InvalidDate(m.group(0))
        return {"from": date_to_int(d), "to": date_to_int(d), "matched": m.group(0), "granularity": "day"}
    m = _US_DATE.search(text)
    if m:
        try:
            d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            raise InvalidDate(m.group(0))
        return {"from": date_to_int(d), "to": date_to_int(d), "matched": m.group(0), "granularity": "day"}
    m = _MONTH_YEAR.search(text)
    if m:
        month, year = _MONTHS[m.group(1).lower()], int(m.group(2))
        last = calendar.monthrange(year, month)[1]
        return {"from": date_to_int(date(year, month, 1)), "to": date_to_int(date(year, month, last)),
                "matched": m.group(0), "granularity": "month"}
    m = re.fullmatch(r"\s*(" + _MONTH_NAME[3:-1] + r")\.?\s*", text, re.IGNORECASE)
    if m and default_year:  # bare month inside a range: "March to June 2025"
        month = _MONTHS[m.group(1).lower()]
        last = calendar.monthrange(default_year, month)[1]
        return {"from": date_to_int(date(default_year, month, 1)), "to": date_to_int(date(default_year, month, last)),
                "matched": m.group(0).strip(), "granularity": "month"}
    m = re.fullmatch(r"\s*((?:19|20)\d{2})\s*", text)
    if m:
        year = int(m.group(1))
        return {"from": year * 10000 + 101, "to": year * 10000 + 1231, "matched": m.group(1), "granularity": "year"}
    return None


def _find_date_range(query: str) -> Optional[Dict[str, Any]]:
    """
    The visit-date constraint in a question, as an inclusive range.
    Explicit ranges ("between March 2025 and June 2025") come first, then a
    single date, month or year. A malformed date gives an "invalid" result
    (no filter), never a silently different one.
    """
    try:
        m = _RANGE.search(query)
        if m:
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            year_b = re.search(r"\d{4}", b)
            start = _single_date(a, default_year=int(year_b.group(0)) if year_b else None)
            end = _single_date(b)
            if start and end and start["from"] <= end["to"]:
                return {"from": start["from"], "to": end["to"], "matched": m.group(0), "granularity": "range"}
        return _find_single_date(query)
    except InvalidDate as e:
        return {"invalid": str(e), "matched": str(e)}


def _find_single_date(query: str) -> Optional[Dict[str, Any]]:
    for rx in (_ISO_DATE, _US_DATE):
        m = rx.search(query)
        if m:
            return _single_date(m.group(0))
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

    dxs = _find_diagnoses(query, diagnoses)
    if dxs:
        # One diagnosis -> a string; several ("asthma or COPD") -> a list
        values = [d["value"] for d in dxs]
        filters["diagnosis"] = values[0] if len(values) == 1 else values
        extracted.extend({"type": "diagnosis", **d} for d in dxs)

    dr = _find_date_range(query)
    if dr and "invalid" in dr:
        # Report it, apply no date filter, and keep the text in the ranking query
        extracted.append({"type": "invalid_date", "matched": dr["matched"]})
        dr = None
    if dr:
        filters["date_from"], filters["date_to"] = dr["from"], dr["to"]
        extracted.append({"type": "date", "value": [dr["from"], dr["to"]], **{k: dr[k] for k in ("matched", "granularity")}})

    # Scope: does the question target one note or several? An exact date
    # (with or without a diagnosis) pins one visit; a month/year range or a
    # plural "patients" question can span several notes and needs more context.
    if len(dxs) > 1:
        scope = "multi_note"
    elif dr and dr["granularity"] == "day":
        scope = "single_note"
    elif dr or re.search(r"\bpatients\b", query, re.IGNORECASE):
        scope = "multi_note"
    else:
        scope = "single_note"

    residual = residual_query(query, extracted)
    out = {"filters": filters, "extracted": extracted, "scope": scope, "residual_query": residual}
    if scope == "multi_note":
        out.update(split_condition(residual) or {})
    return out


_RELATIVE = re.compile(r"\s+(?:who|whose|that|which)\s+", re.IGNORECASE)


def split_condition(residual: str) -> Optional[Dict[str, str]]:
    """
    Split a multi-note question into what is ASKED and the CONDITION a note
    must meet, at the first relative clause:

        "Which medications were prescribed to patients who presented with fatigue ?"
        -> ask       "Which medications were prescribed to patients"
           condition "presented with fatigue"

    These are two hops that usually live in different sections of a note
    (the symptom in SUBJECTIVE, the drug in PLAN). Retrieval finds the notes
    with the whole question, then picks each note's answer section with the
    ask alone. Returns None if there is no relative clause or a side is empty.
    """
    m = _RELATIVE.search(residual)
    if not m:
        return None
    ask = residual[:m.start()].strip(" ?.!,")
    condition = residual[m.end():].strip(" ?.!,")
    if len(ask.split()) < 2 or len(condition.split()) < 2:
        return None
    return {"ask": ask, "condition": condition}


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
        if e["type"] == "invalid_date":
            continue
        out = re.sub(re.escape(e["matched"]), " ", out, flags=re.IGNORECASE)
    # Drop connective words left dangling by the removal ("dated", "in", "on")
    out = re.sub(r"\b(?:whose note is dated|dated|seen on|in|on|during|from|between)\s*(?=[?.!,]|$)", " ", out)
    out = re.sub(r"\b(?:or|and)\s*(?=[?.!,]|$)", " ", out)  # "with asthma or COPD" -> "with"
    out = re.sub(r"\b(?:or|and)\s+(?=(?:or|and)\b)", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def matches(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Does a chunk satisfy the filters? (Python-side twin of chroma_where.)"""
    if "diagnosis" in filters:
        allowed = filters["diagnosis"]
        if chunk["diagnosis"] not in ([allowed] if isinstance(allowed, str) else allowed):
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
        dx = filters["diagnosis"]
        conds.append({"diagnosis": {"$eq": dx}} if isinstance(dx, str) else {"diagnosis": {"$in": list(dx)}})
    if "date_from" in filters:
        conds.append({"visit_date_int": {"$gte": filters["date_from"]}})
        conds.append({"visit_date_int": {"$lte": filters["date_to"]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
