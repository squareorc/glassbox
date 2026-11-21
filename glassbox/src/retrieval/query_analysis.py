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

_MONTH_NAME = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_MON = _MONTH_NAME[3:-1]
_ORD = r"(?:st|nd|rd|th)?"
# Day-level forms, tried in this order (robustness fix: only ISO and US numeric dates were
# understood; "November 10, 2024", "Nov 10 2024", "10 Nov 2024" and "the 16th of June 2025"
# were ignored or shrunk to the whole month).
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(r"\b(" + _MON + r")\.?\s+(\d{1,2})" + _ORD + r",?\s+(\d{4})\b", re.IGNORECASE)
_DAY_FIRST = re.compile(r"\b(?:the\s+)?(\d{1,2})" + _ORD + r"(?:\s+of)?\s+(" + _MON + r")\.?,?\s+(\d{4})\b", re.IGNORECASE)
_US_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
# Month-level forms
_MONTH_YEAR = re.compile(r"\b(" + _MON + r")\.?,?\s+(\d{4})\b", re.IGNORECASE)
_NUM_MONTH = re.compile(r"(?<![\d/])(\d{1,2})/(\d{4})\b")            # 5/2025
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")               # 2025-05
# "between March 2025 and June 2025", "from 2025-03-01 to 2025-06-30", "March to June 2025"
_DATE_EXPR = (r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|" + _MONTH_NAME + r"\.?(?:\s+\d{1,2}(?!\d)" + _ORD
              + r")?(?:,?\s+\d{4})?|(?:19|20)\d{2})")
_RANGE = re.compile(r"\b(?:between|from)\s+(" + _DATE_EXPR + r")\s+(?:and|to|until|through)\s+(" + _DATE_EXPR + r")"
                    r"|\b(" + _DATE_EXPR + r")\s*(?:-|to|through|until)\s*(" + _DATE_EXPR + r")(?=\W|$)", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:in|during|from)\s+((?:19|20)\d{2})\b", re.IGNORECASE)


def _day(y: int, m: int, d: int, matched: str) -> Dict[str, Any]:
    try:
        dt = date(y, m, d)
    except ValueError:
        raise InvalidDate(matched)
    return {"from": date_to_int(dt), "to": date_to_int(dt), "matched": matched, "granularity": "day"}


def _month(y: int, m: int, matched: str) -> Dict[str, Any]:
    if not 1 <= m <= 12:
        raise InvalidDate(matched)
    last = calendar.monthrange(y, m)[1]
    return {"from": date_to_int(date(y, m, 1)), "to": date_to_int(date(y, m, last)),
            "matched": matched, "granularity": "month"}


class InvalidDate(ValueError):
    pass


def _single_date(text: str, default_year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Parse the first date expression in `text` into a range: a day, else a month, else
    (only for a bare year) a year. Returns None if there is none; raises InvalidDate for
    one that looks like a date but isn't (2025-13-01), so the caller doesn't silently
    fall back to a coarser match.
    """
    m = _ISO_DATE.search(text)
    if m:
        return _day(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(0))
    m = _LONG_DATE.search(text)
    if m:
        return _day(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)), m.group(0))
    m = _DAY_FIRST.search(text)
    if m:
        return _day(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)), m.group(0))
    m = _US_DATE.search(text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # US month/day order; a first number above 12 can only be a day (23/08/2025)
        return _day(y, b, a, m.group(0)) if a > 12 and b <= 12 else _day(y, a, b, m.group(0))
    m = _MONTH_YEAR.search(text)
    if m:
        return _month(int(m.group(2)), _MONTHS[m.group(1).lower()], m.group(0))
    m = _NUM_MONTH.search(text)
    if m:
        return _month(int(m.group(2)), int(m.group(1)), m.group(0))
    m = _ISO_MONTH.search(text)
    if m:
        return _month(int(m.group(1)), int(m.group(2)), m.group(0))
    m = re.fullmatch(r"\s*(" + _MON + r")\.?\s*", text, re.IGNORECASE)
    if m and default_year:  # bare month inside a range: "March to June 2025"
        return _month(default_year, _MONTHS[m.group(1).lower()], m.group(0).strip())
    m = re.fullmatch(r"\s*((?:19|20)\d{2})\s*", text)
    if m:
        year = int(m.group(1))
        return {"from": year * 10000 + 101, "to": year * 10000 + 1231, "matched": m.group(1), "granularity": "year"}
    return None


# Common short forms -> a word that identifies the canonical diagnosis.
# Applied only if exactly one known diagnosis contains the target word.
DIAGNOSIS_ALIASES = {
    # Robustness fix: extended with standard clinical abbreviations and synonyms (from common
    # clinical-abbreviation usage, not only the forms that appeared in paraphrase set A, which
    # did include HTN, MDD, OA, CKD3, coronary heart disease, emphysema, reactive airway disease).
    # Matched as whole words, optionally plural ("diabetics", "asthmatics").
    "hypertension": "hypertension", "high blood pressure": "hypertension", "htn": "hypertension",
    "hypertensive": "hypertension",
    "diabetes": "diabetes", "diabetic": "diabetes", "t2dm": "diabetes", "t2d": "diabetes", "dm2": "diabetes",
    "dm": "diabetes", "niddm": "diabetes", "type 2 diabetes": "diabetes", "type ii diabetes": "diabetes",
    "depression": "depressive", "major depression": "depressive", "mdd": "depressive",
    "ckd": "kidney", "ckd3": "kidney", "kidney disease": "kidney", "renal disease": "kidney",
    "renal insufficiency": "kidney",
    "cad": "coronary", "chd": "coronary", "ihd": "coronary", "coronary disease": "coronary",
    "coronary heart disease": "coronary", "ischemic heart disease": "coronary", "ischaemic heart disease": "coronary",
    "copd": "copd", "emphysema": "copd", "chronic bronchitis": "copd",
    "chronic obstructive pulmonary disease": "copd",
    "asthma": "asthma", "asthmatic": "asthma", "reactive airway disease": "asthma",
    "osteoarthritis": "osteoarthritis", "oa": "osteoarthritis", "osteoarthritic": "osteoarthritis",
    "degenerative joint disease": "osteoarthritis", "djd": "osteoarthritis",
    "gerd": "gastroesophageal", "reflux": "gastroesophageal",
    "anxiety": "anxiety", "hypothyroidism": "hypothyroidism",
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
        for m in list(re.finditer(r"\b" + re.escape(alias) + r"s?\b", q)):
            take(m, hits[0], "alias")
    found.sort(key=lambda f: f.pop("pos"))
    unique, seen = [], set()
    for f in found:
        if f["value"] not in seen:
            seen.add(f["value"])
            unique.append(f)
    return unique


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
    found = _single_date(query)
    if found and found["granularity"] in ("day", "month"):
        return found
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
# Robustness fix: the split used to need a relative clause ("patients who presented with X"),
# so "patients with X", "patients complaining of X" and "X: patients presenting with Y" got no
# two-hop selection. A condition is now also a marker + the words up to a boundary.
_CONDITION = re.compile(r"\b(?:presenting with|presented with|complaining of|complained of|reporting|reported|"
                        r"having|had|with)\s+(.+?)(?=\s+(?:given|prescribed|receive|received|get|got|put|on|in|during|"
                        r"what|which|-|—)\b|[?.,:;!\-—]|$)", re.IGNORECASE)


def split_condition(residual: str) -> Optional[Dict[str, str]]:
    """
    Split a multi-note question into what is ASKED and the CONDITION a note must meet:

        "Which medications were prescribed to patients who presented with fatigue ?"
        -> ask "Which medications were prescribed to patients", condition "presented with fatigue"
        "What drugs were the patients with dizziness given?"
        -> ask "What drugs were the patients given", condition "with dizziness"

    These are two hops that usually live in different sections of a note (the symptom
    in SUBJECTIVE, the drug in PLAN). Retrieval finds the notes with the whole question,
    then picks each note's answer section with the ask alone. Returns None if no
    condition is found or a side is empty.
    """
    m = _RELATIVE.search(residual)
    if m:
        ask, condition = residual[:m.start()].strip(" ?.!,"), residual[m.end():].strip(" ?.!,")
    else:
        m = _CONDITION.search(residual)
        if not m:
            return None
        condition = m.group(0).strip(" ?.!,")
        ask = re.sub(r"\s+", " ", (residual[:m.start()] + " " + residual[m.end():])).strip(" ?.!,:-—")
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
    # A preposition left in front of another function word or the end ("prescribed in to patients",
    # "patients in given") once its date was removed
    out = re.sub(r"\b(?:in|on|during|from|at)\s+(?=(?:to|given|for|of|the|visit|patients?|-)\b|[?.!,:]|$)", " ", out)
    out = re.sub(r"^\s*[:\-—,]\s*", "", out)
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
