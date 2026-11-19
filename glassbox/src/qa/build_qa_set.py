"""
QA evaluation set builder.

Generates three kinds of questions from the generators' per-note `facts`,
each with a gold answer and a small set of gold relevant documents:

- clinical_fact: a single note, identified only by NON-identifying details
  (diagnosis + note date). The answer is clinical content (medication,
  symptom, vitals, labs, follow-up), which redaction should NOT remove, so
  utility should survive redaction.

- identity: a single note, identified by the patient's NAME or PATIENT_ID.
  Under redaction the identifier is gone from the index, so the correct
  behaviour is to abstain. This measures the intended privacy cost directly,
  and any confident answer is a privacy-induced hallucination the verifier
  should catch.

- aggregate: 2-4 notes sharing (diagnosis, symptom, time period). Tests
  multi-document retrieval and synthesis.

Every answer item carries `answer_keywords`: lower-cased strings that must ALL
appear in a generated answer for that item to count as correct. This gives a
deterministic, model-free accuracy metric.

Usage (rebuild from an existing notes file, from the repo root):
    python -m glassbox.src.qa.build_qa_set \\
        --notes glassbox/data/raw/realistic/synthetic_notes.jsonl --profile realistic
"""

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

# How many questions of each type per dataset profile, and the time
# granularity used to form aggregate groups (finer for the larger corpus so
# groups stay at 2-4 documents).
PROFILES = {
    "realistic": {"clinical_fact": 60, "identity": 40, "aggregate": 30, "period": "month"},
    "synthetic": {"clinical_fact": 40, "identity": 25, "aggregate": 15, "period": "year"},
}

DEV_FRACTION = 0.3  # dev split is for tuning (e.g. NLI thresholds); report on test
AGGREGATE_GROUP_SIZE = (2, 4)

# Clinical-fact answer fields -> question template. {ref} is the
# non-identifying reference to the note ("the Asthma patient whose note is
# dated 2025-04-02").
CLINICAL_FIELDS = {
    "medication": "What medication was prescribed for {ref}?",
    "symptom": "What symptom or complaint was documented for {ref}?",
    "followup_weeks": "In how many weeks is follow-up planned for {ref}?",
    "hr": "What heart rate was recorded for {ref}?",
    "bp": "What blood pressure was recorded for {ref}?",
    "lab_orders": "Which labs were ordered for {ref}?",
}

IDENTITY_QUESTIONS = [
    # (identifier field, answer field, template)
    ("name", "medication", "What medication is {ident} prescribed?"),
    ("name", "diagnosis", "What condition is {ident} being treated for?"),
    ("name", "symptom", "What symptom did {ident} report?"),
    ("patient_id", "diagnosis", "What is the diagnosis recorded for patient {ident}?"),
    ("patient_id", "medication", "What medication is patient {ident} on?"),
]


# ----------------------------------------------------------------------------
# Answer normalisation / keywords
# ----------------------------------------------------------------------------

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|units|g)\b", re.IGNORECASE)


def medication_keywords(med: str) -> List[str]:
    """Drug name (first word) + dose if present, e.g. 'Metformin 500mg PO BID' -> ['metformin', '500mg']."""
    keywords = [med.split()[0].lower()]
    dose = _DOSE_RE.search(med)
    if dose:
        keywords.append(f"{dose.group(1)}{dose.group(2).lower()}".replace("units", " units"))
    return keywords


def answer_items(field: str, doc: Dict[str, Any]) -> List[str]:
    """Gold answer items for a field of a document."""
    facts = doc["facts"]
    if field == "medication":
        return list(doc["medications"])
    if field == "lab_orders":
        return [x.strip() for x in facts["lab_orders"].split(",")]
    if field == "followup_weeks":
        return [f"{facts['followup_weeks']} weeks"]
    return [str(facts[field])]


def item_keywords(field: str, item: str) -> List[str]:
    if field == "medication":
        return medication_keywords(item)
    if field == "followup_weeks":
        return [item.split()[0]]  # the number; matcher checks it near "week"
    return [item.lower()]


def answer_item_found(text: str, keywords: List[str], field: str) -> bool:
    """
    Does `text` contain one gold answer item? Shared by retrieval evidence
    recall and generation answer scoring.

    - All keywords must appear (case-insensitive).
    - Pure numbers match only as standalone numbers, so HR "94" doesn't match
      inside "BP 157/94".
    - Follow-up weeks must appear as "<n> week(s)".
    """
    t = text.lower()
    for kw in keywords:
        kw = kw.lower()
        if field == "followup_weeks":
            if not re.search(rf"(?<![\d/.]){re.escape(kw)}\s*-?\s*weeks?\b", t):
                return False
        elif kw.replace(".", "", 1).isdigit():
            if not re.search(rf"(?<![\d/.]){re.escape(kw)}(?![\d/]|\.\d)", t):
                return False
        elif kw not in t:
            return False
    return True


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------

def _ref(doc: Dict[str, Any]) -> str:
    return f"the {doc['diagnosis']} patient whose note is dated {doc['facts']['visit_date']}"


def _period(visit_date: str, granularity: str) -> str:
    d = date.fromisoformat(visit_date)
    return d.strftime("%B %Y") if granularity == "month" else str(d.year)


def _record(qtype, question, field, docs, items, requires_identifier, extra=None):
    rec = {
        "type": qtype,
        "question": question,
        "answer_field": field,
        "answer": items,
        "answer_keywords": [item_keywords(field, i) for i in items],
        "relevant_doc_ids": [d["doc_id"] for d in docs],
        "requires_identifier": requires_identifier,
    }
    if extra:
        rec.update(extra)
    return rec


def build_clinical_fact(documents, n, rng) -> List[Dict[str, Any]]:
    # The (diagnosis, date) reference must pick out exactly one note.
    key_counts = defaultdict(int)
    for d in documents:
        key_counts[(d["diagnosis"], d["facts"]["visit_date"])] += 1
    eligible = [d for d in documents if key_counts[(d["diagnosis"], d["facts"]["visit_date"])] == 1]
    rng.shuffle(eligible)

    fields = list(CLINICAL_FIELDS)
    out, i = [], 0
    for doc in eligible:
        if len(out) >= n:
            break
        # Round-robin over fields so the set is balanced; skip fields the
        # note's template didn't render.
        for _ in range(len(fields)):
            field = fields[i % len(fields)]
            i += 1
            source_key = "medication" if field == "medication" else field
            if source_key in doc["fields_in_text"]:
                out.append(_record(
                    "clinical_fact",
                    CLINICAL_FIELDS[field].format(ref=_ref(doc)),
                    field, [doc], answer_items(field, doc), False,
                ))
                break
    return out


def build_identity(documents, n, rng) -> List[Dict[str, Any]]:
    # Identifiers must be unique across ALL person names in the corpus
    # (patients and providers), otherwise the gold doc is ambiguous.
    name_counts, id_counts = defaultdict(int), defaultdict(int)
    for d in documents:
        for e in d["entities"]:
            if e["type"] == "NAME":
                name_counts[e["value"]] += 1
        id_counts[d["facts"]["patient_id"]] += 1

    def unique(doc, ident_field):
        v = doc["facts"][ident_field]
        if ident_field == "name":
            # A patient name appears once per note in the templates; >1 means
            # it collides with another person somewhere in the corpus.
            return name_counts[v] == sum(1 for e in doc["entities"] if e["value"] == v)
        return id_counts[v] == 1

    docs = list(documents)
    rng.shuffle(docs)
    out, i = [], 0
    for doc in docs:
        if len(out) >= n:
            break
        for _ in range(len(IDENTITY_QUESTIONS)):
            ident_field, field, tmpl = IDENTITY_QUESTIONS[i % len(IDENTITY_QUESTIONS)]
            i += 1
            if ident_field in doc["fields_in_text"] and field in doc["fields_in_text"] and unique(doc, ident_field):
                out.append(_record(
                    "identity",
                    tmpl.format(ident=doc["facts"][ident_field]),
                    field, [doc], answer_items(field, doc), True,
                    extra={"identifier_field": ident_field},
                ))
                break
    return out


def build_aggregate(documents, n, rng, granularity) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for d in documents:
        if "symptom" not in d["fields_in_text"]:
            continue
        key = (d["diagnosis"], d["facts"]["symptom"], _period(d["facts"]["visit_date"], granularity))
        groups[key].append(d)
    lo, hi = AGGREGATE_GROUP_SIZE
    eligible = sorted(k for k, v in groups.items() if lo <= len(v) <= hi)
    rng.shuffle(eligible)

    out = []
    for dx, symptom, period in eligible[:n]:
        docs = groups[(dx, symptom, period)]
        items = sorted({m for d in docs for m in d["medications"]})
        out.append(_record(
            "aggregate",
            f"Which medications were prescribed to {dx} patients who presented with {symptom} in {period}?",
            "medication", docs, items, False,
        ))
    return out


def build_qa_set(documents: List[Dict[str, Any]], profile: str = "realistic", seed: int = 1234) -> List[Dict[str, Any]]:
    """Build the full QA set for a dataset profile ('realistic' or 'synthetic')."""
    cfg = PROFILES[profile]
    rng = random.Random(seed)

    by_type = {
        "clinical_fact": build_clinical_fact(documents, cfg["clinical_fact"], rng),
        "identity": build_identity(documents, cfg["identity"], rng),
        "aggregate": build_aggregate(documents, cfg["aggregate"], rng, cfg["period"]),
    }

    qa_set = []
    abbrev = {"clinical_fact": "cf", "identity": "id", "aggregate": "ag"}
    for qtype, items in by_type.items():
        n_dev = round(len(items) * DEV_FRACTION)
        for i, rec in enumerate(items):
            rec["qid"] = f"{profile}-{abbrev[qtype]}-{i:03d}"
            rec["split"] = "dev" if i < n_dev else "test"
            qa_set.append({"qid": rec.pop("qid"), **rec})
    return qa_set


def write_qa_set(qa_set: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in qa_set:
            f.write(json.dumps(rec) + "\n")


def load_qa_set(path: Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
    with Path(path).open() as f:
        recs = [json.loads(line) for line in f]
    return [r for r in recs if split is None or r["split"] == split]


def main():
    parser = argparse.ArgumentParser(description="Build the GlassBox QA eval set from a notes file")
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--profile", choices=list(PROFILES), required=True)
    parser.add_argument("--output", type=Path, default=None, help="Default: qa_eval_set.jsonl next to --notes")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    with args.notes.open() as f:
        documents = [json.loads(line) for line in f]
    qa_set = build_qa_set(documents, args.profile, args.seed)
    out = args.output or args.notes.parent / "qa_eval_set.jsonl"
    write_qa_set(qa_set, out)

    counts = defaultdict(int)
    for r in qa_set:
        counts[(r["type"], r["split"])] += 1
    print(f"Wrote {len(qa_set)} questions to {out}")
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:<14} {k[1]:<5} {v}")


if __name__ == "__main__":
    main()
