"""
Corrupted-claim test set for the verifier, with exact ground truth.

Source: the GlassBox system's own claims (redacted config, generation run of
Day 3). A claim is GENUINE if it cites a chunk of a gold note, contains a gold
answer item found in that cited text, and every drug name and every number in
it appears in the cited text. Genuine claims are expected SUPPORTED.

From each genuine claim, every applicable corruption is made (expected: not
SUPPORTED; the "ideal" label is what a perfect verifier would say):

    drug_swap             a drug replaced by one absent from the evidence     ideal CONTRADICTED
    dose_change           a dose number changed ("25mg" -> "50mg")            ideal CONTRADICTED
    number_change         a vital/follow-up number changed (HR, BP, weeks)    ideal CONTRADICTED
    negation              "was prescribed" -> "was not prescribed"            ideal CONTRADICTED
    symptom_swap          the gold symptom replaced by another                ideal CONTRADICTED
    unsupported_addition  an invented extra drug appended                     ideal UNSUPPORTED
    wrong_citation        evidence replaced by the same section of another
                          note (same diagnosis) that lacks the claim's item   ideal UNSUPPORTED

Each item stores its premise text(s), so the set is self-contained and frozen.
Splits follow the source question's dev/test split.

    python -m glassbox.src.verification.corruptions
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from glassbox.src.qa.build_qa_set import answer_item_found
from glassbox.src.redaction.clinical_terms import DRUGS
from glassbox.src.retrieval.index import index_dir

ANSWERS = Path("glassbox/results/generation/realistic_t0.50_answers.jsonl")
QA = Path("glassbox/data/raw/realistic/qa_eval_set.jsonl")
OUT = Path("glassbox/data/verification/corrupted_claims.jsonl")

REPLACEMENT_DRUGS = ["Warfarin", "Digoxin", "Furosemide", "Levothyroxine", "Allopurinol", "Amoxicillin",
                     "Prednisone", "Sertraline", "Metformin", "Lisinopril", "Gabapentin", "Omeprazole"]
SYMPTOMS = ["chest pain", "headache", "fatigue", "dizziness", "nausea", "joint pain", "cough", "wheezing",
            "edema", "palpitations", "insomnia", "back pain"]
_DOSE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|units|g|ml)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w./])(\d{1,3})(?:/(\d{1,3}))?(?![\w/])")
_AUX = re.compile(r"\b(was|were|is|are|has been|have been)\b", re.IGNORECASE)


def _drugs_in(text):
    return [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() in DRUGS]


def _numbers_in(text):
    return re.findall(r"\d+(?:\.\d+)?", text)


def genuine(claim, cited_texts, q):
    if not cited_texts:
        return False
    evidence = " ".join(cited_texts).lower()
    items = [kw for kw in q["answer_keywords"] if answer_item_found(claim, kw, q["answer_field"])
             and answer_item_found(evidence, kw, q["answer_field"])]
    if not items:
        return False
    return all(d.lower() in evidence for d in _drugs_in(claim)) and all(n in evidence for n in _numbers_in(claim))


def asserts_uncited_condition(claim, evidence, q):
    """
    Does a genuine claim also restate the question's condition (symptom / month) when the
    cited evidence doesn't show it? E.g. "Tiotropium was prescribed to the COPD patients who
    presented with dyspnea in September 2024" citing only the PLAN section. A verifier is
    RIGHT to call such a claim unsupported by its citation: the generator cited incompletely.
    Kept as a flag (not a relabel) so results are reported both ways.
    """
    m = re.search(r"presented with (.+?) in ([A-Za-z]+) (\d{4})", q["question"])
    if not m:
        return False
    c, e = claim.lower(), evidence.lower()
    return any(part.lower() in c and part.lower() not in e for part in (m.group(1), m.group(2)))


def corruptions(claim, evidence, q, rng):
    out = []
    low = evidence.lower()
    drugs = _drugs_in(claim)
    if drugs:
        new = rng.choice([d for d in REPLACEMENT_DRUGS if d.lower() not in low])
        out.append(("drug_swap", claim.replace(drugs[0], new, 1), "CONTRADICTED"))
    m = _DOSE.search(claim)
    if m:
        n = float(m.group(1))
        new = n * 2 if f"{n * 2:g}" not in low else n + 10
        out.append(("dose_change", claim[:m.start(1)] + f"{new:g}" + claim[m.end(1):], "CONTRADICTED"))
    if q["answer_field"] in ("hr", "bp", "followup_weeks"):
        for m in _NUMBER.finditer(claim):
            if _DOSE.match(claim, m.start()):
                continue
            if m.group(2):
                rep = f"{int(m.group(1)) + 12}/{int(m.group(2)) + 9}"
            else:
                rep = str(int(m.group(1)) + (13 if q["answer_field"] == "hr" else 3))
            out.append(("number_change", claim[:m.start()] + rep + claim[m.end():], "CONTRADICTED"))
            break
    m = _AUX.search(claim)
    if m:
        out.append(("negation", claim[:m.end()] + " not" + claim[m.end():], "CONTRADICTED"))
    if q["answer_field"] == "symptom":
        gold = next((a for a in q["answer"] if a.lower() in claim.lower()), None)
        if gold:
            new = rng.choice([s for s in SYMPTOMS if s not in low and s != gold.lower()])
            out.append(("symptom_swap", re.sub(re.escape(gold), new, claim, count=1, flags=re.IGNORECASE),
                        "CONTRADICTED"))
    extra = rng.choice([d for d in REPLACEMENT_DRUGS if d.lower() not in low])
    out.append(("unsupported_addition", claim.rstrip(". ") + f". The patient was also started on {extra} 5mg PO daily.",
                "UNSUPPORTED"))
    return out


def build(seed: int = 7):
    qa = {q["qid"]: q for q in map(json.loads, QA.open())}
    with (index_dir("realistic", "redacted", "section") / "chunks.jsonl").open() as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    by_section = {}
    for c in chunks.values():
        by_section.setdefault((c["section"], c["diagnosis"]), []).append(c)
    rng = random.Random(seed)
    items, n, seen = [], 0, set()
    for r in map(json.loads, ANSWERS.open()):
        if r["config"] != "redacted" or r["type"] == "identity":
            continue
        q = qa[r["qid"]]
        for claim in r["record"]["claims"]:
            cited = [c for c in claim["citations"] if c.split("#")[0] in q["relevant_doc_ids"] and c in chunks]
            texts = [chunks[c]["text"] for c in cited]
            if not genuine(claim["text"], texts, q) or (claim["text"], tuple(texts)) in seen:
                continue
            seen.add((claim["text"], tuple(texts)))   # the same claim + evidence counted once
            base = {"qid": q["qid"], "split": q["split"], "answer_field": q["answer_field"], "source_claim": claim["text"],
                    "asserts_uncited_condition": asserts_uncited_condition(claim["text"], " ".join(texts), q)}
            items.append({**base, "id": f"vc{n:04d}", "kind": "genuine", "claim": claim["text"], "premises": texts,
                          "expected": "SUPPORTED", "ideal": "SUPPORTED"}); n += 1
            for kind, text, ideal in corruptions(claim["text"], " ".join(texts), q, rng):
                items.append({**base, "id": f"vc{n:04d}", "kind": kind, "claim": text, "premises": texts,
                              "expected": "NOT_SUPPORTED", "ideal": ideal}); n += 1
            # wrong citation: same section type and diagnosis, another note, lacking the claim's items
            first = chunks[cited[0]]
            pool = [c for c in by_section[(first["section"], first["diagnosis"])]
                    if c["doc_id"] not in q["relevant_doc_ids"]
                    and not any(answer_item_found(c["text"], kw, q["answer_field"]) for kw in q["answer_keywords"])]
            if pool:
                other = rng.choice(pool)
                items.append({**base, "id": f"vc{n:04d}", "kind": "wrong_citation", "claim": claim["text"],
                              "premises": [other["text"]], "expected": "NOT_SUPPORTED", "ideal": "UNSUPPORTED"}); n += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"{len(items)} items -> {OUT}")
    print(Counter((it["split"], it["kind"]) for it in items))


if __name__ == "__main__":
    build()
