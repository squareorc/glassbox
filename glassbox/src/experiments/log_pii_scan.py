"""
PII scan of every audit log and trace: does any identifying value reach a log
that the redacted system writes?

    python -m glassbox.src.experiments.log_pii_scan

Three checks, for each record (one JSON line of a log, or one trace file):
    1. full values   every ground-truth PII value of the dataset's notes (names, patient
                     IDs, dates of birth, phones, emails, addresses, SSNs; values of 5+
                     characters), matched as word sequences, ignoring case and punctuation;
    2. questions     the original text of a QA identity question (it names a person);
    3. name parts    for an identity question, the first or last name of the person it is
                     about, as a capitalised word anywhere in that question's record
                     (catches a half-redacted name like "[NAME] Price").

Also scanned: the committed generation outputs (results/generation/*_answers.jsonl),
which are public on GitHub; their raw-baseline records are skipped.

Logs of the raw (no-privacy) baseline contain PII by design: they are scanned and
reported separately, as the positive control that shows the scan can find leaks.

Output: glassbox/results/experiments/log_pii_scan.json
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from glassbox.src.qa.build_qa_set import load_qa_set
from glassbox.src.retrieval.index import DATA_ROOT

LOG_ROOT = Path("glassbox/logs")
RESULTS_GEN = Path("glassbox/results/generation")
OUT = Path("glassbox/results/experiments/log_pii_scan.json")
MIN_LEN = 5
TITLES = {"mr", "mrs", "ms", "miss", "dr", "mx"}


def is_baseline(path: Path) -> bool:
    """Raw-index / no-privacy logs, where PII is expected."""
    parts = set(path.parts)
    return path.stem.startswith("raw") or "raw" in parts


def records(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if path.suffix == ".json":
        rec = json.loads(path.read_text())
        yield json.dumps(rec), rec
    else:
        for line in path.open():
            if line.strip():
                rec = json.loads(line)
                yield line, rec


_TOKEN = re.compile(r"\w+")


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class ValueMatcher:
    """
    Finds PII values as token sequences (case- and punctuation-insensitive), indexed by
    first token: one pass over a record instead of one regex per value. "(555) 867-5309"
    therefore also matches "555 867 5309", which is the intended, stricter behaviour.
    """

    def __init__(self, values):
        self.by_first: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
        for v in values:
            t = tuple(tokens(v))
            if t:
                self.by_first[t[0]].append(t)

    def finditer(self, text: str) -> Iterator[Tuple[str, ...]]:
        toks = tokens(text)
        for i, tok in enumerate(toks):
            for cand in self.by_first.get(tok, ()):
                if tuple(toks[i:i + len(cand)]) == cand:
                    yield cand


def dataset_secrets(dataset: str):
    notes = [json.loads(l) for l in (DATA_ROOT / "raw" / dataset / "synthetic_notes.jsonl").open()]
    values = {e["value"].strip() for n in notes for e in n["entities"] if len(e["value"].strip()) >= MIN_LEN}
    name_of_doc = {n["doc_id"]: n["patient_name"] for n in notes}
    qa = load_qa_set(DATA_ROOT / "raw" / dataset / "qa_eval_set.jsonl")
    identity = {q["qid"]: q for q in qa if q["type"] == "identity"}
    name_parts = {}
    for qid, q in identity.items():
        # Titles ("Mrs. Jane Doe") are kept by design and reported separately (PROGRESS.md section 12)
        parts = {p for d in q["relevant_doc_ids"] for p in re.findall(r"[A-Z][a-z'\-]{2,}", name_of_doc.get(d, ""))
                 if p.lower() not in TITLES}
        name_parts[qid] = [re.compile(rf"(?<![\w\[]){re.escape(p)}(?!\w)") for p in parts]
    return ValueMatcher(values), {q["question"] for q in identity.values()}, name_parts, len(values)


def scan_file(path: Path, matcher, questions, name_parts) -> Dict[str, Any]:
    n, skipped, hits = 0, 0, defaultdict(list)
    for raw, rec in records(path):
        if rec.get("config") == "raw":      # baseline records inside a mixed results file
            skipped += 1
            continue
        n += 1
        qid = rec.get("qid")
        for m in matcher.finditer(raw):
            hits["full_value"].append({"qid": qid, "value_tokens": len(m)})  # never store the value
        if any(q in raw for q in questions):
            hits["identity_question_text"].append({"qid": qid})
        for rx in name_parts.get(qid, []):
            if rx.search(raw):
                hits["name_part"].append({"qid": qid})
                break
    return {"records": n, "skipped_baseline_records": skipped, **{k: len(v) for k, v in hits.items()},
            "examples": {k: v[:5] for k, v in hits.items()}}


def main():
    report: Dict[str, Any] = {"files": {}, "totals": {}}
    for dataset in ("realistic", "synthetic"):
        files = sorted(p for p in LOG_ROOT.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json")
                       and dataset in p.parts)
        # Committed generation outputs (shared on GitHub): raw-config records are skipped per record
        files += sorted(RESULTS_GEN.glob(f"{dataset}_*_answers.jsonl"))
        if not files:
            continue
        matcher, questions, name_parts, n_values = dataset_secrets(dataset)
        # "live": traces of questions asked in the dashboard (logs/traces/live/). Scanned like the
        # rest, but totalled apart so the committed evaluation numbers don't change with every demo.
        totals = {"privacy": defaultdict(int), "baseline": defaultdict(int), "live": defaultdict(int),
                  "live_baseline": defaultdict(int)}
        by_dir: Dict[str, Dict[str, Any]] = {}
        for path in files:
            r = scan_file(path, matcher, questions, name_parts)
            group = ("live_" if "live" in path.parts else "") + ("baseline" if is_baseline(path) else "privacy")
            group = "live" if group == "live_privacy" else group
            key = str(path.parent) if path.suffix == ".json" else str(path)   # traces: one entry per folder
            agg = by_dir.setdefault(key, {"group": group, "files": 0, "records": 0, "full_value": 0,
                                          "identity_question_text": 0, "name_part": 0, "examples": {}})
            agg["files"] += 1
            for k in ("records", "full_value", "identity_question_text", "name_part"):
                agg[k] += r.get(k, 0)
                totals[group][k] += r.get(k, 0)
            for k, ex in r["examples"].items():
                agg["examples"].setdefault(k, []).extend(ex[: 5 - len(agg["examples"].get(k, []))])
        report["files"].update(by_dir)
        report["totals"][dataset] = {"pii_values_searched": n_values, **{g: dict(t) for g, t in totals.items() if t}}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    for key, r in report["files"].items():
        flag = "" if r["group"].endswith("baseline") or not (r["full_value"] or r["identity_question_text"] or r["name_part"]) else "  <-- LEAK"
        print(f"{r['group']:<8} {key:<62} records {r['records']:>6}  full {r['full_value']:>6}  "
              f"question {r['identity_question_text']:>4}  name-part {r['name_part']:>4}{flag}")
    print(json.dumps(report["totals"], indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
