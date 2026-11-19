"""
Retrieval evaluation and index leak audit.

Retrieval metrics (note-level: a retrieved chunk counts for the note it came
from):
    recall@k  fraction of a question's gold notes that appear among the
              notes of the top-k chunks, averaged over questions
    MRR       mean of 1 / rank of the first chunk from a gold note
              (0 if none in the top MAX_K)

Reported per index (redacted / raw), chunking (section / note), method
(dense / bm25 / hybrid), question type and split.

Leak audit:
    For every ground-truth PII entity, checks whether its value is still
    visible in the chunks of its note in an index:
        full_leak     the whole value appears verbatim
        partial_leak  any identifying component appears (name parts, address
                      parts such as street, city, ZIP) - an upper bound

Usage (from the repo root):
    python -m glassbox.src.retrieval.evaluate --dataset realistic
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from glassbox.src.qa.build_qa_set import load_qa_set

from .index import DATA_ROOT, index_dir
from .retriever import MAX_K, METHODS, Retriever

RESULTS_ROOT = Path("glassbox/results/retrieval")
LOG_ROOT = Path("glassbox/logs/retrieval")
KS = (1, 3, 5, 10)


# ----------------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------------

def score_question(results: List[Dict[str, Any]], gold: List[str]) -> Dict[str, float]:
    gold = set(gold)
    out = {}
    for k in KS:
        retrieved = {r["doc_id"] for r in results[:k]}
        out[f"recall@{k}"] = len(gold & retrieved) / len(gold)
    first = next((r["rank"] for r in results if r["doc_id"] in gold), None)
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in rows[0]} | {"n": len(rows)}


def evaluate_index(retriever: Retriever, qa_set, method: str) -> Dict[str, Any]:
    per_question = []
    for q in qa_set:
        results = retriever.retrieve(q["question"], k=MAX_K, method=method, qid=q["qid"])
        m = score_question(results, q["relevant_doc_ids"])
        per_question.append({"qid": q["qid"], "type": q["type"], "split": q["split"], **m,
                             "retrieved_doc_ids": [r["doc_id"] for r in results]})

    def group(filter_fn):
        return mean_metrics([{k: v for k, v in r.items() if k in ("mrr",) or k.startswith("recall@")}
                             for r in per_question if filter_fn(r)])

    summary = {}
    for split in ("dev", "test", "all"):
        in_split = (lambda r, s=split: s == "all" or r["split"] == s)
        summary[split] = {"overall": group(in_split)}
        for qtype in ("clinical_fact", "identity", "aggregate"):
            summary[split][qtype] = group(lambda r, s=in_split, t=qtype: s(r) and r["type"] == t)
    return {"summary": summary, "per_question": per_question}


# ----------------------------------------------------------------------------
# Leak audit
# ----------------------------------------------------------------------------

_NAME_AFFIXES = {"dr", "mr", "mrs", "ms", "miss", "md", "dds", "phd", "dvm", "jr", "sr", "ii", "iii", "iv"}


def pii_components(entity_type: str, value: str) -> List[str]:
    """Identifying sub-parts of a PII value whose visibility counts as a partial leak."""
    if entity_type == "NAME":
        parts = [p.strip(".,") for p in value.split()]
        return [p for p in parts if len(p) >= 3 and p.lower() not in _NAME_AFFIXES]
    if entity_type == "ADDRESS":
        parts = []
        for piece in value.split(","):
            piece = piece.strip()
            # "CT 47553" -> keep the ZIP; drop bare 2-letter state codes (too generic)
            zip_match = re.search(r"\b\d{5}\b", piece)
            if zip_match:
                parts.append(zip_match.group(0))
            elif len(piece) >= 4:
                parts.append(piece)
        return parts
    return [value]


def leak_audit(dataset: str, idx_dir: Path) -> Dict[str, Any]:
    with (DATA_ROOT / "raw" / dataset / "synthetic_notes.jsonl").open() as f:
        raw_docs = {d["doc_id"]: d for d in map(json.loads, f)}

    indexed_text = defaultdict(str)
    with (idx_dir / "chunks.jsonl").open() as f:
        for c in map(json.loads, f):
            indexed_text[c["doc_id"]] += "\n" + c["text"]

    stats = defaultdict(lambda: {"n": 0, "full_leak": 0, "partial_leak": 0})
    examples = defaultdict(list)
    for doc_id, doc in raw_docs.items():
        text = indexed_text[doc_id]
        for e in doc["entities"]:
            s = stats[e["type"]]
            s["n"] += 1
            full = e["value"] in text
            visible = [p for p in pii_components(e["type"], e["value"]) if p in text]
            s["full_leak"] += full
            s["partial_leak"] += bool(full or visible)
            if (full or visible) and len(examples[e["type"]]) < 5:
                examples[e["type"]].append({"doc_id": doc_id, "value": e["value"], "visible": visible})

    def rates(s):
        return {"n": s["n"],
                "full_leak_rate": round(s["full_leak"] / s["n"], 4) if s["n"] else 0,
                "partial_leak_rate": round(s["partial_leak"] / s["n"], 4) if s["n"] else 0}

    total = {k: sum(s[k] for s in stats.values()) for k in ("n", "full_leak", "partial_leak")}
    return {"index": idx_dir.name, "overall": rates(total),
            "per_type": {t: rates(s) for t, s in sorted(stats.items())},
            "examples": dict(examples)}


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GlassBox retrieval")
    parser.add_argument("--dataset", choices=["synthetic", "realistic"], default="realistic")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    qa_set = load_qa_set(DATA_ROOT / "raw" / args.dataset / "qa_eval_set.jsonl")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    report = {"dataset": args.dataset, "num_questions": len(qa_set), "ks": KS, "runs": {}}
    per_question_all = {}
    for variant in ("redacted", "raw"):
        for chunking in ("section", "note"):
            d = index_dir(args.dataset, variant, chunking, args.threshold)
            log_path = LOG_ROOT / args.dataset / f"{d.name}.jsonl"
            log_path.unlink(missing_ok=True)
            retriever = Retriever(d, baseline=(variant == "raw"), log_path=log_path)
            for method in METHODS:
                run = f"{d.name}/{method}"
                res = evaluate_index(retriever, qa_set, method)
                report["runs"][run] = res["summary"]
                per_question_all[run] = res["per_question"]
                t = res["summary"]["test"]
                print(f"{run:<32} test  R@5 {t['overall']['recall@5']:.3f}  MRR {t['overall']['mrr']:.3f}  |  "
                      f"fact {t['clinical_fact']['recall@5']:.3f}  identity {t['identity']['recall@5']:.3f}  "
                      f"aggregate {t['aggregate']['recall@5']:.3f}")

    report["leak_audit"] = {
        v: leak_audit(args.dataset, index_dir(args.dataset, v, "section", args.threshold))
        for v in ("redacted", "raw")
    }
    print("\nLeak audit (section index):")
    for v, a in report["leak_audit"].items():
        print(f"  {a['index']}: overall full {a['overall']['full_leak_rate']:.3f} partial {a['overall']['partial_leak_rate']:.3f}")
        for t, r in a["per_type"].items():
            print(f"    {t:<14} full {r['full_leak_rate']:.3f}  partial {r['partial_leak_rate']:.3f}  (n={r['n']})")

    out = RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    with (RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}_per_question.json").open("w") as f:
        json.dump(per_question_all, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
