"""
Retrieval evaluation, fusion tuning and index leak audit.

Metrics:
    recall@k     note level: fraction of a question's gold notes that appear
                 among the notes of the top-k chunks
    evidence@k   answer level: fraction of a question's gold answer items
                 (e.g. each medication) that appear in top-k chunks FROM GOLD
                 NOTES - i.e. whether the LLM could actually answer from what
                 it is shown. Stricter than recall@k: a note's symptom chunk
                 counts for recall but not for evidence of its medication.
    MRR          mean of 1 / rank of the first chunk from a gold note
                 (0 if none in the top MAX_K)

Runs: index (redacted / raw) x chunking (section / note) x method
(dense / bm25 / hybrid) x query analysis (off / on), reported per question
type and split. Fusion weights are tuned on the DEV split only (--tune
prints the grid; the chosen default lives in retriever.DEFAULT_WEIGHTS).

Leak audit: for every ground-truth PII entity, whether its value is still
visible in its note's indexed chunks (full value, or any identifying part).

Usage (from the repo root):
    python -m glassbox.src.retrieval.evaluate --dataset realistic
    python -m glassbox.src.retrieval.evaluate --dataset realistic --tune
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from glassbox.src.qa.build_qa_set import answer_item_found, load_qa_set

from .index import DATA_ROOT, index_dir
from .retriever import DEFAULT_WEIGHTS, MAX_K, METHODS, PER_NOTE, Retriever

RESULTS_ROOT = Path("glassbox/results/retrieval")
LOG_ROOT = Path("glassbox/logs/retrieval")
KS = (1, 3, 5, 10)
QTYPES = ("clinical_fact", "identity", "aggregate")
# (dense, bm25) weight grid for tuning: from BM25-only to dense-only
WEIGHT_GRID = ((0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0),
               (3.0, 1.0), (5.0, 1.0), (1.0, 0.0))


# ----------------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------------

def score_question(results: List[Dict[str, Any]], q: Dict[str, Any], auto_k: int = None) -> Dict[str, float]:
    gold = set(q["relevant_doc_ids"])

    def at(k):
        top = results[:k]
        recall = len(gold & {r["doc_id"] for r in top}) / len(gold)
        gold_text = "\n".join(r["text"] for r in top if r["doc_id"] in gold)
        found = sum(answer_item_found(gold_text, kws, q["answer_field"]) for kws in q["answer_keywords"])
        # Data minimisation: how much text the LLM would be shown
        context_chars = sum(len(r["text"]) for r in top)
        return recall, found / len(q["answer_keywords"]), context_chars

    out = {}
    for k in KS:
        out[f"recall@{k}"], out[f"evidence@{k}"], out[f"context_chars@{k}"] = at(k)
    if auto_k:
        out["recall@auto"], out["evidence@auto"], out["context_chars@auto"] = at(auto_k)
    first = next((r["rank"] for r in results if r["doc_id"] in gold), None)
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def _mean(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k == "mrr" or "@" in k]
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys} | {"n": len(rows)}


def bootstrap_ci(values: List[float], n_resamples: int = 5000, seed: int = 0) -> List[float]:
    """95% percentile-bootstrap confidence interval of the mean (questions resampled with replacement)."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples))
    return [round(means[int(0.025 * n_resamples)], 4), round(means[int(0.975 * n_resamples) - 1], 4)]


def summarise(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for split in ("dev", "test", "all"):
        rows = [r for r in per_question if split == "all" or r["split"] == split]
        summary[split] = {
            "overall": _mean(rows),
            # Identity questions are unanswerable under redaction by design;
            # this excludes them to show utility on the answerable ones.
            "answerable": _mean([r for r in rows if r["type"] != "identity"]),
            **{t: _mean([r for r in rows if r["type"] == t]) for t in QTYPES},
            "answerable_evidence@auto_ci95": bootstrap_ci(
                [r["evidence@auto"] for r in rows if r["type"] != "identity"]),
        }
    return summary


def run_questions(retriever, qa_set, method, use_qa, weights=DEFAULT_WEIGHTS, per_note=PER_NOTE, **selection):
    """
    Retrieve MAX_K chunks per question and score every k on prefixes of that
    list (selection is prefix-consistent). "@auto" scores the k query
    analysis would choose (or SINGLE_NOTE_K with query analysis off).
    """
    per_question = []
    for q in qa_set:
        results = retriever.retrieve(q["question"], k=MAX_K, method=method, qid=q["qid"],
                                     use_query_analysis=use_qa, weights=weights, per_note=per_note, **selection)
        auto_k = retriever.auto_k(retriever.last_analysis)
        per_question.append({
            "qid": q["qid"], "type": q["type"], "split": q["split"], "auto_k": auto_k,
            **score_question(results, q, auto_k),
            "filters": retriever.last_analysis["filters"],
            "fallback": retriever.last_analysis["fallback"],
            "retrieved": [r["chunk_id"] for r in results],
        })
    return per_question


# ----------------------------------------------------------------------------
# Fusion weight tuning (dev split only)
# ----------------------------------------------------------------------------

def tune_fusion(dataset: str, threshold: float, qa_set) -> List[Dict[str, Any]]:
    """Grid-search the fusion weights on the redacted section index, dev split, query analysis on."""
    dev = [q for q in qa_set if q["split"] == "dev"]
    retriever = Retriever(index_dir(dataset, "redacted", "section", threshold))
    rows = []
    for wd, wb in WEIGHT_GRID:
        s = summarise(run_questions(retriever, dev, "hybrid", True, (wd, wb)))["dev"]
        rows.append({"dense_weight": wd, "bm25_weight": wb,
                     "answerable_evidence@3": s["answerable"]["evidence@3"],
                     "answerable_evidence@auto": s["answerable"]["evidence@auto"],
                     "fact_evidence@auto": s["clinical_fact"]["evidence@auto"],
                     "aggregate_evidence@auto": s["aggregate"]["evidence@auto"],
                     "answerable_evidence@5": s["answerable"]["evidence@5"],
                     "answerable_recall@5": s["answerable"]["recall@5"],
                     "answerable_mrr": s["answerable"]["mrr"]})
    return rows


# ----------------------------------------------------------------------------
# Selection ablation: each Day 3 retrieval change added in turn
# ----------------------------------------------------------------------------

SELECTION_STEPS = [
    ("Day 2 selection", dict(redact_queries=False, demote_low_info=False, note_ordered=False, two_hop=False)),
    ("+ question redaction", dict(redact_queries=True, demote_low_info=False, note_ordered=False, two_hop=False)),
    ("+ demote low-information chunks", dict(redact_queries=True, demote_low_info=True, note_ordered=False, two_hop=False)),
    ("+ note-ordered grouping", dict(redact_queries=True, demote_low_info=True, note_ordered=True, two_hop=False)),
    ("+ two-hop answer sections (final)", dict(redact_queries=True, demote_low_info=True, note_ordered=True, two_hop=True)),
]


def selection_ablation(dataset: str, threshold: float, qa_set) -> List[Dict[str, Any]]:
    """Redacted section index, hybrid, query analysis on. Steps were chosen on dev; test reported."""
    retriever = Retriever(index_dir(dataset, "redacted", "section", threshold))
    rows = []
    for name, cfg in SELECTION_STEPS:
        cfg = dict(cfg)
        retriever.redact_queries = cfg.pop("redact_queries")
        summary = summarise(run_questions(retriever, qa_set, "hybrid", True, **cfg))
        row = {"step": name}
        for split in ("dev", "test"):
            s = summary[split]
            row[split] = {"answerable_evidence@auto": s["answerable"]["evidence@auto"],
                          "answerable_evidence@auto_ci95": s["answerable_evidence@auto_ci95"],
                          "fact_evidence@auto": s["clinical_fact"]["evidence@auto"],
                          "aggregate_evidence@auto": s["aggregate"]["evidence@auto"],
                          "aggregate_recall@auto": s["aggregate"]["recall@auto"],
                          "context_chars@auto": s["answerable"]["context_chars@auto"],
                          "identity_recall@auto": s["identity"]["recall@auto"]}
        rows.append(row)
    return rows


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
    parser.add_argument("--tune", action="store_true", help="Only run the dev-split fusion weight grid")
    parser.add_argument("--ablate", action="store_true", help="Only run the selection ablation (dev + test)")
    args = parser.parse_args()

    qa_set = load_qa_set(DATA_ROOT / "raw" / args.dataset / "qa_eval_set.jsonl")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.ablate:
        rows = selection_ablation(args.dataset, args.threshold, qa_set)
        print("Selection ablation (redacted section index, hybrid, query analysis on):")
        for r in rows:
            d, t = r["dev"], r["test"]
            print(f"  {r['step']:<36} dev Ev {d['answerable_evidence@auto']:.3f} (agg {d['aggregate_evidence@auto']:.3f}) | "
                  f"test Ev {t['answerable_evidence@auto']:.3f} CI {t['answerable_evidence@auto_ci95']} "
                  f"(fact {t['fact_evidence@auto']:.3f}, agg {t['aggregate_evidence@auto']:.3f}) ctx {t['context_chars@auto']:.0f}")
        with (RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}_selection_ablation.json").open("w") as f:
            json.dump(rows, f, indent=2)
        return

    if args.tune:
        rows = tune_fusion(args.dataset, args.threshold, qa_set)
        print("Fusion weight tuning (redacted section index, query analysis on, DEV split, answerable questions):")
        for r in rows:
            print(f"  dense {r['dense_weight']:<5} bm25 {r['bm25_weight']}  Ev@auto {r['answerable_evidence@auto']:.3f} "
                  f"(fact {r['fact_evidence@auto']:.3f}, agg {r['aggregate_evidence@auto']:.3f})  evidence@3 {r['answerable_evidence@3']:.3f}  evidence@5 {r['answerable_evidence@5']:.3f}  "
                  f"recall@5 {r['answerable_recall@5']:.3f}  MRR {r['answerable_mrr']:.3f}")
        with (RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}_fusion_tuning.json").open("w") as f:
            json.dump(rows, f, indent=2)
        return

    report = {"dataset": args.dataset, "num_questions": len(qa_set), "ks": KS,
              "fusion_weights": list(DEFAULT_WEIGHTS), "runs": {}}
    per_question_all = {}
    for variant in ("redacted", "raw"):
        for chunking in ("section", "note"):
            d = index_dir(args.dataset, variant, chunking, args.threshold)
            log_path = LOG_ROOT / args.dataset / f"{d.name}.jsonl"
            log_path.unlink(missing_ok=True)
            retriever = Retriever(d, baseline=(variant == "raw"), log_path=log_path)
            for use_qa in (False, True):
                for method in METHODS:
                    run = f"{d.name}/{method}/{'qa' if use_qa else 'noqa'}"
                    pq = run_questions(retriever, qa_set, method, use_qa)
                    report["runs"][run] = summarise(pq)
                    per_question_all[run] = pq
                    t = report["runs"][run]["test"]
                    a = t["answerable"]
                    print(f"{run:<36} test@auto answerable R {a['recall@auto']:.3f} Ev {a['evidence@auto']:.3f} "
                          f"ctx {a['context_chars@auto']:>5.0f}ch | fact Ev {t['clinical_fact']['evidence@auto']:.3f} "
                          f"| agg R {t['aggregate']['recall@auto']:.3f} Ev {t['aggregate']['evidence@auto']:.3f} "
                          f"| identity R {t['identity']['recall@auto']:.3f}")

    report["leak_audit"] = {
        v: leak_audit(args.dataset, index_dir(args.dataset, v, "section", args.threshold))
        for v in ("redacted", "raw")
    }
    print("\nLeak audit (section index):")
    for v, a in report["leak_audit"].items():
        print(f"  {a['index']}: overall full {a['overall']['full_leak_rate']:.3f} partial {a['overall']['partial_leak_rate']:.3f}")

    out = RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    with (RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}_per_question.json").open("w") as f:
        json.dump(per_question_all, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
