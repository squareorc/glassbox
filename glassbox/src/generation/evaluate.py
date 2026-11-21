"""
Evaluate cited generation on the QA set.

    python -m glassbox.src.generation.evaluate --dataset realistic --split dev
    python -m glassbox.src.generation.evaluate --dataset realistic            # all configs, all splits

Configurations:
    redacted         - the GlassBox system: redacted index, question redaction, abstain guard
    raw              - no-privacy baseline: raw index, no question redaction (guard still
                       applies to retrieval fallbacks)
    redacted_noguard - redacted index with the guard OFF, run on identity questions only
                       (on every other question it is identical to `redacted`, because the
                       guard never fires there). Shows what the model does on its own.

Per-question metrics:
    grounded_accuracy  HEADLINE. Share of gold answer items that are in the answer AND in a
                    chunk the answer cites from a gold note. Added after the robustness check
                    found plain accuracy credits right items for the wrong reason (an answer
                    listing four drugs from four wrong notes scored 1.0 because one drug
                    coincided with the gold answer).
    accuracy        share of gold answer items anywhere in the answer text (answer + claims),
                    matched with qa.build_qa_set.answer_item_found, the same matcher as
                    retrieval. Secondary: it rewards listing everything.
    all_correct     every gold item found (plain matching)
    med_precision / med_recall / med_f1
                    medication questions: drugs named in the answer vs. drugs in the gold
                    answer (precision catches drugs from notes that don't meet the question's
                    conditions); drug names from redaction.clinical_terms.DRUGS
    abstained       the system refused (guard or model)
    citation checks share of claims with a valid citation; share of cited chunks that come
                    from a gold note; share of found gold items whose keywords appear in a
                    cited chunk ("cited support", a cheap proxy until the Day 4 NLI verifier)
    json            parsed on the first attempt / after the retry

Outputs (glassbox/results/generation/):
    <ds>_t0.50.json                  summaries per config, split and question type, with
                                     bootstrap 95% CIs for answerable accuracy
    <ds>_t0.50_answers.jsonl         every generated record (answer, claims, citations,
                                     raw output, scores): the Day 4 verifier's input
Logs: glassbox/logs/generation/<ds>/<config>.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from glassbox.src.qa.build_qa_set import answer_item_found, load_qa_set
from glassbox.src.redaction.clinical_terms import DRUGS
from glassbox.src.retrieval.evaluate import bootstrap_ci
from glassbox.src.retrieval.index import DATA_ROOT, index_dir
from glassbox.src.retrieval.retriever import Retriever

from .generator import Generator

RESULTS_ROOT = Path("glassbox/results/generation")
LOG_ROOT = Path("glassbox/logs/generation")
CONFIGS = ("redacted", "raw", "redacted_noguard")
QTYPES = ("clinical_fact", "aggregate", "identity")


def drugs_in(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w in DRUGS}


def score(record: Dict[str, Any], q: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = " ".join([record["answer"]] + [c["text"] for c in record["claims"]])
    found = [answer_item_found(text, kw, q["answer_field"]) for kw in q["answer_keywords"]]
    text_of = {c["chunk_id"]: c["text"] for c in chunks}
    gold_cited = " ".join(text_of.get(cid, "") for c in record["claims"] for cid in c["citations"]
                          if cid.split("#")[0] in q["relevant_doc_ids"])
    grounded = [f and answer_item_found(gold_cited, kw, q["answer_field"])
                for f, kw in zip(found, q["answer_keywords"])]
    s: Dict[str, Any] = {
        # float(): with one gold item the mean of one bool is a bool, which summaries would skip
        "grounded_accuracy": float(mean(grounded)) if grounded else 0.0,
        "accuracy": float(mean(found)) if found else 0.0,
        "all_correct": float(all(found)),
        "abstained": float(record["abstained"]),
        "abstain_reason": record["abstain_reason"],
        "ok": float(record["ok"]),
        "json_first_try": float(record["model_called"] and not record["errors"]),
        "n_claims": len(record["claims"]),
    }
    if q["answer_field"] == "medication" and not record["abstained"]:
        named = drugs_in(text)
        gold = {d for a in q["answer"] for d in drugs_in(a)}
        p = len(named & gold) / len(named) if named else 0.0
        r = len(named & gold) / len(gold) if gold else 0.0
        s.update(med_precision=p, med_recall=r, med_f1=2 * p * r / (p + r) if p + r else 0.0)
    if record["claims"]:
        cited = [cid for c in record["claims"] for cid in c["citations"]]
        s["claims_cited"] = mean(0.0 if c["uncited"] else 1.0 for c in record["claims"])
        s["invalid_citations"] = sum(len(c["invalid_citations"]) for c in record["claims"])
        if cited:
            s["cited_from_gold_note"] = float(mean(cid.split("#")[0] in q["relevant_doc_ids"] for cid in cited))
            cited_text = " ".join(text_of.get(cid, "") for cid in set(cited))
            hits = [answer_item_found(cited_text, kw, q["answer_field"])
                    for kw, f in zip(q["answer_keywords"], found) if f]
            if hits:
                s["cited_support"] = float(mean(hits))
    return s


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(rs):
        if not rs:
            return {}
        keys = {k for r in rs for k, v in r.items() if isinstance(v, float)}
        out = {k: round(mean(r[k] for r in rs if k in r), 4) for k in sorted(keys)}
        out["n"] = len(rs)
        out["latency_ms"] = round(mean(r["latency_ms"] for r in rs), 1)
        return out

    summary = {}
    for split in ("dev", "test", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        if not rs:
            continue
        answerable = [r for r in rs if r["type"] != "identity"]
        summary[split] = {
            "answerable": agg(answerable),
            "answerable_grounded_accuracy_ci95": bootstrap_ci([r["grounded_accuracy"] for r in answerable]),
            "answerable_accuracy_ci95": bootstrap_ci([r["accuracy"] for r in answerable]),
            **{t: agg([r for r in rs if r["type"] == t]) for t in QTYPES},
        }
        ident = [r for r in rs if r["type"] == "identity"]
        if ident:
            summary[split]["identity"]["false_attribution"] = round(
                mean(1.0 - r["abstained"] for r in ident), 4)  # answered a question about a person
    return summary


def run_config(config: str, dataset: str, threshold: float, qa_set, generator_log: Path) -> List[Dict[str, Any]]:
    raw = config == "raw"
    retriever = Retriever(index_dir(dataset, "raw" if raw else "redacted", "section", threshold), baseline=raw)
    generator_log.unlink(missing_ok=True)
    gen = Generator(log_path=generator_log)
    questions = [q for q in qa_set if config != "redacted_noguard" or q["type"] == "identity"]
    rows = []
    for n, q in enumerate(questions, 1):
        chunks = retriever.retrieve(q["question"], qid=q["qid"])
        question = retriever.last_query_redaction["text"] if retriever.last_query_redaction else q["question"]
        rec = gen.answer(question, chunks, retriever.last_analysis, retriever.last_query_redaction,
                         retriever.manifest["source"], qid=q["qid"], use_guard=(config != "redacted_noguard"))
        s = score(rec, q, chunks)
        rows.append({"config": config, "qid": q["qid"], "type": q["type"], "split": q["split"],
                     "latency_ms": rec["latency_ms"], **s, "record": rec})
        if n % 25 == 0 or n == len(questions):
            done = [r for r in rows if r["type"] != "identity"]
            ident = [r for r in rows if r["type"] == "identity"]
            acc = f"{mean(r['grounded_accuracy'] for r in done):.3f}" if done else "n/a (identity-only run)"
            ans = f"{mean(1 - r['abstained'] for r in ident):.3f}" if ident else "n/a"
            print(f"  [{config}] {n}/{len(questions)}  answerable grounded accuracy so far {acc} | "
                  f"identity questions answered {ans}", flush=True)
    return rows


def rescore_config(config: str, dataset: str, threshold: float, qa_set, generator_log: Path) -> List[Dict[str, Any]]:
    """Score saved generation records again (e.g. after a metric change), without calling the model."""
    variant = "raw" if config == "raw" else "redacted"
    with (index_dir(dataset, variant, "section", threshold) / "chunks.jsonl").open() as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    by_qid = {q["qid"]: q for q in qa_set}
    rows = []
    for rec in map(json.loads, generator_log.open()):
        q = by_qid.get(rec["qid"])
        if q is None:
            continue
        s = score(rec, q, [chunks[c] for c in rec["chunk_ids"]])
        rows.append({"config": config, "qid": q["qid"], "type": q["type"], "split": q["split"],
                     "latency_ms": rec["latency_ms"], **s, "record": rec})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate GlassBox cited generation")
    parser.add_argument("--dataset", choices=["synthetic", "realistic"], default="realistic")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="First N questions (quick checks)")
    parser.add_argument("--tag", default="", help="Suffix for output files (e.g. prompt experiments)")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score the saved generation logs with the current metrics (no model calls)")
    args = parser.parse_args()

    qa_set = load_qa_set(DATA_ROOT / "raw" / args.dataset / "qa_eval_set.jsonl",
                         None if args.split == "all" else args.split)[: args.limit]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_t{args.threshold:.2f}{'_' + args.split if args.split != 'all' else ''}{args.tag}"

    report = {"dataset": args.dataset, "split": args.split, "num_questions": len(qa_set), "configs": {}}
    all_rows = []
    for config in args.configs:
        run = rescore_config if args.rescore else run_config
        rows = run(config, args.dataset, args.threshold, qa_set, LOG_ROOT / args.dataset / f"{config}{args.tag}.jsonl")
        all_rows += rows
        report["configs"][config] = summarise(rows)
        for split, s in report["configs"][config].items():
            if (args.split == "all" and split == "all") or (args.split != "all" and split != args.split):
                continue
            a, idt = s["answerable"], s.get("identity", {})
            if not a:  # identity-only run (redacted_noguard)
                print(f"{config:<17} {split:<4} identity n={idt.get('n', 0)} answered (false attribution) "
                      f"{idt.get('false_attribution', 0):.3f} | answer contains the person's true value "
                      f"{idt.get('accuracy', 0):.3f}")
                continue
            print(f"{config:<17} {split:<4} answerable grounded {a.get('grounded_accuracy', 0):.3f} "
                  f"{s['answerable_grounded_accuracy_ci95']} plain {a.get('accuracy', 0):.3f} med-F1 {a.get('med_f1', 0):.3f} "
                  f"all-correct {a.get('all_correct', 0):.3f} med-precision {a.get('med_precision', 0):.3f} "
                  f"abstain {a.get('abstained', 0):.3f} | identity abstain {idt.get('abstained', 0):.3f} "
                  f"| cited {a.get('claims_cited', 0):.3f} gold-note {a.get('cited_from_gold_note', 0):.3f} "
                  f"support {a.get('cited_support', 0):.3f} json1 {a.get('json_first_try', 0):.3f} "
                  f"| {a.get('latency_ms', 0):.0f} ms")

    with (RESULTS_ROOT / f"{stem}.json").open("w") as f:
        json.dump(report, f, indent=2)
    with (RESULTS_ROOT / f"{stem}_answers.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {RESULTS_ROOT / stem}.json")


if __name__ == "__main__":
    main()
