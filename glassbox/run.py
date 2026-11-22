"""
GlassBox command line: question in, verified cited answer and audit trace out.

    python -m glassbox.run "Which medications were prescribed to patients who presented with fatigue?"
    python -m glassbox.run --qa-id realistic-cf-000            # one QA-set question (scored)
    python -m glassbox.run --qa-set --split test               # batch over the QA set (scored)
    python -m glassbox.run --batch questions.txt               # one question per line (or JSONL with "question")
    python -m glassbox.run --config raw "..."                  # no-privacy baseline
    python -m glassbox.run --config redacted_noguard --qa-set --types identity

Traces go to glassbox/logs/traces/<dataset>/<config>/<qid or trace_id>.json.
A QA-set batch also writes glassbox/results/pipeline/<dataset>_<config>[_<split>].json:
grounded accuracy (with a bootstrap 95% CI), identity questions answered, answer
verdicts and stage latencies. Because the pipeline uses the same stages and
settings as the evaluations, those numbers must equal generation/evaluate.py's.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from glassbox.src.generation.evaluate import score
from glassbox.src.pipeline import CONFIGS, GlassBox
from glassbox.src.qa.build_qa_set import load_qa_set
from glassbox.src.retrieval.evaluate import bootstrap_ci
from glassbox.src.retrieval.index import DATA_ROOT

RESULTS_ROOT = Path("glassbox/results/pipeline")
SCORE_KEYS = ("grounded_accuracy", "accuracy", "all_correct", "med_precision", "med_recall", "med_f1",
              "cited_from_gold_note")


def run_qa(gb: GlassBox, q: Dict[str, Any]) -> Dict[str, Any]:
    """Answer a QA-set question and attach its scores (never the question text or gold answer)."""
    trace = gb.answer(q["question"], qid=q["qid"], save=False)
    gen = trace["generation"]
    s = score(gen, q, trace["retrieval"]["chunks"])
    trace["qa"] = {"qid": q["qid"], "type": q["type"], "split": q["split"], "answer_field": q["answer_field"],
                   "gold_doc_ids": q["relevant_doc_ids"],
                   "scores": {k: s[k] for k in SCORE_KEYS if k in s}}
    gb.save(trace)
    return trace


def summarise(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [t for t in traces if t["qa"]["type"] != "identity"]
    identity = [t for t in traces if t["qa"]["type"] == "identity"]
    ga = [t["qa"]["scores"]["grounded_accuracy"] for t in answerable]
    out: Dict[str, Any] = {"n": len(traces), "n_answerable": len(answerable), "n_identity": len(identity)}
    if answerable:
        out["answerable_grounded_accuracy"] = round(mean(ga), 4)
        out["answerable_grounded_accuracy_ci95"] = bootstrap_ci(ga)
        for qtype in ("clinical_fact", "aggregate"):
            rs = [t["qa"]["scores"]["grounded_accuracy"] for t in answerable if t["qa"]["type"] == qtype]
            if rs:
                out[f"{qtype}_grounded_accuracy"] = round(mean(rs), 4)
        out["answerable_abstained"] = round(mean(t["generation"]["abstained"] for t in answerable), 4)
    if identity:
        out["identity_answered"] = sum(not t["generation"]["abstained"] for t in identity)
    out["verdicts"] = dict(Counter(t["verification"]["verdict"] for t in traces))
    by_verdict = {}
    for v in ("VERIFIED", "PARTLY_VERIFIED", "NOT_VERIFIED", "CONTRADICTED"):
        rs = [t["qa"]["scores"]["grounded_accuracy"] for t in answerable if t["verification"]["verdict"] == v]
        if rs:
            by_verdict[v] = {"n": len(rs), "grounded_accuracy": round(mean(rs), 4)}
    out["grounded_accuracy_by_verdict"] = by_verdict
    stages = traces[0]["timings_ms"].keys()
    out["latency_ms_median"] = {s: round(median(t["timings_ms"][s] for t in traces), 1) for s in stages}
    called = [t["timings_ms"]["total"] for t in traces if t["generation"]["model_called"]]
    if called:
        out["latency_ms_median_when_model_called"] = round(median(called), 1)
    return out


def print_trace(trace: Dict[str, Any]) -> None:
    q, gen, ver = trace["question"], trace["generation"], trace["verification"]
    print(f"Question (as processed): {q['text']}")
    if q["identifiers"]:
        print(f"  identifiers removed: {', '.join(q['identifiers'])}")
    a = trace["query_analysis"]
    print(f"Query analysis: filters {a.get('filters') or '{}'} | scope {a.get('scope')} | fallback {a.get('fallback')}")
    print(f"Retrieved {trace['retrieval']['k']} chunks ({trace['retrieval']['context_chars']} chars):")
    for c in trace["retrieval"]["chunks"]:
        print(f"  {c['rank']:>2}. {c['chunk_id']:<28} {c['selected_by']:<8} score {c['score']:.4f}")
    if gen["abstained"]:
        print(f"ABSTAINED ({gen['abstain_reason']}): {gen['answer']}")
    else:
        print(f"Answer: {gen['answer']}")
    labels = {c["claim"]: c["label"] for c in ver["claims"]}
    for c in gen["claims"]:
        print(f"  - [{labels.get(c['text'], '-')}] {c['text']}  cites {', '.join(c['citations']) or 'nothing'}")
    print(f"Verdict: {ver['verdict']} | total {trace['timings_ms']['total']:.0f} ms | trace {trace['trace_id']}")
    if "qa" in trace:
        print(f"Scores: {trace['qa']['scores']}")


def read_batch(path: Path) -> List[Dict[str, Any]]:
    items = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            items.append({"question": obj["question"], "qid": obj.get("qid") or f"{path.stem}-{n:04d}"})
        else:
            items.append({"question": line, "qid": f"{path.stem}-{n:04d}"})
    return items


def main(argv=None):
    p = argparse.ArgumentParser(description="GlassBox: verified, cited answers with an audit trace")
    p.add_argument("question", nargs="?")
    p.add_argument("--qa-id", help="Answer one QA-set question by id (scored)")
    p.add_argument("--qa-set", action="store_true", help="Answer every QA-set question (scored)")
    p.add_argument("--batch", type=Path, help="File with one question per line, or JSONL with 'question'")
    p.add_argument("--dataset", choices=["synthetic", "realistic"], default="realistic")
    p.add_argument("--config", choices=CONFIGS, default="redacted")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--types", nargs="+", choices=["clinical_fact", "aggregate", "identity"])
    p.add_argument("--limit", type=int)
    p.add_argument("--no-verify", action="store_true", help="Skip claim verification (faster)")
    args = p.parse_args(argv)
    if sum(bool(x) for x in (args.question, args.qa_id, args.qa_set, args.batch)) != 1:
        p.error("give exactly one of: a question, --qa-id, --qa-set, --batch")

    gb = GlassBox(args.dataset, args.config, args.threshold, verify=not args.no_verify)
    qa_path = DATA_ROOT / "raw" / args.dataset / "qa_eval_set.jsonl"

    if args.question:
        print_trace(gb.answer(args.question))
        return
    if args.qa_id:
        q = next((q for q in load_qa_set(qa_path) if q["qid"] == args.qa_id), None)
        if q is None:
            sys.exit(f"no QA question with id {args.qa_id}")
        print_trace(run_qa(gb, q))
        return
    if args.batch:
        items = read_batch(args.batch)[: args.limit]
        for n, it in enumerate(items, 1):
            t = gb.answer(it["question"], qid=it["qid"])
            print(f"[{n}/{len(items)}] {it['qid']}: {t['verification']['verdict']}", flush=True)
        print(f"Traces in {gb.trace_dir}")
        return

    questions = load_qa_set(qa_path, None if args.split == "all" else args.split)
    if args.types:
        questions = [q for q in questions if q["type"] in args.types]
    questions = questions[: args.limit]
    traces = []
    for n, q in enumerate(questions, 1):
        traces.append(run_qa(gb, q))
        if n % 25 == 0 or n == len(questions):
            s = summarise(traces)
            print(f"[{n}/{len(questions)}] grounded {s.get('answerable_grounded_accuracy', 'n/a')} | "
                  f"identity answered {s.get('identity_answered', 'n/a')} | {s['verdicts']}", flush=True)
    report = {"dataset": args.dataset, "config": gb.config_record(), "split": args.split, "types": args.types,
              "limit": args.limit, "summary": {}}
    for split in ("dev", "test", "all"):
        ts = [t for t in traces if split == "all" or t["qa"]["split"] == split]
        if ts:
            report["summary"][split] = summarise(ts)
    if args.limit is None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        tag = "".join(f"_{x}" for x in ([args.split] if args.split != "all" else []) + (args.types or []))
        out = RESULTS_ROOT / f"{args.dataset}_{args.config}{tag}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved {out}")
    print(json.dumps(report["summary"], indent=2))
    print(f"Traces in {gb.trace_dir}")


if __name__ == "__main__":
    main()
