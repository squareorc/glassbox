"""
Privacy-utility trade-off: sweep the redaction confidence threshold.

For each threshold t in 0.3 ... 0.8 (realistic dataset, all 390 QA questions):
    1. redaction at t            -> P/R/F1, masked recall, char leakage, over-redaction count
    2. redacted section index at t
    3. retrieval (system config) -> evidence, recall, context size; leak audit of the index
    4. cited generation (system config: question redaction + abstain guard)
                                 -> grounded accuracy, drug-list F1, identity questions answered

A lower threshold redacts more (more privacy, risk of destroying clinical text);
a higher one redacts less. Only the NOTE redaction threshold varies; question
redaction stays at the system default (0.5), so every point uses the same guard.

    python -m glassbox.src.experiments.threshold_sweep            # ~2.5 h
    python -m glassbox.src.experiments.threshold_sweep --skip-generation

Results are written after every threshold to glassbox/results/experiments/threshold_sweep.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from glassbox.src.qa.build_qa_set import load_qa_set
from glassbox.src.retrieval.evaluate import leak_audit, run_questions, summarise
from glassbox.src.retrieval.index import DATA_ROOT, index_dir
from glassbox.src.retrieval.retriever import Retriever

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
OUT = Path("glassbox/results/experiments/threshold_sweep.json")
DATASET = "realistic"


def _run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def over_redactions(t: float) -> int:
    """Redactions that overlap no ground-truth PII span (clinical text destroyed)."""
    with (DATA_ROOT / "raw" / DATASET / "synthetic_notes.jsonl").open() as f:
        gold = {d["doc_id"]: d["entities"] for d in map(json.loads, f)}
    n = 0
    with (DATA_ROOT / "processed" / DATASET / f"t{t:.2f}" / "redaction_audit_log.jsonl").open() as f:
        for r in map(json.loads, f):
            n += not any(r["start"] < e["end"] and e["start"] < r["end"] for e in gold[r["doc_id"]])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    qa_set = load_qa_set(DATA_ROOT / "raw" / DATASET / "qa_eval_set.jsonl")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    for t in args.thresholds:
        key = f"{t:.2f}"
        t0 = time.time()
        print(f"=== threshold {key}", flush=True)
        _run("glassbox.src.redaction.pipeline", "--dataset", DATASET, "--threshold", str(t))
        red = json.loads(Path(f"glassbox/results/redaction/{DATASET}_t{key}.json").read_text())["evaluation"]
        _run("glassbox.src.retrieval.index", "--dataset", DATASET, "--variant", "redacted",
             "--chunking", "section", "--threshold", str(t))

        idx = index_dir(DATASET, "redacted", "section", t)
        retr = summarise(run_questions(Retriever(idx), qa_set, "hybrid", True))
        leak = leak_audit(DATASET, idx)
        row = {
            "threshold": t,
            "redaction": {"overall": red["overall"], "masking": red["masking"]["overall"],
                          "per_type_f1": {k: v["f1"] for k, v in red["per_type"].items()},
                          "over_redactions": over_redactions(t)},
            "leak_audit": leak["overall"] | {"per_type": leak["per_type"]},
            "retrieval_test": {k: retr["test"][k] for k in ("answerable", "clinical_fact", "aggregate", "identity")}
                              | {"answerable_evidence@auto_ci95": retr["test"]["answerable_evidence@auto_ci95"]},
        }
        a = row["retrieval_test"]["answerable"]
        print(f"  redaction F1 {red['overall']['f1']:.3f}, char leakage {red['masking']['overall']['char_leakage']:.4f}, "
              f"over-redactions {row['redaction']['over_redactions']} | retrieval evidence {a['evidence@auto']:.3f}",
              flush=True)

        if not args.skip_generation:
            from glassbox.src.generation.evaluate import run_config, summarise as gen_summarise
            rows = run_config("redacted", DATASET, t, qa_set,
                              Path(f"glassbox/logs/generation/{DATASET}/sweep_t{key}.jsonl"))
            g = gen_summarise(rows)["test"]
            row["generation_test"] = {"answerable": g["answerable"], "clinical_fact": g["clinical_fact"],
                                      "aggregate": g["aggregate"], "identity": g["identity"],
                                      "answerable_grounded_accuracy_ci95": g["answerable_grounded_accuracy_ci95"]}
            print(f"  grounded accuracy {g['answerable']['grounded_accuracy']:.3f} "
                  f"{g['answerable_grounded_accuracy_ci95']} | identity answered "
                  f"{g['identity'].get('false_attribution', 0):.3f}", flush=True)

        row["minutes"] = round((time.time() - t0) / 60, 1)
        results[key] = row
        OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
