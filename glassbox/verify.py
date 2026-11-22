"""
Live reproducibility check (about a minute): re-run a sample of the evaluation
questions through the full pipeline right now and compare with the stored results.

    python -m glassbox.verify               # 10 questions: 4 clinical fact, 3 aggregate, 3 identity
    python -m glassbox.verify --n 30        # bigger sample
    python -m glassbox.verify --tests       # also run the fast test suite first (~30 s)

What it shows: the reported numbers come from the code as it is, not from a lucky run or an
edited file. Generation is greedy, so the same question must give the same retrieved
chunks, the same answer text, the same citations and the same score as the stored
evaluation run (`glassbox/results/generation/realistic_t0.50_answers.jsonl`, committed).
Any difference is printed and the exit code is 1.

The questions are a fixed, stratified sample drawn with a fixed seed, printed in their
REDACTED form (identity questions contain names).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from glassbox.src.generation.evaluate import score
from glassbox.src.pipeline import GlassBox
from glassbox.src.qa.build_qa_set import load_qa_set
from glassbox.src.retrieval.index import DATA_ROOT

STORED = Path("glassbox/results/generation/realistic_t0.50_answers.jsonl")
SHARE = {"clinical_fact": 0.4, "aggregate": 0.3, "identity": 0.3}
SEED = 0


def sample(qa, n):
    rng = random.Random(SEED)
    out = []
    for qtype, share in SHARE.items():
        pool = sorted((q for q in qa if q["type"] == qtype), key=lambda q: q["qid"])
        out += rng.sample(pool, max(1, round(n * share)))
    return out[:n] if len(out) > n else out


def compare(trace, stored, q):
    gen, rec = trace["generation"], stored["record"]
    checks = {
        "chunks": [c["chunk_id"] for c in trace["retrieval"]["chunks"]] == rec["chunk_ids"],
        "answer": gen["answer"] == rec["answer"],
        "claims": [(c["text"], c["citations"]) for c in gen["claims"]] == [(c["text"], c["citations"]) for c in rec["claims"]],
        "refusal": gen["abstain_reason"] == rec["abstain_reason"],
        "score": abs(score(gen, q, trace["retrieval"]["chunks"])["grounded_accuracy"] - stored["grounded_accuracy"]) < 1e-9,
    }
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-run stored evaluation questions live and compare")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tests", action="store_true", help="Run the fast test suite first")
    args = ap.parse_args(argv)

    if args.tests:
        print("Running the fast test suite…", flush=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "glassbox/tests", "-q", "-p", "no:warnings"],
                           capture_output=True, text=True)
        print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1], flush=True)
        if r.returncode:
            print(r.stdout[-3000:])
            sys.exit(1)

    stored = {r["qid"]: r for r in map(json.loads, STORED.open()) if r["config"] == "redacted"}
    qa = [q for q in load_qa_set(DATA_ROOT / "raw" / "realistic" / "qa_eval_set.jsonl") if q["qid"] in stored]
    questions = sample(qa, args.n)

    print(f"Loading the pipeline (retriever, 7B generator, NLI verifier)…", flush=True)
    t0 = time.perf_counter()
    gb = GlassBox("realistic", "redacted", preload=True)
    print(f"  loaded in {time.perf_counter() - t0:.0f} s\n", flush=True)

    print(f"Re-running {len(questions)} stored evaluation questions live:\n")
    ok_all, t_run = 0, time.perf_counter()
    for q in questions:
        trace = gb.answer(q["question"], qid=q["qid"], save=False)
        checks = compare(trace, stored[q["qid"]], q)
        same = all(checks.values())
        ok_all += same
        gen = trace["generation"]
        outcome = f"refused ({gen['abstain_reason']})" if gen["abstained"] else trace["verification"]["verdict"]
        mark = "identical" if same else "DIFFERS: " + ", ".join(k for k, v in checks.items() if not v)
        print(f"  {q['qid']:<18} {q['type']:<13} {outcome:<34} {trace['timings_ms']['total'] / 1000:4.1f} s  {mark}")
        print(f"      {trace['question']['text'][:110]}")
    elapsed = time.perf_counter() - t_run
    print(f"\n{ok_all}/{len(questions)} identical to the stored evaluation run "
          f"(retrieved chunks, answer, citations, refusal, score) · {elapsed:.0f} s for {len(questions)} questions")
    sys.exit(0 if ok_all == len(questions) else 1)


if __name__ == "__main__":
    main()
