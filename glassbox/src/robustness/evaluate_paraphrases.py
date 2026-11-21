"""
Evaluate a paraphrase set end to end: retrieval evidence (and optionally cited
generation) on the paraphrased questions vs. the original wording.

    python glassbox/src/robustness/evaluate_paraphrases.py --set a
    python glassbox/src/robustness/evaluate_paraphrases.py --set b --generate
    # pre-fix code on the same questions (before/after comparison):
    PYTHONPATH=<worktree of commit 2f0bb6b> python glassbox/src/robustness/evaluate_paraphrases.py --set b --tag _prefix

Absolute imports only, so the same file can run against an older checkout of the
package via PYTHONPATH. Results: glassbox/results/robustness/paraphrases_<set><tag>.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from glassbox.src.retrieval.evaluate import bootstrap_ci, score_question
from glassbox.src.retrieval.index import index_dir
from glassbox.src.retrieval.retriever import MAX_K, Retriever

DATA = Path("glassbox/data/robustness")
RESULTS = Path("glassbox/results/robustness")


def group_key(p):
    if "variant" in p:
        return [f"date_format={p['variant']['date_format']}", f"dx_form={p['variant']['dx_form']}"]
    return [f"source={p['source']}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["a", "b"], required=True)
    ap.add_argument("--generate", action="store_true", help="Also run cited generation on the paraphrases")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    paras = [json.loads(l) for l in (DATA / f"paraphrases_{args.set}.jsonl").open()]

    retriever = Retriever(index_dir("realistic", "redacted", "section"))
    gen = None
    if args.generate:
        from glassbox.src.generation.evaluate import score as gen_score
        from glassbox.src.generation.generator import Generator
        gen = Generator(log_path=Path("glassbox/logs/generation/realistic") / f"paraphrases_{args.set}{args.tag}.jsonl")

    rows = []
    for p in paras:
        row = {"qid": p["qid"], "type": p["type"], "question": p["question"], "groups": group_key(p)}
        for which, key in (("original", "original_question"), ("para", "question")):
            res = retriever.retrieve(p[key], k=MAX_K)
            a = retriever.last_analysis
            s = score_question(res, p, retriever.auto_k(a))
            row[f"evidence_{which}"] = s["evidence@auto"]
            row[f"recall_{which}"] = s["recall@auto"]
            if which == "para":
                row["filters"] = a["filters"]
                row["redacted_question"] = (retriever.last_query_redaction or {}).get("text", p[key])
                if gen:
                    k = retriever.auto_k(a)
                    rec = gen.answer(row["redacted_question"], res[:k], a, retriever.last_query_redaction,
                                     retriever.manifest["source"], qid=p["qid"])
                    gs = gen_score(rec, p, res[:k])
                    row.update(grounded_accuracy=gs["grounded_accuracy"], accuracy=gs["accuracy"],
                               abstained=gs["abstained"], abstain_reason=gs["abstain_reason"],
                               med_f1=gs.get("med_f1"), answer=rec["answer"])
        rows.append(row)

    def summ(rs):
        out = {"n": len(rs),
               "evidence_original": round(mean(r["evidence_original"] for r in rs), 4),
               "evidence_paraphrase": round(mean(r["evidence_para"] for r in rs), 4),
               "evidence_paraphrase_ci95": bootstrap_ci([r["evidence_para"] for r in rs]),
               "recall_paraphrase": round(mean(r["recall_para"] for r in rs), 4)}
        if gen:
            out.update(grounded_accuracy=round(mean(r["grounded_accuracy"] for r in rs), 4),
                       grounded_accuracy_ci95=bootstrap_ci([r["grounded_accuracy"] for r in rs]),
                       accuracy=round(mean(r["accuracy"] for r in rs), 4),
                       abstained=round(mean(r["abstained"] for r in rs), 4))
            f1 = [r["med_f1"] for r in rs if r.get("med_f1") is not None]
            if f1:
                out["med_f1"] = round(mean(f1), 4)
        return out

    report = {"set": args.set, "tag": args.tag, "overall": summ(rows),
              "by_type": {t: summ([r for r in rows if r["type"] == t]) for t in ("clinical_fact", "aggregate")}}
    groups = defaultdict(list)
    for r in rows:
        for g in r["groups"]:
            groups[(r["type"], g)].append(r)
    report["by_group"] = {f"{t} {g}": summ(rs) for (t, g), rs in sorted(groups.items())}

    def line(name, s):
        extra = (f" | grounded acc {s['grounded_accuracy']:.3f} {s['grounded_accuracy_ci95']} abstain {s['abstained']:.3f}"
                 if gen else "")
        print(f"  {name:<44} n={s['n']:3d} evidence orig {s['evidence_original']:.3f} -> para "
              f"{s['evidence_paraphrase']:.3f} {s['evidence_paraphrase_ci95']}{extra}")

    print(f"Paraphrase set {args.set}{args.tag}:")
    line("ALL", report["overall"])
    for t, s in report["by_type"].items():
        line(t, s)
    for g, s in report["by_group"].items():
        line(g, s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"paraphrases_{args.set}{args.tag}.json").write_text(json.dumps({**report, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
