"""
Evaluate the claim verifier.

    python -m glassbox.src.verification.evaluate

1. Corrupted-claim set (corruptions.py), DEV split: score both candidate NLI
   models; choose the model by balanced accuracy (SUPPORTED vs not), then the
   UNCERTAIN margin as the smallest margin at which the confident labels are
   >= TARGET_ACCURACY correct (selective prediction).
2. TEST split with the chosen model and margin: per corruption type, how often
   the verifier says SUPPORTED (should be ~1 for genuine, ~0 for corruptions),
   agreement with the ideal label, coverage (share not UNCERTAIN).
3. Real answers (Day 3 generation run, test split): label distribution, and
   whether VERIFIED answers are more often correct (grounded accuracy).
4. Guard-off identity answers: labels with and without subject rewriting.

Results: glassbox/results/verification/verifier_eval.json
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from glassbox.src.retrieval.evaluate import bootstrap_ci
from glassbox.src.retrieval.index import index_dir

from .verifier import SUPPORTED, UNCERTAIN, Verifier

ITEMS = Path("glassbox/data/verification/corrupted_claims.jsonl")
ANSWERS = Path("glassbox/results/generation/realistic_t0.50_answers.jsonl")
OUT = Path("glassbox/results/verification/verifier_eval.json")
MODELS = ["cross-encoder/nli-deberta-v3-base", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]
TARGET_ACCURACY = 0.95
MARGINS = [round(0.05 * i, 2) for i in range(20)]


def best_probs(verifier, items):
    """Highest-entailment premise option per item (each premise, and all together)."""
    jobs = []
    for k, it in enumerate(items):
        for p in Verifier.premises(it["premises"]):
            jobs.append((k, (p, it["claim"])))
    probs = verifier.probabilities([pair for _, pair in jobs])
    best = {}
    for (k, _), pr in zip(jobs, probs):
        if k not in best or pr["entailment"] > best[k]["entailment"]:
            best[k] = pr
    return [best[k] for k in range(len(items))]


def binary_correct(label, item):
    return (label == SUPPORTED) == (item["expected"] == SUPPORTED)


def balanced_accuracy(labels, items):
    pos = [binary_correct(l, it) for l, it in zip(labels, items) if it["expected"] == SUPPORTED]
    neg = [binary_correct(l, it) for l, it in zip(labels, items) if it["expected"] != SUPPORTED]
    return (mean(pos) + mean(neg)) / 2


def margin_table(verifier, probs, items):
    rows = []
    for m in MARGINS:
        labels = [verifier.label(p, m) for p in probs]
        covered = [(l, it) for l, it in zip(labels, items) if l != UNCERTAIN]
        rows.append({"margin": m, "coverage": round(len(covered) / len(items), 4),
                     "accuracy_on_covered": round(mean(binary_correct(l, it) for l, it in covered), 4) if covered else None})
    return rows


def per_kind(labels, items):
    out = {}
    for kind in sorted({it["kind"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["kind"] == kind]
        dist = Counter(labels[i] for i in idx)
        out[kind] = {"n": len(idx), "said_supported": round(dist[SUPPORTED] / len(idx), 4),
                     "ideal_label_agreement": round(mean(labels[i] == items[i]["ideal"] for i in idx), 4),
                     "labels": dict(dist)}
    return out


def main():
    items = [json.loads(l) for l in ITEMS.open()]
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    report = {"n_items": {"dev": len(dev), "test": len(test)}, "target_accuracy": TARGET_ACCURACY, "models": {}}

    # ---- 1. dev: model and margin choice
    for name in MODELS:
        v = Verifier(name, margin=0.0)
        probs = best_probs(v, dev)
        labels = [v.label(p, 0.0) for p in probs]
        table = margin_table(v, probs, dev)
        report["models"][name] = {"dev_balanced_accuracy": round(balanced_accuracy(labels, dev), 4),
                                  "dev_per_kind": per_kind(labels, dev), "dev_margin_table": table}
        print(f"{name.split('/')[-1]:<45} dev balanced accuracy {report['models'][name]['dev_balanced_accuracy']:.3f}")
    chosen = max(MODELS, key=lambda n: report["models"][n]["dev_balanced_accuracy"])
    table = report["models"][chosen]["dev_margin_table"]
    ok = [r for r in table if r["accuracy_on_covered"] is not None and r["accuracy_on_covered"] >= TARGET_ACCURACY]
    margin = ok[0]["margin"] if ok else max(table, key=lambda r: r["accuracy_on_covered"] or 0)["margin"]
    report["chosen"] = {"model": chosen, "margin": margin}
    print(f"chosen: {chosen}, UNCERTAIN margin {margin}")

    # ---- 2. test
    v = Verifier(chosen, margin=margin)
    t0 = time.perf_counter()
    probs = best_probs(v, test)
    ms_per_item = (time.perf_counter() - t0) * 1000 / len(test)
    labels = [v.label(p) for p in probs]
    covered = [(l, it) for l, it in zip(labels, test) if l != UNCERTAIN]
    report["test"] = {
        "balanced_accuracy": round(balanced_accuracy(labels, test), 4),
        "coverage": round(len(covered) / len(test), 4),
        "accuracy_on_covered": round(mean(binary_correct(l, it) for l, it in covered), 4),
        "false_alarm_on_genuine": round(mean(l != SUPPORTED for l, it in zip(labels, test) if it["kind"] == "genuine"), 4),
        # genuine claims that assert nothing beyond their cited evidence (see corruptions.asserts_uncited_condition)
        "false_alarm_on_genuine_clean": round(mean(l != SUPPORTED for l, it in zip(labels, test)
                                                   if it["kind"] == "genuine" and not it["asserts_uncited_condition"]), 4),
        "genuine_asserting_uncited_condition": sum(it["kind"] == "genuine" and it["asserts_uncited_condition"] for it in test),
        "corruptions_caught": round(mean(l != SUPPORTED for l, it in zip(labels, test) if it["kind"] != "genuine"), 4),
        "corruptions_caught_ci95": bootstrap_ci([float(l != SUPPORTED) for l, it in zip(labels, test) if it["kind"] != "genuine"]),
        "per_kind": per_kind(labels, test),
        "ms_per_item": round(ms_per_item, 1),
    }
    t = report["test"]
    print(f"TEST: balanced accuracy {t['balanced_accuracy']:.3f} | corruptions caught {t['corruptions_caught']:.3f} "
          f"{t['corruptions_caught_ci95']} | false alarms on genuine {t['false_alarm_on_genuine']:.3f} "
          f"(clean claims {t['false_alarm_on_genuine_clean']:.3f}; {t['genuine_asserting_uncited_condition']} assert an uncited condition) | "
          f"coverage {t['coverage']:.3f}")
    for kind, s in t["per_kind"].items():
        print(f"   {kind:<22} n={s['n']:3d} said SUPPORTED {s['said_supported']:.3f} | ideal label {s['ideal_label_agreement']:.3f} | {s['labels']}")

    # ---- 3. real answers + 4. guard-off identity answers
    with (index_dir("realistic", "redacted", "section") / "chunks.jsonl").open() as f:
        chunk_text = {c["chunk_id"]: c["text"] for c in map(json.loads, f)}
    rows = [json.loads(l) for l in ANSWERS.open()]
    real = [r for r in rows if r["config"] == "redacted" and r["split"] == "test" and r["type"] != "identity"]
    verdicts, claim_labels = defaultdict(list), Counter()
    for r in real:
        res = v.verify_answer(r["record"], chunk_text, {"identifiers": []})
        claim_labels.update(c["label"] for c in res["claims"])
        verdicts[res["verdict"]].append(r["grounded_accuracy"])
    report["real_answers_test"] = {
        "claim_labels": dict(claim_labels),
        "answer_verdicts": {k: {"n": len(x), "grounded_accuracy": round(mean(x), 4)} for k, x in verdicts.items()},
    }
    print("REAL answers (test):", dict(claim_labels))
    for k, x in verdicts.items():
        print(f"   {k:<16} n={len(x):3d} mean grounded accuracy {mean(x):.3f}")

    noguard = [r for r in rows if r["config"] == "redacted_noguard" and r["split"] == "test"]
    ident = {}
    for mode, ids in (("without_rewriting", []), ("with_rewriting", ["NAME"])):
        c = Counter()
        for r in noguard:
            c.update(x["label"] for x in v.verify_claims(r["record"]["claims"], chunk_text, ids))
        ident[mode] = dict(c)
        print(f"GUARD-OFF identity answers, claims {mode}: {dict(c)}")
    report["guard_off_identity_claims"] = ident

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
