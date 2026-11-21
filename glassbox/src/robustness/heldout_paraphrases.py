"""
Held-out paraphrase set B: built and committed BEFORE the robustness fixes,
and never used while developing them.

    part 1 - 60 test questions reworded by hand (manual_paraphrases.py)
    part 2 - the other 129 answerable test questions reworded by the local
             Qwen model, which has never seen GlassBox's query-analysis code,
             so its wording is independent of the fixes' author

Together they cover every answerable realistic test question exactly once.

Qwen part: sampled (temperature 0.9, fixed seed per question, so it is
reproducible). Each paraphrase is checked for the original's facts (visit
date or month, diagnosis, symptom). A paraphrase that fails the check is
resampled (up to 4 tries); failures after that are written to a review list
and decided by hand, with the decision recorded in the file.

    python -m glassbox.src.robustness.heldout_paraphrases generate   # Qwen part -> review
    python -m glassbox.src.robustness.heldout_paraphrases freeze     # writes paraphrases_b.jsonl
"""

import calendar
import json
import re
import sys
from pathlib import Path

from .manual_paraphrases import MANUAL

QA = Path("glassbox/data/raw/realistic/qa_eval_set.jsonl")
OUT_DIR = Path("glassbox/data/robustness")
QWEN_RAW = OUT_DIR / "paraphrases_b_qwen_raw.jsonl"
OUT = OUT_DIR / "paraphrases_b.jsonl"

PROMPT = """Rewrite the question below the way a busy doctor might type it into a clinical records search tool.
Keep EVERY fact exactly: the diagnosis, the date (day, month and year), any symptom, and what is being asked.
You may change the sentence structure, use common medical abbreviations, or write the date in another format.
Reply with only the rewritten question.

Question: {q}"""

DX_WORDS = {"Essential hypertension": ["hypertens", "htn", "high blood pressure", "bp"],
            "Type 2 diabetes mellitus": ["diabet", "t2dm", "dm2", "dm"],
            "Major depressive disorder": ["depress", "mdd"],
            "Chronic kidney disease stage 3": ["ckd", "kidney", "renal"],
            "Coronary artery disease": ["coronary", "cad", "chd"],
            "COPD": ["copd", "obstructive"], "Asthma": ["asthma"], "Osteoarthritis": ["osteoarthritis", "oa"]}


def fact_problems(q: dict, text: str) -> list:
    """Which of the original question's facts are not visibly kept in `text`."""
    t, probs = text.lower(), []
    if q["type"] == "clinical_fact":
        dx = re.search(r"for the (.+?) patient whose note", q["question"]).group(1)
        y, m, d = map(int, re.search(r"(\d{4})-(\d{2})-(\d{2})", q["question"]).groups())
        if str(y) not in t and f"'{str(y)[2:]}" not in t and not re.search(rf"/{str(y)[2:]}\b", t):
            probs.append("year")
        if not re.search(rf"(?<!\d)0?{d}(?:st|nd|rd|th)?(?!\d)", t):
            probs.append("day")
        if calendar.month_abbr[m].lower() not in t and not re.search(rf"(?<!\d)0?{m}(?=[/\-.])|-{m:02d}-", t):
            probs.append("month")
    else:
        dx, sym, month, year = re.search(r"to (.+?) patients who presented with (.+?) in ([A-Z][a-z]+) (\d{4})",
                                         q["question"]).groups()
        m = list(calendar.month_name).index(month)
        if year not in t and f"'{year[2:]}" not in t:
            probs.append("year")
        if month.lower()[:3] not in t and not re.search(rf"(?<!\d)0?{m}[/\-]|-{m:02d}\b", t):
            probs.append("month")
        if sym.split()[0].lower()[:5] not in t:
            probs.append("symptom")
    if not any(re.search(rf"\b{re.escape(w)}", t) for w in DX_WORDS[dx]):
        probs.append("diagnosis")
    return probs


def _questions():
    qa = [json.loads(l) for l in QA.open()]
    return [q for q in qa if q["split"] == "test" and q["type"] != "identity"]


def generate():
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    from glassbox.src.generation.generator import MODEL_ID, _load
    model, tok = _load(MODEL_ID)
    sampler = make_sampler(temp=0.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, q in enumerate(q for q in _questions() if q["qid"] not in MANUAL):
        tries = []
        for attempt in range(4):
            mx.random.seed(1000 * n + attempt)
            prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q["question"])}],
                                             add_generation_prompt=True, tokenize=False)
            text = mlx_generate(model, tok, prompt=prompt, max_tokens=80, sampler=sampler).strip().strip('"')
            text = text.splitlines()[0].strip() if text else text
            probs = fact_problems(q, text)
            tries.append({"text": text, "problems": probs})
            if not probs:
                break
        rows.append({"qid": q["qid"], "original_question": q["question"], "tries": tries,
                     "accepted": tries[-1]["text"] if not tries[-1]["problems"] else None})
        print(f"{q['qid']}: {'ok' if rows[-1]['accepted'] else 'REVIEW'} | {tries[-1]['text']}", flush=True)
    with QWEN_RAW.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{sum(r['accepted'] is not None for r in rows)}/{len(rows)} passed the fact check; rest need review")


def freeze(review_decisions: dict = None):
    """review_decisions: {qid: text or None} for paraphrases that failed the automatic check."""
    review_decisions = review_decisions or {}
    qa = {q["qid"]: q for q in _questions()}
    out = []
    for qid, text in MANUAL.items():
        out.append({**qa[qid], "original_question": qa[qid]["question"], "question": text, "source": "manual"})
    for r in map(json.loads, QWEN_RAW.open()):
        text = r["accepted"] if r["accepted"] else review_decisions.get(r["qid"])
        if text:
            out.append({**qa[r["qid"]], "original_question": r["original_question"], "question": text,
                        "source": "qwen" if r["accepted"] else "qwen (hand-reviewed)"})
    with OUT.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"froze {len(out)} paraphrases -> {OUT}")


# Hand-review decisions for Qwen paraphrases that failed the automatic fact check (2026-11-21).
# Decided on meaning alone, before set B was evaluated.
REVIEW_DECISIONS = {
    # "decreased ROM" = "reduced range of motion" (ROM is the standard abbreviation); facts kept
    "realistic-ag-047": "Which meds were dispensed to OA pts with decreased ROM in Jun 2026?",
}

if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(REVIEW_DECISIONS)
    else:
        generate()
