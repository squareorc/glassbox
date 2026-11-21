"""
Paraphrase set A (development): template-based rewordings of the realistic
TEST questions, written on 2026-11-21 to differ from the QA builder's single
template per field. It revealed that query analysis was fitted to the
builder's wording (PROGRESS.md, robustness section), so robustness fixes were
developed against it; the held-out check is set B (heldout_paraphrases.py).

    python -m glassbox.src.robustness.paraphrase    # writes glassbox/data/robustness/paraphrases_a.jsonl

Paraphrases of the TEST questions, written to differ from the QA
builder's single template per field. Each paraphrase records which variant it
uses (sentence template, date/period format, diagnosis form) so drops can be
attributed. Formats are cycled deterministically so each gets ~equal counts.
"""
import calendar
import json
import random
import re
import sys
from datetime import date

QA = "glassbox/data/raw/realistic/qa_eval_set.jsonl"

DX_FORMS = {  # canonical, alias in our table, common abbreviation NOT in our table
    "Essential hypertension": ["Essential hypertension", "high blood pressure", "HTN"],
    "Type 2 diabetes mellitus": ["Type 2 diabetes mellitus", "diabetes", "T2DM"],
    "Major depressive disorder": ["Major depressive disorder", "depression", "MDD"],
    "Chronic kidney disease stage 3": ["Chronic kidney disease stage 3", "CKD", "CKD3"],
    "Coronary artery disease": ["Coronary artery disease", "CAD", "coronary heart disease"],
    "COPD": ["COPD", "COPD", "emphysema"],
    "Asthma": ["Asthma", "asthma", "reactive airway disease"],
    "Osteoarthritis": ["Osteoarthritis", "osteoarthritis", "OA"],
}
DX_KIND = ["canonical", "alias", "unlisted_abbrev"]

DATE_FORMATS = {
    "iso": lambda d: d.isoformat(),
    "us": lambda d: f"{d.month}/{d.day}/{d.year}",
    "long": lambda d: f"{calendar.month_name[d.month]} {d.day}, {d.year}",
    "short": lambda d: f"{calendar.month_abbr[d.month]} {d.day} {d.year}",
    "day_first": lambda d: f"{d.day} {calendar.month_abbr[d.month]} {d.year}",
}
PERIOD_FORMATS = {
    "month_name": lambda y, m: f"{calendar.month_name[m]} {y}",
    "month_abbr": lambda y, m: f"{calendar.month_abbr[m]} {y}",
    "numeric_us": lambda y, m: f"{m}/{y}",
    "iso_month": lambda y, m: f"{y}-{m:02d}",
}

CF = {
    "medication": ["Which drug did the {dx} patient seen on {date} get?",
                   "{dx} visit on {date}: what was prescribed?",
                   "For the {date} {dx} encounter, what medication was given?"],
    "symptom": ["What did the {dx} patient seen on {date} complain of?",
                "Chief complaint at the {date} {dx} visit?",
                "Which symptoms were noted at the {dx} visit on {date}?"],
    "followup_weeks": ["When should the {dx} patient seen on {date} come back?",
                       "Follow-up interval after the {date} {dx} visit?",
                       "How soon is the next appointment for the {dx} patient from {date}?"],
    "hr": ["What was the pulse of the {dx} patient on {date}?",
           "Heart rate at the {date} {dx} visit?",
           "{dx} patient, {date}: what was the HR?"],
    "bp": ["What was the BP of the {dx} patient seen on {date}?",
           "Blood pressure reading for the {dx} encounter on {date}?",
           "{dx} patient on {date} - blood pressure?"],
    "lab_orders": ["What tests were ordered for the {dx} patient on {date}?",
                   "Labs requested at the {date} {dx} visit?",
                   "Which investigations did the {dx} patient seen {date} get?"],
}
AG = ["What drugs were the {dx} patients with {sym} in {period} given?",
      "{period}: {dx} patients presenting with {sym} - what were they prescribed?",
      "List the medications for {dx} patients who had {sym} during {period}.",
      "Medications prescribed in {period} to {dx} patients complaining of {sym}?"]

CF_RE = re.compile(r"for the (.+?) patient whose note is dated (\d{4}-\d{2}-\d{2})\?$")
AG_RE = re.compile(r"to (.+?) patients who presented with (.+?) in ([A-Z][a-z]+) (\d{4})\?$")


def build(split="test"):
    qa = [json.loads(l) for l in open(QA)]
    out, i = [], 0
    for q in qa:
        if q["split"] != split or q["type"] == "identity":
            continue
        i += 1
        rng = random.Random(f"para-{q['qid']}")
        if q["type"] == "clinical_fact":
            dx, iso = CF_RE.search(q["question"]).groups()
            kind = i % 3
            fmt = list(DATE_FORMATS)[i % len(DATE_FORMATS)]
            tmpl = rng.randrange(3)
            text = CF[q["answer_field"]][tmpl].format(dx=DX_FORMS[dx][kind], date=DATE_FORMATS[fmt](date.fromisoformat(iso)))
            variant = {"template": f"cf{tmpl}", "date_format": fmt, "dx_form": DX_KIND[kind]}
        else:
            dx, sym, month, year = AG_RE.search(q["question"]).groups()
            m = list(calendar.month_name).index(month)
            kind = i % 3
            fmt = list(PERIOD_FORMATS)[i % len(PERIOD_FORMATS)]
            tmpl = rng.randrange(len(AG))
            text = AG[tmpl].format(dx=DX_FORMS[dx][kind], sym=sym, period=PERIOD_FORMATS[fmt](int(year), m))
            variant = {"template": f"ag{tmpl}", "date_format": fmt, "dx_form": DX_KIND[kind]}
        text = text[0].upper() + text[1:]
        out.append({**q, "original_question": q["question"], "question": text, "variant": variant})
    return out


OUT = "glassbox/data/robustness/paraphrases_a.jsonl"

if __name__ == "__main__":
    rows = build()
    with open(sys.argv[1] if len(sys.argv) > 1 else OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows[:4] + rows[-4:]:
        print(r["variant"], "|", r["question"])
    print(len(rows), "paraphrases")
