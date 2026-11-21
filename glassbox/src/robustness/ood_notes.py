"""
Out-of-distribution redaction test sets: free-text clinical notes with sentence
structures and PII formats the dataset generators never produce, and names
from many Faker locales. Every PII value's exact span is recorded.

    Set A (development): 3 narrative templates, 8 name locales, address
        formats incl. full state name / no commas. Used on 2026-11-21 to
        find redaction weaknesses, so fixes were developed against it.
    Set B (held out): written and committed BEFORE the fixes; never used
        while developing them. 4 new templates (referral letter, nursing
        note, ED triage, phone message), 8 different locales, and PII formats
        in neither set A nor the data (PO boxes, multi-line addresses,
        "D.O.B. 3.4.1975", "Acct #", extensions...).

    python -m glassbox.src.robustness.ood_notes --set a      # build + evaluate
    python -m glassbox.src.robustness.ood_notes --set b
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from faker import Faker

OUT_DIR = Path("glassbox/data/robustness")
RESULTS_DIR = Path("glassbox/results/robustness")

STATES = {"IL": "Illinois", "TX": "Texas", "CA": "California", "NY": "New York", "OH": "Ohio", "GA": "Georgia"}
MEDS = ["Metformin 500mg PO BID", "Lisinopril 10mg daily", "Glipizide 5mg daily", "Sertraline 50mg daily",
        "Atorvastatin 40mg QHS", "Albuterol inhaler PRN", "Apixaban 5mg BID", "Gabapentin 300mg TID"]
SYMPTOMS = ["chest pain", "shortness of breath", "dizziness", "low back pain", "fatigue", "palpitations"]

SETS = {
    "a": {"seed": 99, "n": 240, "locales": ["en_US", "es_MX", "en_IN", "de_DE", "fr_FR", "it_IT", "pt_BR", "pl_PL"]},
    "b": {"seed": 2026, "n": 240, "locales": ["en_GB", "es_ES", "nl_NL", "tr_TR", "vi_VN", "sv_SE", "en_IE", "da_DK"]},
}


def _assemble(segs) -> Dict[str, Any]:
    text, ents, pos = "", [], 0
    for s, t in segs:
        if t:
            ents.append({"type": t, "value": s, "start": pos, "end": pos + len(s)})
        text += s
        pos += len(s)
    return {"text": text, "entities": ents}


# ---------------------------------------------------------------- set A (as run on 2026-11-21)

def _note_a(rng, fk_name, fk_us) -> Dict[str, Any]:
    name = fk_name.name()
    relative = fk_name.first_name() + " " + fk_name.last_name()
    provider = fk_us.last_name()
    st = rng.choice(list(STATES))
    street, city, zip_ = fk_us.street_address(), fk_us.city(), fk_us.zipcode()
    addr_fmt = rng.randrange(3)
    address = [f"{street}, {city}, {st} {zip_}", f"{street}, {city}, {STATES[st]} {zip_}",
               f"{street} {city} {st} {zip_}"][addr_fmt]
    phone = rng.choice([fk_us.numerify("(###) ###-####"), fk_us.numerify("###.###.####"),
                        fk_us.numerify("+1 ### ### ####")])
    dob_y = rng.randint(1940, 2000)
    dob = rng.choice([f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{dob_y}",
                      f"{rng.choice(['March', 'June', 'October'])} {rng.randint(1, 28)}, {dob_y}"])
    mrn = rng.choice([f"MRN {rng.randint(1000000, 9999999)}", f"MRN: {rng.randint(100000, 999999)}"])
    email = fk_us.user_name() + "@" + rng.choice(["gmail.com", "yahoo.com", "outlook.com"])
    age = 2026 - dob_y
    sym, med = rng.choice(SYMPTOMS), rng.choice(MEDS)
    v = rng.randrange(3)
    if v == 0:
        segs = [(name, "NAME"), (f", a {age}-year-old, was seen today for {sym}. ", None),
                ("Lives at ", None), (address, "ADDRESS"), (" with ", None), (relative, "NAME"),
                (", who can be reached at ", None), (phone, "PHONE_NUMBER"), (". ", None),
                ("Record number ", None), (mrn.split()[-1], "PATIENT_ID"),
                (f". Plan: continue {med}; follow up with Dr. ", None), (provider, "NAME"), (".", None)]
    elif v == 1:
        segs = [("Pt seen in clinic. Name: ", None), (name, "NAME"), (". DOB ", None), (dob, "DATE_OF_BIRTH"),
                (f". Complains of {sym} x 3 days. Started {med}. Pt's spouse ", None), (relative, "NAME"),
                (" present. Email on file ", None), (email, "EMAIL"), (". Home: ", None), (address, "ADDRESS"),
                (".", None)]
    else:
        segs = [("Telephone encounter. Called ", None), (name, "NAME"), (" at ", None), (phone, "PHONE_NUMBER"),
                (f" re: {sym}. Advised to keep taking {med}. ", None), (mrn.split(":")[0].split()[0] + " ", None),
                (mrn.split()[-1], "PATIENT_ID"), (". Message left for Dr. ", None), (provider, "NAME"),
                (".", None)]
    return {**_assemble(segs), "name": name, "addr_fmt": ["comma+code", "full state name", "no commas"][addr_fmt],
            "med": med, "sym": sym, "template": f"a{v}"}


# ---------------------------------------------------------------- set B (held out)

def _note_b(rng, fk_name, fk_us) -> Dict[str, Any]:
    name = fk_name.name()
    relative = fk_name.first_name() + " " + fk_name.last_name()
    provider = fk_us.first_name() + " " + fk_us.last_name()
    st = rng.choice(list(STATES))
    city, zip_ = fk_us.city(), fk_us.zipcode()
    addr_fmt = rng.randrange(4)
    address = [f"{fk_us.street_address()}, {city}, {st} {zip_}",
               f"P.O. Box {rng.randint(10, 9999)}, {city}, {st} {zip_}",
               f"{fk_us.street_address()}\n{city}, {st} {zip_}",
               f"{fk_us.building_number()} {fk_us.street_name()} Apt {rng.randint(1, 40)}{rng.choice('ABCD')}, {city} {st}"][addr_fmt]
    phone = rng.choice([fk_us.numerify("###-###-#### ext ##"), fk_us.numerify("##########"),
                        fk_us.numerify("(###)###-####")])
    dob_y = rng.randint(1938, 2004)
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    dob = rng.choice([f"{dob_y}-{m:02d}-{d:02d}", f"{d:02d}-{['Jan', 'Mar', 'May', 'Aug', 'Nov'][m % 5]}-{dob_y}",
                      f"{m}.{d}.{dob_y}"])
    ident = rng.choice([("Medical Record #: ", str(rng.randint(10000000, 99999999))),
                        ("Acct # ", str(rng.randint(100000, 999999))),
                        ("MRN#", f"A{rng.randint(1000000, 9999999)}")])
    email = (fk_name.first_name() + "." + fk_name.last_name()).lower().replace(" ", "") + "@" + rng.choice(["mail.com", "icloud.com"])
    sym, med = rng.choice(SYMPTOMS), rng.choice(MEDS)
    v = rng.randrange(4)
    if v == 0:    # referral letter
        segs = [("Dear Dr. ", None), (provider, "NAME"), (",\n\nThank you for seeing ", None), (name, "NAME"),
                (" (D.O.B. ", None), (dob, "DATE_OF_BIRTH"), (f"), who has had {sym} for several weeks despite {med}. ", None),
                ("Address:\n", None), (address, "ADDRESS"), ("\nPhone: ", None), (phone, "PHONE_NUMBER"),
                ("\n\nKind regards,\n", None), (fk_us.first_name() + " " + fk_us.last_name(), "NAME")]
    elif v == 1:  # nursing note
        segs = [("Nursing note. ", None), (ident[0], None), (ident[1], "PATIENT_ID"), (". Pt ", None), (name, "NAME"),
                (f" resting comfortably, reports {sym} 4/10. Given {med} as ordered. Daughter ", None), (relative, "NAME"),
                (" visited; she asked to be called on ", None), (phone, "PHONE_NUMBER"), (".", None)]
    elif v == 2:  # ED triage
        segs = [("ED TRIAGE\nPatient: ", None), (name, "NAME"), ("\nDOB: ", None), (dob, "DATE_OF_BIRTH"),
                (f"\nCC: {sym}\nMeds: {med}\nEmergency contact: ", None), (relative, "NAME"), (" ", None),
                (phone, "PHONE_NUMBER"), ("\nEmail: ", None), (email, "EMAIL")]
    else:         # phone message
        segs = [("Msg from ", None), (name, "NAME"), (f" re refill of {med}. Pls call back ", None),
                (phone, "PHONE_NUMBER"), (". Mailing address updated to ", None), (address, "ADDRESS"), (". ", None),
                (ident[0], None), (ident[1], "PATIENT_ID"), (".", None)]
    return {**_assemble(segs), "name": name,
            "addr_fmt": ["comma+code", "PO box", "multi-line", "apt, no ZIP"][addr_fmt],
            "med": med, "sym": sym, "template": ["b-referral", "b-nursing", "b-triage", "b-phone"][v]}


def build(set_name: str) -> List[Dict[str, Any]]:
    cfg = SETS[set_name]
    rng = random.Random(cfg["seed"])
    fk_us = Faker("en_US"); fk_us.seed_instance(cfg["seed"])
    fakers = {loc: Faker(loc) for loc in cfg["locales"]}
    for i, f in enumerate(fakers.values()):
        f.seed_instance(cfg["seed"] + 1 + i)
    make = _note_a if set_name == "a" else _note_b
    notes = []
    for i in range(cfg["n"]):
        loc = cfg["locales"][i % len(cfg["locales"])]
        n = make(rng, fakers[loc], fk_us)
        n.update(note_id=f"ood{set_name}_{i:03d}", name_locale=loc)
        notes.append(n)
    return notes


def evaluate(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fully-masked rate per type / name locale / address format, over-redactions, clinical damage."""
    from glassbox.src.redaction.detector import PIIDetector
    from glassbox.src.redaction.redactor import PIIRedactor
    det, red = PIIDetector(0.5), PIIRedactor()
    stats = defaultdict(lambda: [0, 0])
    over, clinical_damage = [], 0
    for nt in notes:
        text, log = red.redact(nt["text"], det.detect(nt["text"]))
        spans = [(r["start"], r["end"]) for r in log]
        for e in nt["entities"]:
            covered = sum(max(0, min(e["end"], b) - max(e["start"], a)) for a, b in spans)
            full = covered >= e["end"] - e["start"]
            keys = [e["type"]]
            if e["type"] == "NAME" and e["value"] == nt["name"]:
                keys.append(f"NAME[patient, {nt['name_locale']}]")
            elif e["type"] == "NAME":
                keys.append("NAME[relative/provider]")
            if e["type"] == "ADDRESS":
                keys.append(f"ADDRESS[{nt['addr_fmt']}]")
            for k in keys:
                stats[k][0] += 1
                stats[k][1] += full
        over += [r["original_value"] for r in log
                 if not any(r["start"] < e["end"] and e["start"] < r["end"] for e in nt["entities"])]
        clinical_damage += nt["med"].split()[0] not in text or nt["sym"] not in text
    total = [sum(v[0] for k, v in stats.items() if "[" not in k), sum(v[1] for k, v in stats.items() if "[" not in k)]
    return {
        "notes": len(notes),
        "overall_fully_masked": round(total[1] / total[0], 4),
        "fully_masked": {k: {"n": n, "rate": round(m / n, 4)} for k, (n, m) in sorted(stats.items())},
        "over_redactions": len(over),
        "over_redaction_examples": sorted(set(over))[:15],
        "notes_with_clinical_damage": clinical_damage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / f"ood_notes_{args.set}.json"
    if data_path.exists():
        notes = json.loads(data_path.read_text())   # frozen copy
    else:
        notes = build(args.set)
        data_path.write_text(json.dumps(notes, indent=1))
    res = evaluate(notes)
    print(f"OOD set {args.set}: {res['notes']} notes, overall fully masked {res['overall_fully_masked']:.3f}")
    for k, v in res["fully_masked"].items():
        print(f"  {k:<30} {v['rate']:.2f}  (n={v['n']})")
    print(f"  over-redactions: {res['over_redactions']}  e.g. {res['over_redaction_examples'][:8]}")
    print(f"  notes with a drug or symptom redacted: {res['notes_with_clinical_damage']}")
    (RESULTS_DIR / f"ood_redaction_{args.set}{args.tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
