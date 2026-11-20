"""
Clinical-term guard for NER detections.

spaCy's general-purpose NER was trained on news and web text, so it sometimes
tags clinical vocabulary as people or places: the drug "Glipizide" as a
PERSON, "SpO2" as a PERSON, a blood pressure "158/62" as a LOCATION, or the
field label "Email:" as a PERSON. Each such false positive destroys clinical
content (a redacted drug name makes a medication question unanswerable) without
protecting anything.

The guard drops a NAME/ADDRESS detection from the NER model when the span is
made up entirely of clinical vocabulary or is a measurement. It is applied
ONLY to NER-sourced types; pattern recognizers (IDs, phones, emails, DOBs,
structured addresses) are never filtered.

The drug list is a general list of commonly prescribed US generics (roughly
the top 150 by prescription volume) plus common insulin/inhaler names - not
only the drugs that appear in the synthetic data - so the guard generalises
beyond the generator's templates.
"""

import re

DRUGS = {
    # cardiovascular
    "amlodipine", "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
    "lisinopril", "enalapril", "ramipril", "benazepril", "quinapril", "losartan", "valsartan",
    "irbesartan", "olmesartan", "telmisartan", "candesartan", "metoprolol", "atenolol",
    "carvedilol", "propranolol", "bisoprolol", "nebivolol", "labetalol", "hydrochlorothiazide",
    "chlorthalidone", "furosemide", "torsemide", "bumetanide", "spironolactone", "eplerenone",
    "diltiazem", "verapamil", "nifedipine", "hydralazine", "clonidine", "isosorbide",
    "nitroglycerin", "digoxin", "amiodarone", "warfarin", "apixaban", "rivaroxaban",
    "dabigatran", "clopidogrel", "ticagrelor", "prasugrel", "aspirin", "ezetimibe",
    "fenofibrate", "gemfibrozil", "sacubitril", "ranolazine",
    # diabetes / endocrine
    "metformin", "glipizide", "glimepiride", "glyburide", "pioglitazone", "sitagliptin",
    "linagliptin", "empagliflozin", "dapagliflozin", "canagliflozin", "liraglutide",
    "semaglutide", "dulaglutide", "tirzepatide", "insulin", "glargine", "lispro", "aspart",
    "detemir", "degludec", "levothyroxine", "liothyronine", "methimazole",
    # respiratory
    "albuterol", "levalbuterol", "ipratropium", "tiotropium", "umeclidinium", "fluticasone",
    "budesonide", "mometasone", "beclomethasone", "salmeterol", "formoterol", "vilanterol",
    "montelukast", "theophylline", "roflumilast", "prednisone", "prednisolone",
    "methylprednisolone", "dexamethasone",
    # psychiatric / neuro
    "sertraline", "escitalopram", "citalopram", "fluoxetine", "paroxetine", "venlafaxine",
    "desvenlafaxine", "duloxetine", "bupropion", "mirtazapine", "trazodone", "buspirone",
    "amitriptyline", "nortriptyline", "quetiapine", "aripiprazole", "risperidone",
    "olanzapine", "lamotrigine", "lithium", "alprazolam", "lorazepam", "clonazepam",
    "diazepam", "zolpidem", "gabapentin", "pregabalin", "topiramate", "levetiracetam",
    "sumatriptan", "donepezil", "memantine", "carbidopa", "levodopa",
    # GI
    "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "famotidine", "ondansetron",
    "metoclopramide", "docusate", "senna", "polyethylene", "loperamide",
    # pain / musculoskeletal
    "acetaminophen", "ibuprofen", "naproxen", "meloxicam", "celecoxib", "diclofenac",
    "tramadol", "oxycodone", "hydrocodone", "morphine", "cyclobenzaprine", "methocarbamol",
    "tizanidine", "baclofen", "allopurinol", "colchicine", "febuxostat",
    # anti-infective
    "amoxicillin", "clavulanate", "azithromycin", "doxycycline", "cephalexin",
    "ciprofloxacin", "levofloxacin", "nitrofurantoin", "sulfamethoxazole", "trimethoprim",
    "clindamycin", "metronidazole", "valacyclovir", "acyclovir", "fluconazole",
    # renal / other
    "epo", "erythropoietin", "epoetin", "calcitriol", "sevelamer", "calcium", "sodium",
    "bicarbonate", "potassium", "chloride", "magnesium", "iron", "ferrous", "sulfate",
    "carbonate", "citrate", "vitamin", "cholecalciferol", "finasteride", "tamsulosin",
    "oxybutynin", "sildenafil", "alendronate",
}

# Clinical abbreviations, vitals, labs and routes/frequencies
ABBREVIATIONS = {
    "spo2", "sao2", "o2", "bp", "hr", "rr", "temp", "bmi", "ecg", "ekg", "cbc", "bmp", "cmp",
    "hba1c", "a1c", "egfr", "gfr", "ldl", "hdl", "tsh", "inr", "ptt", "bun", "wbc", "rbc",
    "po", "iv", "im", "sc", "sq", "prn", "qd", "qhs", "bid", "tid", "qid", "qam", "qpm",
    "mg", "mcg", "ml", "xl", "er", "sr", "dr", "hfa", "mdi", "copd", "ckd", "cad", "gerd",
    "s1", "s2", "ra", "nka", "nkda", "sob", "doe", "nad", "rrr", "cta",
}

# Field labels / section headings that NER mistakes for names at line starts
FIELD_LABELS = {
    "email", "e-mail", "phone", "tel", "telephone", "cell", "mobile", "fax", "address",
    "patient", "provider", "attending", "physician", "contact", "emergency", "date", "dob",
    "mrn", "id", "signature", "signed", "electronically", "referring", "diagnosis",
    "assessment", "plan", "subjective", "objective", "impression", "history", "allergies",
    "medications", "vitals",
}

CLINICAL_TERMS = DRUGS | ABBREVIATIONS | FIELD_LABELS

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
# A measurement, optionally preceded by a vitals label: "158/62", "BP 152/68", "97%", "98.6"
_MEASUREMENT = re.compile(r"^(?:[A-Za-z][A-Za-z0-9]{0,4}\s*:?\s*)?\d+(?:[/.]\d+)?\s*%?$")


def is_clinical_term(span: str) -> bool:
    """
    True if a NER span is clinical vocabulary or a measurement, not PII.

    A span is clinical if it is a measurement ("158/62", "BP 152/68"), or if
    every word in it is in the clinical lexicon ("Glipizide", "SpO2",
    "Email", "Calcium Carbonate").
    """
    s = span.strip().strip(":,.;")
    if not s:
        return False
    if _MEASUREMENT.match(s):
        return True
    tokens = [t.lower() for t in _TOKEN.findall(s)]
    return bool(tokens) and all(t in CLINICAL_TERMS for t in tokens)
