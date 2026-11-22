"""
The scripted demo: six questions that each show one property of the system.

Used by the dashboard's Demo page (one-click buttons) and described in DEMO.md.
The questions were chosen from the stored evaluation runs because they behave
reliably (decoding is greedy, so each one gives the same answer every time):
    fact / aggregate   VERIFIED with grounded accuracy 1.0 in the evaluation
    named person       redacted system refuses; raw baseline answers with the true value
    no matching notes  query analysis finds the conditions match no note -> refusal
"""

DEMO_STEPS = [
    {
        "key": "fact",
        "title": "1 · A clinical fact",
        "question": "What medication was prescribed for the Essential hypertension patient whose note is dated 2024-11-10?",
        "configs": ["redacted"],
        "point_out": [
            "Query analysis turned 'Essential hypertension' and the date into filters: only that visit's note is searched.",
            "The answer is one claim citing the PLAN section it came from (click the citation to jump to the chunk).",
            "The verifier, a separate NLI model, checked the claim against that chunk: SUPPORTED.",
            "The chunk text has [NAME], [PATIENT_ID] etc. The model never saw who the patient is.",
        ],
    },
    {
        "key": "aggregate",
        "title": "2 · A question across several patients",
        "question": "Which medications were prescribed to Osteoarthritis patients who presented with reduced range of motion in February 2025?",
        "configs": ["redacted"],
        "point_out": [
            "Two-hop retrieval: notes are chosen by the symptom (selected_by = question), then each note's medication "
            "section is added (selected_by = ask).",
            "One claim per qualifying patient, each citing that patient's own sections; every claim is verified.",
        ],
    },
    {
        "key": "identity",
        "title": "3 · Asking about a named person: privacy",
        "question": "What medication is Steven Cox prescribed?",
        "configs": ["redacted", "raw"],
        "point_out": [
            "GlassBox (left): the name is redacted before anything else sees it, and the guard refuses BEFORE the "
            "model is called. The records are de-identified, so any answer would describe someone else.",
            "Raw baseline (right), the same pipeline without privacy: it finds the note by name and gives the person's "
            "medication. This is the leak GlassBox prevents.",
            "Evaluation: GlassBox answered 0 of 84 such test questions; the raw baseline answered 42 and gave the true value for 16.",
        ],
    },
    {
        "key": "fallback",
        "title": "4 · Conditions that match no record",
        "question": "What medication was prescribed for the Asthma patient whose note is dated 2019-03-14?",
        "configs": ["redacted"],
        "point_out": [
            "No asthma note exists for that date. Rather than answer from unrelated notes, the system says so.",
            "The refusal is a code rule, not a model judgement, so it can't be talked out of it, and the trace states why.",
        ],
    },
]

# Claim edits that should make the verifier object (step 5 of the demo: "try to fool the verifier")
VERIFIER_TIPS = [
    "Change the dose (e.g. 25mg → 50mg): expect CONTRADICTED.",
    "Swap the drug for another one: expect CONTRADICTED or UNSUPPORTED.",
    "Add a fact the note doesn't state (\"… and was referred to cardiology\"): expect UNSUPPORTED.",
]
