"""
GlassBox synthetic dataset generator.

Generates synthetic clinical notes with injected PII (with known ground-truth
character spans, for redaction precision/recall evaluation) plus a companion
QA evaluation set (for retrieval and answer-utility evaluation later).

Usage (from the repo root):
    python -m glassbox.src.generate_synthetic_dataset
"""

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from glassbox.src.qa.build_qa_set import build_qa_set, write_qa_set

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT_DIR = Path("glassbox/data/raw/synthetic")

# Fixed "today" for the synthetic world, so regeneration is reproducible
# regardless of the date the script is run on.
REFERENCE_DATE = date(2026, 9, 16)

N_DOCUMENTS = 250

# Each condition bundles a diagnosis with plausible medications and symptoms,
# so generated notes stay clinically coherent instead of random word salad.
CONDITIONS = [
    {
        "diagnosis": "Type 2 diabetes",
        "medications": ["Metformin 500mg twice daily", "Metformin 1000mg once daily"],
        "symptoms": ["increased thirst", "fatigue", "blurred vision", "frequent urination"],
    },
    {
        "diagnosis": "Hypertension",
        "medications": ["Lisinopril 10mg daily", "Amlodipine 5mg daily"],
        "symptoms": ["headache", "dizziness", "shortness of breath"],
    },
    {
        "diagnosis": "Asthma",
        "medications": ["Albuterol inhaler as needed", "Fluticasone inhaler twice daily"],
        "symptoms": ["wheezing", "chest tightness", "shortness of breath"],
    },
    {
        "diagnosis": "Major depressive disorder",
        "medications": ["Sertraline 50mg daily", "Escitalopram 10mg daily"],
        "symptoms": ["low mood", "fatigue", "difficulty concentrating", "sleep disturbance"],
    },
    {
        "diagnosis": "Hypothyroidism",
        "medications": ["Levothyroxine 75mcg daily", "Levothyroxine 100mcg daily"],
        "symptoms": ["fatigue", "weight gain", "cold intolerance"],
    },
    {
        "diagnosis": "Generalized anxiety disorder",
        "medications": ["Buspirone 15mg twice daily", "Escitalopram 10mg daily"],
        "symptoms": ["excessive worry", "restlessness", "muscle tension"],
    },
    {
        "diagnosis": "Osteoarthritis",
        "medications": ["Ibuprofen 400mg as needed", "Acetaminophen 500mg as needed"],
        "symptoms": ["joint pain", "stiffness", "reduced range of motion"],
    },
    {
        "diagnosis": "Gastroesophageal reflux disease",
        "medications": ["Omeprazole 20mg daily", "Pantoprazole 40mg daily"],
        "symptoms": ["heartburn", "regurgitation", "chest discomfort"],
    },
]

# Multiple templates keep the corpus from looking mechanically repetitive.
NOTE_TEMPLATES = [
    "{name}, Patient ID {patient_id}, presented on {visit_date} with {symptom}. "
    "The patient was diagnosed with {diagnosis} and prescribed {medication}. "
    "Contact number on file: {phone}. Follow-up scheduled in {followup_weeks} weeks.",

    "Patient {name} (DOB: {dob}, ID {patient_id}) reported {symptom} during a visit on "
    "{visit_date}. Assessment: {diagnosis}. Treatment plan includes {medication}. "
    "Patient may be reached at {email} for follow-up scheduling.",

    "On {visit_date}, {name} (Patient ID: {patient_id}) was evaluated for {symptom}. "
    "Diagnosis: {diagnosis}. Current medication: {medication}. "
    "Home address on file: {address}.",
]


def make_entities_and_text(template, values):
    """
    Fill a template and track the character span of each injected PII value,
    so we have ground truth for redaction evaluation.

    Strategy: parse the template into tokens (literal text vs {placeholder}),
    build the result string from left to right, tracking the exact character
    offset where each value lands.
    """
    # Split template by placeholders like {name}, {patient_id}, etc.
    pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")
    tokens = []
    last_end = 0

    for match in pattern.finditer(template):
        # Literal text before placeholder
        if match.start() > last_end:
            tokens.append({"type": "literal", "text": template[last_end:match.start()]})

        # Placeholder
        key = match.group(1)
        tokens.append({"type": "placeholder", "key": key})
        last_end = match.end()

    # Any trailing literal text
    if last_end < len(template):
        tokens.append({"type": "literal", "text": template[last_end:]})

    # Assemble the final string left-to-right, tracking entity spans
    full_text = []
    entities = []
    current_offset = 0

    for token in tokens:
        if token["type"] == "literal":
            full_text.append(token["text"])
            current_offset += len(token["text"])
        else:
            key = token["key"]
            if key in values:
                val, etype = values[key]
                val_str = str(val)
                start_idx = current_offset
                end_idx = current_offset + len(val_str)

                full_text.append(val_str)
                current_offset = end_idx

                if etype is not None:
                    entities.append({
                        "type": etype,
                        "value": val_str,
                        "start": start_idx,
                        "end": end_idx,
                    })

    return "".join(full_text), entities


def generate_document(doc_id):
    condition = random.choice(CONDITIONS)
    template = random.choice(NOTE_TEMPLATES)

    name = fake.name()
    patient_id = fake.bothify(text="??######")
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    # Ages 18-90 as of REFERENCE_DATE; visits in the 2 years before it
    dob = (REFERENCE_DATE - timedelta(days=random.randint(18 * 365 + 5, 90 * 365))).isoformat()
    visit_date = (REFERENCE_DATE - timedelta(days=random.randint(0, 730))).isoformat()
    medication = random.choice(condition["medications"])
    symptom = random.choice(condition["symptoms"])
    followup_weeks = str(random.choice([2, 4, 6, 8, 12]))

    # (value, entity_type) — entity_type is None for clinically relevant but
    # non-identifying content, so it's never marked as PII to redact.
    values = {
        "name": (name, "NAME"),
        "patient_id": (patient_id, "PATIENT_ID"),
        "phone": (phone, "PHONE_NUMBER"),
        "email": (email, "EMAIL"),
        "address": (address, "ADDRESS"),
        "dob": (dob, "DATE_OF_BIRTH"),
        "visit_date": (visit_date, None),
        "diagnosis": (condition["diagnosis"], None),
        "medication": (medication, None),
        "symptom": (symptom, None),
        "followup_weeks": (followup_weeks, None),
    }

    text, entities = make_entities_and_text(template, values)

    return {
        "doc_id": f"doc_{doc_id:04d}",
        "text": text,
        "entities": entities,      # ground truth for redaction precision/recall
        "diagnosis": condition["diagnosis"],
        "medications": [medication],
        "patient_name": name,      # kept for building/inspecting the QA set
        # Structured ground truth for the QA set. Contains PII, so it must
        # never be passed to retrieval/generation.
        "facts": {k: v[0] for k, v in values.items()},
        "fields_in_text": sorted(set(re.findall(r"\{([a-zA-Z0-9_]+)\}", template))),
    }


def main():
    documents = [generate_document(i) for i in range(N_DOCUMENTS)]
    qa_set = build_qa_set(documents, profile="synthetic")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    docs_path = OUTPUT_DIR / "synthetic_notes.jsonl"
    with docs_path.open("w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    qa_path = OUTPUT_DIR / "qa_eval_set.jsonl"
    write_qa_set(qa_set, qa_path)

    print(f"Wrote {len(documents)} documents to {docs_path}")
    print(f"Wrote {len(qa_set)} QA eval examples to {qa_path}")


if __name__ == "__main__":
    main()