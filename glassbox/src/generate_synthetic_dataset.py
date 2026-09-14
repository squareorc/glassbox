"""
GlassBox synthetic dataset generator.

Generates synthetic clinical notes with injected PII (with known ground-truth
character spans, for redaction precision/recall evaluation) plus a companion
QA evaluation set (for retrieval and answer-utility evaluation later).

Usage:
    pip install faker
    python generate_synthetic_dataset.py
"""

import json
import random
from pathlib import Path

from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT_DIR = Path("glassbox/data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    so we have ground truth for redaction evaluation. Placeholders are
    substituted one at a time so offsets stay correct as the string grows.
    """
    text = template
    entities = []
    for key, (value, etype) in values.items():
        placeholder = "{" + key + "}"
        idx = text.find(placeholder)
        if idx == -1:
            continue
        text = text[:idx] + value + text[idx + len(placeholder):]
        if etype is not None:
            entities.append({
                "type": etype,
                "value": value,
                "start": idx,
                "end": idx + len(value),
            })
    return text, entities


def generate_document(doc_id):
    condition = random.choice(CONDITIONS)
    template = random.choice(NOTE_TEMPLATES)

    name = fake.name()
    patient_id = fake.bothify(text="??######")
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat()
    visit_date = fake.date_between(start_date="-2y", end_date="today").isoformat()
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
    }


def build_qa_eval_set(documents, n_queries=20):
    """
    Build a small ground-truth QA set: for each query, which documents are
    relevant (by diagnosis) and what the expected answer content is. Used to
    evaluate retrieval precision/recall and answer utility later, and as a
    source of "known correct" claims to contrast against injected wrong ones
    when testing the verification stage.
    """
    by_diagnosis = {}
    for doc in documents:
        by_diagnosis.setdefault(doc["diagnosis"], []).append(doc["doc_id"])

    diagnoses = list(by_diagnosis.keys())
    random.shuffle(diagnoses)

    qa_set = []
    for diagnosis in diagnoses[:n_queries]:
        relevant_docs = by_diagnosis[diagnosis]
        meds = sorted({
            m for doc in documents if doc["diagnosis"] == diagnosis for m in doc["medications"]
        })
        qa_set.append({
            "query": f"What medications are associated with patients diagnosed with {diagnosis}?",
            "relevant_doc_ids": relevant_docs,
            "expected_medications": meds,
        })
    return qa_set


def main():
    documents = [generate_document(i) for i in range(N_DOCUMENTS)]
    qa_set = build_qa_eval_set(documents)

    docs_path = OUTPUT_DIR / "synthetic_notes.jsonl"
    with docs_path.open("w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    qa_path = OUTPUT_DIR / "qa_eval_set.jsonl"
    with qa_path.open("w") as f:
        for qa in qa_set:
            f.write(json.dumps(qa) + "\n")

    print(f"Wrote {len(documents)} documents to {docs_path}")
    print(f"Wrote {len(qa_set)} QA eval examples to {qa_path}")


if __name__ == "__main__":
    main()