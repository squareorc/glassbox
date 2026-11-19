"""
Enhanced GlassBox synthetic dataset generator - realistic clinical notes.

Generates synthetic clinical notes that more closely mimic real-world EHR data:
- Multiple note types (progress notes, discharge summaries, consultation notes)
- Realistic clinical language and structure
- Varied PII formats and placements
- Lab values, vital signs, and clinical observations
- Multi-condition patients
- Longitudinal visit patterns

Usage:
    python -m glassbox.src.generate_realistic_dataset --size large
"""

import json
import random
import argparse
from pathlib import Path
import re
from datetime import date, timedelta

from faker import Faker

from glassbox.src.qa.build_qa_set import build_qa_set, write_qa_set

fake = Faker()

# Fixed "today" for the synthetic world, so regeneration is byte-for-byte
# reproducible regardless of the date the script is run on.
REFERENCE_DATE = date(2026, 9, 16)


class RealisticClinicalGenerator:
    """Generates realistic synthetic clinical documentation."""

    def __init__(self, seed=42):
        random.seed(seed)
        Faker.seed(seed)
        self.fake = fake

        # Expanded clinical content
        self.conditions = [
            {
                "diagnosis": "Type 2 diabetes mellitus",
                "icd10": "E11.9",
                "medications": [
                    "Metformin 500mg PO BID",
                    "Metformin 1000mg PO daily",
                    "Glipizide 5mg PO daily",
                    "Insulin glargine 20 units SubQ at bedtime"
                ],
                "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision", "neuropathy"],
                "labs": ["HbA1c", "fasting glucose", "lipid panel"],
            },
            {
                "diagnosis": "Essential hypertension",
                "icd10": "I10",
                "medications": [
                    "Lisinopril 10mg PO daily",
                    "Amlodipine 5mg PO daily",
                    "Losartan 50mg PO daily",
                    "Hydrochlorothiazide 25mg PO daily"
                ],
                "symptoms": ["headache", "dizziness", "shortness of breath", "chest pain"],
                "labs": ["BMP", "renal function panel"],
            },
            {
                "diagnosis": "Asthma",
                "icd10": "J45.909",
                "medications": [
                    "Albuterol HFA 90mcg inhaler 2 puffs Q4H PRN",
                    "Fluticasone 110mcg inhaler 2 puffs BID",
                    "Montelukast 10mg PO QHS"
                ],
                "symptoms": ["wheezing", "dyspnea", "chest tightness", "cough"],
                "labs": ["spirometry", "peak flow"],
            },
            {
                "diagnosis": "Major depressive disorder",
                "icd10": "F32.9",
                "medications": [
                    "Sertraline 50mg PO daily",
                    "Escitalopram 10mg PO daily",
                    "Bupropion XL 150mg PO daily",
                    "Fluoxetine 20mg PO daily"
                ],
                "symptoms": ["depressed mood", "anhedonia", "insomnia", "fatigue", "poor concentration"],
                "labs": ["TSH", "CBC"],
            },
            {
                "diagnosis": "Chronic kidney disease stage 3",
                "icd10": "N18.3",
                "medications": [
                    "Sodium bicarbonate 650mg PO TID",
                    "Calcium carbonate 500mg PO TID with meals",
                    "EPO 4000 units SubQ weekly"
                ],
                "symptoms": ["fatigue", "edema", "nausea", "decreased appetite"],
                "labs": ["BMP", "eGFR", "creatinine", "phosphorus"],
            },
            {
                "diagnosis": "Coronary artery disease",
                "icd10": "I25.10",
                "medications": [
                    "Aspirin 81mg PO daily",
                    "Atorvastatin 40mg PO QHS",
                    "Metoprolol tartrate 25mg PO BID",
                    "Clopidogrel 75mg PO daily"
                ],
                "symptoms": ["chest pain", "dyspnea on exertion", "fatigue"],
                "labs": ["troponin", "lipid panel", "ECG"],
            },
            {
                "diagnosis": "COPD",
                "icd10": "J44.9",
                "medications": [
                    "Tiotropium 18mcg inhaler daily",
                    "Albuterol/Ipratropium inhaler QID",
                    "Prednisone 10mg PO daily"
                ],
                "symptoms": ["chronic cough", "sputum production", "dyspnea", "wheezing"],
                "labs": ["spirometry", "chest X-ray", "ABG"],
            },
            {
                "diagnosis": "Osteoarthritis",
                "icd10": "M19.90",
                "medications": [
                    "Ibuprofen 400mg PO TID with food",
                    "Acetaminophen 650mg PO Q6H PRN",
                    "Celecoxib 200mg PO daily",
                    "Tramadol 50mg PO Q6H PRN"
                ],
                "symptoms": ["joint pain", "morning stiffness", "reduced range of motion", "swelling"],
                "labs": ["ESR", "CRP", "X-ray"],
            },
        ]

        # Note templates with realistic EHR structure
        self.note_templates = {
            "progress": [
                self._progress_note_template_1,
                self._progress_note_template_2,
                self._progress_note_template_3,
            ],
            "discharge": [
                self._discharge_summary_template,
            ],
            "consultation": [
                self._consultation_note_template,
            ],
        }

        # Varied phone formats
        self.phone_formats = [
            lambda: self.fake.numerify("(###) ###-####"),
            lambda: self.fake.numerify("###-###-####"),
            lambda: self.fake.numerify("###.###.####"),
            lambda: self.fake.numerify("+1-###-###-#### x###"),
            lambda: self.fake.numerify("001-###-###-####"),
            lambda: self.fake.numerify("##########"),
        ]

        # Varied patient ID formats
        self.patient_id_formats = [
            lambda: self.fake.bothify("??######"),
            lambda: self.fake.bothify("?-######"),
            lambda: f"MRN{self.fake.numerify('#######')}",
            lambda: f"P{self.fake.numerify('######')}",
        ]

    def _progress_note_template_1(self, values):
        """Standard progress note format."""
        return f"""PROGRESS NOTE

Patient: {values['name']}
MRN: {values['patient_id']}
DOB: {values['dob']}
Date of Service: {values['visit_date']}

CHIEF COMPLAINT: {values['symptom']}

HPI: {values['age']}-year-old {values['gender']} with history of {values['diagnosis']} presents for follow-up. Patient reports {values['symptom']} {values['symptom_duration']}. {values['review_of_systems']}

VITALS:
BP: {values['bp']} | HR: {values['hr']} | Temp: {values['temp']}°F | RR: {values['rr']} | SpO2: {values['spo2']}%

PHYSICAL EXAM: {values['physical_exam']}

ASSESSMENT/PLAN:
1. {values['diagnosis']} (ICD-10: {values['icd10']})
   - Continue {values['medication']}
   - {values['plan_item']}
   - Follow-up in {values['followup_weeks']} weeks

Lab orders: {values['lab_orders']}

Contact: Patient can be reached at {values['phone']} or {values['email']}.

Electronically signed by {values['provider']}, MD
"""

    def _progress_note_template_2(self, values):
        """SOAP note format."""
        return f"""SOAP NOTE - {values['visit_date']}

PATIENT INFORMATION
Name: {values['name']}
Patient ID: {values['patient_id']}
Date of Birth: {values['dob']} (Age: {values['age']})
Phone: {values['phone']}

SUBJECTIVE: Patient presents with {values['symptom']}. {values['subjective_detail']}

OBJECTIVE:
Vitals: BP {values['bp']}, HR {values['hr']}, Temp {values['temp']}°F, RR {values['rr']}, SpO2 {values['spo2']}%
{values['physical_exam']}

ASSESSMENT:
{values['diagnosis']} ({values['icd10']})

PLAN:
- Prescribed {values['medication']}
- {values['plan_item']}
- Patient education provided
- RTC {values['followup_weeks']} weeks

Contact email: {values['email']}
Address on file: {values['address']}

Provider: {values['provider']}, MD
"""

    def _progress_note_template_3(self, values):
        """Brief progress note."""
        return f"""Date: {values['visit_date']}
Patient: {values['name']} (ID: {values['patient_id']}, DOB {values['dob']})

S: {values['symptom']}. Patient reports {values['symptom_duration']}.

O: BP {values['bp']}, HR {values['hr']}, Temp {values['temp']}°F. {values['physical_exam']}

A: {values['diagnosis']}

P: Continue {values['medication']}. {values['plan_item']}. Follow-up {values['followup_weeks']} weeks. Labs: {values['lab_orders']}.

Contact: {values['phone']} (cell), {values['email']}

- {values['provider']}, MD
"""

    def _discharge_summary_template(self, values):
        """Hospital discharge summary."""
        return f"""DISCHARGE SUMMARY

PATIENT: {values['name']}
MRN: {values['patient_id']}
DOB: {values['dob']}
ADMISSION DATE: {values['admission_date']}
DISCHARGE DATE: {values['visit_date']}

PRINCIPAL DIAGNOSIS: {values['diagnosis']} ({values['icd10']})

HOSPITAL COURSE:
{values['age']}-year-old {values['gender']} admitted with {values['symptom']}. {values['hospital_course']}

DISCHARGE MEDICATIONS:
1. {values['medication']}
2. {values['additional_med']}

DISCHARGE INSTRUCTIONS:
- {values['discharge_instruction']}
- Follow-up with PCP in {values['followup_weeks']} weeks
- Contact number: {values['phone']}
- Emergency contact: {values['emergency_contact']} at {values['emergency_phone']}

Address: {values['address']}
Email: {values['email']}

Attending: {values['provider']}, MD
"""

    def _consultation_note_template(self, values):
        """Specialist consultation note."""
        return f"""CONSULTATION NOTE

Date: {values['visit_date']}
Patient: {values['name']}
MRN: {values['patient_id']} | DOB: {values['dob']}
Referring Physician: {values['referring_provider']}

REASON FOR CONSULTATION: {values['symptom']}

HISTORY: {values['age']}-year-old {values['gender']} referred for evaluation of {values['diagnosis']}. {values['consultation_history']}

REVIEW OF RECORDS: {values['record_review']}

EXAMINATION:
Vitals: BP {values['bp']}, HR {values['hr']}, Temp {values['temp']}°F
{values['physical_exam']}

IMPRESSION: {values['diagnosis']} ({values['icd10']})

RECOMMENDATIONS:
1. Start {values['medication']}
2. {values['consultation_recommendation']}
3. {values['plan_item']}
4. Recheck labs: {values['lab_orders']}

Patient contact: {values['phone']} or {values['email']}
Address: {values['address']}

Thank you for this referral.

{values['provider']}, MD
Specialty: {values['specialty']}
"""

    def generate_note_values(self, condition, note_type="progress"):
        """Generate all values needed for a clinical note."""
        name = self.fake.name()
        gender = random.choice(["male", "female"])
        age = random.randint(25, 85)

        # DOB consistent with age as of REFERENCE_DATE (birth years ~1940-2001)
        dob_date = REFERENCE_DATE.replace(year=REFERENCE_DATE.year - age) - timedelta(
            days=random.randint(0, 364)
        )
        dob = dob_date.isoformat()

        # Visit date in the 2 years before REFERENCE_DATE
        visit_dt = REFERENCE_DATE - timedelta(days=random.randint(0, 730))
        visit_date = visit_dt.isoformat()

        # Patient ID - varied formats
        patient_id = random.choice(self.patient_id_formats)()

        # Phone - varied formats
        phone = random.choice(self.phone_formats)()

        # Email
        email = self.fake.email()

        # Address
        address = self.fake.address().replace("\n", ", ")

        # Vitals
        bp_systolic = random.randint(110, 160)
        bp_diastolic = random.randint(60, 95)
        bp = f"{bp_systolic}/{bp_diastolic}"
        hr = random.randint(60, 100)
        temp = round(random.uniform(97.0, 99.5), 1)
        rr = random.randint(12, 20)
        spo2 = random.randint(95, 100)

        # Clinical content
        symptom = random.choice(condition["symptoms"])
        medication = random.choice(condition["medications"])
        lab_orders = ", ".join(random.sample(condition["labs"], min(2, len(condition["labs"]))))

        # Additional varied content
        symptom_duration = random.choice([
            "for the past 2 weeks",
            "since last visit",
            "x3 days",
            "ongoing for several months"
        ])

        physical_exam = random.choice([
            "No acute distress. Heart RRR, no murmurs. Lungs CTAB.",
            "Alert and oriented x3. HEENT normal. CV regular. Resp unlabored.",
            "General: Appears well. CV: S1 S2 normal. Pulm: Clear bilaterally.",
        ])

        plan_item = random.choice([
            "Continue current regimen",
            "Lifestyle modifications discussed",
            "Patient counseled on medication adherence",
            "Dietary modifications advised",
        ])

        provider = self.fake.name()
        specialty = random.choice([
            "Internal Medicine", "Cardiology", "Endocrinology",
            "Pulmonology", "Nephrology", "Primary Care"
        ])

        values = {
            "name": (name, "NAME"),
            "patient_id": (patient_id, "PATIENT_ID"),
            "dob": (dob, "DATE_OF_BIRTH"),
            "visit_date": (visit_date, None),
            "phone": (phone, "PHONE_NUMBER"),
            "email": (email, "EMAIL"),
            "address": (address, "ADDRESS"),
            "diagnosis": (condition["diagnosis"], None),
            "icd10": (condition["icd10"], None),
            "medication": (medication, None),
            "age": (str(age), None),
            "gender": (gender, None),
            "bp": (bp, None),
            "hr": (str(hr), None),
            "temp": (str(temp), None),
            "rr": (str(rr), None),
            "spo2": (str(spo2), None),
            "symptom": (symptom, None),
            "symptom_duration": (symptom_duration, None),
            "physical_exam": (physical_exam, None),
            "plan_item": (plan_item, None),
            "lab_orders": (lab_orders, None),
            "followup_weeks": (str(random.choice([2, 4, 6, 8, 12])), None),
            "provider": (provider, "NAME"),  # Provider names are PII
            "specialty": (specialty, None),
            "referring_provider": (self.fake.name(), "NAME"),  # Referring provider names are PII
        }

        # Additional values for specific note types
        if note_type == "discharge":
            values.update({
                # Admission precedes discharge (visit_date) by 1-14 days
                "admission_date": ((visit_dt - timedelta(days=random.randint(1, 14))).isoformat(), None),
                "hospital_course": ("Patient responded well to treatment.", None),
                "additional_med": (random.choice(condition["medications"]), None),
                "discharge_instruction": ("Resume normal activities as tolerated", None),
                "emergency_contact": (self.fake.name(), "NAME"),  # Emergency contact names are PII
                "emergency_phone": (random.choice(self.phone_formats)(), "PHONE_NUMBER"),
            })

        if note_type == "consultation":
            values.update({
                "consultation_history": ("Patient has been experiencing symptoms intermittently.", None),
                "record_review": ("Previous labs reviewed.", None),
                "consultation_recommendation": ("Consider additional imaging", None),
            })

        if note_type == "progress":
            values.update({
                "review_of_systems": ("Denies fever, chills, or weight changes.", None),
                "subjective_detail": ("Reports some improvement with current medications.", None),
            })

        return values

    def make_entities_and_text(self, template_func, values):
        """
        Generate text from template and track PII entity positions.
        Uses the same token-based approach as the original generator.
        """
        # Unpack tuples for template - templates expect just the values, not (value, etype) tuples
        template_values = {k: v[0] for k, v in values.items()}

        # Generate the text by calling the template function
        text = template_func(template_values)

        # Now find all PII entities in the generated text
        entities = []

        for key, (value, etype) in values.items():
            if etype is None:
                continue

            # Find all occurrences of this PII value in the text
            value_str = str(value)
            start = 0
            while True:
                idx = text.find(value_str, start)
                if idx == -1:
                    break

                entities.append({
                    "type": etype,
                    "value": value_str,
                    "start": idx,
                    "end": idx + len(value_str),
                })
                start = idx + len(value_str)

        # Sort by start position and remove duplicates at same position
        entities.sort(key=lambda e: (e["start"], e["end"]))
        unique_entities = []
        seen_spans = set()
        for e in entities:
            span_key = (e["start"], e["end"])
            if span_key not in seen_spans:
                unique_entities.append(e)
                seen_spans.add(span_key)

        return text, unique_entities

    def generate_document(self, doc_id):
        """Generate one realistic clinical document."""
        # Pick condition and note type
        condition = random.choice(self.conditions)
        note_type = random.choice(["progress", "progress", "progress", "discharge", "consultation"])

        # Generate values
        values = self.generate_note_values(condition, note_type)

        # Pick template
        template_func = random.choice(self.note_templates[note_type])

        # Generate text and track entities
        text, entities = self.make_entities_and_text(template_func, values)

        medications = [values["medication"][0]]
        if "additional_med" in values and values["additional_med"][0] not in medications:
            medications.append(values["additional_med"][0])

        return {
            "doc_id": f"doc_{doc_id:04d}",
            "text": text,
            "entities": entities,
            "diagnosis": condition["diagnosis"],
            "note_type": note_type,
            "medications": medications,
            "patient_name": values["name"][0],
            # Structured ground truth for building the QA eval set. Contains PII,
            # so it must never be passed to retrieval/generation.
            "facts": {k: v[0] for k, v in values.items()},
            # Which fields the chosen template actually rendered into the text,
            # so QA questions only ask about information present in the note.
            "fields_in_text": self._fields_rendered(template_func, values),
        }

    @staticmethod
    def _fields_rendered(template_func, values):
        """Render the template with sentinel values to see which keys it uses."""
        probe = template_func({k: f"<<{k}>>" for k in values})
        return sorted(set(re.findall(r"<<([a-z0-9_]+)>>", probe)))

    def generate_dataset(self, n_documents, output_dir):
        """Generate full dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = [self.generate_document(i) for i in range(n_documents)]

        # Save documents
        docs_path = output_dir / "synthetic_notes.jsonl"
        with docs_path.open("w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        # Build QA evaluation set from per-note facts (see glassbox/src/qa/)
        qa_set = build_qa_set(documents, profile="realistic")
        qa_path = output_dir / "qa_eval_set.jsonl"
        write_qa_set(qa_set, qa_path)

        print(f"Generated {len(documents)} realistic clinical notes")
        print(f"  Written to: {docs_path}")
        print(f"  QA eval set: {qa_path}")
        print(f"\nNote type distribution:")
        note_types = {}
        for doc in documents:
            note_types[doc["note_type"]] = note_types.get(doc["note_type"], 0) + 1
        for nt, count in sorted(note_types.items()):
            print(f"    {nt}: {count}")

        # Entity statistics
        entity_counts = {}
        for doc in documents:
            for ent in doc["entities"]:
                entity_counts[ent["type"]] = entity_counts.get(ent["type"], 0) + 1
        print(f"\nPII entity counts:")
        for etype, count in sorted(entity_counts.items()):
            print(f"    {etype}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic clinical notes")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "xlarge"],
        default="large",
        help="Dataset size (small=250, medium=1000, large=2500, xlarge=5000)"
    )
    parser.add_argument(
        "--output",
        default="glassbox/data/raw/realistic",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    size_map = {
        "small": 250,
        "medium": 1000,
        "large": 2500,
        "xlarge": 5000,
    }
    n_documents = size_map[args.size]

    print(f"Generating {args.size} dataset ({n_documents} documents)...")
    generator = RealisticClinicalGenerator(seed=args.seed)
    generator.generate_dataset(n_documents, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
