"""
Generalization test suite for PII redaction.

Tests the redaction module on unseen formats not present in the synthetic training data.
"""
import pytest
from glassbox.src.redaction.detector import PIIDetector
from glassbox.src.redaction.redactor import PIIRedactor


@pytest.fixture
def redactor_pipeline():
    detector = PIIDetector(confidence_threshold=0.5)
    redactor = PIIRedactor()
    return detector, redactor


def test_generalization_patient_ids(redactor_pipeline):
    """Test patient ID formats beyond the base 2-letter+6-digit format."""
    detector, redactor = redactor_pipeline

    # Dash format
    text = "Patient Mary Johnson (ID: AB-123456) seen today."
    entities = detector.detect(text)
    _, log = redactor.redact(text, entities)
    redacted_vals = [e["original_value"] for e in log]
    assert "AB-123456" in redacted_vals

    # MRN format
    text = "Patient Tom Wilson, MRN0012345, presented with fever."
    entities = detector.detect(text)
    _, log = redactor.redact(text, entities)
    redacted_vals = [e["original_value"] for e in log]
    assert "MRN0012345" in redacted_vals


def test_generalization_dates(redactor_pipeline):
    """Test date formats beyond ISO YYYY-MM-DD."""
    detector, redactor = redactor_pipeline

    # US format MM/DD/YYYY
    text = "Patient Sarah Lee (DOB: 03/20/1985) visited on 09/15/2026."
    entities = detector.detect(text)
    _, log = redactor.redact(text, entities)
    redacted_vals = [e["original_value"] for e in log]
    assert "03/20/1985" in redacted_vals
    assert "09/15/2026" not in redacted_vals  # Visit date preserved

    # Written format
    text = "The patient was born March 20, 1985 and lives alone."
    entities = detector.detect(text)
    _, log = redactor.redact(text, entities)
    assert any("March 20, 1985" in e["original_value"] and e["type"] == "DATE_OF_BIRTH" for e in log)


def _redacted_values(detector, redactor, text, entity_type):
    _, log = redactor.redact(text, detector.detect(text))
    return [e["original_value"] for e in log if e["type"] == entity_type]


@pytest.mark.parametrize("phone", [
    "(555) 123-4567",        # parentheses
    "555-123-4567 x272",     # extension
    "+1-555-123-4567",       # international prefix
])
def test_generalization_phone_formats(redactor_pipeline, phone):
    detector, redactor = redactor_pipeline
    text = f"Patient can be reached at {phone} for scheduling."
    assert phone in _redacted_values(detector, redactor, text, "PHONE_NUMBER")


def test_generalization_full_address(redactor_pipeline):
    detector, redactor = redactor_pipeline
    address = "123 Main Street, Springfield, IL 62701"
    text = f"Home address on file: {address}."
    redacted, _ = redactor.redact(text, detector.detect(text))
    assert "Main Street" not in redacted and "62701" not in redacted


# ---------------------------------------------------------------------------
# Regression tests for issues found on the 2,500-note realistic dataset
# ---------------------------------------------------------------------------

def test_patient_id_not_mislabelled_as_name(redactor_pipeline):
    """spaCy tags bare IDs like 'pm010651' as PERSON at the same score; ID must win the tie."""
    detector, redactor = redactor_pipeline
    text = "Patient: Allison Hill (ID: pm010651, DOB 1986-12-06)"
    assert "pm010651" in _redacted_values(detector, redactor, text, "PATIENT_ID")


def test_ner_span_does_not_cross_newline(redactor_pipeline):
    """NER spans like 'MRN0033092\\nDate' must not swallow the ID and the next field label."""
    detector, redactor = redactor_pipeline
    text = "Patient ID: g-177589\nDate of Birth: 1986-12-06 (Age: 40)"
    redacted, log = redactor.redact(text, detector.detect(text))
    assert "g-177589" in [e["original_value"] for e in log if e["type"] == "PATIENT_ID"]
    assert "Date of Birth:" in redacted


def test_mrn_beats_dea_license_suffix(redactor_pipeline):
    """Presidio's DEA recognizer matches 'RN6659051' inside 'MRN6659051' at score 1.0."""
    detector, redactor = redactor_pipeline
    text = "Name: Allison Hill\nMRN: MRN6659051\nDOB: 1986-12-06"
    assert "MRN6659051" in _redacted_values(detector, redactor, text, "PATIENT_ID")


def test_raw_phone_not_tagged_uk_nhs(redactor_pipeline):
    detector, redactor = redactor_pipeline
    text = "Contact number: 2727956878"
    _, log = redactor.redact(text, detector.detect(text))
    assert [e["type"] for e in log if e["original_value"] == "2727956878"] == ["PHONE_NUMBER"]


def test_md_credential_not_redacted_as_address(redactor_pipeline):
    detector, redactor = redactor_pipeline
    text = "Electronically signed by Lindsey Roman, MD\n"
    redacted, _ = redactor.redact(text, detector.detect(text))
    assert redacted.strip().endswith("[NAME], MD")


def test_overlap_resolution_prefers_higher_score_regardless_of_position():
    """An earlier, lower-confidence span must not beat a later, higher-confidence one."""
    redactor = PIIRedactor()
    entities = [
        {"type": "NAME", "start": 0, "end": 10, "score": 0.6},
        {"type": "PATIENT_ID", "start": 5, "end": 15, "score": 0.9},
    ]
    kept = redactor._resolve_overlaps(entities)
    assert [e["type"] for e in kept] == ["PATIENT_ID"]


# ---------------------------------------------------------------------------
# Address recognizer, name-in-context recognizer and clinical-term guard
# ---------------------------------------------------------------------------

def _redact(redactor_pipeline, text):
    detector, redactor = redactor_pipeline
    return redactor.redact(text, detector.detect(text))[0]


@pytest.mark.parametrize("address", [
    "0288 Robert Lake Suite 860, West Lorihaven, AL 34055",
    "826 Bobby Lakes Apt. 533, West Kimberlychester, RI 98736",
    "Unit 4724 Box 3586, DPO AE 64879",
    "PSC 1234, Box 5678, APO AA 12345",
    "USNS Long, FPO AP 12345",
])
def test_address_is_one_span(redactor_pipeline, address):
    assert _redact(redactor_pipeline, f"Address: {address}\nEmail: a@b.com") == "Address: [ADDRESS]\nEmail: [EMAIL]"


def test_street_line_needs_context(redactor_pipeline):
    assert _redact(redactor_pipeline, "She lives at 42 Oak Street.") == "She lives at [ADDRESS]."
    assert "42 Oak Street" in _redact(redactor_pipeline, "Seen 42 Oak Street ago")


def test_address_patterns_are_case_sensitive(redactor_pipeline):
    """Presidio's default IGNORECASE would let [A-Z] match lowercase words."""
    assert _redact(redactor_pipeline, "Take 5 tabs in the am, in 12345") == "Take 5 tabs in the am, in 12345"


@pytest.mark.xfail(reason="Known limitation: the address patterns use capitalisation; "
                          "an all-lowercase address is only partly caught", strict=True)
def test_lowercase_address(redactor_pipeline):
    assert "62704" not in _redact(redactor_pipeline, "lives at 42 oak street, springfield, il 62704")


@pytest.mark.parametrize("text", [
    "Started Metformin 500mg and Glipizide 5mg.",
    "SpO2 97% on RA.",
    "BP 158/62, HR 72.",
])
def test_clinical_terms_not_redacted(redactor_pipeline, text):
    assert _redact(redactor_pipeline, text) == text


def test_field_label_not_redacted(redactor_pipeline):
    assert _redact(redactor_pipeline, "Email: jdoe@mail.com") == "Email: [EMAIL]"


@pytest.mark.parametrize("text,expected", [
    ("Thank you.\n\nLance Hebert, MD\nCardiology", "Thank you.\n\n[NAME], MD\nCardiology"),
    ("Referring Physician: Autumn Sharp\n", "Referring Physician: [NAME]\n"),
    ("Electronically signed by Tiffany Wilson, MD", "Electronically signed by [NAME], MD"),
    ("Dr. Patel to follow up.", "Dr. [NAME] to follow up."),
])
def test_names_in_clinical_context(redactor_pipeline, text, expected):
    assert _redact(redactor_pipeline, text) == expected


def test_numeric_mrn_after_label(redactor_pipeline):
    assert _redact(redactor_pipeline, "MRN: 00482913.") == "MRN: [PATIENT_ID]."



# ---------------------------------------------------------------------------
# Robustness fixes (OOD redaction set A)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("address", [
    "42 Oak Street, Springfield, Illinois 62704",     # full state name
    "42 Oak Street Springfield IL 62704",             # no commas
])
def test_address_variants(redactor_pipeline, address):
    assert _redact(redactor_pipeline, f"Lives at {address}.") == "Lives at [ADDRESS]."


@pytest.mark.parametrize("dob", ["3/4/1975", "3-4-1975", "04-Mar-1975"])
def test_dob_variants(redactor_pipeline, dob):
    assert _redact(redactor_pipeline, f"DOB {dob}.") == "DOB [DATE_OF_BIRTH]."


def test_record_number_with_context(redactor_pipeline):
    assert _redact(redactor_pipeline, "Record number 1234567.") == "Record number [PATIENT_ID]."
    assert _redact(redactor_pipeline, "Walked 1234567 steps.") == "Walked 1234567 steps."


def test_accented_name_after_label(redactor_pipeline):
    assert _redact(redactor_pipeline, "Patient: Nurşan Çamurcuoğlu\n") == "Patient: [NAME]\n"


@pytest.mark.parametrize("text", ["Pt seen in clinic.", "Visit on 9/7/2026.", "Emphysema patient, stable."])
def test_more_non_pii_kept(redactor_pipeline, text):
    assert _redact(redactor_pipeline, text) == text
