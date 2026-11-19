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


@pytest.mark.xfail(reason="Known limitation: spaCy NER fragments addresses; full-span address "
                          "detection is scheduled work (README, Known issues)", strict=True)
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
