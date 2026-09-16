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
