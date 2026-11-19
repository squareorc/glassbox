"""
PII detection using Presidio with custom recognizers.
"""

import re
from typing import List, Dict, Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer


class PIIDetector:
    """
    Wraps Presidio's AnalyzerEngine with custom recognizers for healthcare PII.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            confidence_threshold: Minimum confidence score to keep a detection.
                Lower = higher recall (catches more PII) but lower precision (over-redacts).
        """
        self.confidence_threshold = confidence_threshold
        self.analyzer = AnalyzerEngine()
        # Domain is US clinical records. The UK NHS-number recognizer fires on
        # raw 10-digit US phone numbers that happen to pass its checksum.
        self.analyzer.registry.remove_recognizer("NhsRecognizer")
        self._register_custom_recognizers()

    def _register_custom_recognizers(self):
        """
        Register custom pattern-based recognizers for healthcare-specific PII.

        Includes multiple pattern variants to improve generalization beyond
        the synthetic dataset's specific formats.
        """
        # Patient ID recognizers: multiple common formats
        patient_id_patterns = [
            # 1-2 letters + 6 digits (e.g., "Bc321819", "P010651")
            Pattern(name="patient_id_alpha_numeric", regex=r"\b[A-Za-z]{1,2}\d{6}\b", score=0.85),
            # Dash-separated (e.g., "P-123456", "AB-123456")
            Pattern(name="patient_id_dash", regex=r"\b[A-Za-z]{1,2}-\d{6}\b", score=0.85),
            # MRN format (e.g., "MRN0012345", "MRN-12345"). The explicit prefix
            # is unambiguous, so it scores 1.0 - otherwise Presidio's DEA
            # checksum recognizer claims the "RN0012345" suffix as a
            # MEDICAL_LICENSE at 1.0 and wins the overlap.
            Pattern(name="patient_id_mrn", regex=r"\bMRN[-]?\d{5,7}\b", score=1.0),
        ]
        patient_id_recognizer = PatternRecognizer(
            supported_entity="PATIENT_ID",
            patterns=patient_id_patterns,
        )
        self.analyzer.registry.add_recognizer(patient_id_recognizer)

        # Date of birth patterns: multiple formats
        dob_patterns = [
            # ISO format YYYY-MM-DD with plausible DOB range (1936-2008)
            Pattern(name="dob_iso_plausible", regex=r"\b(19[3-9]\d|20[0][0-8])-\d{2}-\d{2}\b", score=0.85),
            # MM/DD/YYYY format with plausible years
            Pattern(name="dob_us_format", regex=r"\b\d{2}/\d{2}/(19[3-9]\d|20[0][0-8])\b", score=0.85),
            # Written format with context (e.g., "born March 20, 1985", "DOB: March 20, 1985")
            Pattern(name="dob_written", regex=r"\b(?:born|DOB:?|date of birth:?)\s+[A-Z][a-z]+\s+\d{1,2},?\s+(19[3-9]\d|20[0][0-8])\b", score=0.8),
        ]
        dob_recognizer = PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=dob_patterns,
        )
        self.analyzer.registry.add_recognizer(dob_recognizer)

        # Phone number patterns: comprehensive coverage for clinical notes
        # Covers formatted numbers, extensions, and context-gated raw numbers
        phone_patterns = [
            # Formatted numbers with all common separators (parens, dashes, dots, spaces)
            # + optional extensions (x123, ext. 123, #123)
            # e.g., (555) 123-4567, 555-123-4567, +1-555-123-4567 ext. 402, 001-286-928-9023x303
            Pattern(
                name="phone_formatted_with_ext",
                regex=r"(?<!\w)(?:\+?1[-.\s]?|001[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?:\s*(?:x|ext\.?|#|extension)\s*\d{1,6})?(?!\w)",
                score=0.85,
            ),
            # Raw 10-digit numbers (e.g., 8135389083) with LOW base score
            # Context words boost above threshold to avoid false positives on medical codes
            Pattern(
                name="phone_raw_10digit_context",
                regex=r"\b\d{10}\b",
                score=0.40,  # Below threshold; requires context boost
            ),
        ]
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=phone_patterns,
            context=[
                "phone", "call", "contact", "cell", "tel", "telephone",
                "mobile", "number", "reached", "dial", "fax", "extension"
            ],
        )
        self.analyzer.registry.add_recognizer(phone_recognizer)

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text.

        Args:
            text: Raw clinical note text.

        Returns:
            List of detected entities, each with:
                - type: Entity type (NAME, PATIENT_ID, PHONE_NUMBER, etc.)
                - start: Character offset start
                - end: Character offset end
                - score: Confidence score [0, 1]
        """
        # Presidio returns RecognizerResult objects
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=None,  # Detect all entity types
        )

        # Filter by confidence threshold, normalize types, and filter noise
        entities = []
        for result in results:
            if result.score >= self.confidence_threshold:
                normalized_type = self._normalize_entity_type(result.entity_type)

                # Skip entity types we don't care about
                if normalized_type in ["DATE_TIME", "NRP", "URL", "US_BANK_NUMBER"]:
                    continue

                # spaCy tags clinical credentials ("Lindsey Roman, MD") as
                # LOCATION (Maryland). Drop them unless followed by a ZIP
                # code, which signals a real state abbreviation in an address.
                if normalized_type == "ADDRESS" and self._is_credential(text, result.start, result.end):
                    continue

                # spaCy NER spans sometimes run across a line break into the
                # next field label ("MRN0033092\nDate"). No PII value in a
                # note spans lines, so cut NER spans at the first newline.
                start, end = result.start, result.end
                if normalized_type in ("NAME", "ADDRESS"):
                    newline = text.find("\n", start, end)
                    if newline != -1:
                        end = newline
                    if end <= start:
                        continue

                entities.append({
                    "type": normalized_type,
                    "start": start,
                    "end": end,
                    "score": result.score,
                })

        # Sort by start position for consistent processing
        entities.sort(key=lambda e: e["start"])

        return entities

    CREDENTIALS = {"MD", "DO", "RN", "NP", "PA", "PA-C", "PhD", "DDS", "MBBS", "FACP"}
    _ZIP_AFTER = re.compile(r"\s+\d{5}(?:-\d{4})?\b")

    def _is_credential(self, text: str, start: int, end: int) -> bool:
        if text[start:end] not in self.CREDENTIALS:
            return False
        return not self._ZIP_AFTER.match(text, end)

    def _normalize_entity_type(self, presidio_type: str) -> str:
        """
        Map Presidio's entity types to our dataset's ground-truth types.

        Presidio uses: PERSON, PHONE_NUMBER, EMAIL_ADDRESS, LOCATION, etc.
        Our dataset uses: NAME, PATIENT_ID, PHONE_NUMBER, EMAIL, ADDRESS, DATE_OF_BIRTH.
        """
        type_map = {
            "PERSON": "NAME",
            "EMAIL_ADDRESS": "EMAIL",
            "LOCATION": "ADDRESS",
            "US_DRIVER_LICENSE": "PATIENT_ID",  # Patient IDs detected as driver licenses
            "PHONE_NUMBER": "PHONE_NUMBER",
            "PATIENT_ID": "PATIENT_ID",  # Our custom recognizer
            "DATE_OF_BIRTH": "DATE_OF_BIRTH",  # Our custom recognizer
        }
        return type_map.get(presidio_type, presidio_type)
