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
        self._register_custom_recognizers()

    def _register_custom_recognizers(self):
        """
        Register custom pattern-based recognizers for healthcare-specific PII.

        The synthetic dataset uses Faker's bothify() for patient IDs:
        two letters + six digits (e.g., "Bc321819", "Km618495").
        """
        # Patient ID recognizer: 2 letters + 6 digits
        patient_id_pattern = Pattern(
            name="patient_id_pattern",
            regex=r"\b[A-Za-z]{2}\d{6}\b",
            score=0.85,
        )
        patient_id_recognizer = PatternRecognizer(
            supported_entity="PATIENT_ID",
            patterns=[patient_id_pattern],
        )
        self.analyzer.registry.add_recognizer(patient_id_recognizer)

        # Date of birth pattern: ISO format YYYY-MM-DD, but only for plausible DOB
        # (18-90 years ago from current date 2026-09-16)
        # This avoids flagging recent visit dates as DOB
        # DOB range: 1936-09-16 to 2008-09-16
        dob_pattern = Pattern(
            name="dob_plausible_range",
            regex=r"\b(19[3-9]\d|20[0][0-8])-\d{2}-\d{2}\b",
            score=0.85,
        )
        dob_recognizer = PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=[dob_pattern],
        )
        self.analyzer.registry.add_recognizer(dob_recognizer)

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

                entities.append({
                    "type": normalized_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                })

        # Sort by start position for consistent processing
        entities.sort(key=lambda e: e["start"])

        return entities

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
