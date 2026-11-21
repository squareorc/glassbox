"""
PII detection using Presidio with custom recognizers.
"""

import re
from typing import List, Dict, Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from .clinical_terms import is_clinical_term

# ---------------------------------------------------------------- addresses
# spaCy's NER splits a US address into fragments ("Rodriguez Mews" -> PERSON,
# "South Aaron" -> LOCATION) and leaves house numbers and ZIP codes visible.
# These patterns match the STRUCTURE of a US postal address instead, so the
# whole address is one span. They don't depend on the note templates' labels
# ("Address:") or on a list of street-suffix words.
#
# Every street pattern must end in a valid USPS state/territory code + ZIP,
# which is what keeps them precise: "5 mg daily, Take PO" can't match.
US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH "
    "NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC "
    "AS GU MP PR VI FM MH PW"  # territories and freely associated states
).split()
US_STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
    "Puerto Rico", "Guam",
]
# Robustness fix (OOD set A): full state names ("Springfield, Illinois 62704") were not matched.
_STATE = "(?:" + "|".join(sorted(US_STATE_NAMES, key=len, reverse=True) + US_STATES) + ")"
_ZIP = r"\d{5}(?:-\d{4})?"
_WORD = r"[A-Z][A-Za-z'\-]*"  # a capitalised word (street or city name)
_UNIT = r"(?:,?[ ]+(?:Apt\.?|Apartment|Suite|Ste\.?|Unit|Rm\.?|Room|Fl\.?|Floor|#)[ ]*[A-Za-z0-9\-]+)"
_CITY = rf"{_WORD}(?:[ ]+{_WORD}){{0,3}}"
ADDRESS_PATTERNS = [
    # "0288 Robert Lake Suite 860, West Lorihaven, AL 34055", "42 Oak Street, Springfield, IL 62704"
    Pattern(
        name="us_street_address",
        # Commas optional (robustness fix, OOD set A: "12 Oak St Springfield IL 62704" was missed)
        regex=rf"\b\d{{1,6}}[ ]+{_WORD}(?:[ ]+{_WORD}){{0,4}}\.?{_UNIT}?,?[ ]+{_CITY},?[ ]+{_STATE}[ ]+{_ZIP}\b",
        score=0.95,
    ),
    # Military mail: "Unit 4724 Box 3586, DPO AE 64879", "PSC 1234, Box 5678, APO AA 12345",
    # "USNS Long, FPO AP 12345"
    Pattern(
        name="us_military_address",
        regex=rf"\b(?:Unit[ ]+\d{{3,5}}[ ]+Box[ ]+\d{{3,5}}|PSC[ ]+\d{{3,5}},[ ]+Box[ ]+\d{{3,5}}"
              rf"|US(?:NS|NV|S|CGC)[ ]+{_WORD}),[ ]+(?:APO|FPO|DPO)[ ]+(?:AA|AE|AP)[ ]+{_ZIP}\b",
        score=0.95,
    ),
    # Street line without city/state ("Lives at 42 Oak Street"): only a common
    # street-type word makes this an address, so it scores below threshold
    # and needs a context word ("address", "lives", ...) nearby.
    Pattern(
        name="us_street_line",
        regex=rf"\b\d{{1,6}}[ ]+{_WORD}(?:[ ]+{_WORD}){{0,3}}[ ]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd"
              rf"|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Circle|Highway|Hwy|Parkway|Pkwy)\.?{_UNIT}?\b",
        score=0.45,
    ),
]

# Presidio compiles patterns case-INsensitively by default, which would let
# "[A-Z]" match lowercase words ("Dr. Patel to follow"). Capitalisation is the
# signal in the address and name patterns, so they compile case-sensitively.
CASE_SENSITIVE = re.MULTILINE | re.DOTALL

# ---------------------------------------------------------------- names in context
# spaCy misses some names in two standard clinical-document positions:
# right before a credential ("Lance Hebert, MD" on a signature line) and
# right after a form label ("Referring Physician: Autumn Sharp",
# "Electronically signed by Tiffany Wilson"). A name is 2-3 capitalised words.
# \p{Lu}/\p{Ll}: any script's upper/lower-case letters, so accented names (Müller, Çamurcuoğlu)
# match too (robustness fix; Presidio compiles patterns with the `regex` module).
_NAME = r"\p{Lu}[\p{Ll}'\-]+(?:[ ]\p{Lu}\.)?(?:[ ]\p{Lu}[\p{Ll}'\-]+){1,2}"
_CREDENTIAL = r"(?:MD|DO|RN|NP|PA-C|PA|PhD|DDS|DMD|MBBS|FACP|FNP|APRN)"
_LABELS = ["Patient: ", "PATIENT: ", "Name: ", "Physician: ", "Provider: ", "Attending: ",
           "Clinician: ", "Nurse: ", "signed by ", "Signed by ", "Dr. ", "Dr "]
NAME_PATTERNS = [
    Pattern(name="name_before_credential", regex=rf"\b{_NAME}(?=,?[ ]{_CREDENTIAL}\b)", score=0.85),
    Pattern(name="name_after_label",
            regex="(?:" + "|".join(f"(?<={re.escape(l)})" for l in _LABELS) + rf"){_NAME}\b", score=0.85),
]



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
            # Purely numeric record number after an explicit label ("MRN: 00482913")
            Pattern(name="patient_id_labelled_number",
                    regex=r"(?:(?<=MRN: )|(?<=MRN:)|(?<=MRN# )|(?<=MRN #)|(?<=MRN ))\d{5,10}\b", score=0.95),
            # A bare 6-9 digit number scores below the threshold and only counts with a record-type
            # word nearby ("Record number 1234567", "Medical Record #", "Acct", "chart").
            # Robustness fix (OOD set A: IDs without an "MRN" label were missed). 10 digits are
            # left to the phone recognizer.
            Pattern(name="patient_id_context_number", regex=r"(?<![\d\-])\d{6,9}(?![\d\-])", score=0.4),
        ]
        patient_id_recognizer = PatternRecognizer(
            supported_entity="PATIENT_ID",
            patterns=patient_id_patterns,
            context=["record", "mrn", "chart", "account", "acct", "patient", "id", "identifier", "medical"],
        )
        self.analyzer.registry.add_recognizer(patient_id_recognizer)

        # Date of birth patterns: multiple formats
        dob_patterns = [
            # ISO format YYYY-MM-DD with plausible DOB range (1936-2008)
            Pattern(name="dob_iso_plausible", regex=r"\b(19[3-9]\d|20[0][0-8])-\d{2}-\d{2}\b", score=0.85),
            # MM/DD/YYYY format with plausible years
            # 1-2 digit month/day with / - or . (robustness fix, OOD set A: "3/4/1975" was missed)
            Pattern(name="dob_us_format", regex=r"\b\d{1,2}[/.\-]\d{1,2}[/.\-](19[3-9]\d|20[0][0-8])\b", score=0.85),
            Pattern(name="dob_day_mon_year", regex=r"\b\d{1,2}[ \-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?[ \-](19[3-9]\d|20[0][0-8])\b", score=0.85),
            # Written format with context (e.g., "born March 20, 1985", "DOB: March 20, 1985")
            Pattern(name="dob_written", regex=r"\b(?:born|DOB:?|date of birth:?)\s+[A-Z][a-z]+\s+\d{1,2},?\s+(19[3-9]\d|20[0][0-8])\b", score=0.8),
            # "March 5, 1975" without a keyword right before it: below threshold unless a
            # birth-related word is nearby (context boost)
            Pattern(name="dob_month_day_year", regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19[3-9]\d|20[0][0-8])\b", score=0.45),
        ]
        dob_recognizer = PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=dob_patterns,
            context=["dob", "born", "birth", "d.o.b", "birthday"],
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

        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="ADDRESS",
            patterns=ADDRESS_PATTERNS,
            context=["address", "live", "lives", "reside", "resides", "residence", "home", "mailing"],
            global_regex_flags=CASE_SENSITIVE,
        ))
        self.analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PERSON", patterns=NAME_PATTERNS, name="NameContextRecognizer",
            global_regex_flags=CASE_SENSITIVE,
        ))

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

                # Clinical-term guard (clinical_terms.py): NER sometimes tags
                # drugs ("Glipizide"), vitals ("SpO2", "158/62") and field
                # labels ("Email") as people or places. Applied to every NAME
                # (spaCy or the name-in-context heuristic) and to spaCy
                # locations; structural patterns (IDs, phones, emails, DOBs,
                # postal addresses) are never filtered.
                source = (result.recognition_metadata or {}).get("recognizer_name")
                guarded = normalized_type == "NAME" or (normalized_type == "ADDRESS" and source == "SpacyRecognizer")
                if guarded and is_clinical_term(text[result.start:result.end]):
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
