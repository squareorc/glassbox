"""
PII redaction - replaces detected entities with type tags.
"""

from typing import List, Dict, Any, Tuple

# Tie-break order for overlapping spans with equal score and length: specific
# pattern-based types first, generic NER types last. Lower = preferred.
TYPE_PRIORITY = {
    "PATIENT_ID": 0,
    "EMAIL": 1,
    "PHONE_NUMBER": 2,
    "DATE_OF_BIRTH": 3,
    "ADDRESS": 4,
    "NAME": 5,
}


class PIIRedactor:
    """
    Redacts PII by replacing detected entities with [TYPE] tags.
    Handles overlapping spans and logs redactions for audit trail.
    """

    def __init__(self, redaction_format: str = "[{type}]"):
        """
        Initialize the redactor.

        Args:
            redaction_format: Template for redacted text. {type} is replaced
                with the entity type. Default: "[NAME]", "[PHONE_NUMBER]", etc.
        """
        self.redaction_format = redaction_format

    def redact(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII entities from text.

        Args:
            text: Original text.
            entities: List of detected entities from PIIDetector, each with
                type, start, end, score.

        Returns:
            Tuple of:
                - Redacted text with PII replaced by type tags
                - Audit log: list of redaction records with original value,
                  replacement, position, type, score
        """
        if not entities:
            return text, []

        # Handle overlapping spans: keep the one with higher confidence
        entities = self._resolve_overlaps(entities)

        # Build the redacted text and audit log
        redacted_text = []
        audit_log = []
        prev_end = 0

        for entity in entities:
            start, end = entity["start"], entity["end"]
            entity_type = entity["type"]
            original_value = text[start:end]

            # Append text before this entity
            redacted_text.append(text[prev_end:start])

            # Append the redaction tag
            replacement = self.redaction_format.format(type=entity_type)
            redacted_text.append(replacement)

            # Log the redaction
            audit_log.append({
                "type": entity_type,
                "original_value": original_value,
                "replacement": replacement,
                "start": start,
                "end": end,
                "score": entity.get("score", 1.0),
            })

            prev_end = end

        # Append remaining text
        redacted_text.append(text[prev_end:])

        return "".join(redacted_text), audit_log

    def _resolve_overlaps(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Handle overlapping entity spans, greedily keeping the best span.

        Candidates are ranked globally (not by position), and each is kept
        unless it overlaps an already-kept, better-ranked span. Ranking:
        1. Higher confidence score
        2. If tied, the more specific entity type (pattern recognizers like
           PATIENT_ID beat generic NER types like NAME/ADDRESS, which spaCy
           assigns to ID-like tokens such as "pm010651")
        3. If still tied, longer span
        4. If still tied, earlier span

        Args:
            entities: Detected entities (any order).

        Returns:
            Filtered list with no overlaps, sorted by start position.
        """
        if not entities:
            return []

        entities = sorted(
            entities,
            key=lambda e: (
                -e.get("score", 0),
                TYPE_PRIORITY.get(e["type"], len(TYPE_PRIORITY)),
                -(e["end"] - e["start"]),
                e["start"],
            ),
        )

        non_overlapping = []
        for entity in entities:
            # Check if this entity overlaps with any already-selected entity
            overlaps = False
            for selected in non_overlapping:
                if self._spans_overlap(
                    (entity["start"], entity["end"]),
                    (selected["start"], selected["end"]),
                ):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(entity)

        # Re-sort by start position for sequential processing
        non_overlapping.sort(key=lambda e: e["start"])
        return non_overlapping

    @staticmethod
    def _spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two character spans overlap."""
        start1, end1 = span1
        start2, end2 = span2
        return not (end1 <= start2 or end2 <= start1)
