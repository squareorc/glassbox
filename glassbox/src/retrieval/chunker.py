"""
Chunking: split clinical notes into retrievable pieces.

Two strategies:

- "section" (default): one chunk per note section. The note templates separate
  sections with blank lines ("SUBJECTIVE: ...", "PLAN: ...", "S: ...").
  Small chunks give precise citations and short NLI premises for the verifier,
  and expose less of a note to the LLM than whole-note retrieval.

- "note": one chunk per note. Kept as an ablation baseline.

Every chunk's text starts with a context line built only from
non-identifying metadata, e.g.

    [Progress note | 2024-11-10 | Essential hypertension]
    PLAN: - Continue Amlodipine 5mg PO daily ...

so a section on its own still says which visit it belongs to. Character
offsets (`start`, `end`) of the section body within the source note are kept
so the dashboard can highlight it in context.
"""

import re
from typing import Any, Dict, List

NOTE_TYPE_LABELS = {
    "progress": "Progress note",
    "discharge": "Discharge summary",
    "consultation": "Consultation note",
}

# Blocks shorter than this are merged into a neighbour (e.g. the one-line
# "PROGRESS NOTE" title, or "Thank you for this referral.").
MIN_BLOCK_CHARS = 40

_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")
_LABEL = re.compile(r"^([A-Za-z][A-Za-z /-]{0,40}?):")
# A heading line with no colon, e.g. "PATIENT INFORMATION" (not note titles,
# which are handled separately)
_CAPS_HEADING = re.compile(r"^(?!SOAP NOTE|PROGRESS NOTE|DISCHARGE SUMMARY|CONSULTATION NOTE)([A-Z][A-Z /]{3,40})$")

# Normalised section labels -> canonical section names.
SECTION_ALIASES = {
    "s": "subjective",
    "o": "objective",
    "a": "assessment",
    "p": "plan",
    "patient": "patient_info",
    "patient_information": "patient_info",
    "date": "patient_info",
    "contact": "contact",
    "contact_email": "contact",
    "patient_contact": "contact",
    "address": "contact",
    "attending": "signature",
    "provider": "signature",
    "vitals": "vitals",
    "assessment_plan": "assessment_plan",
}


def context_line(doc: Dict[str, Any]) -> str:
    note_type = NOTE_TYPE_LABELS.get(doc.get("note_type"), "Clinical note")
    return f"[{note_type} | {doc.get('visit_date', 'date unknown')} | {doc['diagnosis']}]"


def section_name(block: str) -> str:
    """Canonical section name for a block of note text."""
    first_line = block.lstrip().split("\n", 1)[0]
    match = _LABEL.match(first_line) or _CAPS_HEADING.match(first_line)
    if match:
        slug = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
        return SECTION_ALIASES.get(slug, slug)
    if re.search(r",\s*MD\b", block) and len(block) < 120:
        return "signature"
    if re.match(r"(SOAP NOTE|PROGRESS NOTE|DISCHARGE SUMMARY|CONSULTATION NOTE)", first_line):
        return "title"
    if block.lstrip().startswith("Electronically signed"):
        return "signature"
    return "note"


def _blocks_with_offsets(text: str) -> List[Dict[str, Any]]:
    """Split text at blank lines, keeping each block's character offsets."""
    blocks, pos = [], 0
    for match in _BLOCK_SPLIT.finditer(text):
        blocks.append((pos, match.start()))
        pos = match.end()
    blocks.append((pos, len(text)))

    out = []
    for start, end in blocks:
        # Trim surrounding whitespace but keep offsets exact
        seg = text[start:end]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        start, end = start + lead, end - trail
        if end > start:
            out.append({"start": start, "end": end})
    return out


def _is_labelled(block: str) -> bool:
    first_line = block.lstrip().split("\n", 1)[0]
    return bool(_LABEL.match(first_line) or _CAPS_HEADING.match(first_line))


def _merge_small(blocks: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Merge short UNLABELLED blocks (titles, sign-offs like "Thank you for this
    referral.") into the following block, or the previous one at the end.
    Short labelled sections ("CHIEF COMPLAINT: headache") stay separate.
    """
    merged: List[Dict[str, Any]] = []
    carry_start = None
    for b in blocks:
        start = carry_start if carry_start is not None else b["start"]
        body = text[b["start"]:b["end"]]
        if b["end"] - b["start"] < MIN_BLOCK_CHARS and not _is_labelled(body):
            carry_start = start
            continue
        merged.append({"start": start, "end": b["end"]})
        carry_start = None
    if carry_start is not None:
        if merged:
            merged[-1]["end"] = blocks[-1]["end"]
        else:
            merged.append({"start": carry_start, "end": blocks[-1]["end"]})
    return merged


def chunk_document(doc: Dict[str, Any], strategy: str = "section") -> List[Dict[str, Any]]:
    """
    Split one note into chunks.

    Args:
        doc: A note with at least doc_id, text, diagnosis; optionally
            note_type and visit_date (non-identifying metadata).
        strategy: "section" or "note".

    Returns:
        Chunks with chunk_id, doc_id, section, text (context line + body),
        body, start/end offsets of the body in doc["text"], and metadata.
    """
    text = doc["text"]
    if strategy == "note":
        spans = [{"start": 0, "end": len(text.rstrip())}]
    elif strategy == "section":
        spans = _merge_small(_blocks_with_offsets(text), text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    header = context_line(doc)
    chunks = []
    seen = {}
    for span in spans:
        body = text[span["start"]:span["end"]]
        # A merged title block takes the name of the section it was merged into
        section = "note" if strategy == "note" else section_name(_strip_title(body))
        seen[section] = seen.get(section, 0) + 1
        suffix = section if seen[section] == 1 else f"{section}_{seen[section]}"
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{suffix}",
            "doc_id": doc["doc_id"],
            "section": section,
            "text": f"{header}\n{body}",
            "body": body,
            "start": span["start"],
            "end": span["end"],
            "note_type": doc.get("note_type", "note"),
            "diagnosis": doc["diagnosis"],
            "visit_date": doc.get("visit_date", ""),
        })
    return chunks


def _strip_title(body: str) -> str:
    """Drop a leading title line so the section is named after its content."""
    lines = body.split("\n", 1)
    if len(lines) == 2 and section_name(lines[0]) == "title":
        return lines[1].lstrip()
    return body
