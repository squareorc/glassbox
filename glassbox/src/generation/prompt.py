"""
Prompt construction for cited generation.

The retrieved chunks are shown as numbered sources [C1]..[Ck]. Each chunk's
text already begins with its non-PII context line ("[Progress note |
2024-11-10 | Essential hypertension]"), so the model can tell which visit a
section belongs to without seeing any identifier.

The model must reply with one JSON object:
    {"answer": str,
     "claims": [{"text": str, "citations": ["C1", ...]}, ...],
     "insufficient": bool}

Prompt versions were compared on the DEV split only (PROGRESS.md §7):
    v1  basic rules                                          accuracy 0.890
    v2  + "Note n" labels, note-by-note rule, exact doses    accuracy 0.941  <- current
    v3  v2 with "claims" before "answer" in the JSON         accuracy 0.924 (rejected)

JSON (rather than free text with inline [C1] markers) gives the Day 4
verifier clean, claim-by-claim input and makes malformed output detectable.
"""

import hashlib
import json
from typing import Any, Dict, List, Tuple

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You answer questions about clinical notes using ONLY the numbered sources provided.

Each source is one section of a clinical note. Sources marked with the same "Note" number come from the same visit of the same patient.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge and never guess.
2. Every claim is one complete sentence and must cite the source(s) it comes from, e.g. ["C2"].
3. Copy clinical values exactly as written, including doses, routes and frequencies (e.g. "Metformin 500mg PO BID", not "Metformin").
4. Values in square brackets such as [NAME], [PATIENT_ID], [DATE_OF_BIRTH], [PHONE_NUMBER], [EMAIL] and [ADDRESS] are hidden for privacy. Never guess them, and never assume a hidden [NAME] is a person mentioned in the question.
5. If the question is about several patients, go through the notes one by one. A note counts only if its sources show it meets EVERY condition in the question (diagnosis, date, symptom, ...); some notes shown will not. For each note that counts, make one claim listing ALL the requested items in that note. The answer must combine the items from every note that counts.
6. If the sources do not contain the answer, set "insufficient" to true, give an empty "claims" list, and say in "answer" that the records provided do not contain it.

Reply with ONLY one JSON object, no other text:
{"answer": "<one or two sentences>", "claims": [{"text": "<claim>", "citations": ["C1"]}], "insufficient": false}"""


def format_sources(chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
    """
    Number the chunks. Returns the sources block and {"C1": chunk_id, ...}.

    Each source also gets an opaque per-prompt note number ("Note 2"), so the
    model can tell which sections belong to the same visit: a note's SUBJECTIVE
    (symptom) and PLAN (drug) otherwise look like unrelated sources. The number
    is local to this prompt and carries no identifying information.
    """
    lines, id_map, note_no = [], {}, {}
    for n, c in enumerate(chunks, 1):
        label = f"C{n}"
        id_map[label] = c["chunk_id"]
        note = note_no.setdefault(c["doc_id"], len(note_no) + 1)
        lines.append(f"[{label}] (Note {note}) {c['text'].strip()}")
    return "\n\n".join(lines), id_map


def build_messages(question: str, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Chat messages for the model, plus the source-label -> chunk_id map."""
    sources, id_map = format_sources(chunks)
    user = f"Sources:\n\n{sources}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], id_map


def retry_message(error: str) -> Dict[str, str]:
    """Follow-up turn after an unparseable reply."""
    return {"role": "user", "content": f"Your reply could not be used: {error}. "
                                       "Reply again with ONLY the JSON object described above."}


def prompt_hash(messages: List[Dict[str, str]]) -> str:
    """Short fingerprint of the exact prompt, for the audit log."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
