"""
Parsing and validation of the model's JSON reply.

Accepts the reply with or without ```json fences or surrounding prose, takes
the first complete JSON object, checks its shape, and maps citations to
chunk IDs. Problems are recorded, never raised:

    - a citation to a source that wasn't shown ("C9" with 5 sources) is
      dropped and listed in the claim's `invalid_citations`;
    - a claim left with no valid citation is flagged `uncited`.

Citations are normalised: "C2", "[C2]", "c2", "C 2" and "Source 2" all mean
source 2 (the model writes several of these).
"""

import json
import re
from typing import Any, Dict, List, Optional

_CITATION = re.compile(r"^\s*\[?\s*(?:c|source|src)\s*(\d+)\s*\]?\s*$", re.IGNORECASE)


class ParseError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """The first complete JSON object in `text`."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ParseError("no JSON object found")
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ParseError(f"invalid JSON ({e.msg})")
                if not isinstance(obj, dict):
                    raise ParseError("top-level JSON value is not an object")
                return obj
    raise ParseError("JSON object is not closed (reply may be truncated)")


def normalise_citation(raw: Any) -> Optional[str]:
    """'[C2]' / 'c2' / 'Source 2' / 2 -> 'C2'; None if unrecognisable."""
    if isinstance(raw, int):
        return f"C{raw}"
    if not isinstance(raw, str):
        return None
    m = _CITATION.match(raw)
    return f"C{int(m.group(1))}" if m else None


def parse_reply(text: str, id_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse a reply. Raises ParseError only if no usable JSON object exists
    (the caller retries); everything else is recorded on the result.
    """
    obj = extract_json(text)
    answer = obj.get("answer")
    if not isinstance(answer, str):
        raise ParseError('"answer" is missing or not a string')
    raw_claims = obj.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ParseError('"claims" is not a list')

    claims: List[Dict[str, Any]] = []
    for rc in raw_claims:
        if isinstance(rc, str):          # tolerate a bare string claim
            rc = {"text": rc, "citations": []}
        if not isinstance(rc, dict) or not isinstance(rc.get("text"), str) or not rc["text"].strip():
            continue
        cites = rc.get("citations", [])
        if not isinstance(cites, list):
            cites = [cites]
        valid, invalid = [], []
        for c in cites:
            label = normalise_citation(c)
            if label in id_map:
                if id_map[label] not in valid:
                    valid.append(id_map[label])
            else:
                invalid.append(str(c))
        claims.append({"text": rc["text"].strip(), "citations": valid,
                       "invalid_citations": invalid, "uncited": not valid})

    return {
        "answer": answer.strip(),
        "claims": claims,
        "insufficient": bool(obj.get("insufficient", False)),
    }
