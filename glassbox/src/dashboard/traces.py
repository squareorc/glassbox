"""
Trace loading and summarising for the audit dashboard (no Streamlit here, so it can be tested).

Traces are the JSON files written by glassbox/src/pipeline.py under
glassbox/logs/traces/<dataset>/<config>/. The dashboard only reads them; it
never recomputes a stored trace.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

TRACE_ROOT = Path("glassbox/logs/traces")

LABEL_COLOUR = {"SUPPORTED": "green", "CONTRADICTED": "red", "UNSUPPORTED": "orange", "UNCERTAIN": "gray"}
VERDICT_COLOUR = {"VERIFIED": "green", "PARTLY_VERIFIED": "orange", "NOT_VERIFIED": "orange",
                  "CONTRADICTED": "red", "ABSTAINED": "blue", "NO_CLAIMS": "gray", "NOT_RUN": "gray"}
ABSTAIN_EXPLANATION = {
    "identifier_in_question": "The question names or identifies a specific person. Identities are removed from the "
                              "records, so any answer would attribute someone else's note to that person. "
                              "Refused before the model was called.",
    "retrieval_fallback": "No record matches the diagnosis/date conditions in the question, so the retrieved notes "
                          "would not satisfy it. Refused before the model was called.",
    "model_insufficient": "The model read the sources and reported that they do not contain the answer.",
}


def trace_sets(root: Path = TRACE_ROOT) -> List[str]:
    """'<dataset>/<config>' folders that hold traces."""
    if not root.exists():
        return []
    return sorted(str(d.relative_to(root)) for d in root.glob("*/*") if d.is_dir() and any(d.glob("*.json")))


def load_trace(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def summary_row(trace: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """One table row per trace for the batch view."""
    gen, qa = trace["generation"], trace.get("qa", {})
    return {
        "id": trace.get("qid") or trace["trace_id"],
        "type": qa.get("type", "ad hoc"),
        "split": qa.get("split", ""),
        "question": trace["question"]["text"],
        "verdict": trace["verification"]["verdict"],
        "abstain_reason": gen.get("abstain_reason") or "",
        "claims": len(gen.get("claims") or []),
        "grounded_accuracy": qa.get("scores", {}).get("grounded_accuracy"),
        "chunks": trace["retrieval"]["k"],
        "total_ms": trace["timings_ms"]["total"],
        "path": str(path),
    }


def load_rows(trace_set: str, root: Path = TRACE_ROOT) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted((root / trace_set).glob("*.json")):
        try:
            rows.append(summary_row(load_trace(p), p))
        except (KeyError, json.JSONDecodeError):
            continue  # an older or partial trace: skip rather than break the table
    return rows


def chunk_anchor(chunk_id: str) -> str:
    """HTML id for a chunk card, so a claim's citation can link to it."""
    return "chunk-" + "".join(ch if ch.isalnum() else "-" for ch in chunk_id)


def source_label(trace: Dict[str, Any]) -> Dict[str, str]:
    """chunk_id -> the 'C<n>' label the model saw (from the generation source map)."""
    return {cid: label for label, cid in (trace["generation"].get("source_map") or {}).items()}
