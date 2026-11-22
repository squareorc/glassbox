"""
End-to-end GlassBox pipeline: one question in, one audit trace out.

    question -> question redaction -> query analysis -> access-controlled retrieval
             -> abstain guard -> cited generation -> NLI claim verification -> trace

Every stage already exists as its own module; this file only connects them,
loads each model once, times each stage and writes the trace.

Configurations (the same three as generation/evaluate.py, so a batch run over
the QA set reproduces the evaluation numbers):
    redacted          the GlassBox system: redacted index, question redaction, guard on
    raw               no-privacy baseline: raw index, no question redaction (needs
                      baseline access, see retrieval/retriever.py)
    redacted_noguard  redacted index with the abstain guard off (shows what the model does
                      on its own when asked about a named person)

The trace (one JSON file per question) is the audit record the dashboard reads:
    trace_id, timestamp, config     what ran, with which models / prompt / weights
    question                        the REDACTED question and the types/offsets removed
    query_analysis                  filters, scope, residual and ranking query, ask/condition
    retrieval                       every chunk shown to the model, with scores and ranks
    generation                      guard decision, answer, claims with citations, raw output
    verification                    one NLI label per claim, and the answer verdict
    timings_ms                      per stage and total
    qa (QA-set questions only)      qid, type, split and the evaluation scores, never the
                                    question text or the gold answer

Privacy rule: on a redacted configuration the original question is never
written anywhere. Only the redacted text and the TYPES of what was removed are
kept (tested in tests/test_pipeline.py). The raw baseline has no question
redaction by definition, so its traces hold the question as typed and are marked
`baseline_mode: true`.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .generation.generator import MAX_TOKENS, Generator
from .generation.prompt import PROMPT_VERSION
from .redaction.query import redact_query
from .retrieval.index import index_dir
from .retrieval.retriever import (DEFAULT_WEIGHTS, MULTI_NOTE_K, PER_NOTE, SINGLE_NOTE_K,
                                  Retriever)
from .verification.verifier import Verifier

CONFIGS = ("redacted", "raw", "redacted_noguard")
TRACE_ROOT = Path("glassbox/logs/traces")
TRACE_VERSION = 1


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class GlassBox:
    """Holds the loaded retriever, generator and verifier; `answer()` runs one question."""

    def __init__(self, dataset: str = "realistic", config: str = "redacted", threshold: float = 0.5,
                 verify: bool = True, trace_dir: Optional[Path] = None):
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {CONFIGS}")
        self.dataset, self.config, self.threshold = dataset, config, threshold
        self.baseline = config == "raw"
        self.use_guard = config != "redacted_noguard"
        # Question redaction is done here (not inside the retriever) so it can be timed
        # and recorded as its own stage. The result is identical: the retriever would run
        # the same redact_query on the same text.
        self.retriever = Retriever(index_dir(dataset, "raw" if self.baseline else "redacted", "section", threshold),
                                   baseline=self.baseline, redact_queries=False)
        self.redact_questions = self.retriever.manifest["source"] == "redacted"
        self.generator = Generator()
        self.verifier = Verifier() if verify else None
        self.trace_dir = Path(trace_dir) if trace_dir else TRACE_ROOT / dataset / config

    # ------------------------------------------------------------------
    def config_record(self) -> Dict[str, Any]:
        m = self.retriever.manifest
        return {
            "name": self.config,
            "dataset": self.dataset,
            "index": m["name"],
            "index_source": m["source"],
            "baseline_mode": self.baseline,
            "redaction_threshold": self.threshold,
            "question_redaction": self.redact_questions,
            "abstain_guard": self.use_guard,
            "embedding_model": m.get("embedding_model"),
            "retrieval_method": "hybrid",
            "fusion_weights": list(DEFAULT_WEIGHTS),
            "per_note": PER_NOTE,
            "auto_k": {"single_note": SINGLE_NOTE_K, "multi_note": MULTI_NOTE_K},
            "generator_model": self.generator.model_id,
            "prompt_version": PROMPT_VERSION,
            "max_tokens": MAX_TOKENS,
            "verifier_model": self.verifier.model_name if self.verifier else None,
            "verifier_margin": self.verifier.margin if self.verifier else None,
        }

    def answer(self, question: str, qid: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        """Run the whole pipeline on one question and return its trace (also saved as JSON)."""
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        # 1. Question redaction
        t0 = time.perf_counter()
        if self.redact_questions:
            qr = redact_query(question, self.threshold)
            asked = qr["text"]
        else:
            qr, asked = None, question
        timings["question_redaction"] = _ms(t0)

        # 2-3. Query analysis + retrieval (one call: analysis decides filters and k)
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(asked, qid=qid)
        analysis = self.retriever.last_analysis
        timings["retrieval"] = _ms(t0)

        # 4-5. Abstain guard + generation (the guard runs inside answer(), before the model)
        t0 = time.perf_counter()
        gen = self.generator.answer(asked, chunks, analysis, qr, self.retriever.manifest["source"],
                                    qid=qid, use_guard=self.use_guard)
        timings["generation"] = _ms(t0)

        # 6. Claim verification against the cited chunks
        t0 = time.perf_counter()
        if self.verifier:
            chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
            verification = self.verifier.verify_answer(gen, chunk_text, qr)
        else:
            verification = {"verdict": "NOT_RUN", "claims": []}
        timings["verification"] = _ms(t0)
        timings["total"] = _ms(t_start)

        trace = {
            "trace_version": TRACE_VERSION,
            "trace_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "qid": qid,
            "config": self.config_record(),
            "question": {
                "text": asked,  # redacted unless baseline_mode
                "redacted": qr["redacted"] if qr else [],
                "identifiers": qr["identifiers"] if qr else [],
                "redaction_applied": qr is not None,
            },
            "query_analysis": analysis,
            "retrieval": {
                "k": len(chunks),
                "context_chars": sum(len(c["text"]) for c in chunks),
                "chunks": chunks,
            },
            "generation": {
                "guard_decision": ("abstain" if gen["abstained"] and not gen["model_called"] else "pass")
                                  if self.use_guard else "disabled",
                **{k: gen.get(k) for k in (
                    "abstained", "abstain_reason", "model_called", "ok", "answer", "claims", "source_map",
                    "prompt_hash", "attempts", "errors", "raw_outputs", "prompt_tokens", "generation_tokens",
                    "latency_ms")},
            },
            "verification": verification,
            "timings_ms": timings,
        }
        if save:
            self.save(trace)
        return trace

    def save(self, trace: Dict[str, Any]) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{trace['qid'] or trace['trace_id']}.json"
        path.write_text(json.dumps(trace, indent=2))
        return path
