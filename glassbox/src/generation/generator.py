"""
Cited answer generation with a local LLM (Qwen2.5-7B-Instruct, 4-bit, MLX).

Stage 3 of the GlassBox pipeline. Input: the question (already redacted by
the retriever), the retrieved chunks, and the retriever's query analysis and
question-redaction records. Output: an answer split into claims, each citing
the chunk(s) it comes from, or an explicit abstention.

Abstain guard (runs BEFORE the model is called):
    1. identifier_in_question - question redaction found a name/ID/... and
       the index is the redacted one. The records can't say who [NAME] is,
       so any answer would attribute a redacted note to the person asked
       about. The audit showed the model does exactly that when asked.
    2. retrieval_fallback - the question's diagnosis/date constraints match
       no note, so the retrieved notes don't satisfy the question.
    A code rule rather than a prompt instruction: it can't be argued out of,
    costs nothing, and the log states exactly why the system refused.
    The model can also abstain itself ("insufficient": true) when the
    sources don't contain the answer -> reason "model_insufficient".

Decoding is greedy (temperature 0), so a rerun reproduces the same output.
Every call can append one JSON line to a log: stage 3 of the audit trail.
The log never contains the original question (only the redacted one).
"""

import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parse import ParseError, parse_reply
from .prompt import PROMPT_VERSION, build_messages, prompt_hash, retry_message

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 400
MAX_RETRIES = 1

ABSTAIN_ANSWERS = {
    "identifier_in_question": "This can't be determined from the de-identified records: the question asks about a "
                              "specific person, and identities are removed from the records.",
    "retrieval_fallback": "No records match the conditions in the question (diagnosis and/or date).",
    "model_insufficient": "The records provided do not contain the answer.",
}


@lru_cache(maxsize=1)
def _load(model_id: str):
    from mlx_lm import load
    return load(model_id)


def abstain_reason(analysis: Dict[str, Any], query_redaction: Optional[Dict[str, Any]],
                   index_source: str) -> Optional[str]:
    """The guard: why the system must not answer, or None."""
    if index_source == "redacted" and query_redaction and query_redaction.get("identifiers"):
        return "identifier_in_question"
    if analysis.get("fallback"):
        return "retrieval_fallback"
    return None


class Generator:
    def __init__(self, model_id: str = MODEL_ID, max_tokens: int = MAX_TOKENS,
                 log_path: Optional[Path] = None):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _complete(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        from mlx_lm import stream_generate
        model, tokenizer = _load(self.model_id)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text, last = "", None
        for last in stream_generate(model, tokenizer, prompt, max_tokens=self.max_tokens):  # greedy
            text += last.text
        return {"text": text,
                "prompt_tokens": last.prompt_tokens if last else 0,
                "generation_tokens": last.generation_tokens if last else 0}

    def answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        query_redaction: Optional[Dict[str, Any]],
        index_source: str,
        qid: Optional[str] = None,
        use_guard: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer `question` from `chunks`. `question` must be the text the
        retriever actually used (redacted when question redaction is on).
        """
        t0 = time.perf_counter()
        record: Dict[str, Any] = {
            "qid": qid, "question": question, "index_source": index_source,
            "chunk_ids": [c["chunk_id"] for c in chunks], "guard": use_guard,
            "model": self.model_id, "prompt_version": PROMPT_VERSION,
        }

        reason = abstain_reason(analysis, query_redaction, index_source) if use_guard else None
        if reason:
            record.update(answer=ABSTAIN_ANSWERS[reason], claims=[], abstained=True, abstain_reason=reason,
                          model_called=False, ok=True, attempts=0, raw_outputs=[], errors=[])
        else:
            messages, id_map = build_messages(question, chunks)
            record["prompt_hash"] = prompt_hash(messages)
            record["source_map"] = id_map
            raw_outputs, errors, parsed = [], [], None
            tokens = {"prompt_tokens": 0, "generation_tokens": 0}
            for _ in range(1 + MAX_RETRIES):
                out = self._complete(messages)
                raw_outputs.append(out["text"])
                for key in tokens:
                    tokens[key] += out[key]
                try:
                    parsed = parse_reply(out["text"], id_map)
                    break
                except ParseError as e:
                    errors.append(str(e))
                    messages = messages + [{"role": "assistant", "content": out["text"]}, retry_message(str(e))]
            record.update(model_called=True, attempts=len(raw_outputs), raw_outputs=raw_outputs,
                          errors=errors, **tokens)
            if parsed is None:
                record.update(ok=False, answer="", claims=[], abstained=False, abstain_reason=None)
            elif parsed["insufficient"]:
                record.update(ok=True, answer=parsed["answer"] or ABSTAIN_ANSWERS["model_insufficient"],
                              claims=parsed["claims"], abstained=True, abstain_reason="model_insufficient")
            else:
                record.update(ok=True, answer=parsed["answer"], claims=parsed["claims"],
                              abstained=False, abstain_reason=None)

        record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                    **record}) + "\n")
        return record
