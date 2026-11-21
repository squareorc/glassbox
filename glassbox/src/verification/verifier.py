"""
Claim verification with a natural-language-inference (NLI) model.

Stage 4 of the GlassBox pipeline. Each claim of a generated answer is checked
against the chunk(s) it cites, independently of the model that wrote it:

    premise    = a cited chunk's text (with its non-PII context line)
    hypothesis = the claim
    NLI model  -> P(entailment), P(neutral), P(contradiction)

Labels:
    SUPPORTED     the evidence entails the claim
    CONTRADICTED  the evidence conflicts with the claim
    UNSUPPORTED   the evidence neither entails nor contradicts it (or the claim cites nothing)
    UNCERTAIN     the model's top two probabilities are closer than MARGIN: it can't
                  call it reliably. MARGIN is chosen on the dev corrupted-claim set as the
                  smallest margin at which the confident labels are >= 95% correct
                  (selective prediction), see evaluate.py.

With several citations, every cited chunk on its own and all of them together
are tried; the premise with the highest entailment probability is used (a claim
may combine facts from two sections of one note).

Subject rewriting (tested, limited value): when question
redaction found an identifier ("What is [NAME] prescribed?"), a claim's generic
subject ("The patient", "[NAME]", "They", ...) is rewritten to "The person named in
the question". Under redaction no note can show that it is about that person, so
such a claim is (correctly) UNSUPPORTED. Without the rewrite, NLI models read
"[NAME] is prescribed X" as entailed by any note that prescribes X.
Measured on guard-off identity answers (test): SUPPORTED claims 93/94 without the rewrite,
57/94 with it (bare claims like "Acetaminophen 650mg PO Q6H PRN" stay entailed). And it only
triggers when question redaction found an identifier, which is exactly when the guard has
already refused. So claim verification cannot catch misattribution under redaction;
identity protection rests on the abstain guard.
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

SUPPORTED, CONTRADICTED, UNSUPPORTED, UNCERTAIN = "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNCERTAIN"
LABELS = (SUPPORTED, CONTRADICTED, UNSUPPORTED, UNCERTAIN)
# Chosen on the DEV corrupted-claim set (evaluate.py; PROGRESS.md section 7.5):
#   balanced accuracy  large 0.967 vs base 0.830 -> large
#   UNCERTAIN margin   0.0: the large model's dev accuracy is >= 0.95 with no margin, and stays
#                      flat (~0.978) for margins up to 0.45 because its errors are confident
#                      (margins 0.7-1.0), so a margin can't separate errors from correct labels.
#                      UNCERTAIN remains a label (it fires for a non-zero margin, e.g. with
#                      another model); with this model and rule it does not occur.
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_MARGIN = 0.0
MODEL_SETTINGS: Dict[str, Dict[str, float]] = {DEFAULT_MODEL: {"margin": 0.0}}

_SUBJECT = re.compile(r"^\s*(?:the\s+)?(?:patient|pt|person|individual|\[NAME\]|\[PATIENT_ID\]|they|he|she)(?!\w)(?:'s)?",
                      re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\[(?:NAME|PATIENT_ID)\]")
QUESTION_SUBJECT = "The person named in the question"


def rewrite_claim(claim: str, identifiers: Optional[List[str]]) -> str:
    """Make the claim's subject explicit when the question named a specific person."""
    if not identifiers:
        return claim
    if _SUBJECT.match(claim):
        out = _SUBJECT.sub(QUESTION_SUBJECT, claim, count=1)
    else:
        out = f"{QUESTION_SUBJECT}: {claim}"
    return _PLACEHOLDER.sub(QUESTION_SUBJECT.lower(), out)


@lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    order = {key: next(i for i, n in names.items() if n.startswith(key)) for key in ("entail", "neutral", "contra")}
    return tok, model, order, torch


class Verifier:
    def __init__(self, model_name: str = DEFAULT_MODEL, margin: Optional[float] = None,
                 device: Optional[str] = None, batch_size: int = 16):
        import torch
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.margin = margin if margin is not None else MODEL_SETTINGS.get(model_name, {}).get("margin", DEFAULT_MARGIN)
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    def probabilities(self, pairs: List[tuple]) -> List[Dict[str, float]]:
        """[(premise, hypothesis)] -> [{"entailment", "neutral", "contradiction"}]."""
        tok, model, order, torch = _load(self.model_name, self.device)
        out = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = tok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
            out += [{"entailment": p[order["entail"]], "neutral": p[order["neutral"]],
                     "contradiction": p[order["contra"]]} for p in probs]
        return out

    def label(self, probs: Dict[str, float], margin: Optional[float] = None) -> str:
        margin = self.margin if margin is None else margin
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        if ranked[0][1] - ranked[1][1] < margin:
            return UNCERTAIN
        return {"entailment": SUPPORTED, "contradiction": CONTRADICTED, "neutral": UNSUPPORTED}[ranked[0][0]]

    @staticmethod
    def premises(cited_texts: List[str]) -> List[str]:
        options = list(cited_texts)
        if len(cited_texts) > 1:
            options.append("\n\n".join(cited_texts))
        return options

    def verify_claims(self, claims: List[Dict[str, Any]], chunk_text: Dict[str, str],
                      identifiers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        claims: [{"text", "citations": [chunk_id, ...]}]. Returns one verdict per claim:
        {"claim", "checked_as", "label", "probabilities", "premise_chunks", "margin"}.
        All premise/hypothesis pairs of all claims are scored in one batched pass.
        """
        jobs, verdicts = [], []
        for ci, claim in enumerate(claims):
            hyp = rewrite_claim(claim["text"], identifiers)
            cited = [c for c in claim.get("citations", []) if c in chunk_text]
            verdicts.append({"claim": claim["text"], "checked_as": hyp, "citations": cited})
            options = self.premises([chunk_text[c] for c in cited])
            groups = [[c] for c in cited] + ([cited] if len(cited) > 1 else [])
            jobs += [(ci, g, (p, hyp)) for g, p in zip(groups, options)]
        probs = self.probabilities([pair for _, _, pair in jobs])
        best: Dict[int, tuple] = {}
        for (ci, group, _), pr in zip(jobs, probs):
            if ci not in best or pr["entailment"] > best[ci][1]["entailment"]:
                best[ci] = (group, pr)
        for ci, v in enumerate(verdicts):
            if ci in best:
                group, pr = best[ci]
                v.update(label=self.label(pr), probabilities={k: round(x, 4) for k, x in pr.items()},
                         premise_chunks=group)
            else:
                v.update(label=UNSUPPORTED, probabilities=None, premise_chunks=[], reason="no valid citation")
            v["margin"] = self.margin
        return verdicts

    def verify_answer(self, record: Dict[str, Any], chunk_text: Dict[str, str],
                      query_redaction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify a generation record; adds an answer-level verdict."""
        if record.get("abstained"):
            return {"verdict": "ABSTAINED", "claims": []}
        identifiers = (query_redaction or {}).get("identifiers") or []
        claims = self.verify_claims(record.get("claims", []), chunk_text, identifiers)
        labels = [c["label"] for c in claims]
        if not labels:
            verdict = "NO_CLAIMS"
        elif all(l == SUPPORTED for l in labels):
            verdict = "VERIFIED"
        elif CONTRADICTED in labels:
            verdict = "CONTRADICTED"
        else:
            verdict = "PARTLY_VERIFIED" if SUPPORTED in labels else "NOT_VERIFIED"
        return {"verdict": verdict, "claims": claims, "model": self.model_name}
