"""
Access-controlled, logged retrieval over one index.

Methods:
    dense   - cosine similarity of bge-small embeddings (ChromaDB)
    bm25    - keyword relevance (exact dates, IDs, drug names)
    hybrid  - weighted reciprocal rank fusion (RRF) of the two rankings:
              score(c) = sum over lists L containing c of w_L / (RRF_K + rank_L(c))
              (Cormack et al., 2009). Rank-based, so the two methods' very
              different score scales need no calibration. The weights are
              chosen on the QA dev split (see evaluate.py --tune).

Query analysis (query_analysis.py), on by default:
    Diagnosis and visit-date constraints found in the question become
    metadata filters applied to both methods before ranking. If no chunk
    satisfies them, retrieval falls back to unfiltered search and the log
    records the fallback. With filters applied, ranking uses the question
    minus the constraint phrases ("residual query"). Query analysis also
    sets k: SINGLE_NOTE_K for an exact-date question, MULTI_NOTE_K for one
    spanning several notes.

Question redaction (redaction/query.py), on by default for redacted indexes:
    The question itself is redacted before anything else sees it. Only the
    redacted question is embedded, ranked and logged; the log records the
    TYPES removed, never the values. `last_query_redaction["identifiers"]`
    tells generation that the question names a specific person.

Low-information chunks:
    After redaction, header and contact sections ("PATIENT: [NAME] / MRN:
    [PATIENT_ID] / DOB: [DATE_OF_BIRTH]") are only placeholders and form
    labels. They carry nothing a question can be answered from, yet they
    match "patient" questions well. They are moved behind every other
    candidate (not dropped). Raw-index chunks hold real values, so they are
    never low-information and the baseline is unaffected.

Note-grouped selection:
    Notes are ordered by their best chunk, and each note contributes its top
    PER_NOTE chunks together before the next note; leftover slots are filled
    in score order. For a single-note question this is identical to plain
    ranking.

Two-hop answer sections:
    A multi-note question like "Which medications were prescribed to
    patients who presented with fatigue?" has two hops that live in
    different sections of each note: the CONDITION (symptom, in SUBJECTIVE)
    and the ASK (drug, in PLAN). Query analysis splits them. Notes are
    chosen by the whole question (their best chunk is the qualifying one);
    each chosen note's remaining PER_NOTE - 1 slots go to that note's best
    sections for the ASK alone, ranked with the same hybrid method. Each
    result records `selected_by` ("question" or "ask") for the audit trail.

Dense search is EXACT (Day 5 fix):
    Cosine similarity of the query against every chunk embedding (one matrix
    product over ~18k x 384 floats, a few ms), ties broken by chunk order. Until
    Day 5 the dense ranking came from ChromaDB's approximate HNSW index, whose
    top-50 shared as few as 23-31 chunks with the exact top-50 on identity
    questions (hundreds of near-identical header chunks) and could differ between
    processes, so two runs of the same question could retrieve different chunks.
    ChromaDB is still the store the embeddings are read from;
    dense_backend="hnsw" keeps the old behaviour for comparison.

Access control:
    The Retriever opens only indexes whose manifest says they were built from
    redacted data. Opening the raw (no-privacy) baseline requires an explicit
    `baseline=True`, and every log record states which index and source were
    used. Results per query are capped at MAX_K chunks (data minimisation).

Logging:
    With `log_path` set, every query appends one JSON line: query id/text,
    index name/source, method, k, latency, and every returned chunk with its
    rank and dense / BM25 / fused scores. This is the retrieval stage of the
    audit trail and feeds the dashboard.
"""

import json
import re
import time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bm25 import BM25
from .index import COLLECTION, embed_query
from .query_analysis import analyze_query, chroma_where, matches

MAX_K = 10
# k chosen by query analysis when not given: one note needs few chunks; a
# question spanning several notes needs ~2 chunks per note (see PER_NOTE).
SINGLE_NOTE_K = 5
MULTI_NOTE_K = 10
# Note-grouped selection: at most this many chunks per note in the first
# pass (e.g. the "why this note qualifies" chunk + the "answer" chunk), so
# one note's sections can't crowd out the other relevant notes. Remaining
# slots are filled in score order. None disables grouping.
PER_NOTE = 2
CANDIDATES = 50  # per-method candidate pool before fusion
RRF_K = 60
METHODS = ("dense", "bm25", "hybrid")
DENSE_BACKENDS = ("exact", "hnsw")
# Fusion weights (dense, bm25), chosen on the DEV split (evaluate.py --tune).
# Day 2 (39 dev questions): plateau for dense weights 1.0-5.0 -> (1.0, 1.0).
# Day 3 (117 dev questions, two-hop selection): (1.0, 1.0) 0.943, plateau
# 0.968 for dense weights 2.0-5.0 -> the simplest plateau point, (2.0, 1.0).
DEFAULT_WEIGHTS = (2.0, 1.0)

_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
# Form labels and header words that carry no answerable content
_FORM_WORDS = {
    "patient", "name", "mrn", "dob", "date", "admission", "discharge", "summary", "consultation",
    "progress", "note", "referring", "physician", "provider", "attending", "address", "email",
    "phone", "contact", "cell", "home", "work", "mobile", "emergency", "signed", "electronically",
    "file", "information", "and", "the", "for", "this", "thank", "you", "referral", "specialty", "soap", "birth", "age",
}


def is_low_information(body: str) -> bool:
    """A redacted chunk made only of placeholders, form labels and numbers."""
    if not _PLACEHOLDER.search(body):
        return False
    words = [w for w in re.findall(r"[A-Za-z]{3,}", _PLACEHOLDER.sub(" ", body)) if w.lower() not in _FORM_WORDS]
    return len(words) < 3


class AccessDenied(PermissionError):
    pass


class Retriever:
    def __init__(self, index_dir: Path, baseline: bool = False, log_path: Optional[Path] = None,
                 redact_queries: Optional[bool] = None, dense_backend: str = "exact"):
        import chromadb

        self.index_dir = Path(index_dir)
        with (self.index_dir / "manifest.json").open() as f:
            self.manifest = json.load(f)

        if self.manifest["source"] != "redacted" and not baseline:
            raise AccessDenied(
                f"Index '{self.manifest['name']}' was built from {self.manifest['source']} "
                f"(un-redacted) data. Pass baseline=True to use it for baseline experiments."
            )
        self.baseline = baseline

        with (self.index_dir / "chunks.jsonl").open() as f:
            self.chunks: List[Dict[str, Any]] = [json.loads(line) for line in f]
        self.position = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.by_doc: Dict[str, List[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c["doc_id"], []).append(i)
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.diagnoses = sorted({c["diagnosis"] for c in self.chunks})
        self.low_info = {i for i, c in enumerate(self.chunks) if is_low_information(c["body"])}
        # Redact questions by default whenever the index is the redacted one
        self.redact_queries = (self.manifest["source"] == "redacted") if redact_queries is None else redact_queries
        self.last_query_redaction: Optional[Dict[str, Any]] = None

        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        self.collection = client.get_collection(COLLECTION)
        if dense_backend not in DENSE_BACKENDS:
            raise ValueError(f"dense_backend must be one of {DENSE_BACKENDS}")
        self.dense_backend = dense_backend
        if dense_backend == "exact":
            # Every chunk's (normalised) embedding, in chunk order, for exact cosine search
            stored = self.collection.get(include=["embeddings"])
            emb = np.asarray(stored["embeddings"], dtype=np.float32)
            self.embeddings = np.zeros((len(self.chunks), emb.shape[1]), dtype=np.float32)
            self.embeddings[[self.position[cid] for cid in stored["ids"]]] = emb

        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _exact_scores(self, query: str, idxs: np.ndarray) -> List[tuple]:
        """(chunk index, cosine similarity) for `idxs`, best first; ties by chunk order."""
        sims = self.embeddings[idxs] @ np.asarray(embed_query(query), dtype=np.float32)
        order = np.lexsort((idxs, -sims))
        return [(int(idxs[j]), float(sims[j])) for j in order]

    def _dense(self, query: str, n: int, filters: Dict[str, Any]) -> List[tuple]:
        if self.dense_backend == "exact":
            idxs = (np.array([i for i, c in enumerate(self.chunks) if matches(c, filters)], dtype=np.int64)
                    if filters else np.arange(len(self.chunks)))
            return self._exact_scores(query, idxs)[:n] if len(idxs) else []
        res = self.collection.query(
            query_embeddings=[embed_query(query)], n_results=n, where=chroma_where(filters)
        )
        # Chroma returns cosine distance; similarity = 1 - distance
        return [(self.position[cid], 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def _bm25(self, query: str, n: int, filters: Dict[str, Any]) -> List[tuple]:
        if not filters:
            return self.bm25.top_k(query, n)
        # Every chunk satisfying the filters is a candidate, including ones
        # sharing no word with the query (score 0, ranked last): after
        # filtering to one note, "heart rate" must still reach "HR 94".
        scores = self.bm25.scores(query)
        scored = [(i, scores.get(i, 0.0)) for i, c in enumerate(self.chunks) if matches(c, filters)]
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:n]

    @staticmethod
    def auto_k(analysis: Dict[str, Any]) -> int:
        return MULTI_NOTE_K if analysis.get("scope") == "multi_note" else SINGLE_NOTE_K

    def _group_by_note(self, ranked: List[tuple], per_note: Optional[int], note_ordered: bool = True) -> List[tuple]:
        """
        Cap each note at `per_note` chunks before any note gets more; the rest fill in score order.
        note_ordered=True emits each note's chunks together, notes ordered by their best chunk.
        note_ordered=False (the Day 2 version) keeps the capped chunks in score order, interleaving notes.
        """
        if not per_note:
            return ranked
        groups: Dict[str, List[tuple]] = {}
        first, rest = [], []
        for item in ranked:
            group = groups.setdefault(self.chunks[item[0]]["doc_id"], [])
            if len(group) < per_note:
                group.append(item)
                first.append(item)
            else:
                rest.append(item)
        if note_ordered:
            first = [item for group in groups.values() for item in group]
        return first + rest

    def _rank_within_notes(self, query: str, doc_ids: List[str], method: str, weights: tuple) -> Dict[str, List[int]]:
        """Each note's chunks ranked against `query` (same method/fusion as the main ranking)."""
        cands = [i for d in doc_ids for i in self.by_doc[d]]
        ranks: List[Dict[int, int]] = []
        if method in ("dense", "hybrid"):
            if self.dense_backend == "exact":
                ranked = self._exact_scores(query, np.array(cands, dtype=np.int64))
                ranks.append({i: r for r, (i, _) in enumerate(ranked, 1)})
            else:
                res = self.collection.query(query_embeddings=[embed_query(query)], n_results=len(cands),
                                            where={"doc_id": {"$in": list(doc_ids)}})
                ranks.append({self.position[cid]: r for r, cid in enumerate(res["ids"][0], 1)})
        if method in ("bm25", "hybrid"):
            scores = self.bm25.scores(query)
            ordered = sorted(cands, key=lambda i: (-scores.get(i, 0.0), i))
            ranks.append({i: r for r, i in enumerate(ordered, 1)})
        w = weights if method == "hybrid" else (1.0,)
        fused = {i: sum(wt / (RRF_K + rk.get(i, len(cands) + 1)) for rk, wt in zip(ranks, w)) for i in cands}
        out: Dict[str, List[int]] = {}
        for i in sorted(cands, key=lambda i: (-fused[i], i)):
            out.setdefault(self.chunks[i]["doc_id"], []).append(i)
        return out

    def _two_hop(self, ranked: List[tuple], ask: str, per_note: int, k: int,
                 method: str, weights: tuple) -> List[tuple]:
        """Notes by their best chunk for the whole question; their other slots from the ask."""
        score = dict(ranked)
        notes: List[str] = []
        for i, _ in ranked:
            d = self.chunks[i]["doc_id"]
            if d not in notes:
                notes.append(d)
            if len(notes) * per_note >= k:
                break
        by_ask = self._rank_within_notes(ask, notes, method, weights)
        chosen, origin = [], {}
        for d in notes:
            best = next(i for i, _ in ranked if self.chunks[i]["doc_id"] == d)
            picks = [best] + [i for i in by_ask[d] if i != best and i not in self.low_info][: per_note - 1]
            chosen.extend(picks)
            origin.update({best: "question", **{i: "ask" for i in picks[1:]}})
        taken = set(chosen)
        self._origin = origin
        return [(i, score.get(i, 0.0)) for i in chosen] + [x for x in ranked if x[0] not in taken]

    def _demote_low_information(self, ranked: List[tuple]) -> List[tuple]:
        return [x for x in ranked if x[0] not in self.low_info] + [x for x in ranked if x[0] in self.low_info]

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        method: str = "hybrid",
        qid: Optional[str] = None,
        use_query_analysis: bool = True,
        weights: tuple = DEFAULT_WEIGHTS,
        per_note: Optional[int] = PER_NOTE,
        demote_low_info: bool = True,
        two_hop: bool = True,
        note_ordered: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k chunks for a query (each with rank and scores).
        k=None lets query analysis choose (SINGLE_NOTE_K or MULTI_NOTE_K).
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

        t0 = time.perf_counter()

        if self.redact_queries:
            from ..redaction.query import redact_query
            self.last_query_redaction = redact_query(query)
            query = self.last_query_redaction["text"]
        else:
            self.last_query_redaction = None

        if use_query_analysis:
            analysis = analyze_query(query, self.diagnoses)
        else:
            analysis = {"filters": {}, "extracted": [], "scope": "single_note", "residual_query": query}
        if k is None:
            k = self.auto_k(analysis)
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")
        filters = analysis["filters"]
        analysis["fallback"] = False
        if filters and not any(matches(c, filters) for c in self.chunks):
            # Constraints match nothing in the index: search unfiltered rather than return nothing
            filters, analysis["fallback"] = {}, True
        self.last_analysis = analysis

        # With filters applied, rank on what's being asked, not on the constraints
        ranking_query = analysis["residual_query"] if filters else query
        analysis["ranking_query"] = ranking_query

        dense = self._dense(ranking_query, CANDIDATES, filters) if method in ("dense", "hybrid") else []
        sparse = self._bm25(ranking_query, CANDIDATES, filters) if method in ("bm25", "hybrid") else []

        dense_score = {i: s for i, s in dense}
        bm25_score = {i: s for i, s in sparse}
        dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        bm25_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}

        if method == "dense":
            ranked = [(i, s) for i, s in dense]
        elif method == "bm25":
            ranked = [(i, s) for i, s in sparse]
        else:
            fused: Dict[int, float] = {}
            for ranks, w in ((dense_rank, weights[0]), (bm25_rank, weights[1])):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + w / (RRF_K + r)
            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))

        if demote_low_info:
            ranked = self._demote_low_information(ranked)
        self._origin = {}
        if two_hop and per_note and per_note > 1 and analysis.get("ask"):
            ranked = self._two_hop(ranked, analysis["ask"], per_note, k, method, weights)
        else:
            ranked = self._group_by_note(ranked, per_note, note_ordered)

        results = []
        for rank, (i, score) in enumerate(ranked[:k], 1):
            c = self.chunks[i]
            results.append({
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "section": c["section"],
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
                "score": round(score, 6),
                "dense_score": round(dense_score[i], 6) if i in dense_score else None,
                "bm25_score": round(bm25_score[i], 6) if i in bm25_score else None,
                "dense_rank": dense_rank.get(i),
                "bm25_rank": bm25_rank.get(i),
                "low_information": i in self.low_info,
                "selected_by": self._origin.get(i, "question"),
            })
        latency_ms = (time.perf_counter() - t0) * 1000

        if self.log_path:
            self._log(qid, query, method, k, latency_ms, results, analysis, weights, per_note, demote_low_info)
        return results

    def _log(self, qid, query, method, k, latency_ms, results, analysis, weights, per_note, demote_low_info):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "qid": qid,
            "query": query,  # already redacted when redact_queries is on
            "query_redaction": (
                {k: self.last_query_redaction[k] for k in ("redacted", "identifiers")}
                if self.last_query_redaction else None
            ),
            "demote_low_info": demote_low_info,
            "two_hop": bool(self._origin),
            "index": self.manifest["name"],
            "dataset": self.manifest["dataset"],
            "source": self.manifest["source"],
            "baseline_mode": self.baseline,
            "method": method,
            "fusion_weights": list(weights) if method == "hybrid" else None,
            "query_analysis": analysis,
            "per_note": per_note,
            "k": k,
            "latency_ms": round(latency_ms, 2),
            "results": [{key: r[key] for key in r if key != "text"} for r in results],
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
