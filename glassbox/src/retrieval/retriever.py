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

Note-grouped selection:
    The ranked chunks are re-ordered so each note contributes at most
    PER_NOTE chunks before any note contributes more; leftover slots are
    filled in score order. For a single-note question this is identical to
    plain ranking.

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
import time
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
# Fusion weights (dense, bm25). Equal weights (1.0, 1.0) were the Day 2 baseline;
# DEV-split tuning (evaluate.py --tune) found a plateau for dense weights 1.0-5.0;
# the simplest point, equal weights, is the default.
DEFAULT_WEIGHTS = (1.0, 1.0)


class AccessDenied(PermissionError):
    pass


class Retriever:
    def __init__(self, index_dir: Path, baseline: bool = False, log_path: Optional[Path] = None):
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
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self.diagnoses = sorted({c["diagnosis"] for c in self.chunks})

        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        self.collection = client.get_collection(COLLECTION)

        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _dense(self, query: str, n: int, filters: Dict[str, Any]) -> List[tuple]:
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

    def _group_by_note(self, ranked: List[tuple], per_note: Optional[int]) -> List[tuple]:
        if not per_note:
            return ranked
        taken: Dict[str, int] = {}
        first, rest = [], []
        for item in ranked:
            doc = self.chunks[item[0]]["doc_id"]
            if taken.get(doc, 0) < per_note:
                taken[doc] = taken.get(doc, 0) + 1
                first.append(item)
            else:
                rest.append(item)
        return first + rest

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        method: str = "hybrid",
        qid: Optional[str] = None,
        use_query_analysis: bool = True,
        weights: tuple = DEFAULT_WEIGHTS,
        per_note: Optional[int] = PER_NOTE,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k chunks for a query (each with rank and scores).
        k=None lets query analysis choose (SINGLE_NOTE_K or MULTI_NOTE_K).
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

        t0 = time.perf_counter()

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

        ranked = self._group_by_note(ranked, per_note)

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
            })
        latency_ms = (time.perf_counter() - t0) * 1000

        if self.log_path:
            self._log(qid, query, method, k, latency_ms, results, analysis, weights, per_note)
        return results

    def _log(self, qid, query, method, k, latency_ms, results, analysis, weights, per_note):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "qid": qid,
            "query": query,
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
