"""
Access-controlled, logged retrieval over one index.

Methods:
    dense   - cosine similarity of bge-small embeddings (ChromaDB)
    bm25    - keyword relevance (exact dates, IDs, drug names)
    hybrid  - reciprocal rank fusion (RRF) of the two rankings:
              score(c) = sum over lists L containing c of 1 / (RRF_K + rank_L(c))
              (Cormack et al., 2009). Rank-based, so the two methods' very
              different score scales need no calibration or tuning.

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

DEFAULT_K = 5
MAX_K = 10
CANDIDATES = 50  # per-method candidate pool before fusion
RRF_K = 60
METHODS = ("dense", "bm25", "hybrid")


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

        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        self.collection = client.get_collection(COLLECTION)

        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _dense(self, query: str, n: int) -> List[tuple]:
        res = self.collection.query(query_embeddings=[embed_query(query)], n_results=n)
        # Chroma returns cosine distance; similarity = 1 - distance
        return [(self.position[cid], 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def _bm25(self, query: str, n: int) -> List[tuple]:
        return self.bm25.top_k(query, n)

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        method: str = "hybrid",
        qid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-k chunks for a query (each with rank and scores)."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")

        t0 = time.perf_counter()
        dense = self._dense(query, CANDIDATES) if method in ("dense", "hybrid") else []
        sparse = self._bm25(query, CANDIDATES) if method in ("bm25", "hybrid") else []

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
            for ranks in (dense_rank, bm25_rank):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))

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
            self._log(qid, query, method, k, latency_ms, results)
        return results

    def _log(self, qid, query, method, k, latency_ms, results):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "qid": qid,
            "query": query,
            "index": self.manifest["name"],
            "dataset": self.manifest["dataset"],
            "source": self.manifest["source"],
            "baseline_mode": self.baseline,
            "method": method,
            "k": k,
            "latency_ms": round(latency_ms, 2),
            "results": [{key: r[key] for key in r if key != "text"} for r in results],
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
