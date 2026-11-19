"""
Index building: chunk notes, embed them, store in ChromaDB + chunk file.

Each index lives in its own directory:

    glassbox/data/indexes/<dataset>/<variant>-<chunking>/
        manifest.json   provenance: source (redacted|raw), dataset, threshold,
                        chunking, embedding model, chunk count, build time
        chunks.jsonl    every chunk (text, offsets, metadata); BM25 is rebuilt
                        from this at load time (fast, ~1s)
        chroma/         ChromaDB persistent store of dense embeddings

Variants:
    redacted-t<thr>  built from data/processed/<dataset>/t<thr>/ (the system)
    raw              built from the `text` field of data/raw/<dataset>/ (the
                     no-privacy baseline). Never reads the PII-bearing `facts`.

Usage (from the repo root):
    python -m glassbox.src.retrieval.index --dataset realistic --variant redacted --chunking section
    python -m glassbox.src.retrieval.index --dataset realistic --all
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import chunk_document
from .query_analysis import iso_to_int

DATA_ROOT = Path("glassbox/data")
INDEX_ROOT = DATA_ROOT / "indexes"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction on queries (not on passages)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
COLLECTION = "chunks"

_embedder_cache: Dict[str, Any] = {}


def get_embedder(model_name: str = EMBEDDING_MODEL):
    """Load (once) a sentence-transformers model on Apple GPU if available."""
    if model_name not in _embedder_cache:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _embedder_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[model_name]


def embed_passages(texts: List[str], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    model = get_embedder(model_name)
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache(maxsize=4096)
def _embed_query_cached(query: str, model_name: str) -> tuple:
    model = get_embedder(model_name)
    return tuple(model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0].tolist())


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> List[float]:
    # Cached: evaluation runs the same questions against many index/method combinations
    return list(_embed_query_cached(query, model_name))


def index_name(variant: str, chunking: str, threshold: float = 0.5) -> str:
    base = f"redacted-t{threshold:.2f}" if variant == "redacted" else "raw"
    return f"{base}-{chunking}"


def index_dir(dataset: str, variant: str, chunking: str, threshold: float = 0.5) -> Path:
    return INDEX_ROOT / dataset / index_name(variant, chunking, threshold)


def load_source_documents(dataset: str, variant: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Documents to index, with only the fields retrieval may see.

    For the raw baseline, the PII-bearing ground-truth fields (`facts`,
    `entities`, `patient_name`) are dropped here so they can never enter an
    index; only the note text and non-identifying metadata are kept.
    """
    if variant == "redacted":
        path = DATA_ROOT / "processed" / dataset / f"t{threshold:.2f}" / "redacted_notes.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run: python -m glassbox.src.redaction.pipeline "
                f"--dataset {dataset} --threshold {threshold}"
            )
    elif variant == "raw":
        path = DATA_ROOT / "raw" / dataset / "synthetic_notes.jsonl"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    docs = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            docs.append({
                "doc_id": d["doc_id"],
                "text": d["text"],
                "diagnosis": d["diagnosis"],
                "note_type": d.get("note_type", "note"),
                "visit_date": d.get("visit_date") or d.get("facts", {}).get("visit_date", ""),
            })
    return docs


def build_index(
    dataset: str,
    variant: str,
    chunking: str = "section",
    threshold: float = 0.5,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build (or rebuild) one index. Returns its manifest."""
    import chromadb

    out_dir = Path(out_dir or index_dir(dataset, variant, chunking, threshold))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t0 = time.time()
    docs = load_source_documents(dataset, variant, threshold)
    chunks = [c for d in docs for c in chunk_document(d, chunking)]

    with (out_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    embeddings = embed_passages([c["text"] for c in chunks])

    client = chromadb.PersistentClient(path=str(out_dir / "chroma"))
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    batch = 2000
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=embeddings[i:i + batch],
            documents=[c["text"] for c in part],
            metadatas=[
                {**{k: c[k] for k in ("doc_id", "section", "note_type", "diagnosis", "visit_date")},
                 # integer YYYYMMDD so query-analysis date ranges can filter in Chroma
                 "visit_date_int": iso_to_int(c["visit_date"]) or 0}
                for c in part
            ],
        )

    manifest = {
        "name": out_dir.name,
        "dataset": dataset,
        "source": variant,  # "redacted" | "raw" - checked by the Retriever's access control
        "redaction_threshold": threshold if variant == "redacted" else None,
        "chunking": chunking,
        "embedding_model": EMBEDDING_MODEL,
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "build_seconds": round(time.time() - t0, 1),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build GlassBox retrieval indexes")
    parser.add_argument("--dataset", choices=["synthetic", "realistic"], default="realistic")
    parser.add_argument("--variant", choices=["redacted", "raw"], default="redacted")
    parser.add_argument("--chunking", choices=["section", "note"], default="section")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--all", action="store_true", help="Build redacted+raw x section+note")
    args = parser.parse_args()

    combos = (
        [(v, c) for v in ("redacted", "raw") for c in ("section", "note")]
        if args.all else [(args.variant, args.chunking)]
    )
    for variant, chunking in combos:
        m = build_index(args.dataset, variant, chunking, args.threshold)
        print(f"Built {args.dataset}/{m['name']}: {m['num_chunks']} chunks from "
              f"{m['num_documents']} notes in {m['build_seconds']}s")


if __name__ == "__main__":
    main()
