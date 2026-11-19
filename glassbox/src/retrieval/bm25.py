"""
BM25 (Okapi) keyword retrieval.

Complements dense embeddings, which blur exact strings: to an embedding model
"2024-11-10" and "2024-11-19" (or two MRNs) look nearly identical, but the
clinical questions hinge on such exact tokens.

score(q, d) = sum over query terms t of
    IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Keep dates (2024-11-10), BP (157/94), IDs (MRN0033092, f-819600) and
# decimals (98.5) as single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")

# Very common words that carry no retrieval signal in these notes/questions.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "was", "were", "what", "which", "who",
    "with", "whose", "how", "many", "did", "does", "this", "that", "patient",
    "patients",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: List[int] = []
        # Inverted index: term -> list of (doc index, term frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n_docs = len(documents)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.idf = {
            t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: str) -> Dict[int, float]:
        """BM25 score for every document containing at least one query term."""
        out: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        scored = self.scores(query)
        return sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:k]
