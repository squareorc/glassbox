"""
GlassBox retrieval module.

Chunks sanitized notes, indexes them (dense embeddings in ChromaDB + BM25),
and retrieves the most relevant chunks for a query with access control and
a per-query retrieval log for the audit trail.
"""
