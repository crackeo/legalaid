"""Hybrid retrieval: BM25 + vector similarity, fused with reciprocal rank fusion.

Legal queries mix exact terms of art ("Section 410", "felony of the fourth
degree") where keyword search wins, with paraphrases ("what happens if I hit
someone") where embeddings win. RRF combines both without score calibration.
"""

from .store import Hit, Store

RRF_K = 60  # standard RRF damping constant


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(hit.chunk_id, hit)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for cid, score in fused:
        hit = best[cid]
        hit.score = score
        out.append(hit)
    return out


def retrieve(store: Store, embedder, query: str, k: int = 6,
             candidates: int = 20) -> list[Hit]:
    """Top-k provisions for a query via hybrid BM25 + vector search."""
    bm25_hits = store.search_bm25(query, k=candidates)
    query_vec = embedder.embed([query])[0]
    vec_hits = store.search_vector(query_vec, k=candidates)
    return reciprocal_rank_fusion([bm25_hits, vec_hits])[:k]
