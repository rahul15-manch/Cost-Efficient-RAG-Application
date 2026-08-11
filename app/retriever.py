import math
import time
from app.config import settings
from app import vector_store
from app.lance_db import embed_texts
from app.models import RetrievedChunk, RetrievalResponse
from app.telemetry import logger

def choose_top_k(query: str) -> int:
    query_lower = query.lower()
    words = query.split()
    
    if len(words) <= 5:
        k = 2
    elif any(word in query_lower for word in ["compare", "summarize", "all", "differences", "explain", "overview"]):
        k = 8
    else:
        k = 5
        
    return max(settings.MIN_TOP_K, min(settings.MAX_TOP_K, k))

def embed_query(query: str) -> list[float]:
    embeddings = embed_texts([query])
    if not embeddings:
        return []
    return embeddings[0]

def normalize_score(distance: float) -> float:
    return round(math.exp(-distance), 3)

def compute_confidence(scores: list[float]) -> float:
    if not scores:
        return 0.0
    
    mean_similarity = sum(scores) / len(scores)
    top_similarity = max(scores)
    
    # Weighted combination: 60% top similarity, 40% mean similarity
    confidence = 0.6 * top_similarity + 0.4 * mean_similarity
    return round(confidence, 3)

def compute_evidence_coverage(scores: list[float]) -> float:
    if not scores:
        return 0.0
    
    top_scores = scores[:3]
    coverage = sum(top_scores) / len(top_scores)
    return round(coverage, 3)

def none_if_nan(value):
    """Convert pandas/numpy NaN to None."""
    if value is None:
        return None
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return value

def retrieve(query: str, top_k: int | None, source_filter: str | None, section_filter: str | None) -> RetrievalResponse:
    start_time = time.perf_counter()
    
    if vector_store.count_vectors() == 0:
        return RetrievalResponse(
            question=query,
            top_k_used=0,
            confidence=0.0,
            evidence_coverage=0.0,
            retrieval_time_ms=0.0,
            chunks=[]
        )
    
    k = top_k if top_k is not None else choose_top_k(query)
    
    query_embedding = embed_query(query)
    if not query_embedding:
        return RetrievalResponse(
            question=query,
            top_k_used=k,
            confidence=0.0,
            evidence_coverage=0.0,
            retrieval_time_ms=0.0,
            chunks=[]
        )
        
    results = vector_store.search(query_embedding, k, source_filter, section_filter)
    if results.empty:
        return RetrievalResponse(
            question=query,
            top_k_used=k,
            confidence=0.0,
            evidence_coverage=0.0,
            retrieval_time_ms=0.0,
            chunks=[]
        )
    
    chunks = []
    scores = []
    
    for i, row in results.iterrows():
        distance = row.get("_distance", 0.0)
        score = normalize_score(distance)
        scores.append(score)
        
        chunks.append(RetrievedChunk(
            chunk_id=row["chunk_id"],
            source=row["source"],
            page=none_if_nan(row.get("page")),
            section=none_if_nan(row.get("section")),
            text=row["text"],
            score=score,
            rank=i + 1
        ))
        
    confidence = compute_confidence(scores)
    evidence_coverage = compute_evidence_coverage(scores)
    
    end_time = time.perf_counter()
    retrieval_time_ms = round((end_time - start_time) * 1000, 2)
    
    return RetrievalResponse(
        question=query,
        top_k_used=k,
        confidence=confidence,
        evidence_coverage=evidence_coverage,
        retrieval_time_ms=retrieval_time_ms,
        chunks=chunks
    )
