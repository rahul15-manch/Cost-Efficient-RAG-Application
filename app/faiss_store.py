import os
import pickle
import numpy as np
import pandas as pd
from app.config import settings
from app.models import ChunkedDocument
from app.lance_db import embed_texts

FAISS_DIR = "data/processed/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

_index = None
_metadata = None

def load_index():
    global _index, _metadata
    if _index is not None and _metadata is not None:
        return
        
    import faiss
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)
    else:
        _index = None
        _metadata = []

def save_index():
    import faiss
    os.makedirs(FAISS_DIR, exist_ok=True)
    if _index is not None:
        faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

def create_index(dim: int):
    global _index, _metadata
    import faiss
    _index = faiss.IndexFlatL2(dim)
    _metadata = []

def index_chunks(chunks: list[ChunkedDocument]) -> dict:
    global _index, _metadata
    
    if not chunks:
        return {"indexed": 0, "skipped": 0, "total": count_vectors()}
        
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    dim = len(embeddings[0])
    
    load_index()
    if _index is None:
        create_index(dim)
        
    seen_ids = set(m["chunk_id"] for m in _metadata)
    
    vectors_to_add = []
    metadata_to_add = []
    skipped = 0
    
    for chunk, emb in zip(chunks, embeddings):
        if chunk.chunk_id in seen_ids:
            skipped += 1
            continue
            
        vectors_to_add.append(emb)
        metadata_to_add.append({
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index
        })
        
    if vectors_to_add:
        vectors_np = np.array(vectors_to_add, dtype=np.float32)
        _index.add(vectors_np)
        _metadata.extend(metadata_to_add)
        save_index()
        
    return {
        "indexed": len(vectors_to_add),
        "skipped": skipped,
        "total": count_vectors(),
        "embedding_model": settings.EMBED_MODEL,
        "embedding_dim": dim
    }

def count_vectors() -> int:
    load_index()
    return _index.ntotal if _index else 0

def search(query_embedding: list[float], top_k: int, source_filter: str | None = None, section_filter: str | None = None) -> pd.DataFrame:
    load_index()
    if _index is None or _index.ntotal == 0:
        return pd.DataFrame()
        
    fetch_k = top_k if not (source_filter or section_filter) else min(_index.ntotal, max(top_k * 5, 100))
    
    q_np = np.array([query_embedding], dtype=np.float32)
    distances, indices = _index.search(q_np, fetch_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = _metadata[idx]
        
        if source_filter and meta.get("source") != source_filter:
            continue
        if section_filter and meta.get("section") != section_filter:
            continue
            
        row = meta.copy()
        row["_distance"] = float(dist)
        results.append(row)
        
        if len(results) == top_k:
            break
            
    return pd.DataFrame(results)
