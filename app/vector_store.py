from app.config import settings
from app.models import ChunkedDocument
import pandas as pd
from app.lance_db import embed_texts

def backend_name() -> str:
    return settings.VECTOR_BACKEND.lower()

def index_chunks(chunks: list[ChunkedDocument]) -> dict:
    if backend_name() == "faiss":
        from app.faiss_store import index_chunks as faiss_index
        return faiss_index(chunks)
    else:
        from app.lance_db import index_chunks as lance_index
        return lance_index(chunks)

def count_vectors() -> int:
    if backend_name() == "faiss":
        from app.faiss_store import count_vectors as faiss_count
        return faiss_count()
    else:
        from app.lance_db import count_records
        return count_records()

def search(query_embedding: list[float], top_k: int, source_filter: str | None = None, section_filter: str | None = None) -> pd.DataFrame:
    if backend_name() == "faiss":
        from app.faiss_store import search as faiss_search
        return faiss_search(query_embedding, top_k, source_filter, section_filter)
    else:
        from app.lance_db import table_exists, get_table
        if not table_exists():
            return pd.DataFrame()
            
        table = get_table()
        search_req = table.search(query_embedding)
        
        filters = []
        if source_filter is not None:
            filters.append(f"source = '{source_filter}'")
        if section_filter is not None:
            filters.append(f"section = '{section_filter}'")
            
        if filters:
            search_req = search_req.where(" AND ".join(filters))
            
        search_req = search_req.limit(top_k)
        return search_req.to_pandas()
