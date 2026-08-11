import os
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.models import VectorRecord, ChunkedDocument
from app.telemetry import logger

_db = None
_model = None

def get_db():
    global _db
    if _db is None:
        os.makedirs(settings.LANCEDB_PATH, exist_ok=True)
        _db = lancedb.connect(settings.LANCEDB_PATH)
    return _db

def load_embedding_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

def table_exists() -> bool:
    db = get_db()
    return "documents" in db.table_names()

def get_table():
    db = get_db()
    return db.open_table("documents")

def create_table_if_missing(dim: int):
    if not table_exists():
        db = get_db()
        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32(), nullable=True),
            pa.field("section", pa.string(), nullable=True),
            pa.field("text", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("token_estimate", pa.int32()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_dim", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("created_at", pa.timestamp('us'))
        ])
        db.create_table("documents", schema=schema)

def existing_chunk_ids() -> set[str]:
    if not table_exists():
        return set()
    table = get_table()
    try:
        df = table.search().select(["chunk_id"]).to_pandas()
    except Exception:
        df = table.to_pandas()
    return set(df["chunk_id"].tolist())

def add_records(records: list[VectorRecord]) -> int:
    if not records:
        return 0
    table = get_table()
    data = [record.model_dump() for record in records]
    table.add(data)
    return len(records)

def count_records() -> int:
    if not table_exists():
        return 0
    table = get_table()
    return len(table)

def search_by_metadata(source: str | None = None, section: str | None = None) -> list[dict]:
    if not table_exists():
        return []
    table = get_table()
    
    filters = []
    if source is not None:
        filters.append(f"source = '{source}'")
    if section is not None:
        filters.append(f"section = '{section}'")
        
    try:
        if filters:
            query_str = " AND ".join(filters)
            df = table.search().where(query_str).to_pandas()
        else:
            df = table.to_pandas()
    except Exception:
        df = table.to_pandas()
        if source is not None:
            df = df[df["source"] == source]
        if section is not None:
            df = df[df["section"] == section]
            
    return df.to_dict(orient="records")

def index_chunks(chunks: list[ChunkedDocument]) -> dict:
    if not chunks:
        return {
            "indexed": 0,
            "skipped": 0,
            "total": count_records(),
            "embedding_dim": 0,
            "embedding_model": settings.EMBED_MODEL
        }
        
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    dim = len(embeddings[0])
    
    create_table_if_missing(dim)
    seen_ids = existing_chunk_ids()
    
    new_records = []
    skipped = 0
    
    for i, chunk in enumerate(chunks):
        if chunk.chunk_id in seen_ids:
            skipped += 1
            continue
            
        record = VectorRecord(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            file_hash=chunk.file_hash,
            chunk_index=chunk.chunk_index,
            token_estimate=chunk.token_estimate,
            embedding_model=settings.EMBED_MODEL,
            embedding_dim=dim,
            embedding=embeddings[i],
            created_at=chunk.created_at
        )
        new_records.append(record)
        
    indexed = add_records(new_records)
    total = count_records()
    
    return {
        "indexed": indexed,
        "skipped": skipped,
        "total": total,
        "embedding_dim": dim,
        "embedding_model": settings.EMBED_MODEL
    }
