from pydantic import BaseModel
from datetime import datetime

class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    text: str
    file_hash: str
    created_at: datetime

class ChunkedDocument(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    text: str
    file_hash: str
    chunk_index: int
    token_estimate: int
    created_at: datetime

class VectorRecord(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    text: str
    file_hash: str
    chunk_index: int
    token_estimate: int
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    created_at: datetime

class IngestRequest(BaseModel):
    path: str

class IngestResponse(BaseModel):
    source: str
    file_type: str
    chunks_created: int
    pages_detected: int | None = None
    sections_detected: int
    file_hash: str

class ChunkRequest(BaseModel):
    path: str

class ChunkResponse(BaseModel):
    source: str
    chunks_created: int
    duplicates_skipped: int
    total_after_dedup: int
    avg_tokens_per_chunk: int

class IndexRequest(BaseModel):
    path: str

class IndexResponse(BaseModel):
    source: str
    chunks_indexed: int
    chunks_skipped: int
    total_vectors: int
    embedding_model: str
    embedding_dim: int

class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    source_filter: str | None = None
    section_filter: str | None = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    text: str
    score: float
    rank: int

class RetrievalResponse(BaseModel):
    question: str
    top_k_used: int
    confidence: float
    evidence_coverage: float
    retrieval_time_ms: float
    chunks: list[RetrievedChunk]
