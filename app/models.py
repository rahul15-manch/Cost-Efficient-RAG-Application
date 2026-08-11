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

class IngestRequest(BaseModel):
    path: str

class IngestResponse(BaseModel):
    source: str
    file_type: str
    chunks_created: int
    pages_detected: int | None = None
    sections_detected: int
    file_hash: str
