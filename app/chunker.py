import os
import nltk
from datetime import datetime, timezone
from app.models import DocumentChunk, ChunkedDocument
from app.utils import make_chunk_id, load_seen_chunk_ids, save_seen_chunk_ids
from app.config import settings

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception:
        pass

def split_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)

def estimate_tokens(text: str) -> int:
    """A simple token estimation: ~4 chars per token."""
    return len(text) // 4

def chunk_document(doc: DocumentChunk, chunk_size: int, overlap: int) -> list[ChunkedDocument]:
    sentences = split_sentences(doc.text)
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    chunk_index = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_length + sentence_len + (1 if current_length > 0 else 0) > chunk_size and current_chunk_sentences:
            text = " ".join(current_chunk_sentences)
            chunk_id = make_chunk_id(doc.source, doc.page, doc.section, text, chunk_index)
            chunks.append(ChunkedDocument(
                chunk_id=chunk_id,
                source=doc.source,
                page=doc.page,
                section=doc.section,
                text=text,
                file_hash=doc.file_hash,
                chunk_index=chunk_index,
                token_estimate=estimate_tokens(text),
                created_at=datetime.utcnow()
            ))
            chunk_index += 1
            
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk_sentences):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_length = sum(len(s) + 1 for s in current_chunk_sentences) - (1 if current_chunk_sentences else 0)
            
        current_chunk_sentences.append(sentence)
        current_length += sentence_len + (1 if current_length > 0 else 0)
        
    if current_chunk_sentences:
        text = " ".join(current_chunk_sentences)
        chunk_id = make_chunk_id(doc.source, doc.page, doc.section, text, chunk_index)
        chunks.append(ChunkedDocument(
            chunk_id=chunk_id,
            source=doc.source,
            page=doc.page,
            section=doc.section,
            text=text,
            file_hash=doc.file_hash,
            chunk_index=chunk_index,
            token_estimate=estimate_tokens(text),
            created_at=datetime.utcnow()
        ))
        
    return chunks

def chunk_documents(docs: list[DocumentChunk]) -> list[ChunkedDocument]:
    seen_path = "data/processed/seen_chunks.json"
    seen_ids = load_seen_chunk_ids(seen_path)
    
    new_chunks = []
    
    for doc in docs:
        doc_chunks = chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for chunk in doc_chunks:
            if chunk.chunk_id not in seen_ids:
                new_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)
                
    if new_chunks:
        save_seen_chunk_ids(seen_path, seen_ids)
        
    return new_chunks
