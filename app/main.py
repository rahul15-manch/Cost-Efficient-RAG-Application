from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
import groq

from app.telemetry import logger
from app.models import (
    IngestRequest, IngestResponse, 
    ChunkRequest, ChunkResponse, 
    IndexRequest, IndexResponse,
    QueryRequest, RetrievalResponse,
    AnswerRequest, AnswerResponse, CitedChunk,
    TrustResponse, PipelineRequest, PipelineResponse, StageMetrics, HeatmapItem, FailureAnalysis, MetricsSnapshot
)
from app.ingest import load_document
from app.chunker import chunk_documents, chunk_document
from app.lance_db import table_exists, get_table
from app import vector_store
from app.retriever import retrieve
from app.generator import generate_answer
from app.trust import build_heatmap, analyze_failure
from app.config import settings
from app.request_context import generate_request_id, attach_request_id
from app.analytics import record_success, record_failure, snapshot, reset_metrics
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service startup: Trust-Aware RAG initialized")
    yield
    logger.info("Service shutdown")

app = FastAPI(
    title="Trust-Aware Cost-Efficient RAG",
    description="Production-grade FastAPI foundation for Trust-Aware RAG",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    logger.info("Health request received")
    return {"status": "ok", "service": "trust-aware-rag"}

@app.get("/config/check")
def config_check_endpoint():
    from app.config import is_groq_configured, settings
    from app.lance_db import table_exists, count_records
    
    return {
        "groq_configured": is_groq_configured(),
        "embedding_model": settings.EMBED_MODEL,
        "lancedb_exists": table_exists(),
        "vector_count": count_records() if table_exists() else 0
    }

@app.post("/ingest", response_model=IngestResponse)
def ingest_document_endpoint(request: IngestRequest):
    logger.info(f"Ingest request received for {request.path}")
    try:
        file_type, chunks = load_document(request.path)
        pages = set()
        sections = set()
        for chunk in chunks:
            if chunk.page is not None:
                pages.add(chunk.page)
            if chunk.section is not None:
                sections.add(chunk.section)
        return IngestResponse(
            source=os.path.basename(request.path),
            file_type=file_type,
            chunks_created=len(chunks),
            pages_detected=len(pages) if pages else None,
            sections_detected=len(sections),
            file_hash=chunks[0].file_hash
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chunk", response_model=ChunkResponse)
def chunk_endpoint(request: ChunkRequest):
    try:
        file_type, docs = load_document(request.path)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
            
        total_attempted = len(all_chunks)
        total_tokens = sum(c.token_estimate for c in all_chunks)
        avg_tokens = total_tokens // total_attempted if total_attempted > 0 else 0
        new_chunks = chunk_documents(docs)
        chunks_created = len(new_chunks)
        
        return ChunkResponse(
            source=os.path.basename(request.path),
            chunks_created=chunks_created,
            duplicates_skipped=total_attempted - chunks_created,
            total_after_dedup=total_attempted,
            avg_tokens_per_chunk=avg_tokens
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/index", response_model=IndexResponse)
def index_document_endpoint(request: IndexRequest):
    try:
        file_type, docs = load_document(request.path)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
            
        result = vector_store.index_chunks(all_chunks)
        
        return IndexResponse(
            source=os.path.basename(request.path),
            chunks_indexed=result["indexed"],
            chunks_skipped=result["skipped"],
            total_vectors=result["total"],
            embedding_model=result["embedding_model"],
            embedding_dim=result["embedding_dim"],
            vector_backend=vector_store.backend_name()
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="File not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_endpoint(request: QueryRequest):
    try:
        return retrieve(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            section_filter=request.section_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(request: AnswerRequest):
    start_time = time.perf_counter()
    try:
        retrieval_response = retrieve(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            section_filter=request.section_filter
        )
        if not retrieval_response.chunks or retrieval_response.confidence < 0.35 or retrieval_response.evidence_coverage < 0.30:
            answer = "No relevant context found."
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            generation_time_ms = 0.0
        else:
            answer, token_usage, generation_time_ms = generate_answer(request.question, retrieval_response)
            
        citations = [
            CitedChunk(chunk_id=c.chunk_id, source=c.source, page=c.page, section=c.section, score=c.score, text=c.text)
            for c in retrieval_response.chunks
        ]
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return AnswerResponse(
            question=request.question, answer=answer, confidence=retrieval_response.confidence,
            evidence_coverage=retrieval_response.evidence_coverage, retrieval_time_ms=retrieval_response.retrieval_time_ms,
            generation_time_ms=generation_time_ms, total_time_ms=total_time_ms,
            token_usage=token_usage, citations=citations
        )
    except groq.GroqError as e:
        raise HTTPException(status_code=500, detail="Groq API failure during generation")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/trust", response_model=TrustResponse)
def trust_endpoint(request: AnswerRequest):
    start_time = time.perf_counter()
    try:
        retrieval_response = retrieve(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            section_filter=request.section_filter
        )
        if not retrieval_response.chunks or retrieval_response.confidence < 0.35 or retrieval_response.evidence_coverage < 0.30:
            answer = "No relevant context found."
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            generation_time_ms = 0.0
        else:
            answer, token_usage, generation_time_ms = generate_answer(request.question, retrieval_response)
            
        heatmap = build_heatmap(retrieval_response.chunks)
        failure_analysis = analyze_failure(
            question=request.question, confidence=retrieval_response.confidence,
            evidence_coverage=retrieval_response.evidence_coverage, chunks=retrieval_response.chunks
        )
        citations = [CitedChunk(chunk_id=c.chunk_id, source=c.source, page=c.page, section=c.section, score=c.score, text=c.text) for c in retrieval_response.chunks]
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return TrustResponse(
            question=request.question, answer=answer, confidence=retrieval_response.confidence,
            evidence_coverage=retrieval_response.evidence_coverage, heatmap=heatmap,
            failure_analysis=failure_analysis, citations=citations, retrieval_time_ms=retrieval_response.retrieval_time_ms,
            generation_time_ms=generation_time_ms, total_time_ms=total_time_ms, token_usage=token_usage
        )
    except groq.GroqError as e:
        raise HTTPException(status_code=500, detail="Groq API failure during generation")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/pipeline", response_model=PipelineResponse)
def pipeline_endpoint(request: PipelineRequest):
    request_id = generate_request_id()
    req_logger = attach_request_id(logger, request_id)
    
    start_time = time.perf_counter()
    
    try:
        # 1. Retrieval
        retrieval_start = time.perf_counter()
        retrieval_response = retrieve(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            section_filter=request.section_filter
        )
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
        
        # 2. Generation
        gen_start = time.perf_counter()
        if not retrieval_response.chunks or retrieval_response.confidence < 0.35 or retrieval_response.evidence_coverage < 0.30:
            answer = "No relevant context found."
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            answer, token_usage, _ = generate_answer(request.question, retrieval_response)
        generation_ms = round((time.perf_counter() - gen_start) * 1000, 2)
        
        # 3. Trust Layer
        trust_start = time.perf_counter()
        heatmap = build_heatmap(retrieval_response.chunks)
        failure_analysis = analyze_failure(
            question=request.question,
            confidence=retrieval_response.confidence,
            evidence_coverage=retrieval_response.evidence_coverage,
            chunks=retrieval_response.chunks
        )
        citations = [
            CitedChunk(chunk_id=c.chunk_id, source=c.source, page=c.page, section=c.section, score=c.score, text=c.text)
            for c in retrieval_response.chunks
        ]
        trust_ms = round((time.perf_counter() - trust_start) * 1000, 2)
        
        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        req_logger.info(
            f"stage=pipeline query='{request.question}' top_k={retrieval_response.top_k_used} "
            f"confidence={retrieval_response.confidence} evidence_coverage={retrieval_response.evidence_coverage} "
            f"retrieval_ms={retrieval_ms} generation_ms={generation_ms} trust_ms={trust_ms} "
            f"total_ms={total_ms} prompt_tokens={token_usage['prompt_tokens']} "
            f"completion_tokens={token_usage['completion_tokens']} citation_count={len(citations)}"
        )
        
        record_success(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            confidence=retrieval_response.confidence,
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"]
        )
        
        return PipelineResponse(
            request_id=request_id,
            question=request.question,
            answer=answer,
            confidence=retrieval_response.confidence,
            evidence_coverage=retrieval_response.evidence_coverage,
            heatmap=heatmap,
            failure_analysis=failure_analysis,
            citations=citations,
            token_usage=token_usage,
            metrics=StageMetrics(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                trust_ms=trust_ms,
                total_ms=total_ms
            )
        )
        
    except Exception as e:
        record_failure()
        req_logger.error(f"stage=pipeline error='{str(e)}'")
        raise HTTPException(status_code=500, detail={"error": "Pipeline failed", "request_id": request_id})

@app.get("/metrics", response_model=MetricsSnapshot)
def metrics_endpoint():
    return snapshot()

@app.get("/vector-stats")
def vector_stats_endpoint():
    return {
        "backend": vector_store.backend_name(),
        "vectors": vector_store.count_vectors(),
        "embedding_model": settings.EMBED_MODEL
    }

@app.post("/metrics/reset")
def reset_metrics_endpoint():
    reset_metrics()
    return {"status": "reset"}

@app.get("/documents/stats")
def document_stats_endpoint():
    if not table_exists():
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "unique_sources": 0,
            "unique_sections": 0
        }
        
    try:
        table = get_table()
        df = table.to_pandas()
        
        total_vectors = len(df)
        total_chunks = df["chunk_id"].nunique()
        unique_sources = df["source"].nunique()
        unique_sections = df["section"].nunique()
        
        return {
            "total_documents": unique_sources,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
            "unique_sources": unique_sources,
            "unique_sections": unique_sections
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate document statistics")
