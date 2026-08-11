from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
import groq

from app.telemetry import logger, Timer
from app.models import (
    IngestRequest, IngestResponse, 
    ChunkRequest, ChunkResponse, 
    IndexRequest, IndexResponse,
    QueryRequest, RetrievalResponse,
    AnswerRequest, AnswerResponse, CitedChunk,
    TrustResponse
)
from app.ingest import load_document
from app.chunker import chunk_documents, chunk_document
from app.lance_db import index_chunks
from app.retriever import retrieve
from app.generator import generate_answer
from app.trust import build_heatmap, analyze_failure
from app.config import settings
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log service startup
    logger.info("Service startup: Trust-Aware RAG initialized")
    yield
    # Log service shutdown
    logger.info("Service shutdown")

# Create a FastAPI application with title, description, and version
app = FastAPI(
    title="Trust-Aware Cost-Efficient RAG",
    description="Production-grade FastAPI foundation for Trust-Aware RAG",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    # Wrap the endpoint with the Timer context manager
    with Timer("Health check"):
        # Log health requests
        logger.info("Health request received")
        return {"status": "ok", "service": "trust-aware-rag"}

@app.post("/ingest", response_model=IngestResponse)
def ingest_document_endpoint(request: IngestRequest):
    with Timer(f"Ingest document {request.path}"):
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
                    
            pages_detected = len(pages) if pages else None
            sections_detected = len(sections)
            
            logger.info(f"Ingested {request.path}: type={file_type}, chunks={len(chunks)}")
            
            return IngestResponse(
                source=os.path.basename(request.path),
                file_type=file_type,
                chunks_created=len(chunks),
                pages_detected=pages_detected,
                sections_detected=sections_detected,
                file_hash=chunks[0].file_hash
            )
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail="File not found")
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during ingestion")

@app.post("/chunk", response_model=ChunkResponse)
def chunk_endpoint(request: ChunkRequest):
    with Timer(f"Chunk document {request.path}"):
        logger.info(f"Chunk request received for {request.path}")
        
        try:
            file_type, docs = load_document(request.path)
            
            # Pre-calculate all chunks to get accurate stats for the response
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
                
            total_attempted = len(all_chunks)
            total_tokens = sum(c.token_estimate for c in all_chunks)
            avg_tokens = total_tokens // total_attempted if total_attempted > 0 else 0
            
            new_chunks = chunk_documents(docs)
            
            chunks_created = len(new_chunks)
            duplicates_skipped = total_attempted - chunks_created
            
            logger.info(f"Chunked {request.path}: {len(docs)} raw docs, {chunks_created} created, {duplicates_skipped} skipped, {avg_tokens} avg tokens")
            
            return ChunkResponse(
                source=os.path.basename(request.path),
                chunks_created=chunks_created,
                duplicates_skipped=duplicates_skipped,
                total_after_dedup=total_attempted,
                avg_tokens_per_chunk=avg_tokens
            )
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail="File not found")
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during chunking")

@app.post("/index", response_model=IndexResponse)
def index_document_endpoint(request: IndexRequest):
    with Timer(f"Index document {request.path}"):
        logger.info(f"Index request received for {request.path}")
        
        try:
            file_type, docs = load_document(request.path)
            
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
                
            result = index_chunks(all_chunks)
            
            logger.info(f"Indexed {request.path}: {len(all_chunks)} chunks received, "
                        f"{result['indexed']} indexed, {result['skipped']} skipped. "
                        f"Model={result['embedding_model']}, dim={result['embedding_dim']}, "
                        f"total_vectors={result['total']}")
            
            return IndexResponse(
                source=os.path.basename(request.path),
                chunks_indexed=result["indexed"],
                chunks_skipped=result["skipped"],
                total_vectors=result["total"],
                embedding_model=result["embedding_model"],
                embedding_dim=result["embedding_dim"]
            )
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail="File not found")
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during indexing")

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_endpoint(request: QueryRequest):
    with Timer(f"Retrieve {request.question}"):
        logger.info(f"Retrieve request received for query: {request.question}")
        
        try:
            response = retrieve(
                query=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter,
                section_filter=request.section_filter
            )
            
            logger.info(f"Retrieved: query='{request.question}', top_k={response.top_k_used}, "
                        f"source={request.source_filter}, section={request.section_filter}, "
                        f"confidence={response.confidence}, coverage={response.evidence_coverage}, "
                        f"time={response.retrieval_time_ms}ms, results={len(response.chunks)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during retrieval")

@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(request: AnswerRequest):
    start_time = time.perf_counter()
    with Timer(f"Answer {request.question}"):
        logger.info(f"Answer request received for query: {request.question}")
        
        try:
            # 1. Retrieval
            retrieval_response = retrieve(
                query=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter,
                section_filter=request.section_filter
            )
            
            # 2. Hallucination check
            if not retrieval_response.chunks or retrieval_response.confidence < 0.35 or retrieval_response.evidence_coverage < 0.30:
                answer = "No relevant context found."
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                generation_time_ms = 0.0
            else:
                # 3. Generation
                answer, token_usage, generation_time_ms = generate_answer(request.question, retrieval_response)
                
            # 4. Citations extraction
            citations = [
                CitedChunk(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page=c.page,
                    section=c.section,
                    score=c.score
                ) for c in retrieval_response.chunks
            ]
            
            end_time = time.perf_counter()
            total_time_ms = round((end_time - start_time) * 1000, 2)
            
            logger.info(f"Answered: query='{request.question}', top_k={retrieval_response.top_k_used}, "
                        f"confidence={retrieval_response.confidence}, coverage={retrieval_response.evidence_coverage}, "
                        f"retrieval_time={retrieval_response.retrieval_time_ms}ms, gen_time={generation_time_ms}ms, "
                        f"total_time={total_time_ms}ms, "
                        f"prompt_tokens={token_usage['prompt_tokens']}, comp_tokens={token_usage['completion_tokens']}, "
                        f"citations={len(citations)}")
                        
            return AnswerResponse(
                question=request.question,
                answer=answer,
                confidence=retrieval_response.confidence,
                evidence_coverage=retrieval_response.evidence_coverage,
                retrieval_time_ms=retrieval_response.retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=total_time_ms,
                token_usage=token_usage,
                citations=citations
            )
            
        except groq.GroqError as e:
            logger.error(f"Groq API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Groq API failure during generation")
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during answer generation")

@app.post("/trust", response_model=TrustResponse)
def trust_endpoint(request: AnswerRequest):
    start_time = time.perf_counter()
    with Timer(f"Trust {request.question}"):
        logger.info(f"Trust request received for query: {request.question}")
        
        try:
            # 1. Retrieval
            retrieval_response = retrieve(
                query=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter,
                section_filter=request.section_filter
            )
            
            # 2. Hallucination check & Generation
            if not retrieval_response.chunks or retrieval_response.confidence < 0.35 or retrieval_response.evidence_coverage < 0.30:
                answer = "No relevant context found."
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                generation_time_ms = 0.0
            else:
                answer, token_usage, generation_time_ms = generate_answer(request.question, retrieval_response)
                
            # 3. Trust layer
            heatmap = build_heatmap(retrieval_response.chunks)
            failure_analysis = analyze_failure(
                question=request.question,
                confidence=retrieval_response.confidence,
                evidence_coverage=retrieval_response.evidence_coverage,
                chunks=retrieval_response.chunks
            )
            
            citations = [
                CitedChunk(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page=c.page,
                    section=c.section,
                    score=c.score
                ) for c in retrieval_response.chunks
            ]
            
            end_time = time.perf_counter()
            total_time_ms = round((end_time - start_time) * 1000, 2)
            
            strong_count = sum(1 for h in heatmap if h.strength == "strong")
            medium_count = sum(1 for h in heatmap if h.strength == "medium")
            weak_count = sum(1 for h in heatmap if h.strength == "weak")
            
            logger.info(f"Trust Analyzed: query='{request.question}', confidence={retrieval_response.confidence}, "
                        f"coverage={retrieval_response.evidence_coverage}, strong={strong_count}, medium={medium_count}, "
                        f"weak={weak_count}, low_confidence={failure_analysis.low_confidence}, total_time={total_time_ms}ms")
                        
            return TrustResponse(
                question=request.question,
                answer=answer,
                confidence=retrieval_response.confidence,
                evidence_coverage=retrieval_response.evidence_coverage,
                heatmap=heatmap,
                failure_analysis=failure_analysis,
                citations=citations,
                retrieval_time_ms=retrieval_response.retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=total_time_ms,
                token_usage=token_usage
            )
            
        except groq.GroqError as e:
            logger.error(f"Groq API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Groq API failure during generation")
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during trust analysis")
