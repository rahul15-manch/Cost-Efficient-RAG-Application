from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.telemetry import logger, Timer
from app.models import IngestRequest, IngestResponse
from app.ingest import load_document
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
def ingest_document(request: IngestRequest):
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
