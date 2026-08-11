from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.telemetry import logger, Timer

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
    with Timer("Health check"):
        logger.info("Health request received")
        return {"status": "ok", "service": "trust-aware-rag"}
