import os
import shutil
import pytest
from fastapi.testclient import TestClient
import fitz

# Override environment variables for isolation before loading app
os.environ["LANCEDB_PATH"] = "tests/test_lancedb"
os.environ["GROQ_API_KEY"] = "test_key"
os.environ["CHUNK_SIZE"] = "600"
os.environ["CHUNK_OVERLAP"] = "100"

from app.main import app

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    # Setup directories
    os.makedirs("tests/test_lancedb", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Copy text samples to working directory
    shutil.copy("tests/sample.md", "data/raw/sample.md")
    shutil.copy("tests/sample.html", "data/raw/sample.html")
    
    # Generate PDF programmatically for testing
    pdf_path = "data/raw/sample.pdf"
    if not os.path.exists(pdf_path):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "PDF Content Page 1")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "PDF Content Page 2")
        doc.save(pdf_path)
        doc.close()
    
    # Generate empty and unsupported files
    with open("data/raw/empty.md", "w") as f:
        pass
    with open("data/raw/unsupported.txt", "w") as f:
        f.write("Text file")
        
    # Clear existing state
    seen_path = "data/processed/seen_chunks.json"
    if os.path.exists(seen_path):
        os.remove(seen_path)
    if os.path.exists("tests/test_lancedb"):
        shutil.rmtree("tests/test_lancedb")
        os.makedirs("tests/test_lancedb")

    yield

    # Teardown / Cleanup
    if os.path.exists(seen_path):
        os.remove(seen_path)
    if os.path.exists("tests/test_lancedb"):
        shutil.rmtree("tests/test_lancedb")
        
    for f in ["sample.md", "sample.html", "sample.pdf", "empty.md", "unsupported.txt"]:
        path = os.path.join("data/raw", f)
        if os.path.exists(path):
            os.remove(path)

# --- Milestone 1: Health & Telemetry ---
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "trust-aware-rag"}
    assert os.path.exists("logs/app.log")

# --- Milestone 2: Ingestion ---
def test_ingest_markdown():
    response = client.post("/ingest", json={"path": "data/raw/sample.md"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "sample.md"
    assert data["file_type"] == "markdown"
    assert data["chunks_created"] > 0
    assert data["sections_detected"] > 0

def test_ingest_html():
    response = client.post("/ingest", json={"path": "data/raw/sample.html"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "sample.html"
    assert data["file_type"] == "html"
    assert data["chunks_created"] > 0
    assert data["sections_detected"] > 0

def test_ingest_pdf():
    response = client.post("/ingest", json={"path": "data/raw/sample.pdf"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "sample.pdf"
    assert data["file_type"] == "pdf"
    assert data["pages_detected"] == 2

def test_stable_hash():
    r1 = client.post("/ingest", json={"path": "data/raw/sample.md"}).json()
    r2 = client.post("/ingest", json={"path": "data/raw/sample.md"}).json()
    assert r1["file_hash"] == r2["file_hash"]

# --- Milestone 3: Chunking & Idempotency ---
def test_chunking_idempotency_and_overlap():
    seen_path = "data/processed/seen_chunks.json"
    if os.path.exists(seen_path):
        os.remove(seen_path)
        
    r1 = client.post("/chunk", json={"path": "data/raw/sample.md"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["chunks_created"] > 0
    assert d1["duplicates_skipped"] == 0
    
    r2 = client.post("/chunk", json={"path": "data/raw/sample.md"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["chunks_created"] == 0
    assert d2["duplicates_skipped"] == d1["total_after_dedup"]
    assert d2["total_after_dedup"] == d1["total_after_dedup"]

# --- Milestone 4: Indexing & Persistence ---
def test_indexing():
    r1 = client.post("/index", json={"path": "data/raw/sample.md"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["chunks_indexed"] > 0
    assert d1["chunks_skipped"] == 0
    assert d1["embedding_dim"] > 0
    assert d1["total_vectors"] > 0
    
    # Idempotency checks
    r2 = client.post("/index", json={"path": "data/raw/sample.md"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["chunks_indexed"] == 0
    assert d2["chunks_skipped"] == d1["chunks_indexed"]
    
    # Table creation check
    assert os.path.exists("tests/test_lancedb/documents.lance")

# --- Milestone 5: Adaptive Retrieval ---
def test_adaptive_retrieval_top_k():
    # Make sure we have some data across files
    client.post("/index", json={"path": "data/raw/sample.html"})
    client.post("/index", json={"path": "data/raw/sample.pdf"})

    # Short query -> k=2
    r_short = client.post("/retrieve", json={"question": "Recall?"})
    assert r_short.status_code == 200
    assert r_short.json()["top_k_used"] == 2
    
    # Normal query -> k=5
    r_normal = client.post("/retrieve", json={"question": "What is the purpose of this project?"})
    assert r_normal.status_code == 200
    assert r_normal.json()["top_k_used"] == 5
    
    # Summarize query -> k=8
    r_sum = client.post("/retrieve", json={"question": "Summarize all sections"})
    assert r_sum.status_code == 200
    assert r_sum.json()["top_k_used"] == 8

def test_retrieval_metrics():
    res = client.post("/retrieve", json={"question": "evaluation cost"}).json()
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["evidence_coverage"] <= 1.0
    
    scores = [c["score"] for c in res["chunks"]]
    # check monotonically non-increasing
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1]

def test_metadata_filtering():
    res = client.post("/retrieve", json={
        "question": "cost",
        "source_filter": "sample.md",
        "section_filter": "Cost"
    }).json()
    
    # All returned chunks must have correct metadata
    for chunk in res["chunks"]:
        assert chunk["source"] == "sample.md"
        assert chunk["section"] == "Cost"

# --- End-to-End Pipeline & Telemetry ---
def test_end_to_end_pipeline():
    path = "data/raw/sample_e2e.md"
    with open(path, "w") as f:
        f.write("# E2E\nThis is an end-to-end test.")
        
    ingest = client.post("/ingest", json={"path": path}).json()
    assert ingest["chunks_created"] > 0
    
    chunk = client.post("/chunk", json={"path": path}).json()
    assert chunk["chunks_created"] > 0
    
    index = client.post("/index", json={"path": path}).json()
    assert index["chunks_indexed"] > 0
    
    retrieve = client.post("/retrieve", json={"question": "end-to-end"}).json()
    assert retrieve["retrieval_time_ms"] < 1000
    
    sources = [c["source"] for c in retrieve["chunks"]]
    assert "sample_e2e.md" in sources

    # Telemetry Log verification
    with open("logs/app.log", "r") as f:
        logs = f.read()
        assert "Ingest request received" in logs
        assert "Chunk request received" in logs
        assert "Index request received" in logs
        assert "Retrieve request received" in logs

    os.remove(path)

# --- Failure Cases ---
def test_failure_cases():
    r_miss = client.post("/ingest", json={"path": "data/raw/missing.md"})
    assert r_miss.status_code == 400
    
    r_empty = client.post("/ingest", json={"path": "data/raw/empty.md"})
    assert r_empty.status_code == 400
    
    r_unsupp = client.post("/ingest", json={"path": "data/raw/unsupported.txt"})
    assert r_unsupp.status_code == 400

    # Retrieve on empty DB (simulate using a filter that doesn't exist)
    r_empty_db = client.post("/retrieve", json={"question": "empty", "source_filter": "nonexistent.md"})
    assert r_empty_db.status_code == 200
    assert r_empty_db.json()["confidence"] == 0.0
    assert len(r_empty_db.json()["chunks"]) == 0
