import os
import shutil
import pytest
from fastapi.testclient import TestClient

# Imported after conftest.py overrides
from app.main import app

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    # Setup test directories
    os.makedirs("tests/test_lancedb", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Text samples
    shutil.copy("tests/sample.md", "data/raw/sample.md")
    shutil.copy("tests/sample.html", "data/raw/sample.html")
    
    # PDF generation programmatically
    import fitz
    pdf_path = "data/raw/sample.pdf"
    if not os.path.exists(pdf_path):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "PDF Content Page 1. This contains some text about RAG and retrieval.")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "PDF Content Page 2. This contains evaluation metrics and costs.")
        doc.save(pdf_path)
        doc.close()
    
    # Empty & Unsupported
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

    # Teardown
    if os.path.exists(seen_path):
        os.remove(seen_path)
    if os.path.exists("tests/test_lancedb"):
        shutil.rmtree("tests/test_lancedb")
    for f in ["sample.md", "sample.html", "sample.pdf", "empty.md", "unsupported.txt"]:
        path = os.path.join("data/raw", f)
        if os.path.exists(path):
            os.remove(path)

# --- 1. Foundation (Milestone 1) ---
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "trust-aware-rag"}
    assert os.path.exists("logs/app.log")

# --- 2. Structured Ingestion (Milestone 2) ---
def test_ingest_markdown():
    res = client.post("/ingest", json={"path": "data/raw/sample.md"}).json()
    assert res["file_type"] == "markdown"
    assert res["sections_detected"] > 0

def test_ingest_html():
    res = client.post("/ingest", json={"path": "data/raw/sample.html"}).json()
    assert res["file_type"] == "html"
    assert res["sections_detected"] > 0

def test_ingest_pdf():
    res = client.post("/ingest", json={"path": "data/raw/sample.pdf"}).json()
    assert res["file_type"] == "pdf"
    assert res["pages_detected"] == 2

def test_ingest_stable_hash():
    r1 = client.post("/ingest", json={"path": "data/raw/sample.md"}).json()
    r2 = client.post("/ingest", json={"path": "data/raw/sample.md"}).json()
    assert r1["file_hash"] == r2["file_hash"]

# --- 3. Smart Chunking & Idempotency (Milestone 3) ---
def test_chunking_idempotency():
    # Delete seen chunks to ensure fresh run
    seen_path = "data/processed/seen_chunks.json"
    if os.path.exists(seen_path):
        os.remove(seen_path)
        
    r1 = client.post("/chunk", json={"path": "data/raw/sample.md"}).json()
    assert r1["chunks_created"] > 0
    assert r1["duplicates_skipped"] == 0
    
    r2 = client.post("/chunk", json={"path": "data/raw/sample.md"}).json()
    assert r2["chunks_created"] == 0
    assert r2["duplicates_skipped"] == r1["total_after_dedup"]
    assert r2["total_after_dedup"] == r1["total_after_dedup"]

# --- 4. Embeddings & LanceDB (Milestone 4) ---
def test_indexing_persistence():
    r1 = client.post("/index", json={"path": "data/raw/sample.md"}).json()
    assert r1["chunks_indexed"] > 0
    assert r1["embedding_dim"] > 0
    assert os.path.exists("tests/test_lancedb/documents.lance")
    
    r2 = client.post("/index", json={"path": "data/raw/sample.md"}).json()
    assert r2["chunks_indexed"] == 0
    assert r2["chunks_skipped"] == r1["chunks_indexed"]
    assert r2["total_vectors"] == r1["total_vectors"]

# --- 5. Adaptive Retrieval (Milestone 5) ---
def test_adaptive_retrieval_k():
    client.post("/index", json={"path": "data/raw/sample.html"})
    
    res1 = client.post("/retrieve", json={"question": "Recall?"}).json()
    assert res1["top_k_used"] == 2
    
    res2 = client.post("/retrieve", json={"question": "What is the purpose of this RAG project?"}).json()
    assert res2["top_k_used"] == 5
    
    res3 = client.post("/retrieve", json={"question": "Summarize all sections"}).json()
    assert res3["top_k_used"] == 8

def test_retrieval_metrics():
    res = client.post("/retrieve", json={"question": "cost"}).json()
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["evidence_coverage"] <= 1.0
    scores = [c["score"] for c in res["chunks"]]
    for i in range(len(scores)-1):
        assert scores[i] >= scores[i+1]

def test_metadata_filtering():
    res = client.post("/retrieve", json={
        "question": "cost",
        "source_filter": "sample.md",
        "section_filter": "Cost"
    }).json()
    for chunk in res["chunks"]:
        assert chunk["source"] == "sample.md"
        assert chunk["section"] == "Cost"

# --- 6. Grounded Generation (Milestone 6) ---
def test_grounded_generation():
    res = client.post("/answer", json={"question": "What measures retrieval quality?"}).json()
    if res["answer"] != "No relevant context found.":
        assert "Recall@k" in res["answer"]
        assert len(res["citations"]) > 0
    
def test_no_hallucination_generation():
    res = client.post("/answer", json={"question": "How do you build a spaceship?"}).json()
    assert res["answer"] == "No relevant context found."
    assert res["token_usage"]["total_tokens"] == 0

# --- 7. Trust Layer (Milestone 7) ---
def test_trust_layer():
    res = client.post("/trust", json={"question": "What measures retrieval quality?"}).json()
    assert "heatmap" in res
    if res["heatmap"]:
        assert res["heatmap"][0]["strength"] in ["strong", "medium", "weak"]
    assert "low_confidence" in res["failure_analysis"]

def test_trust_layer_failure_analysis():
    res = client.post("/trust", json={"question": "How do you build a spaceship?"}).json()
    assert res["failure_analysis"]["low_confidence"] == True
    assert len(res["failure_analysis"]["suggested_queries"]) > 0
    
    # Check uniqueness of reformulations
    suggested = res["failure_analysis"]["suggested_queries"]
    assert len(suggested) == len(set(suggested))
    assert all(len(s) > 0 for s in suggested)

# --- 8. End-to-End Pipeline & 9. Idempotent Pipeline ---
def test_e2e_and_idempotency():
    path = "data/raw/e2e.md"
    with open(path, "w") as f:
        f.write("# E2E\nE2E Pipeline Test.")
        
    client.post("/ingest", json={"path": path})
    client.post("/chunk", json={"path": path})
    idx1 = client.post("/index", json={"path": path}).json()
    
    trust1 = client.post("/trust", json={"question": "E2E Pipeline Test"}).json()
    assert trust1["retrieval_time_ms"] > 0
    assert trust1["total_time_ms"] >= (trust1["retrieval_time_ms"] + trust1["generation_time_ms"])
    
    # Idempotency
    idx2 = client.post("/index", json={"path": path}).json()
    assert idx2["chunks_indexed"] == 0
    assert idx2["total_vectors"] == idx1["total_vectors"]
    
    os.remove(path)

# --- 10. Telemetry ---
def test_telemetry_logs():
    with open("logs/app.log", "r") as f:
        logs = f.read()
        assert "Ingest request received" in logs
        assert "Chunk request received" in logs
        assert "Index request received" in logs
        assert "Retrieve request received" in logs
        assert "Answer request received" in logs
        assert "Trust request received" in logs

# --- 11. Failure Cases ---
def test_failure_cases():
    assert client.post("/ingest", json={"path": "data/raw/missing.md"}).status_code == 400
    assert client.post("/ingest", json={"path": "data/raw/empty.md"}).status_code == 400
    assert client.post("/ingest", json={"path": "data/raw/unsupported.txt"}).status_code == 400
    
    # Empty DB retrieval simulation (using fake filter)
    res = client.post("/retrieve", json={"question": "empty", "source_filter": "nonexistent.md"}).json()
    assert len(res["chunks"]) == 0
    assert res["confidence"] == 0.0
    assert res["evidence_coverage"] == 0.0
    
    res2 = client.post("/answer", json={"question": "empty", "source_filter": "nonexistent.md"}).json()
    assert res2["answer"] == "No relevant context found."

# --- 12. Performance Sanity ---
def test_performance_sanity():
    res = client.post("/trust", json={"question": "Recall?"}).json()
    assert res["retrieval_time_ms"] < 1000
    assert res["total_time_ms"] < 5000
