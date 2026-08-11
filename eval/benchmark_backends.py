import time
import json
import numpy as np
from app.config import settings
from app.retriever import retrieve
from eval.evaluate import load_data

def run_backend(questions, backend_name):
    settings.VECTOR_BACKEND = backend_name
    latencies = []
    
    for q in questions:
        start = time.perf_counter()
        retrieve(query=q["question"], top_k=5, source_filter=None, section_filter=None)
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)
        
    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95))
    }

def main():
    questions, _ = load_data()
    
    print("Benchmarking LanceDB...")
    lancedb_res = run_backend(questions, "lancedb")
    
    print("Benchmarking FAISS...")
    # Make sure FAISS index is built for testing
    from app import vector_store
    from app.ingest import load_document
    from app.chunker import chunk_document
    settings.VECTOR_BACKEND = "faiss"
    if vector_store.count_vectors() == 0:
        print("Building FAISS index for benchmark...")
        import os
        chunks = []
        for f in os.listdir("data/raw"):
            if f.endswith((".md", ".html", ".pdf")):
                path = os.path.join("data/raw", f)
                _, docs = load_document(path)
                for d in docs:
                    chunks.extend(chunk_document(d, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
        vector_store.index_chunks(chunks)
    
    faiss_res = run_backend(questions, "faiss")
    
    results = {
        "lancedb": lancedb_res,
        "faiss": faiss_res
    }
    
    with open("eval/backend_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== Backend Benchmark Results ===")
    print(f"LanceDB Avg Latency: {lancedb_res['avg_latency_ms']:.2f} ms")
    print(f"LanceDB P95 Latency: {lancedb_res['p95_latency_ms']:.2f} ms")
    print(f"FAISS Avg Latency: {faiss_res['avg_latency_ms']:.2f} ms")
    print(f"FAISS P95 Latency: {faiss_res['p95_latency_ms']:.2f} ms")

if __name__ == "__main__":
    main()
