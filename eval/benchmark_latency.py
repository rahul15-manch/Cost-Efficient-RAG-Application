import argparse
import json
import time
import os
import numpy as np
from tqdm import tqdm
from app.config import settings
from app.lance_db import embed_texts
from app import vector_store

def load_queries():
    path = "eval/questions.json"
    if not os.path.exists(path):
        return [
            "What is Cost-Efficient RAG?",
            "How does FAISS compare to LanceDB?",
            "What are the fallback mechanisms?",
            "How do I deploy this application?",
            "What embedding model is used?"
        ]
    with open(path, "r") as f:
        data = json.load(f)
        return [item["question"] for item in data]

def format_table(results):
    print("\n=== Retrieval Latency Benchmark ===\n")
    
    first_key = list(results.keys())[0]
    print(f"Corpus vectors: {results[first_key]['vector_count']}")
    print(f"Embedding model: {results[first_key]['embedding_model']}")
    print(f"Queries: {results[first_key]['queries']}")
    print(f"Runs/query: {results[first_key]['runs_per_query']}\n")
    
    print(f"{'Backend':<10} {'Avg(ms)':>8}  {'Median':>6}  {'P95':>5}   {'Min':>4}   {'Max':>5}   {'Std':>4}")
    print("-" * 65)
    for backend in ["lancedb", "faiss"]:
        if backend in results:
            mets = results[backend]
            name = "LanceDB" if backend == "lancedb" else "FAISS"
            print(f"{name:<10} {mets['avg_ms']:>8.1f}  {mets['median_ms']:>6.1f}  {mets['p95_ms']:>5.1f}  {mets['min_ms']:>4.1f}  {mets['max_ms']:>5.1f}  {mets['std_ms']:>4.1f}")
        
    print()
    if "lancedb" in results and "faiss" in results:
        l_avg = results["lancedb"]["avg_ms"]
        f_avg = results["faiss"]["avg_ms"]
        if f_avg > 0:
            speedup = l_avg / f_avg
            print(f"FAISS is {speedup:.2f}x faster than LanceDB for retrieval-only latency.\n")

    print("Recommendation:")
    print("- LanceDB: metadata-aware retrieval and filtering")
    print("- FAISS: ultra-low-latency ANN retrieval")

def generate_markdown(results):
    md = "# Retrieval Latency Benchmark\n\n"
    
    first_key = list(results.keys())[0]
    md += f"**Corpus vectors:** {results[first_key]['vector_count']}\n"
    md += f"**Embedding model:** {results[first_key]['embedding_model']}\n"
    md += f"**Queries:** {results[first_key]['queries']}\n"
    md += f"**Runs/query:** {results[first_key]['runs_per_query']}\n\n"
    
    md += "| Backend | Avg(ms) | Median | P95 | Min | Max | Std |\n"
    md += "|---------|---------|--------|-----|-----|-----|-----|\n"
    for backend in ["lancedb", "faiss"]:
        if backend in results:
            mets = results[backend]
            name = "LanceDB" if backend == "lancedb" else "FAISS"
            md += f"| {name} | {mets['avg_ms']:.1f} | {mets['median_ms']:.1f} | {mets['p95_ms']:.1f} | {mets['min_ms']:.1f} | {mets['max_ms']:.1f} | {mets['std_ms']:.1f} |\n"
            
    if "lancedb" in results and "faiss" in results:
        l_avg = results["lancedb"]["avg_ms"]
        f_avg = results["faiss"]["avg_ms"]
        if f_avg > 0:
            speedup = l_avg / f_avg
            md += f"\n**Conclusion:** FAISS is {speedup:.2f}x faster than LanceDB for retrieval-only latency.\n"
    
    with open("eval/latency_report.md", "w") as f:
        f.write(md)

def run_benchmark(backend_name: str, queries: list[str], warmup: int, runs: int, top_k: int):
    settings.VECTOR_BACKEND = backend_name
    
    print(f"Embedding {len(queries)} queries for {backend_name}...")
    embedded_queries = embed_texts(queries)
    
    latencies = []
    
    for i, emb in enumerate(tqdm(embedded_queries, desc=f"Benchmarking {backend_name}")):
        # Warm-up
        for _ in range(warmup):
            vector_store.search(emb, top_k)
            
        # Benchmark
        for _ in range(runs):
            start = time.perf_counter()
            vector_store.search(emb, top_k)
            lat = (time.perf_counter() - start) * 1000
            latencies.append(lat)
            
    latencies = np.array(latencies)
    return {
        "backend_name": backend_name,
        "queries": len(queries),
        "runs_per_query": runs,
        "vector_count": vector_store.count_vectors(),
        "embedding_model": settings.EMBED_MODEL,
        "avg_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies))
    }

def main():
    parser = argparse.ArgumentParser(description="Latency Benchmark for LanceDB and FAISS")
    parser.add_argument("--backend", type=str, choices=["lancedb", "faiss", "both"], default="both")
    args = parser.parse_args()
    
    queries = load_queries()
    warmup = 5
    runs = 50
    top_k = 5
    
    backends_to_run = ["lancedb", "faiss"] if args.backend == "both" else [args.backend]
    
    results = {}
    
    for be in backends_to_run:
        results[be] = run_benchmark(be, queries, warmup, runs, top_k)
        
    with open("eval/latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    generate_markdown(results)
    format_table(results)

if __name__ == "__main__":
    main()
