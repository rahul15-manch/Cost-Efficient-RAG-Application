# Retrieval Latency Benchmark

**Corpus vectors:** 9
**Embedding model:** BAAI/bge-small-en-v1.5
**Queries:** 5
**Runs/query:** 50

| Backend | Avg(ms) | Median | P95 | Min | Max | Std |
|---------|---------|--------|-----|-----|-----|-----|
| LanceDB | 1.9 | 1.9 | 2.0 | 1.8 | 2.2 | 0.1 |
| FAISS | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 | 0.0 |

**Conclusion:** FAISS is 23.68x faster than LanceDB for retrieval-only latency.
