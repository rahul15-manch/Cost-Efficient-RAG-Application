# Trust-Aware Cost-Efficient RAG System Notes

## Evaluation Metrics
To evaluate retrieval quality in our RAG system, we use several key metrics: Recall@k, Hit Rate, MRR, nDCG@k, and context precision. These allow us to rigorously measure both the presence of the correct context and its ranking.

## Cost Efficiency with LanceDB
The system is highly cost-efficient because LanceDB reduces infrastructure cost. It is an embedded vector database that integrates directly into the application space, removing the need for a separate database server.

## Confidence and Reliability
The purpose of the confidence score in retrieval is crucial. It helps the system measure how strong the evidence is and skip generation when trust is low. This prevents the LLM from hallucinating answers when relevant context is missing.

## Ingestion and Deduplication
During document ingestion, idempotent deduplication is used. It hashes chunks to generate deterministic IDs, preventing the same content from being indexed multiple times.

## Trust Layer
The main components of the trust layer include evidence heatmaps, failure analysis, and query reformulation. These ensure the system is observable and that users can trust the outputs.
