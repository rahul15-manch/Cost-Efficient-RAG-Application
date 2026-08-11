import json
import os
import sys
from typing import Dict, List

from app.retriever import retrieve
from app.lance_db import existing_chunk_ids, count_records, table_exists

def rebuild_relevance() -> None:
    print("Rebuilding provisional relevance labels...")
    
    if not table_exists() or count_records() == 0:
        print("ERROR: LanceDB database is empty or does not exist.")
        sys.exit(1)
        
    questions_path = "eval/questions.json"
    relevance_path = "eval/relevance.json"
    
    if not os.path.exists(questions_path):
        print(f"ERROR: {questions_path} not found.")
        sys.exit(1)
        
    with open(questions_path, "r") as f:
        questions = json.load(f)
        
    indexed_ids = existing_chunk_ids()
    relevance: Dict[str, List[str]] = {}
    
    total_chunks_written = 0
    missing_ids = 0
    
    for q in questions:
        qid = q["id"]
        res = retrieve(query=q["question"], top_k=None, source_filter=None, section_filter=None)
        
        if not res.chunks:
            print(f"WARNING: Question '{qid}' retrieved no chunks.")
            relevance[qid] = []
            continue
            
        unique_ids: List[str] = []
        for chunk in res.chunks:
            if chunk.chunk_id not in unique_ids:
                if chunk.chunk_id in indexed_ids:
                    unique_ids.append(chunk.chunk_id)
                else:
                    missing_ids += 1
            if len(unique_ids) == 3:
                break
                
        relevance[qid] = unique_ids
        total_chunks_written += len(unique_ids)
        
    with open(relevance_path, "w") as f:
        json.dump(relevance, f, indent=2)
        
    print(f"\nRebuilt {relevance_path}")
    print(f"Questions processed: {len(questions)}")
    print(f"Chunk IDs written: {total_chunks_written}")
    print(f"Missing IDs: {missing_ids}")
    
    # Validation step
    print("\nValidating written file...")
    with open(relevance_path, "r") as f:
        written_relevance = json.load(f)
        
    validation_failures = 0
    for qid, chunk_ids in written_relevance.items():
        for cid in chunk_ids:
            if cid not in indexed_ids:
                print(f"VALIDATION ERROR: Chunk ID {cid} in {relevance_path} not found in index.")
                validation_failures += 1
                
    if validation_failures == 0:
        print("Validation successful: All stored chunk IDs exist in the current LanceDB index.")
    else:
        print(f"Validation failed with {validation_failures} errors.")
        sys.exit(1)

if __name__ == "__main__":
    rebuild_relevance()
