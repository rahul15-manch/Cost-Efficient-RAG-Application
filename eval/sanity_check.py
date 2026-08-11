import os
import sys
import json
from app.config import is_groq_configured
from app.lance_db import table_exists, count_records, existing_chunk_ids

def main():
    print("Running Sanity Check...")
    errors = 0
    
    if not is_groq_configured():
        print("[FAIL] GROQ_API_KEY is not configured or invalid.")
        errors += 1
    else:
        print("[OK] GROQ_API_KEY is configured.")
        
    if not table_exists():
        print("[FAIL] LanceDB database does not exist.")
        errors += 1
    else:
        cnt = count_records()
        if cnt == 0:
            print("[FAIL] LanceDB is empty (vector count = 0).")
            errors += 1
        else:
            print(f"[OK] LanceDB exists and has {cnt} vectors.")
            
    if not os.path.exists("eval/questions.json"):
        print("[FAIL] eval/questions.json not found.")
        errors += 1
    else:
        print("[OK] eval/questions.json exists.")
        
    if not os.path.exists("eval/relevance.json"):
        print("[FAIL] eval/relevance.json not found.")
        errors += 1
    else:
        print("[OK] eval/relevance.json exists.")
        
    if errors == 0:
        with open("eval/relevance.json", "r") as f:
            rel = json.load(f)
        idx = existing_chunk_ids()
        missing = 0
        for q, ids in rel.items():
            for i in ids:
                if i not in idx:
                    print(f"[FAIL] Relevance chunk_id '{i}' not found in index.")
                    missing += 1
        if missing > 0:
            errors += missing
        else:
            print("[OK] All relevance chunk_ids exist in the index.")
            
    if errors > 0:
        print(f"\nSanity check failed with {errors} errors.")
        sys.exit(1)
    else:
        print("\nSanity check passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
