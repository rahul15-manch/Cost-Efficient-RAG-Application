import os
import sys
import glob

from app.config import settings
from app.ingest import load_document
from app.chunker import chunk_document
from app.lance_db import index_chunks

def rebuild_index():
    raw_dir = "data/raw"
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        sys.exit(1)
        
    supported_extensions = (".pdf", ".html", ".md")
    files_to_process = []
    
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                files_to_process.append(os.path.join(root, file))
                
    if not files_to_process:
        print(f"No supported documents (.pdf, .html, .md) found in {raw_dir}.")
        sys.exit(1)
        
    total_docs = len(files_to_process)
    all_chunks = []
    
    print(f"Found {total_docs} documents in {raw_dir}. Starting processing...")
    
    for file_path in files_to_process:
        try:
            file_type, docs = load_document(file_path)
            for doc in docs:
                all_chunks.extend(chunk_document(doc, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if not all_chunks:
        print("No chunks were created from the documents.")
        sys.exit(1)
        
    print(f"\nChunking complete. Attempting to index {len(all_chunks)} chunks using {settings.EMBED_MODEL}...")
    
    try:
        result = index_chunks(all_chunks)
    except Exception as e:
        print(f"Error during indexing: {e}")
        sys.exit(1)
        
    print("\n=== Rebuild Summary ===")
    print(f"Documents processed: {total_docs}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Vectors indexed: {result['indexed']} (Skipped {result['skipped']} duplicates)")
    print(f"Total vectors in DB: {result['total']}")
    print(f"Database path: {settings.LANCEDB_PATH}")
    print(f"Embedding Model: {result['embedding_model']}")

if __name__ == "__main__":
    rebuild_index()
