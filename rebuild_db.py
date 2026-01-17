import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.vectorstore.rag_system import RAGSystem
from src.nlp.nlp_engine import NLPEngine
from src.config.settings import settings

def rebuild_index():
    print("Rebuilding Vector Database from Processed Artifacts...")
    
    # 1. Initialize RAG (384-dim for MiniLM)
    nlp_engine = NLPEngine(device="cpu")
    rag = RAGSystem(nlp_engine)
    
    # Ensure correct dimension if fresh
    if hasattr(rag.vector_store, 'dimension') and rag.vector_store.dimension != 384:
        print(f"Warning: Store dimension is {rag.vector_store.dimension}, expected 384. Resetting...")
        # In a real quick script, just rely on fresh init which defaults to 384 now (I fixed the code)
    
    # 2. Find all Result JSONs
    results_dir = settings.OUTPUT_DIR / "production_runs"
    json_files = list(results_dir.glob("**/*_results.json"))
    
    print(f"Found {len(json_files)} processed artifacts.")
    
    count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Text and Metadata
            # The 'raw_text' might not be in the result JSON to save space? 
            # Let's check keys. Usually result has 'summary', 'topics', but maybe not full content?
            # If full content is missing, we re-read the SOURCE file using metadata['file_path'].
            
            file_path = data.get('file_path')
            if file_path:
                source_path = Path(file_path)
                if source_path.exists():
                    # Read content
                    # Simple text read for now (robust enough for index)
                    try:
                        content = source_path.read_text(encoding='utf-8', errors='ignore')
                    except:
                        continue # Binary file or issue
                    
                    if not content.strip():
                        continue

                    # Ingest
                    rag.ingest_document(
                        content=content,
                        metadata={
                            'file_path': str(source_path),
                            'source': source_path.name
                        }
                    )
                    count += 1
                    print(f"   Indexed: {source_path.name}")
        except Exception as e:
            print(f"   Skipped {json_file.name}: {e}")

    # 3. SAVE CORRECTLY
    db_path = settings.OUTPUT_DIR / "vector_store"
    # Ensure dir exists (My previous fix handled this, but let's be explicit in script too)
    db_path.mkdir(parents=True, exist_ok=True)
    
    rag.vector_store.save(db_path)
    print(f"\nIndex Rebuilt! {count} documents added.")
    print(f"Saved to {db_path}")

if __name__ == "__main__":
    rebuild_index()
