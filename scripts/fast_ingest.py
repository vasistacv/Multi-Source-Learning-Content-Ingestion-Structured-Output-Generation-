import asyncio
import sys
import os
import gc
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

from src.pipeline import ContentIngestionPipeline
from src.config.settings import settings

async def fast_ingest():
    print("STARTING ROBUST SEQUENTIAL INGESTION (Checking every file)")
    
    # 1. CLEAN START (Delete old corrupt/wrong-dim DB)
    db_path = settings.OUTPUT_DIR / "vector_store"
    if db_path.exists():
        print(f"Removing old/corrupt DB at {db_path} to ensure correct dimensions...")
        shutil.rmtree(db_path)
    
    # 2. Initialize Pipeline (Creates NEW 384-dim DB)
    pipeline = ContentIngestionPipeline(use_gpu=True)
    
    # 3. Scan Files (SKIP VIDEO)
    data_path = Path("d:/Navgurukul/data/enterprise_dataset")
    valid_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.md'}
    
    files_to_process = [
        f for f in data_path.glob("**/*") 
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]
    files_to_process = [f for f in files_to_process if "csv" not in f.suffix and "json" not in f.suffix]
    
    print(f"Found {len(files_to_process)} text/document files.")
    
    # 4. Process Sequentially
    success_count = 0
    
    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] Processing {file_path.name}...")
        try:
            # Result path
            output_path = settings.OUTPUT_DIR / "production_runs" / "fast_run" / f"{file_path.stem}_results.json"
            
            # Check if likely already processed? (Skip for speed if we crash/resume?)
            # No, we must ingest to Re-Index into vector DB.
            
            # Process
            result = await pipeline.process_file(
                file_path, 
                generate_artifacts=True, 
                ingest_to_rag=True
            )
            
            # Save Results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            def json_serial(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return str(obj)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=json_serial)
            
            success_count += 1
            
            # Save Vector DB periodically
            if i % 3 == 0:
                print("Saving intermediate DB...")
                pipeline.rag_system.vector_store.save(db_path)
                
            # Cleanup & Sleep (Rate Limit Protection)
            del result
            gc.collect()
            print("Sleeping 10s for Rate Limit...")
            time.sleep(10)
            
        except Exception as e:
            print(f"FAILED {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Final Save
    print("Saving Final Vector Database...")
    try:
        pipeline.rag_system.vector_store.save(db_path)
        print(f"Vector Database saved to {db_path}")
    except Exception as e:
        print(f"Failed to save DB: {e}")
        
    print(f"DONE! Processed {success_count}/{len(files_to_process)} files.")
    
    # Verify
    print("Verifying RAG...")
    if pipeline.rag_system.vector_store.index.ntotal > 0:
        print(f"Index contains {pipeline.rag_system.vector_store.index.ntotal} vectors.")
    else:
        print("Index is EMPTY!")

if __name__ == "__main__":
    asyncio.run(fast_ingest())
