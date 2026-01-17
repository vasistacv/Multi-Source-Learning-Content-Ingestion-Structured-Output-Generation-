import sys
import os
from dotenv import load_dotenv

# Set encoding to utf-8 for stdout/stderr to avoid cp1252 errors
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

from src.pipeline import ContentIngestionPipeline

def debug_query():
    print("Initializing Pipeline...")
    try:
        pipeline = ContentIngestionPipeline(use_gpu=True)
        
        query = "What are the key concepts of Deep Learning?"
        print(f"\nQuerying: {query}")
        
        result = pipeline.query_content(query, top_k=3)
        
        print("\n--- RESULT ---")
        print(f"Answer: {result.get('answer', 'No answer generated')}")
        print("\n--- SOURCES ---")
        for s in result.get('sources', []):
            print(f"- {s['source']} ({s['relevance_score']:.2f})")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_query()
