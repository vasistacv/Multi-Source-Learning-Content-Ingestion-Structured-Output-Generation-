import sys
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from src.nlp.nlp_engine import NLPEngine
from src.vectorstore.rag_system import RAGSystem
from src.config.settings import settings

def check_rag():
    print("Initializing RAG System check...")
    
    # Init NLP Engine (CPU is fine for check)
    nlp_engine = NLPEngine(device="cpu")
    
    # Init RAG
    rag = RAGSystem(nlp_engine)
    
    # Check if index exists and has items
    if rag.vector_store.index.ntotal > 0:
        print(f"\nSUCCESS: Vector Database contains {rag.vector_store.index.ntotal} vectors.")
        
        query = "What is deep learning?"
        print(f"\nTest Query: '{query}'")
        
        results = rag.search(query, k=3)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, res in enumerate(results, 1):
                # Print clean ASCII text
                content_preview = res['content'][:100].replace('\n', ' ')
                print(f"{i}. Score: {res['score']:.4f} | Source: {os.path.basename(res['metadata']['file_path'])}")
                print(f"   Preview: {content_preview}...")
        else:
            print("\nWARNING: Database has vectors but returned no results (threshold might be too high).")
            
    else:
        print("\nPENDING: Vector Database is currently empty. Ingestion is still in progress.")

if __name__ == "__main__":
    check_rag()
