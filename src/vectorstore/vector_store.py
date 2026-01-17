from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    
import faiss
import numpy as np
from pathlib import Path
import pickle
from loguru import logger

from ..config.settings import settings


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ):
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        pass


class ChromaDBStore(BaseVectorStore):
    def __init__(self, collection_name: str = "learning_content", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(settings.DATA_DIR / "chromadb")
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ):
        logger.info(f"Adding {len(documents)} documents to ChromaDB")
        
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadata,
            ids=ids
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count
        }


class FAISSStore(BaseVectorStore):
    def __init__(self, dimension: int = 384, index_type: str = "FlatL2"):
        self.dimension = dimension
        self.index_type = index_type
        
        logger.info(f"Initializing FAISS index: {index_type}")
        
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.documents = []
        self.metadata = []
        self.ids = []
        self.is_trained = False
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ):
        logger.info(f"Adding {len(documents)} documents to FAISS")
        
        embeddings = np.array(embeddings).astype('float32')
        
        if self.index_type == "IVFFlat" and not self.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.is_trained = True
        
        self.index.add(embeddings)
        
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        self.ids.extend(ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_embedding = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'id': self.ids[idx],
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def delete(self, ids: List[str]):
        indices_to_remove = [i for i, doc_id in enumerate(self.ids) if doc_id in ids]
        
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.metadata[idx]
            del self.ids[idx]
        
        logger.warning("FAISS index rebuilt after deletion")
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'ids': self.ids,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"FAISS index saved to {path}")
    
    def load(self, path: Path):
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.ids = data['ids']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        logger.info(f"FAISS index loaded from {path}")


class VectorStoreFactory:
    @staticmethod
    def create(store_type: str = None, **kwargs) -> BaseVectorStore:
        store_type = store_type or settings.VECTOR_DB_TYPE
        
        if store_type == "chromadb":
            if not CHROMADB_AVAILABLE:
                logger.warning("ChromaDB not available, falling back to FAISS")
                return FAISSStore(**kwargs)
            return ChromaDBStore(**kwargs)
        elif store_type == "faiss":
            return FAISSStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

