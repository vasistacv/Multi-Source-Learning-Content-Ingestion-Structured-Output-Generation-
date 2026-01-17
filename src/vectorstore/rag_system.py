from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
from loguru import logger

from .vector_store import BaseVectorStore, VectorStoreFactory
from ..nlp.nlp_engine import NLPEngine
from ..config.settings import settings


@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str


class RAGSystem:
    def __init__(
        self,
        nlp_engine: NLPEngine,
        vector_store: Optional[BaseVectorStore] = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.nlp_engine = nlp_engine
        self.vector_store = vector_store or VectorStoreFactory.create()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        logger.info("RAG System initialized")
    
    def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        document_id = document_id or str(uuid.uuid4())
        
        logger.info(f"Ingesting document: {document_id}")
        
        chunks = self._chunk_text(content)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        embeddings = self.nlp_engine.generate_embeddings(chunks)
        
        chunk_metadata = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **metadata,
                'document_id': document_id,
                'chunk_index': i,
                'chunk_count': len(chunks)
            }
            chunk_metadata.append(chunk_meta)
            chunk_ids.append(f"{document_id}_chunk_{i}")
        
        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadata=chunk_metadata,
            ids=chunk_ids
        )
        
        logger.info(f"Document {document_id} ingested successfully")
        
        return document_id
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        query_embedding = self.nlp_engine.generate_embeddings([query])[0]
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2
        )
        
        if filters:
            results = self._apply_filters(results, filters)
        
        results = results[:top_k]
        
        retrieval_results = []
        for result in results:
            relevance_score = 1.0 - result['distance']
            
            retrieval_results.append(
                RetrievalResult(
                    content=result['document'],
                    metadata=result['metadata'],
                    relevance_score=relevance_score,
                    source=result['metadata'].get('file_path', 'unknown')
                )
            )
        
        logger.info(f"Retrieved {len(retrieval_results)} relevant documents")
        
        return retrieval_results
    
    def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 3,
        max_length: int = 300
    ) -> Dict[str, Any]:
        retrieval_results = self.retrieve(query, top_k=top_k)
        
        context = "\n\n".join([r.content for r in retrieval_results])
        
        augmented_query = f"Based on the following context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        summary = self.nlp_engine.generate_summary(
            augmented_query,
            max_length=max_length
        )
        
        return {
            'query': query,
            'answer': summary,
            'context': context,
            'sources': [
                {
                    'content': r.content[:200] + "...",
                    'source': r.source,
                    'relevance_score': r.relevance_score
                }
                for r in retrieval_results
            ]
        }
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[RetrievalResult]:
        results = self.retrieve(query, top_k=top_k)
        
        filtered_results = [
            r for r in results
            if r.relevance_score >= threshold
        ]
        
        return filtered_results
    
    def find_similar_content(
        self,
        content: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        embedding = self.nlp_engine.generate_embeddings([content])[0]
        
        results = self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k + 1
        )
        
        results = results[1:]
        
        retrieval_results = []
        for result in results:
            relevance_score = 1.0 - result['distance']
            
            retrieval_results.append(
                RetrievalResult(
                    content=result['document'],
                    metadata=result['metadata'],
                    relevance_score=relevance_score,
                    source=result['metadata'].get('file_path', 'unknown')
                )
            )
        
        return retrieval_results
    
    def _chunk_text(self, text: str) -> List[str]:
        doc = self.nlp_engine.nlp(text)
        
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_size = len(s.split())
                    if overlap_size + s_size < self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        filtered_results = []
        
        for result in results:
            metadata = result['metadata']
            
            match = True
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def delete_document(self, document_id: str):
        logger.info(f"Deleting document: {document_id}")
        
        results = self.vector_store.search(
            query_embedding=self.nlp_engine.generate_embeddings([""])[0],
            top_k=10000
        )
        
        ids_to_delete = [
            r['id'] for r in results
            if r['metadata'].get('document_id') == document_id
        ]
        
        if ids_to_delete:
            self.vector_store.delete(ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
        else:
            logger.warning(f"No chunks found for document {document_id}")
