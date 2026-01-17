from typing import List, Dict, Any, Optional
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import re

from ..config.settings import settings

# Robust import handling for spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception as e:
    SPACY_AVAILABLE = False
    logger.warning(f"spaCy not available (crashed on import): {e}. Using regex-based fallback for NLP tasks.")

class NLPEngine:
    def __init__(
        self,
        device: str = "cuda",
        embedding_model: str = None,
        summarization_model: str = "facebook/bart-large-cnn"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing NLP Engine on {self.device}")
        
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        
        if settings.GROQ_API_KEY:
            logger.info("Groq API detected - Skipping local summarization model to save memory.")
            try:
                from ..llm.llm_engine import GroqLLM
                self.groq_llm = GroqLLM(model=settings.LLM_MODEL)
            except Exception as e:
                logger.warning(f"Failed to init Groq in NLP Engine: {e}")
            self.summarizer = None
        else:
            try:
                logger.info(f"Loading summarization model: {summarization_model}")
                self.summarizer = pipeline(
                    "summarization",
                    model=summarization_model,
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                logger.warning(f"Failed to load summarizer (likely OOM): {e}. Local summarization disabled.")
                self.summarizer = None
        
        logger.info("Initializing NLP model...")
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"spaCy model load failed: {e}. Fallback enabled.")
                self.nlp = None
        
        if self.nlp:
            self.nlp.max_length = 2000000
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if self.nlp:
            doc = self.nlp(text[:1000000])
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        else:
            # Simple regex fallback for entities (Capitalized words)
            for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                entities.append({
                    'text': match.group(0),
                    'label': 'ENTITY',
                    'start': match.start(),
                    'end': match.end()
                })
        return entities
    
    def extract_key_concepts(
        self,
        text: str,
        num_concepts: int = 20,
        method: str = "tfidf"
    ) -> List[Dict[str, float]]:
        # TF-IDF doesn't require spaCy
        return self._extract_concepts_tfidf(text, num_concepts)
    
    def _extract_concepts_tfidf(self, text: str, num_concepts: int) -> List[Dict[str, float]]:
        # Simple sentence splitter if spaCy missing
        if self.nlp:
            doc = self.nlp(text[:1000000])
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text)
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=num_concepts,
                stop_words='english',
                ngram_range=(1, 3)
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            concept_scores = [
                {'concept': feature_names[i], 'score': float(scores[i])}
                for i in range(len(feature_names))
            ]
            
            concept_scores.sort(key=lambda x: x['score'], reverse=True)
            return concept_scores[:num_concepts]
        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return []
    
    def extract_topics(
        self,
        text: str,
        num_topics: int = 5
    ) -> List[str]:
        concepts = self.extract_key_concepts(text, num_concepts=num_topics * 2)
        
        topics = []
        for concept_dict in concepts[:num_topics]:
            concept = concept_dict['concept']
            if len(concept.split()) <= 3:
                topics.append(concept.title())
        
        return topics
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 500,
        min_length: int = 100
    ) -> str:
        # Use Groq if available
        if hasattr(self, 'groq_llm'):
             try:
                 return self.groq_llm.generate(f"Summarize the following text:\n{text}")
             except Exception as e:
                 logger.error(f"Groq summary failed: {e}")
                 return text[:500] + "..."

        if len(text) < min_length:
            return text
        
        try:
            if not self.summarizer:
                 return text[:500] + "..."

            chunks = self._chunk_text(text, max_chunk_size=1024)
            
            summaries = []
            # Summarize only first few chunks to save time
            for chunk in chunks[:3]:
                result = self.summarizer(
                    chunk,
                    max_length=max_length // 3,
                    min_length=min(min_length // 3, 30),
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            
            combined_summary = " ".join(summaries)
            return combined_summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:max_length] + "..."
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
             sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_text = sent.strip()
            if not sent_text: continue
            
            sent_size = len(sent_text.split())
            
            if current_size + sent_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_size = sent_size
            else:
                current_chunk.append(sent_text)
                current_size += sent_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def build_concept_graph(
        self,
        text: str,
        top_n_concepts: int = 30
    ) -> nx.Graph:
        concepts = self.extract_key_concepts(text, num_concepts=top_n_concepts)
        concept_texts = [c['concept'] for c in concepts]
        
        G = nx.Graph()
        
        for concept_dict in concepts:
            concept = concept_dict['concept']
            score = concept_dict['score']
            G.add_node(concept, weight=score)
        
        if self.nlp:
            doc = self.nlp(text[:50000])
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text[:50000])
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            concepts_in_sentence = [c for c in concept_texts if c.lower() in sent_lower]
            
            for i, c1 in enumerate(concepts_in_sentence):
                for c2 in concepts_in_sentence[i+1:]:
                    if G.has_edge(c1, c2):
                        G[c1][c2]['weight'] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)
        
        return G

    def extract_learning_hierarchy(self, text: str, max_depth: int = 3) -> Dict[str, Any]:
        """Extract hierarchical learning structure"""
        topics = self.extract_topics(text, num_topics=5)
        concepts = self.extract_key_concepts(text, num_concepts=20)
        
        hierarchy = {'root': 'Learning Content', 'topics': []}
        
        for topic in topics:
            node = {'name': topic, 'concepts': []}
            for c in concepts:
                if c['concept'] in topic.lower() or topic.lower() in c['concept']:
                    node['concepts'].append({'name': c['concept'], 'score': c['score']})
            hierarchy['topics'].append(node)
            
        return hierarchy
