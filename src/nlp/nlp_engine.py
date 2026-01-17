from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from ..config.settings import settings


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
        
        logger.info(f"Loading summarization model: {summarization_model}")
        self.summarizer = pipeline(
            "summarization",
            model=summarization_model,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("Loading spaCy model")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
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
        doc = self.nlp(text[:1000000])
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_key_concepts(
        self,
        text: str,
        num_concepts: int = 20,
        method: str = "tfidf"
    ) -> List[Dict[str, float]]:
        if method == "tfidf":
            return self._extract_concepts_tfidf(text, num_concepts)
        elif method == "noun_chunks":
            return self._extract_concepts_noun_chunks(text, num_concepts)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_concepts_tfidf(self, text: str, num_concepts: int) -> List[Dict[str, float]]:
        doc = self.nlp(text[:1000000])
        
        sentences = [sent.text for sent in doc.sents]
        
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
    
    def _extract_concepts_noun_chunks(self, text: str, num_concepts: int) -> List[Dict[str, float]]:
        doc = self.nlp(text[:1000000])
        
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        chunk_counts = Counter(noun_chunks)
        
        total = sum(chunk_counts.values())
        concepts = [
            {'concept': chunk, 'score': count / total}
            for chunk, count in chunk_counts.most_common(num_concepts)
        ]
        
        return concepts
    
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
        if len(text) < min_length:
            return text
        
        try:
            chunks = self._chunk_text(text, max_chunk_size=1024)
            
            summaries = []
            for chunk in chunks[:5]:
                result = self.summarizer(
                    chunk,
                    max_length=max_length // len(chunks[:5]),
                    min_length=min(min_length // len(chunks[:5]), 30),
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            
            combined_summary = " ".join(summaries)
            
            if len(combined_summary) > max_length:
                result = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]['summary_text']
            
            return combined_summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:max_length] + "..."
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            sent_text = sent.text
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
        doc = self.nlp(text[:1000000])
        
        concepts = self.extract_key_concepts(text, num_concepts=top_n_concepts)
        concept_texts = [c['concept'] for c in concepts]
        
        G = nx.Graph()
        
        for concept_dict in concepts:
            concept = concept_dict['concept']
            score = concept_dict['score']
            G.add_node(concept, weight=score)
        
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences[:500]:
            sent_lower = sentence.lower()
            concepts_in_sentence = [c for c in concept_texts if c.lower() in sent_lower]
            
            for i, c1 in enumerate(concepts_in_sentence):
                for c2 in concepts_in_sentence[i+1:]:
                    if G.has_edge(c1, c2):
                        G[c1][c2]['weight'] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)
        
        return G
    
    def extract_learning_hierarchy(
        self,
        text: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        topics = self.extract_topics(text, num_topics=5)
        concepts = self.extract_key_concepts(text, num_concepts=20)
        
        hierarchy = {
            'root': 'Learning Content',
            'topics': []
        }
        
        for topic in topics:
            topic_node = {
                'name': topic,
                'concepts': []
            }
            
            for concept_dict in concepts[:10]:
                concept = concept_dict['concept']
                if any(word in concept.lower() for word in topic.lower().split()):
                    topic_node['concepts'].append({
                        'name': concept,
                        'score': concept_dict['score']
                    })
            
            if topic_node['concepts']:
                hierarchy['topics'].append(topic_node)
        
        return hierarchy
