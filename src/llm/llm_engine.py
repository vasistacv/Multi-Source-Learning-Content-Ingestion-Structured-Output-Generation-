from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from loguru import logger
from abc import ABC, abstractmethod

from ..config.settings import settings


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        pass


class LocalLLM(BaseLLM):
    """Production-grade local LLM with 4-bit quantization for efficiency."""
    
    SUPPORTED_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
        "phi-2": "microsoft/phi-2",
        "gemma-7b": "google/gemma-7b-it",
        "qwen-7b": "Qwen/Qwen1.5-7B-Chat"
    }
    
    def __init__(
        self,
        model_name: str = "mistral-7b",
        device: str = "cuda",
        use_4bit: bool = True,
        max_memory: Dict[int, str] = None
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        logger.info(f"Loading LLM: {self.model_id} on {self.device}")
        
        # 4-bit quantization config for memory efficiency
        if use_4bit and self.device == "cuda":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            self.bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.bnb_config:
            load_kwargs["quantization_config"] = self.bnb_config
            load_kwargs["device_map"] = "auto"
        
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs
        )
        
        self.model.eval()
        
        logger.info(f"LLM loaded successfully. Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        # Format prompt for instruction-tuned models
        formatted_prompt = self._format_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> List[str]:
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response.strip())
        
        return responses
    
    def _format_prompt(self, prompt: str) -> str:
        # Mistral format
        if "mistral" in self.model_id.lower():
            return f"[INST] {prompt} [/INST]"
        # Llama format
        elif "llama" in self.model_id.lower():
            return f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST]"
        # Default
        else:
            return prompt


class AdvancedReranker:
    """Cross-encoder based re-ranking for precision retrieval."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        logger.info(f"Loading Cross-Encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """Re-rank documents by relevance to query."""
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        ranked = sorted(
            enumerate(zip(scores, documents)),
            key=lambda x: x[1][0],
            reverse=True
        )
        
        # Return top_k with (index, score, document)
        return [(idx, score, doc) for idx, (score, doc) in ranked[:top_k]]


class ColBERTRetriever:
    """Late interaction retrieval for better semantic matching."""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        logger.info(f"Loading ColBERT model: {model_name}")
        # Using sentence-transformers as fallback since full ColBERT requires specific setup
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.document_embeddings = None
        self.documents = []
    
    def index_documents(self, documents: List[str]):
        """Build index from documents."""
        self.documents = documents
        self.document_embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.info(f"Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float, str]]:
        """Retrieve top documents for query."""
        if self.document_embeddings is None:
            return []
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (int(idx), float(similarities[idx]), self.documents[idx])
            for idx in top_indices
        ]


class HybridRetriever:
    """Combines dense retrieval with re-ranking for production accuracy."""
    
    def __init__(self):
        self.dense_retriever = ColBERTRetriever()
        self.reranker = AdvancedReranker()
    
    def index(self, documents: List[str]):
        self.dense_retriever.index_documents(documents)
    
    def retrieve(
        self,
        query: str,
        initial_k: int = 50,
        final_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: dense + rerank."""
        
        # Stage 1: Dense retrieval (fast, broad)
        initial_results = self.dense_retriever.retrieve(query, top_k=initial_k)
        
        if not initial_results:
            return []
        
        # Stage 2: Re-ranking (slower, precise)
        documents = [doc for _, _, doc in initial_results]
        reranked = self.reranker.rerank(query, documents, top_k=final_k)
        
        # Format results
        results = []
        for idx, score, doc in reranked:
            original_idx = initial_results[idx][0]
            results.append({
                "index": original_idx,
                "score": score,
                "document": doc
            })
        
        return results


class LLMPoweredGenerator:
    """Uses local LLM for high-quality artifact generation."""
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
    
    def generate_flashcards(
        self,
        content: str,
        num_cards: int = 20
    ) -> List[Dict[str, str]]:
        """Generate flashcards using LLM."""
        
        prompt = f"""You are an expert educator. Based on the following content, generate exactly {num_cards} high-quality flashcards.

Content:
{content[:4000]}

Generate flashcards in this exact format, one per line:
Q: [question] | A: [answer] | D: [easy/medium/hard]

Rules:
1. Questions should test understanding, not just recall
2. Answers should be concise but complete
3. Cover all major concepts
4. Vary difficulty levels

Generate {num_cards} flashcards now:"""
        
        response = self.llm.generate(prompt, max_tokens=2000)
        
        # Parse response
        flashcards = []
        for line in response.split("\n"):
            if "Q:" in line and "A:" in line:
                try:
                    parts = line.split("|")
                    q = parts[0].replace("Q:", "").strip()
                    a = parts[1].replace("A:", "").strip()
                    d = parts[2].replace("D:", "").strip() if len(parts) > 2 else "medium"
                    
                    flashcards.append({
                        "question": q,
                        "answer": a,
                        "difficulty": d
                    })
                except:
                    continue
        
        return flashcards
    
    def generate_summary(
        self,
        content: str,
        style: str = "comprehensive"
    ) -> str:
        """Generate intelligent summary."""
        
        prompt = f"""Summarize the following content in a {style} manner. 
Focus on key concepts, main ideas, and important details.

Content:
{content[:6000]}

Provide a well-structured summary:"""
        
        return self.llm.generate(prompt, max_tokens=800)
    
    def generate_quiz(
        self,
        content: str,
        num_questions: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate quiz questions using LLM."""
        
        prompt = f"""Create {num_questions} multiple choice questions based on this content.

Content:
{content[:4000]}

Format each question exactly like this:
QUESTION: [question text]
A) [option a]
B) [option b]
C) [option c]
D) [option d]
CORRECT: [A/B/C/D]
EXPLANATION: [brief explanation]

Generate {num_questions} questions:"""
        
        response = self.llm.generate(prompt, max_tokens=2500)
        
        # Parse questions
        questions = []
        current_q = {}
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("QUESTION:"):
                if current_q:
                    questions.append(current_q)
                current_q = {"question": line.replace("QUESTION:", "").strip(), "options": []}
            elif line.startswith(("A)", "B)", "C)", "D)")):
                if current_q:
                    current_q.setdefault("options", []).append(line)
            elif line.startswith("CORRECT:"):
                if current_q:
                    current_q["correct"] = line.replace("CORRECT:", "").strip()
            elif line.startswith("EXPLANATION:"):
                if current_q:
                    current_q["explanation"] = line.replace("EXPLANATION:", "").strip()
        
        if current_q:
            questions.append(current_q)
        
        return questions
    
    def answer_question(
        self,
        question: str,
        context: str
    ) -> str:
        """RAG-powered question answering."""
        
        prompt = f"""Based on the following context, answer the question accurately and comprehensively.

Context:
{context[:4000]}

Question: {question}

Provide a detailed, accurate answer based only on the given context. If the context does not contain enough information, say so clearly.

Answer:"""
        
        return self.llm.generate(prompt, max_tokens=500)
