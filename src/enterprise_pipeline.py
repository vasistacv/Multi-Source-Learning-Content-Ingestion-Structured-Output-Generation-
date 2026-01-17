from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
import torch
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

from .processors.base import ProcessedContent
from .processors.pdf_processor import PDFProcessor
from .processors.video_processor import VideoProcessor
from .processors.document_processor import DocumentProcessor
from .nlp.nlp_engine import NLPEngine
from .generators.artifact_generator import FlashcardGenerator, QuizGenerator
from .generators.knowledge_graph import KnowledgeGraphGenerator
from .vectorstore.rag_system import RAGSystem
from .config.settings import settings

# Advanced components
try:
    from .llm.llm_engine import LocalLLM, HybridRetriever, LLMPoweredGenerator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("Advanced LLM components not available. Using base generators.")

try:
    from .training.fine_tuning import FineTuningPipeline
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

try:
    from .evaluation.metrics import ComprehensiveEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False


class EnterpriseContentPipeline:
    """
    Enterprise-grade content ingestion pipeline that competes with
    Amazon, Microsoft, and Google-level systems.
    
    Features:
    - Multi-modal content processing (PDF, Video, Documents)
    - Local LLM integration (Mistral, Llama, Phi)
    - Hybrid retrieval (Dense + Cross-Encoder Re-ranking)
    - Fine-tuning on ingested content
    - Comprehensive evaluation metrics
    - Distributed processing support
    - MLOps integration (MLflow)
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_local_llm: bool = True,
        llm_model: str = "mistral-7b",
        enable_training: bool = True,
        enable_evaluation: bool = True
    ):
        self.device = settings.GPU_DEVICE if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info("=" * 60)
        logger.info("INITIALIZING ENTERPRISE CONTENT PIPELINE")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info("=" * 60)
        
        # Core NLP Engine
        logger.info("Loading NLP Engine...")
        self.nlp_engine = NLPEngine(device=self.device)
        
        # Content Processors
        logger.info("Initializing Content Processors...")
        self.pdf_processor = PDFProcessor(use_ocr=True)
        self.video_processor = VideoProcessor(device=self.device)
        self.document_processor = DocumentProcessor()
        
        # Base Generators
        logger.info("Initializing Base Artifact Generators...")
        self.flashcard_generator = FlashcardGenerator(self.nlp_engine)
        self.quiz_generator = QuizGenerator(self.nlp_engine)
        self.knowledge_graph_generator = KnowledgeGraphGenerator(self.nlp_engine)
        
        # RAG System
        logger.info("Initializing RAG System...")
        self.rag_system = RAGSystem(self.nlp_engine)
        
        # Advanced LLM (if available)
        self.local_llm = None
        self.llm_generator = None
        self.hybrid_retriever = None
        
        if use_local_llm and LLM_AVAILABLE:
            try:
                logger.info(f"Loading Local LLM: {llm_model}...")
                self.local_llm = LocalLLM(model_name=llm_model, device=self.device)
                self.llm_generator = LLMPoweredGenerator(self.local_llm)
                self.hybrid_retriever = HybridRetriever()
                logger.info("Local LLM loaded successfully!")
            except Exception as e:
                logger.warning(f"Failed to load Local LLM: {e}")
                logger.warning("Falling back to base generators.")
        
        # Fine-tuning Pipeline
        self.fine_tuning = None
        if enable_training and TRAINING_AVAILABLE:
            try:
                logger.info("Initializing Fine-Tuning Pipeline...")
                self.fine_tuning = FineTuningPipeline(use_mlflow=True)
            except Exception as e:
                logger.warning(f"Fine-tuning unavailable: {e}")
        
        # Evaluation Suite
        self.evaluator = None
        if enable_evaluation and EVALUATION_AVAILABLE:
            try:
                logger.info("Initializing Evaluation Suite...")
                self.evaluator = ComprehensiveEvaluator()
            except Exception as e:
                logger.warning(f"Evaluation suite unavailable: {e}")
        
        # Processor mapping
        self.processor_map = {
            '.pdf': self.pdf_processor,
            '.docx': self.document_processor,
            '.pptx': self.document_processor,
            '.xlsx': self.document_processor,
            '.txt': self.document_processor,
            '.md': self.document_processor,
            '.mp4': self.video_processor,
            '.avi': self.video_processor,
            '.mov': self.video_processor,
            '.mkv': self.video_processor,
        }
        
        logger.info("Pipeline initialized successfully!")
    
    async def process_file(
        self,
        file_path: Path,
        generate_artifacts: bool = True,
        use_llm: bool = True,
        ingest_to_rag: bool = True,
        run_evaluation: bool = True
    ) -> Dict[str, Any]:
        """Process a single file with enterprise features."""
        
        logger.info(f"Processing: {file_path}")
        start_time = datetime.now()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.processor_map:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Step 1: Content Extraction
        processor = self.processor_map[suffix]
        processed_content = await processor.process(file_path)
        
        # Step 2: NLP Analysis
        logger.info("Running NLP analysis...")
        entities = self.nlp_engine.extract_entities(processed_content.raw_text)
        concepts = self.nlp_engine.extract_key_concepts(processed_content.raw_text)
        topics = self.nlp_engine.extract_topics(processed_content.raw_text)
        
        processed_content.entities = entities
        processed_content.key_concepts = [c['concept'] for c in concepts]
        processed_content.topics = topics
        
        # Step 3: Generate Embeddings
        logger.info("Generating embeddings...")
        embeddings = self.nlp_engine.generate_embeddings([processed_content.raw_text])
        processed_content.embeddings = embeddings[0].tolist()
        
        # Build result
        result = {
            'file_path': str(file_path),
            'file_type': suffix,
            'processed_at': datetime.now().isoformat(),
            'metadata': processed_content.metadata.__dict__,
            'topics': topics,
            'key_concepts': [c['concept'] for c in concepts[:20]],
            'entity_count': len(entities),
            'word_count': processed_content.metadata.word_count,
        }
        
        # Step 4: Generate Summary
        logger.info("Generating summary...")
        if use_llm and self.llm_generator:
            result['summary'] = self.llm_generator.generate_summary(
                processed_content.raw_text,
                style="comprehensive"
            )
        else:
            result['summary'] = self.nlp_engine.generate_summary(
                processed_content.raw_text,
                max_length=settings.SUMMARY_MAX_LENGTH
            )
        
        # Step 5: Generate Artifacts
        if generate_artifacts:
            logger.info("Generating learning artifacts...")
            
            # Flashcards
            if use_llm and self.llm_generator:
                flashcards = self.llm_generator.generate_flashcards(
                    processed_content.raw_text,
                    num_cards=25
                )
                result['flashcards'] = flashcards
            else:
                flashcards = self.flashcard_generator.generate(
                    processed_content.raw_text,
                    num_cards=20
                )
                result['flashcards'] = [
                    {
                        'question': fc.question,
                        'answer': fc.answer,
                        'topic': fc.topic,
                        'difficulty': fc.difficulty,
                        'confidence_score': fc.confidence_score
                    }
                    for fc in flashcards
                ]
            
            # Quiz Questions
            if use_llm and self.llm_generator:
                quiz = self.llm_generator.generate_quiz(
                    processed_content.raw_text,
                    num_questions=15
                )
                result['quiz_questions'] = quiz
            else:
                quiz = self.quiz_generator.generate(
                    processed_content.raw_text,
                    num_questions=10
                )
                result['quiz_questions'] = [
                    {
                        'question': q.question,
                        'options': q.options,
                        'correct_answer': q.correct_answer,
                        'explanation': q.explanation,
                        'difficulty': q.difficulty,
                        'type': q.question_type
                    }
                    for q in quiz
                ]
            
            # Knowledge Graph
            logger.info("Building knowledge graph...")
            knowledge_graph = self.knowledge_graph_generator.generate(
                processed_content.raw_text,
                max_nodes=40
            )
            
            graph_output_dir = settings.OUTPUT_DIR / "knowledge_graphs" / file_path.stem
            graph_output_dir.mkdir(parents=True, exist_ok=True)
            
            graph_data = self.knowledge_graph_generator.export_to_json(
                knowledge_graph,
                graph_output_dir / "graph.json"
            )
            
            self.knowledge_graph_generator.visualize_matplotlib(
                knowledge_graph,
                graph_output_dir / "graph_visualization.png"
            )
            
            self.knowledge_graph_generator.visualize_plotly(
                knowledge_graph,
                graph_output_dir / "graph_interactive.html"
            )
            
            learning_paths = self.knowledge_graph_generator.extract_learning_paths(
                knowledge_graph,
                max_paths=5
            )
            
            graph_stats = self.knowledge_graph_generator.get_graph_statistics(knowledge_graph)
            
            result['knowledge_graph'] = {
                'statistics': graph_stats,
                'learning_paths': learning_paths,
                'output_directory': str(graph_output_dir),
                'data': graph_data
            }
        
        # Step 6: Ingest to RAG
        if ingest_to_rag:
            logger.info("Ingesting to RAG system...")
            document_id = self.rag_system.ingest_document(
                content=processed_content.raw_text,
                metadata={
                    'file_path': str(file_path),
                    'file_type': suffix,
                    'topics': topics,
                    'processed_at': datetime.now().isoformat()
                }
            )
            result['rag_document_id'] = document_id
            
            # Also index for hybrid retrieval
            if self.hybrid_retriever:
                chunks = self.rag_system._chunk_text(processed_content.raw_text)
                self.hybrid_retriever.index(chunks)
        
        # Step 7: Run Evaluation
        if run_evaluation and self.evaluator:
            logger.info("Running quality evaluation...")
            evaluation_results = self.evaluator.run_full_evaluation(
                result,
                processed_content.raw_text
            )
            result['evaluation'] = evaluation_results
            
            # Save evaluation report
            eval_path = settings.OUTPUT_DIR / "evaluations" / f"{file_path.stem}_eval.json"
            self.evaluator.save_report(evaluation_results, eval_path)
        
        # Timing
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time_seconds'] = processing_time
        
        logger.info(f"File processed in {processing_time:.2f} seconds")
        
        return result
    
    async def process_batch(
        self,
        file_paths: List[Path],
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        """Process multiple files with concurrency control."""
        
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    return await self.process_file(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    return {
                        'file_path': str(file_path),
                        'error': str(e),
                        'status': 'failed'
                    }
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        
        # Aggregate statistics
        success_count = sum(1 for r in results if 'error' not in r)
        logger.info(f"Batch complete: {success_count}/{len(results)} successful")
        
        return results
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """Query the knowledge base with optional LLM-powered answering."""
        
        logger.info(f"Query: {query}")
        
        # Hybrid retrieval if available
        if self.hybrid_retriever:
            retrieval_results = self.hybrid_retriever.retrieve(
                query,
                initial_k=50,
                final_k=top_k
            )
            
            context = "\n\n".join([r['document'] for r in retrieval_results])
            
            if use_llm and self.llm_generator:
                answer = self.llm_generator.answer_question(query, context)
            else:
                answer = self.nlp_engine.generate_summary(
                    f"Question: {query}\nContext: {context}",
                    max_length=300
                )
            
            return {
                'query': query,
                'answer': answer,
                'sources': retrieval_results,
                'method': 'hybrid_llm' if use_llm else 'hybrid_extractive'
            }
        
        else:
            # Fallback to basic RAG
            return self.rag_system.retrieve_and_generate(
                query=query,
                top_k=top_k
            )
    
    def train_on_content(
        self,
        processed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fine-tune models on processed content."""
        
        if not self.fine_tuning:
            logger.warning("Fine-tuning not available")
            return {"error": "Fine-tuning not initialized"}
        
        logger.info("Preparing training data from processed content...")
        
        # Generate QA pairs from flashcards
        contexts, questions, answers = self.fine_tuning.generate_training_data(
            processed_results
        )
        
        if len(questions) < 10:
            logger.warning("Not enough training data. Need at least 10 QA pairs.")
            return {"error": "Insufficient training data"}
        
        logger.info(f"Training QA model on {len(questions)} samples...")
        
        result = self.fine_tuning.train_qa_model(
            contexts=contexts,
            questions=questions,
            answers=answers,
            epochs=3,
            batch_size=4
        )
        
        return result
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ):
        """Save processing results."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
        # Save CSV exports
        if 'flashcards' in results:
            self._save_flashcards_csv(results['flashcards'], output_path)
        
        if 'quiz_questions' in results:
            self._save_quiz_csv(results['quiz_questions'], output_path)
    
    def _save_flashcards_csv(self, flashcards: List[Dict], base_path: Path):
        import csv
        
        csv_path = base_path.parent / f"{base_path.stem}_flashcards.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if not flashcards:
                return
            
            fieldnames = flashcards[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flashcards)
        
        logger.info(f"Flashcards CSV saved to {csv_path}")
    
    def _save_quiz_csv(self, quiz_questions: List[Dict], base_path: Path):
        import csv
        
        csv_path = base_path.parent / f"{base_path.stem}_quiz.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if not quiz_questions:
                return
            
            writer = csv.writer(f)
            writer.writerow(['question', 'options', 'correct_answer', 'explanation', 'difficulty', 'type'])
            
            for q in quiz_questions:
                options = q.get('options', [])
                if isinstance(options, list):
                    options = ' | '.join(str(o) for o in options)
                
                writer.writerow([
                    q.get('question', ''),
                    options,
                    q.get('correct_answer', q.get('correct', '')),
                    q.get('explanation', ''),
                    q.get('difficulty', ''),
                    q.get('type', '')
                ])
        
        logger.info(f"Quiz CSV saved to {csv_path}")
