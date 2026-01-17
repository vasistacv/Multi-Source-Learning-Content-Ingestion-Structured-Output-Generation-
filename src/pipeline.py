from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
from loguru import logger

from .processors.base import ProcessedContent
from .processors.pdf_processor import PDFProcessor
from .processors.video_processor import VideoProcessor
from .processors.document_processor import DocumentProcessor
from .nlp.nlp_engine import NLPEngine
from .generators.artifact_generator import FlashcardGenerator, QuizGenerator
from .generators.knowledge_graph import KnowledgeGraphGenerator
from .vectorstore.rag_system import RAGSystem
from .config.settings import settings


class ContentIngestionPipeline:
    def __init__(self, use_gpu: bool = True):
        device = settings.GPU_DEVICE if use_gpu else "cpu"
        
        logger.info(f"Initializing Content Ingestion Pipeline on {device}")
        
        self.nlp_engine = NLPEngine(device=device)
        
        self.pdf_processor = PDFProcessor(use_ocr=True)
        self.video_processor = VideoProcessor(device=device)
        self.document_processor = DocumentProcessor()
        
        self.flashcard_generator = FlashcardGenerator(self.nlp_engine)
        self.quiz_generator = QuizGenerator(self.nlp_engine)
        self.knowledge_graph_generator = KnowledgeGraphGenerator(self.nlp_engine)
        
        self.rag_system = RAGSystem(self.nlp_engine)
        
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
        
        logger.info("Pipeline initialized successfully")
    
    async def process_file(
        self,
        file_path: Path,
        generate_artifacts: bool = True,
        ingest_to_rag: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path}")
        
        start_time = datetime.now()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.processor_map:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        processor = self.processor_map[suffix]
        processed_content = await processor.process(file_path)
        
        logger.info("Extracting entities and concepts...")
        entities = self.nlp_engine.extract_entities(processed_content.raw_text)
        concepts = self.nlp_engine.extract_key_concepts(processed_content.raw_text)
        topics = self.nlp_engine.extract_topics(processed_content.raw_text)
        
        processed_content.entities = entities
        processed_content.key_concepts = [c['concept'] for c in concepts]
        processed_content.topics = topics
        
        logger.info("Generating embeddings...")
        embeddings = self.nlp_engine.generate_embeddings([processed_content.raw_text])
        processed_content.embeddings = embeddings[0].tolist()
        
        logger.info("Generating summary...")
        summary = self.nlp_engine.generate_summary(
            processed_content.raw_text,
            max_length=settings.SUMMARY_MAX_LENGTH
        )
        
        result = {
            'file_path': str(file_path),
            'file_type': suffix,
            'processed_at': datetime.now().isoformat(),
            'metadata': processed_content.metadata.__dict__,
            'summary': summary,
            'topics': topics,
            'key_concepts': [c['concept'] for c in concepts[:15]],
            'entity_count': len(entities),
            'word_count': processed_content.metadata.word_count,
        }
        
        if generate_artifacts:
            logger.info("Generating learning artifacts...")
            
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
            
            quiz_questions = self.quiz_generator.generate(
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
                for q in quiz_questions
            ]
            
            knowledge_graph = self.knowledge_graph_generator.generate(
                processed_content.raw_text,
                max_nodes=30
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
                'output_directory': str(graph_output_dir)
            }
        
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
        
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time_seconds'] = processing_time
        
        logger.info(f"File processed successfully in {processing_time:.2f} seconds")
        
        return result
    
    async def process_batch(
        self,
        file_paths: List[Path],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        logger.info(f"Processing batch of {len(file_paths)} files")
        
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
        
        return results
    
    def query_content(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        logger.info(f"Querying: {query}")
        
        return self.rag_system.retrieve_and_generate(
            query=query,
            top_k=top_k
        )
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        
        if 'flashcards' in results:
            flashcards_csv_path = output_path.parent / f"{output_path.stem}_flashcards.csv"
            self._save_flashcards_csv(results['flashcards'], flashcards_csv_path)
        
        if 'quiz_questions' in results:
            quiz_csv_path = output_path.parent / f"{output_path.stem}_quiz.csv"
            self._save_quiz_csv(results['quiz_questions'], quiz_csv_path)
    
    def _save_flashcards_csv(self, flashcards: List[Dict], output_path: Path):
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not flashcards:
                return
            
            fieldnames = flashcards[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(flashcards)
        
        logger.info(f"Flashcards saved to {output_path}")
    
    def _save_quiz_csv(self, quiz_questions: List[Dict], output_path: Path):
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not quiz_questions:
                return
            
            fieldnames = ['question', 'option_a', 'option_b', 'option_c', 'option_d', 
                         'correct_answer', 'explanation', 'difficulty', 'type']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for q in quiz_questions:
                row = {
                    'question': q['question'],
                    'correct_answer': q['correct_answer'],
                    'explanation': q['explanation'],
                    'difficulty': q['difficulty'],
                    'type': q['type']
                }
                
                options = q.get('options', [])
                for i, option in enumerate(options[:4]):
                    row[f'option_{chr(97+i)}'] = option
                
                writer.writerow(row)
        
        logger.info(f"Quiz questions saved to {output_path}")
