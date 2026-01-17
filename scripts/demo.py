import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.pipeline import ContentIngestionPipeline
from src.config.settings import settings


console = Console()


async def main():
    console.print(Panel.fit(
        "[bold cyan]Multi-Source Learning Content Ingestion - Demo[/bold cyan]\n"
        "[dim]Enterprise-Grade ML Pipeline Demonstration[/dim]",
        border_style="cyan"
    ))
    
    console.print("\n[bold]Initializing pipeline...[/bold]")
    pipeline = ContentIngestionPipeline(use_gpu=True)
    
    console.print("\n[bold green]Example 1: Processing a PDF Document[/bold green]")
    console.print("[dim]This will extract text, generate flashcards, create a knowledge graph, and ingest into RAG[/dim]\n")
    
    console.print("[yellow]Note: Place your test files in the 'sample_data' directory[/yellow]\n")
    
    sample_data_dir = Path("sample_data")
    sample_data_dir.mkdir(exist_ok=True)
    
    pdf_files = list(sample_data_dir.glob("*.pdf"))
    
    if pdf_files:
        sample_file = pdf_files[0]
        console.print(f"Processing: [cyan]{sample_file}[/cyan]\n")
        
        result = await pipeline.process_file(
            file_path=sample_file,
            generate_artifacts=True,
            ingest_to_rag=True
        )
        
        console.print("[bold green]Processing Complete![/bold green]\n")
        console.print(f"Summary: {result.get('summary', 'N/A')[:200]}...\n")
        console.print(f"Topics: {', '.join(result.get('topics', [])[:5])}\n")
        console.print(f"Flashcards generated: {len(result.get('flashcards', []))}")
        console.print(f"Quiz questions generated: {len(result.get('quiz_questions', []))}\n")
        
        output_file = settings.OUTPUT_DIR / f"{sample_file.stem}_results.json"
        pipeline.save_results(result, output_file)
        console.print(f"Results saved to: [cyan]{output_file}[/cyan]\n")
        
        console.print("\n[bold green]Example 2: Querying the Knowledge Base[/bold green]")
        console.print("[dim]Using RAG to answer questions about the ingested content[/dim]\n")
        
        query = "What are the main concepts discussed?"
        console.print(f"Query: [cyan]{query}[/cyan]\n")
        
        answer_result = pipeline.query_content(query, top_k=3)
        console.print(Panel(
            answer_result['answer'],
            title="[bold]Answer[/bold]",
            border_style="green"
        ))
        
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(answer_result['sources'][:3], 1):
            console.print(f"{i}. Relevance: {source['relevance_score']:.2%}")
            console.print(f"   {source['content'][:150]}...\n")
    
    else:
        console.print("[yellow]No PDF files found in sample_data directory.[/yellow]")
        console.print("[yellow]Please add some PDF files and run again.[/yellow]\n")
        
        console.print("[bold]Creating sample flashcards from text...[/bold]\n")
        
        sample_text = """
        Machine learning is a subset of artificial intelligence that focuses on the development 
        of algorithms that can learn from and make predictions or decisions based on data. 
        Deep learning is a type of machine learning that uses neural networks with multiple layers. 
        Natural language processing refers to the branch of AI that gives computers the ability 
        to understand text and spoken words in much the same way human beings can.
        """
        
        from src.nlp.nlp_engine import NLPEngine
        from src.generators.artifact_generator import FlashcardGenerator, QuizGenerator
        
        nlp_engine = NLPEngine(device="cpu")
        flashcard_gen = FlashcardGenerator(nlp_engine)
        quiz_gen = QuizGenerator(nlp_engine)
        
        flashcards = flashcard_gen.generate(sample_text, num_cards=5)
        console.print(f"Generated {len(flashcards)} flashcards:\n")
        
        for i, fc in enumerate(flashcards[:3], 1):
            console.print(f"[bold]{i}. Question:[/bold] {fc.question}")
            console.print(f"   [green]Answer:[/green] {fc.answer}")
            console.print(f"   [dim]Difficulty: {fc.difficulty} | Confidence: {fc.confidence_score:.2%}[/dim]\n")
        
        quiz_questions = quiz_gen.generate(sample_text, num_questions=3)
        console.print(f"Generated {len(quiz_questions)} quiz questions:\n")
        
        for i, q in enumerate(quiz_questions[:2], 1):
            console.print(f"[bold]{i}. {q.question}[/bold]")
            for j, option in enumerate(q.options, 1):
                console.print(f"   {j}. {option}")
            console.print(f"   [green]Correct: {q.correct_answer}[/green]\n")
    
    console.print("\n[bold cyan]Demo Complete![/bold cyan]")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Add more files to sample_data directory")
    console.print("2. Run: [cyan]python -m src.cli batch --directory sample_data[/cyan]")
    console.print("3. Start API: [cyan]python -m src.api[/cyan]")
    console.print("4. Query: [cyan]python -m src.cli query --query 'your question'[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
