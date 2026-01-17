import sys
import asyncio
from pathlib import Path
import argparse
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from .pipeline import ContentIngestionPipeline
from .config.settings import settings


console = Console()


def setup_logging(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    log_file = settings.LOGS_DIR / "pipeline.log"
    logger.add(
        str(log_file),
        rotation="10 MB",
        retention="7 days",
        level=log_level
    )


async def process_single_file(
    pipeline: ContentIngestionPipeline,
    file_path: Path,
    output_dir: Path
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {file_path.name}...", total=None)
        
        result = await pipeline.process_file(
            file_path=file_path,
            generate_artifacts=True,
            ingest_to_rag=True
        )
        
        progress.update(task, completed=True)
    
    output_path = output_dir / f"{file_path.stem}_results.json"
    pipeline.save_results(result, output_path)
    
    display_results(result)


async def process_batch_files(
    pipeline: ContentIngestionPipeline,
    file_paths: List[Path],
    output_dir: Path
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {len(file_paths)} files...", total=None)
        
        results = await pipeline.process_batch(file_paths, max_concurrent=3)
        
        progress.update(task, completed=True)
    
    for result in results:
        if 'error' not in result:
            output_path = output_dir / f"{Path(result['file_path']).stem}_results.json"
            pipeline.save_results(result, output_path)
    
    display_batch_results(results)


def query_content(pipeline: ContentIngestionPipeline, query: str, top_k: int):
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        
        result = pipeline.query_content(query, top_k=top_k)
        
        progress.update(task, completed=True)
    
    display_query_results(result)


def display_results(result: dict):
    console.print("\n[bold green]Processing Complete![/bold green]\n")
    
    info_table = Table(title="Document Information")
    info_table.add_column("Attribute", style="cyan")
    info_table.add_column("Value", style="magenta")
    
    info_table.add_row("File", result.get('file_path', 'N/A'))
    info_table.add_row("Type", result.get('file_type', 'N/A'))
    info_table.add_row("Word Count", str(result.get('word_count', 0)))
    info_table.add_row("Processing Time", f"{result.get('processing_time_seconds', 0):.2f}s")
    
    console.print(info_table)
    
    if result.get('summary'):
        console.print(Panel(result['summary'], title="[bold]Summary[/bold]", border_style="blue"))
    
    if result.get('topics'):
        console.print(f"\n[bold]Topics:[/bold] {', '.join(result['topics'])}")
    
    if result.get('flashcards'):
        console.print(f"\n[bold]Generated Artifacts:[/bold]")
        console.print(f"  - Flashcards: {len(result['flashcards'])}")
    
    if result.get('quiz_questions'):
        console.print(f"  - Quiz Questions: {len(result['quiz_questions'])}")
    
    if result.get('knowledge_graph'):
        kg_stats = result['knowledge_graph'].get('statistics', {})
        console.print(f"  - Knowledge Graph Nodes: {kg_stats.get('node_count', 0)}")
        console.print(f"  - Knowledge Graph Edges: {kg_stats.get('edge_count', 0)}")


def display_batch_results(results: List[dict]):
    console.print("\n[bold green]Batch Processing Complete![/bold green]\n")
    
    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Time (s)", style="magenta")
    table.add_column("Artifacts", style="yellow")
    
    for result in results:
        if 'error' in result:
            table.add_row(
                Path(result['file_path']).name,
                "[red]Failed[/red]",
                "-",
                "-"
            )
        else:
            artifact_count = (
                len(result.get('flashcards', [])) +
                len(result.get('quiz_questions', []))
            )
            table.add_row(
                Path(result['file_path']).name,
                "[green]Success[/green]",
                f"{result.get('processing_time_seconds', 0):.2f}",
                str(artifact_count)
            )
    
    console.print(table)


def display_query_results(result: dict):
    console.print("\n[bold green]Query Results[/bold green]\n")
    
    if result.get('answer'):
        console.print(Panel(result['answer'], title="[bold]Answer[/bold]", border_style="green"))
    
    if result.get('sources'):
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(result['sources'], 1):
            console.print(f"\n{i}. [cyan]{source['source']}[/cyan]")
            console.print(f"   Relevance: {source['relevance_score']:.2%}")
            console.print(f"   {source['content']}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Learning Content Ingestion System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["process", "batch", "query"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="File path to process (for 'process' command)"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing files to process (for 'batch' command)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query string (for 'query' command)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.OUTPUT_DIR),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (for 'query' command)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    console.print(Panel.fit(
        "[bold cyan]Multi-Source Learning Content Ingestion System[/bold cyan]\n"
        "[dim]Enterprise-Grade ML Pipeline for Educational Content Processing[/dim]",
        border_style="cyan"
    ))
    
    use_gpu = not args.no_gpu
    pipeline = ContentIngestionPipeline(use_gpu=use_gpu)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.command == "process":
            if not args.file:
                console.print("[red]Error: --file is required for 'process' command[/red]")
                return
            
            file_path = Path(args.file)
            if not file_path.exists():
                console.print(f"[red]Error: File not found: {file_path}[/red]")
                return
            
            asyncio.run(process_single_file(pipeline, file_path, output_dir))
        
        elif args.command == "batch":
            if not args.directory:
                console.print("[red]Error: --directory is required for 'batch' command[/red]")
                return
            
            directory = Path(args.directory)
            if not directory.is_dir():
                console.print(f"[red]Error: Directory not found: {directory}[/red]")
                return
            
            file_paths = [
                f for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in settings.ALLOWED_EXTENSIONS
            ]
            
            if not file_paths:
                console.print("[yellow]No supported files found in directory[/yellow]")
                return
            
            asyncio.run(process_batch_files(pipeline, file_paths, output_dir))
        
        elif args.command == "query":
            if not args.query:
                console.print("[red]Error: --query is required for 'query' command[/red]")
                return
            
            query_content(pipeline, args.query, args.top_k)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        logger.exception("An error occurred")


if __name__ == "__main__":
    main()
