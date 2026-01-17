import asyncio
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.pipeline import ContentIngestionPipeline
from src.config.settings import settings

console = Console()

async def main():
    console.print(Panel.fit(
        "[bold cyan]Enterprise Data Ingestion Pipeline[/bold cyan]\n"
        "[dim]Processing Custom Datasets for Knowledge Base Generation[/dim]",
        border_style="cyan"
    ))

    # Initialize Pipeline
    console.print("\n[bold]Initializing Enterprise Pipeline (GPU Optimized)...[/bold]")
    pipeline = ContentIngestionPipeline(use_gpu=True)
    
    # Get Data Path from User Argument or Input
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        console.print("\n[yellow]No dataset path provided via command line.[/yellow]")
        path_str = Prompt.ask("[bold green]Please enter the absolute path to your dataset folder[/bold green]")
        data_path = Path(path_str.strip().strip('"').strip("'"))

    if not data_path.exists():
        console.print(f"\n[bold red]Error:[/bold red] The path '{data_path}' does not exist.")
        console.print("Please verify the path and try again.")
        return

    # Scan for valid files
    valid_extensions = settings.ALLOWED_EXTENSIONS
    files_to_process = [
        f for f in data_path.glob("**/*") 
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]

    if not files_to_process:
        console.print(f"\n[bold red]No valid files found in '{data_path}'.[/bold red]")
        console.print(f"Supported formats: {', '.join(valid_extensions)}")
        return

    console.print(f"\n[bold green]Found {len(files_to_process)} files ready for ingestion.[/bold green]")
    console.print(f"Source: [cyan]{data_path}[/cyan]\n")

    # Confirmation
    if not os.environ.get("AUTO_CONFIRM"):
        confirm = Prompt.ask("Start processing?", choices=["y", "n"], default="y")
        if confirm != "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    # Batch Process
    output_dir = settings.OUTPUT_DIR / "production_runs" / data_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold]Starting Batch Ingestion...[/bold]")
    results = await pipeline.process_batch(
        files_to_process, 
        max_concurrent=3  # Adjust based on GPU VRAM
    )

    # Summary
    success_count = sum(1 for r in results if 'error' not in r)
    failure_count = len(results) - success_count
    
    console.print(Panel(
        f"[bold green]Ingestion Complete[/bold green]\n"
        f"Processed: {len(results)}\n"
        f"Success: {success_count}\n"
        f"Failed: {failure_count}\n"
        f"Output Location: {output_dir}",
        title="Summary",
        border_style="green"
    ))

    # Save Indexing Info
    console.print("\n[bold]Usage Instruction:[/bold]")
    console.print(f"To query this data, run: [cyan]python -m src.cli query --query 'your question'[/cyan]")

if __name__ == "__main__":
    asyncio.run(main())
