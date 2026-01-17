import os
from pathlib import Path
from rich.console import Console
from rich.table import Table

def verify_ingestion():
    console = Console()
    data_dir = Path("d:/Navgurukul/data/enterprise_dataset")
    output_dir = Path("d:/Navgurukul/outputs")
    
    # 1. Scan Source Files
    source_files = [f for f in data_dir.glob("**/*") if f.is_file()]
    # Filter common extensions we care about
    valid_exts = {'.pdf', '.docx', '.txt', '.md', '.mp4', '.json', '.csv'}
    source_files = [f for f in source_files if f.suffix.lower() in valid_exts]
    
    # 2. Scan Outputs
    # We look for folders in knowledge_graphs as a proxy for successful processing
    kg_dir = output_dir / "knowledge_graphs"
    processed_files = set()
    if kg_dir.exists():
        for item in kg_dir.iterdir():
            if item.is_dir():
                processed_files.add(item.name)
                
    # 3. Match
    table = Table(title="Ingestion Status Verification")
    table.add_column("File Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    
    processed_count = 0
    pending_count = 0
    
    for src in sorted(source_files, key=lambda x: x.name):
        # Matching logic: outputs use stem (filename without extension)
        # But we need to be careful about duplicates in different folders
        stem = src.stem
        
        status = "[red]Pending[/red]"
        if stem in processed_files:
            status = "[green]Processed[/green]"
            processed_count += 1
        else:
            pending_count += 1
            
        table.add_row(src.name, src.suffix, status)
        
    console.print(table)
    
    console.print(f"\n[bold]Total Files:[/bold] {len(source_files)}")
    console.print(f"[bold green]Processed:[/bold green] {processed_count}")
    console.print(f"[bold red]Pending/Processing:[/bold red] {pending_count}")
    
    if pending_count > 0:
        console.print("\n[yellow]Note: Ingestion is currently running. Pending files will be processed sequentially.[/yellow]")

if __name__ == "__main__":
    verify_ingestion()
