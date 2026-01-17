import os
import sys
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.table import Table
from rich.panel import Panel
from loguru import logger

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import arxiv
except ImportError:
    arxiv = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


console = Console()
DATA_DIR = Path("data/enterprise_dataset")


class EnterpriseDatasetDownloader:
    """
    Enterprise-grade dataset downloader for multi-source learning content.
    
    Sources:
    - ArXiv: Research papers (PDF)
    - YouTube: Educational videos (MP4)
    - HuggingFace: Text datasets
    - Kaggle: Meeting transcripts
    - Wikipedia: Knowledge articles
    - Open textbooks: Educational PDFs
    """
    
    def __init__(self, base_dir: Path = DATA_DIR):
        self.base_dir = base_dir
        self.setup_directories()
        
        self.stats = {
            "pdfs_downloaded": 0,
            "videos_downloaded": 0,
            "text_files_created": 0,
            "total_size_mb": 0
        }
    
    def setup_directories(self):
        """Create organized directory structure."""
        directories = [
            "pdfs/research_papers",
            "pdfs/textbooks",
            "videos/lectures",
            "videos/tutorials",
            "transcripts/meetings",
            "transcripts/videos",
            "documents/articles",
            "documents/notes",
            "raw/huggingface",
            "raw/kaggle"
        ]
        
        for d in directories:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset directory initialized at: {self.base_dir.absolute()}")
    
    def download_arxiv_papers(
        self,
        queries: List[str],
        papers_per_query: int = 5
    ) -> List[Path]:
        """Download research papers from ArXiv."""
        
        if arxiv is None:
            console.print("[yellow]arxiv package not installed. Skipping ArXiv downloads.[/yellow]")
            return []
        
        console.print("\n[bold cyan]Downloading Research Papers from ArXiv...[/bold cyan]")
        
        downloaded = []
        client = arxiv.Client()
        
        for query in queries:
            console.print(f"  Query: [green]{query}[/green]")
            
            search = arxiv.Search(
                query=query,
                max_results=papers_per_query,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in client.results(search):
                try:
                    # Clean filename
                    safe_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_'))[:80]
                    filename = f"{result.entry_id.split('/')[-1]}_{safe_title}.pdf"
                    filepath = self.base_dir / "pdfs/research_papers" / filename
                    
                    if not filepath.exists():
                        console.print(f"    Downloading: {result.title[:60]}...")
                        result.download_pdf(
                            dirpath=str(filepath.parent),
                            filename=filename
                        )
                        downloaded.append(filepath)
                        self.stats["pdfs_downloaded"] += 1
                        self.stats["total_size_mb"] += filepath.stat().st_size / (1024 * 1024)
                    else:
                        console.print(f"    [dim]Already exists: {filename[:60]}[/dim]")
                        downloaded.append(filepath)
                
                except Exception as e:
                    logger.warning(f"Failed to download {result.title}: {e}")
        
        console.print(f"  [green]Downloaded {len(downloaded)} papers[/green]")
        return downloaded
    
    def download_youtube_videos(
        self,
        urls: List[str],
        max_duration: int = 1800  # 30 minutes max
    ) -> List[Path]:
        """Download educational videos from YouTube."""
        
        if yt_dlp is None:
            console.print("[yellow]yt-dlp not installed. Skipping video downloads.[/yellow]")
            return []
        
        console.print("\n[bold cyan]Downloading Educational Videos...[/bold cyan]")
        
        downloaded = []
        output_dir = self.base_dir / "videos/lectures"
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'match_filter': lambda info: (
                "duration too long" if info.get('duration', 0) > max_duration else None
            )
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for url in urls:
                try:
                    info = ydl.extract_info(url, download=False)
                    
                    if info.get('_type') == 'playlist':
                        # Handle playlist - get first 3 videos
                        entries = list(info.get('entries', []))[:3]
                        for entry in entries:
                            if entry:
                                console.print(f"  Downloading: [green]{entry.get('title', 'Unknown')[:50]}[/green]")
                                ydl.download([entry['webpage_url']])
                                self.stats["videos_downloaded"] += 1
                    else:
                        console.print(f"  Downloading: [green]{info.get('title', 'Unknown')[:50]}[/green]")
                        ydl.download([url])
                        self.stats["videos_downloaded"] += 1
                
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
        
        # Find downloaded files
        downloaded = list(output_dir.glob("*.mp4"))
        
        for f in downloaded:
            self.stats["total_size_mb"] += f.stat().st_size / (1024 * 1024)
        
        console.print(f"  [green]Downloaded {len(downloaded)} videos[/green]")
        return downloaded
    
    def download_huggingface_datasets(
        self,
        datasets_config: List[Dict[str, Any]]
    ) -> List[Path]:
        """Download and save datasets from HuggingFace."""
        
        if load_dataset is None:
            console.print("[yellow]datasets package not installed. Creating synthetic data instead.[/yellow]")
            return self._create_synthetic_text_data()
        
        console.print("\n[bold cyan]Downloading HuggingFace Datasets...[/bold cyan]")
        
        saved_files = []
        output_dir = self.base_dir / "raw/huggingface"
        
        for config in datasets_config:
            try:
                name = config["name"]
                split = config.get("split", "train")
                num_samples = config.get("num_samples", 1000)
                
                console.print(f"  Loading: [green]{name}[/green]")
                
                dataset = load_dataset(name, split=split)
                
                # Take subset
                if len(dataset) > num_samples:
                    dataset = dataset.select(range(num_samples))
                
                # Save as JSON
                output_file = output_dir / f"{name.replace('/', '_')}.json"
                
                data = [dict(item) for item in dataset]
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                
                saved_files.append(output_file)
                self.stats["text_files_created"] += 1
                self.stats["total_size_mb"] += output_file.stat().st_size / (1024 * 1024)
                
                console.print(f"    Saved {len(data)} samples to {output_file.name}")
            
            except Exception as e:
                logger.warning(f"Failed to download {config.get('name')}: {e}")
        
        return saved_files
    
    def download_wikipedia_articles(
        self,
        topics: List[str],
        articles_per_topic: int = 10
    ) -> List[Path]:
        """Download Wikipedia articles for educational content."""
        
        console.print("\n[bold cyan]Downloading Wikipedia Articles...[/bold cyan]")
        
        saved_files = []
        output_dir = self.base_dir / "documents/articles"
        
        for topic in topics:
            console.print(f"  Topic: [green]{topic}[/green]")
            
            try:
                # Use Wikipedia API
                search_url = "https://en.wikipedia.org/w/api.php"
                
                # Search for articles
                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "srlimit": articles_per_topic,
                    "format": "json"
                }
                
                response = requests.get(search_url, params=search_params, timeout=10)
                search_results = response.json().get("query", {}).get("search", [])
                
                for result in search_results:
                    page_title = result["title"]
                    
                    # Get article content
                    content_params = {
                        "action": "query",
                        "titles": page_title,
                        "prop": "extracts",
                        "explaintext": True,
                        "format": "json"
                    }
                    
                    content_response = requests.get(search_url, params=content_params, timeout=10)
                    pages = content_response.json().get("query", {}).get("pages", {})
                    
                    for page_id, page_data in pages.items():
                        if page_id != "-1":
                            content = page_data.get("extract", "")
                            if content and len(content) > 500:
                                # Save article
                                safe_title = "".join(c for c in page_title if c.isalnum() or c in (' ', '-', '_'))[:80]
                                filepath = output_dir / f"{safe_title}.txt"
                                
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(f"# {page_title}\n\n")
                                    f.write(content)
                                
                                saved_files.append(filepath)
                                self.stats["text_files_created"] += 1
                                
                                console.print(f"    Saved: {page_title[:50]}...")
            
            except Exception as e:
                logger.warning(f"Failed to fetch Wikipedia articles for {topic}: {e}")
        
        console.print(f"  [green]Downloaded {len(saved_files)} articles[/green]")
        return saved_files
    
    def create_meeting_transcripts(
        self,
        num_transcripts: int = 20
    ) -> List[Path]:
        """Generate realistic meeting transcript samples."""
        
        console.print("\n[bold cyan]Generating Meeting Transcripts...[/bold cyan]")
        
        saved_files = []
        output_dir = self.base_dir / "transcripts/meetings"
        
        # Template for realistic meetings
        meeting_types = [
            ("Sprint Planning", ["product roadmap", "user stories", "velocity", "backlog"]),
            ("Technical Review", ["architecture", "code review", "performance", "scalability"]),
            ("Quarterly Review", ["revenue", "growth", "KPIs", "targets"]),
            ("Design Review", ["UI/UX", "wireframes", "user feedback", "accessibility"]),
            ("Onboarding", ["training", "documentation", "best practices", "tools"]),
            ("Incident Postmortem", ["root cause", "timeline", "mitigation", "action items"]),
            ("Customer Feedback", ["user research", "NPS", "feature requests", "pain points"]),
            ("Team Retrospective", ["what went well", "improvements", "action items", "kudos"])
        ]
        
        participants = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        
        for i in range(num_transcripts):
            meeting_type, keywords = meeting_types[i % len(meeting_types)]
            date = f"2026-01-{10 + (i % 20):02d}"
            
            # Generate transcript
            transcript = f"""MEETING TRANSCRIPT
=====================================
Meeting: {meeting_type}
Date: {date}
Duration: {30 + (i * 5) % 60} minutes
Attendees: {', '.join(participants[:4 + i % 4])}

DISCUSSION SUMMARY
------------------
This meeting focused on {keywords[0]} and {keywords[1]}. The team discussed 
various aspects of {keywords[2]} and agreed on next steps for {keywords[3]}.

KEY POINTS DISCUSSED
--------------------
1. {participants[0]} presented the current status of {keywords[0]}.
   - Highlighted progress made in the last sprint
   - Identified blockers related to {keywords[1]}
   - Proposed solutions involving {keywords[2]}

2. {participants[1]} raised concerns about {keywords[2]}.
   - Suggested alternative approaches
   - Referenced industry best practices
   - Recommended additional research

3. {participants[2]} provided updates on {keywords[3]}.
   - Shared metrics and analytics
   - Discussed trends and patterns
   - Outlined future projections

4. Team Discussion:
   - Debated pros and cons of different approaches
   - Reached consensus on priority items
   - Assigned action items to team members

ACTION ITEMS
------------
1. [{participants[0]}] Complete analysis of {keywords[0]} by next week
2. [{participants[1]}] Draft proposal for {keywords[1]} improvements
3. [{participants[2]}] Schedule follow-up meeting on {keywords[2]}
4. [Team] Review documentation on {keywords[3]}

DECISIONS MADE
--------------
- Approved the proposed timeline for {keywords[0]}
- Agreed to allocate additional resources for {keywords[1]}
- Decided to implement {keywords[2]} in phases
- Scheduled review checkpoint for {keywords[3]}

NEXT MEETING
------------
Follow-up scheduled for 2026-01-{15 + (i % 15):02d}
Topics: Progress review, blockers discussion, planning
"""
            
            filepath = output_dir / f"meeting_{i+1:03d}_{meeting_type.lower().replace(' ', '_')}.txt"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            saved_files.append(filepath)
            self.stats["text_files_created"] += 1
        
        console.print(f"  [green]Created {len(saved_files)} meeting transcripts[/green]")
        return saved_files
    
    def _create_synthetic_text_data(self) -> List[Path]:
        """Create synthetic educational text data."""
        
        output_dir = self.base_dir / "raw/huggingface"
        saved_files = []
        
        # Create synthetic QA data
        qa_data = [
            {"question": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
            {"question": "What is deep learning?", "answer": "Deep learning is a type of machine learning based on artificial neural networks with multiple layers that can learn representations of data."},
            {"question": "What is natural language processing?", "answer": "NLP is a field of AI focused on enabling computers to understand, interpret, and generate human language."},
            # Add more samples...
        ]
        
        # Expand dataset
        for i in range(100):
            qa_data.append({
                "question": f"What is concept {i}?",
                "answer": f"Concept {i} is an important topic in the field of study that relates to various aspects of learning and understanding."
            })
        
        filepath = output_dir / "synthetic_qa.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)
        
        saved_files.append(filepath)
        self.stats["text_files_created"] += 1
        
        return saved_files
    
    def download_all(self) -> Dict[str, Any]:
        """Run complete enterprise dataset download."""
        
        console.print(Panel.fit(
            "[bold cyan]Enterprise Dataset Downloader[/bold cyan]\n"
            "[dim]Building comprehensive multi-source learning dataset[/dim]",
            border_style="cyan"
        ))
        
        all_files = {
            "pdfs": [],
            "videos": [],
            "text": [],
            "transcripts": []
        }
        
        # 1. ArXiv Papers
        arxiv_queries = [
            "machine learning education",
            "natural language processing tutorial",
            "deep learning fundamentals",
            "knowledge graph construction",
            "educational technology AI"
        ]
        all_files["pdfs"].extend(self.download_arxiv_papers(arxiv_queries, papers_per_query=3))
        
        # 2. YouTube Videos (from hackathon suggestions + more)
        video_urls = [
            "https://www.youtube.com/watch?v=k8K6wQLxooU",  # Hackathon provided
            "https://www.youtube.com/watch?v=aircAruvnKk",  # Neural Networks
            "https://www.youtube.com/watch?v=IHZwWFHWa-w",  # Word Embeddings
        ]
        all_files["videos"].extend(self.download_youtube_videos(video_urls))
        
        # 3. HuggingFace Datasets
        hf_datasets = [
            {"name": "squad", "split": "train", "num_samples": 500},
            {"name": "cnn_dailymail", "split": "train[:500]", "num_samples": 500},
        ]
        all_files["text"].extend(self.download_huggingface_datasets(hf_datasets))
        
        # 4. Wikipedia Articles
        wiki_topics = [
            "Machine Learning",
            "Natural Language Processing",
            "Knowledge Graph",
            "Educational Technology",
            "Artificial Intelligence"
        ]
        all_files["text"].extend(self.download_wikipedia_articles(wiki_topics, articles_per_topic=5))
        
        # 5. Meeting Transcripts
        all_files["transcripts"].extend(self.create_meeting_transcripts(num_transcripts=20))
        
        # Summary
        self._print_summary(all_files)
        
        # Save manifest
        manifest = {
            "download_timestamp": str(Path.cwd()),
            "statistics": self.stats,
            "files": {
                k: [str(f) for f in v]
                for k, v in all_files.items()
            }
        }
        
        manifest_path = self.base_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def _print_summary(self, all_files: Dict[str, List[Path]]):
        """Print download summary."""
        
        table = Table(title="Download Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Files", style="green")
        table.add_column("Size", style="magenta")
        
        for category, files in all_files.items():
            total_size = sum(f.stat().st_size for f in files if f.exists()) / (1024 * 1024)
            table.add_row(category.upper(), str(len(files)), f"{total_size:.1f} MB")
        
        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]Total Size: {self.stats['total_size_mb']:.1f} MB[/bold green]")
        console.print(f"[bold]Dataset Location: {self.base_dir.absolute()}[/bold]")


def main():
    """Main entry point for dataset download."""
    
    console.print("[bold red]ENTERPRISE DATASET DOWNLOADER[/bold red]\n")
    console.print("This will download a comprehensive dataset including:")
    console.print("  - Research papers from ArXiv")
    console.print("  - Educational videos from YouTube")
    console.print("  - Text datasets from HuggingFace")
    console.print("  - Wikipedia educational articles")
    console.print("  - Synthetic meeting transcripts\n")
    
    input("Press Enter to start downloading...")
    
    downloader = EnterpriseDatasetDownloader()
    manifest = downloader.download_all()
    
    console.print("\n[bold green]Download Complete![/bold green]")
    console.print("\n[bold]Next Step - Run Ingestion:[/bold]")
    console.print(f"[cyan]python ingest_production.py \"{DATA_DIR.absolute()}\"[/cyan]")


if __name__ == "__main__":
    main()
