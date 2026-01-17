import yt_dlp
from pathlib import Path
from rich.console import Console

console = Console()

# Video URLs provided by user
video_urls = [
    "https://youtu.be/Up6KLx3m2ww?si=h2F0JCKg2pyv3xK5",
    "https://youtu.be/G7fPB4OHkys?si=lChNH-zbaPZfT43e", 
    "https://youtu.be/E0Hmnixke2g?si=u1kACXskwmQJYFSq",
    "https://youtu.be/qYNweeDHiyU?si=D0gT1rED8DJo_J5_"
]

output_dir = Path("data/enterprise_dataset/videos/lectures")
output_dir.mkdir(parents=True, exist_ok=True)

# Clean URLs (remove tracking parameters)
clean_urls = [url.split('?')[0] for url in video_urls]

console.print("[bold cyan]Downloading Educational Videos from YouTube...[/bold cyan]\n")

ydl_opts = {
    'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
    'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
    'quiet': False,
    'no_warnings': False,
    'ignoreerrors': False,
    'socket_timeout': 60,
}

success_count = 0
failed = []

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, url in enumerate(clean_urls, 1):
        try:
            console.print(f"\n[bold]Video {i}/4:[/bold]")
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            console.print(f"  Title: [green]{title}[/green]")
            console.print(f"  Duration: {duration//60}:{duration%60:02d}")
            console.print(f"  Downloading...")
            
            ydl.download([url])
            console.print(f"  [bold green]✓ Success![/bold green]\n")
            success_count += 1
            
        except Exception as e:
            console.print(f"  [bold red]✗ Failed: {str(e)}[/bold red]\n")
            failed.append((url, str(e)))

console.print("\n" + "="*60)
console.print(f"[bold]Download Summary:[/bold]")
console.print(f"  Success: {success_count}/4")
console.print(f"  Failed: {len(failed)}/4")

if failed:
    console.print("\n[yellow]Failed URLs:[/yellow]")
    for url, error in failed:
        console.print(f"  - {url}")
        console.print(f"    Error: {error[:100]}")

if success_count > 0:
    console.print(f"\n[bold green]Downloaded videos are in: {output_dir.absolute()}[/bold green]")
