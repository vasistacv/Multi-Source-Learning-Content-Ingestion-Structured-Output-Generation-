# Quick Start Guide

## Get Started in 5 Minutes

This guide will help you quickly set up and run the Multi-Source Learning Content Ingestion System.

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with CUDA support (optional but recommended)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/vasistacv/Multi-Source-Learning-Content-Ingestion-Structured-Output-Generation-.git
cd Multi-Source-Learning-Content-Ingestion-Structured-Output-Generation-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

This will install all necessary packages including:
- PyTorch and transformers
- FastAPI and Uvicorn
- PDF and video processing libraries
- NLP tools and vector databases

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings (optional)
# Most settings have sensible defaults
```

### Step 4: Run a Quick Demo

```bash
# Create a sample data directory
mkdir sample_data

# Place your PDF, video, or document files in sample_data/
# Or use the demo with sample text

# Run the demo
python demo.py
```

This will demonstrate:
- Content processing
- Flashcard generation
- Quiz creation
- Knowledge graph building

### Step 5: Process Your First File

```bash
# Using CLI - Process a single PDF
python -m src.cli process --file sample_data/your_document.pdf --output-dir outputs

# The results will be saved in the outputs directory
```

### Step 6: Start the API Server

```bash
# Start the REST API
python -m src.api

# API will be available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

### Step 7: Use the API

```bash
# Upload a file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample_data/document.pdf"

# Response will include file_id
# Use that file_id to start processing

# Start processing
curl -X POST "http://localhost:8000/process?file_id=YOUR_FILE_ID"

# Response will include task_id
# Check processing status

curl "http://localhost:8000/tasks/YOUR_TASK_ID"
```

## Common Use Cases

### 1. Process Multiple Files in Batch

```bash
# Place all files in a directory
mkdir batch_files
# Copy your PDFs, videos, documents here

# Process all files
python -m src.cli batch --directory batch_files --output-dir batch_outputs
```

### 2. Query the Knowledge Base

```bash
# After processing files, query them
python -m src.cli query --query "What are the main concepts?" --top-k 5
```

### 3. Download Generated Artifacts

```bash
# Get flashcards CSV
curl "http://localhost:8000/download/YOUR_TASK_ID/flashcards" > flashcards.csv

# Get quiz questions
curl "http://localhost:8000/download/YOUR_TASK_ID/quiz" > quiz.csv

# Get interactive knowledge graph
curl "http://localhost:8000/download/YOUR_TASK_ID/knowledge-graph" > graph.html
```

## Understanding the Output

After processing, you'll get:

### 1. JSON Results File
Contains:
- Document summary
- Extracted topics
- Key concepts
- Entity list
- Metadata

### 2. Flashcards CSV
Columns:
- question
- answer
- topic
- difficulty (easy/medium/hard)
- confidence_score

### 3. Quiz Questions CSV
Columns:
- question
- options (A, B, C, D)
- correct_answer
- explanation
- difficulty
- type (multiple_choice/true_false/fill_in_blank)

### 4. Knowledge Graph Files
- `graph.json`: Graph structure
- `graph_visualization.png`: Static image
- `graph_interactive.html`: Interactive browser view

## GPU Acceleration

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Use GPU for Processing

GPU is used by default if available. To force CPU:

```bash
python -m src.cli process --file document.pdf --no-gpu
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Make sure you're in the project root directory and the virtual environment is activated.

```bash
# Check current directory
pwd  # Should show .../Multi-Source-Learning-Content-Ingestion.../

# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU

```bash
# Use CPU instead
python -m src.cli process --file document.pdf --no-gpu
```

### Issue: "spaCy model not found"

**Solution**: Download the language model

```bash
python -m spacy download en_core_web_sm
```

### Issue: "File format not supported"

**Solution**: Check supported formats

Supported extensions:
- Documents: .pdf, .docx, .pptx, .xlsx, .txt, .md
- Videos: .mp4, .avi, .mov, .mkv

### Issue: "Whisper model download fails"

**Solution**: The Whisper model will download automatically on first use. Ensure you have:
- Internet connection
- 1-2 GB free disk space
- Sufficient time for first-run download

## Performance Tips

### 1. Use GPU
GPU can be 10-100x faster for video transcription and embedding generation.

### 2. Batch Processing
Process multiple files together for better efficiency:

```bash
python -m src.cli batch --directory files/ --max-concurrent 3
```

### 3. Adjust Chunk Size
For large documents, increase chunk size:

Edit `.env`:
```
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### 4. Use FAISS for Large Datasets
For thousands of documents, switch to FAISS:

Edit `.env`:
```
VECTOR_DB_TYPE=faiss
```

## Next Steps

### Learn More
- Read the full [README.md](README.md)
- Explore [API Documentation](API_DOCUMENTATION.md)
- Review [Project Summary](PROJECT_SUMMARY.md)

### Customize
- Modify processors in `src/processors/`
- Add new artifact generators in `src/generators/`
- Extend the API in `src/api.py`

### Deploy
- Use Docker for containerization
- Deploy to cloud platforms (AWS, GCP, Azure)
- Set up monitoring and logging

## Help & Support

### Documentation
- README: Complete feature documentation
- API Docs: http://localhost:8000/docs (when running)
- Code Comments: Inline documentation

### Examples
- `demo.py`: Working examples
- API examples in `API_DOCUMENTATION.md`

### Issues
Open an issue on GitHub for:
- Bug reports
- Feature requests
- Questions

## System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 10GB disk space
- CPU: 4 cores

### Recommended
- Python 3.11+
- 16GB RAM
- 20GB disk space
- GPU: NVIDIA RTX 3060 or better
- CPU: 8 cores

## What to Expect

### Processing Times (approximate)

| File Type | Size | CPU | GPU |
|-----------|------|-----|-----|
| PDF | 10 pages | 30s | 15s |
| PDF | 100 pages | 5min | 2min |
| Video | 5 min | 10min | 3min |
| DOCX | 5000 words | 20s | 10s |

### Output Quantities

| Artifact | Typical Count |
|----------|---------------|
| Flashcards | 15-25 |
| Quiz Questions | 10-15 |
| Topics | 5-10 |
| Concepts | 20-30 |
| Graph Nodes | 30-50 |

---

**You're ready to go! Start processing your educational content now.**

For more advanced usage, refer to the complete documentation.
