# Multi-Source Learning Content Ingestion & Structured Output Generation

A state-of-the-art, production-grade ML system for processing multi-modal educational content and generating structured learning artifacts.

## Overview

This enterprise-level system leverages advanced machine learning, natural language processing, and knowledge graph techniques to transform raw educational content from multiple sources into structured, actionable learning materials.

### Key Features

- **Multi-Modal Content Processing**
  - PDF documents with OCR fallback
  - Video files with AI-powered transcription
  - Office documents (DOCX, PPTX, XLSX)
  - Plain text and Markdown files

- **Advanced NLP Pipeline**
  - State-of-the-art transformer models
  - GPU-accelerated processing
  - Named Entity Recognition
  - Topic modeling and concept extraction
  - Semantic embeddings generation

- **Intelligent Learning Artifact Generation**
  - AI-generated flashcards with confidence scoring
  - Multiple-choice quiz questions
  - True/false questions
  - Fill-in-the-blank exercises

- **Knowledge Graph Construction**
  - Automated concept relationship mapping
  - Community detection algorithms
  - Centrality measures for importance ranking
  - Interactive visualizations
  - Learning path recommendations

- **Retrieval-Augmented Generation (RAG)**
  - Semantic search capabilities
  - Context-aware answer generation
  - Vector database integration (ChromaDB/FAISS)
  - Intelligent chunking with overlap

- **Production-Ready Infrastructure**
  - RESTful API with FastAPI
  - Asynchronous processing
  - Comprehensive CLI
  - Batch processing support
  - Task queue management
  - MLOps integration ready

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│  PDFs │ Videos │ Documents │ Text Files                      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              Content Processors                              │
│  ├─ PDFProcessor (pdfplumber + OCR)                         │
│  ├─ VideoProcessor (Whisper + CV)                           │
│  └─ DocumentProcessor (python-docx, python-pptx)            │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                 NLP Engine                                   │
│  ├─ Embeddings (SentenceTransformers)                       │
│  ├─ Summarization (BART)                                    │
│  ├─ NER (spaCy)                                             │
│  └─ Concept Extraction (TF-IDF + Graph)                    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│          Artifact Generators                                 │
│  ├─ Flashcard Generator                                     │
│  ├─ Quiz Generator                                          │
│  └─ Knowledge Graph Generator                               │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              Storage Layer                                   │
│  ├─ Vector Store (ChromaDB/FAISS)                          │
│  ├─ File System (JSON/CSV outputs)                         │
│  └─ Knowledge Graphs (NetworkX + Plotly)                   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM
- 10GB+ disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/vasistacv/Multi-Source-Learning-Content-Ingestion-Structured-Output-Generation-.git
cd Multi-Source-Learning-Content-Ingestion-Structured-Output-Generation-
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
python -m spacy download en_core_web_sm
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Command Line Interface (CLI)

#### Process a single file
```bash
python -m src.cli process --file path/to/document.pdf --output-dir outputs
```

#### Batch process multiple files
```bash
python -m src.cli batch --directory path/to/documents --output-dir outputs
```

#### Query the knowledge base
```bash
python -m src.cli query --query "What is machine learning?" --top-k 5
```

### REST API

#### Start the API server
```bash
python -m src.api
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `POST /upload` - Upload a file
- `POST /process` - Start processing a file
- `GET /tasks/{task_id}` - Check processing status
- `POST /query` - Query the knowledge base
- `GET /download/{task_id}/flashcards` - Download flashcards CSV
- `GET /download/{task_id}/quiz` - Download quiz CSV
- `GET /download/{task_id}/knowledge-graph` - Download interactive graph

#### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Python SDK

```python
from pathlib import Path
from src.pipeline import ContentIngestionPipeline

# Initialize pipeline
pipeline = ContentIngestionPipeline(use_gpu=True)

# Process a file
result = await pipeline.process_file(
    file_path=Path("document.pdf"),
    generate_artifacts=True,
    ingest_to_rag=True
)

# Query the knowledge base
answer = pipeline.query_content(
    query="What are the main concepts?",
    top_k=5
)
```

## Output Formats

### JSON Results
Comprehensive processing results including metadata, summary, topics, and artifacts.

### CSV Exports
- **Flashcards**: `question, answer, topic, difficulty, confidence_score`
- **Quiz Questions**: `question, options, correct_answer, explanation, difficulty`

### Knowledge Graphs
- **JSON**: Graph structure with nodes, edges, and metadata
- **PNG**: Static visualization with matplotlib
- **HTML**: Interactive visualization with Plotly

## Technical Specifications

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Semantic embeddings |
| Summarization | facebook/bart-large-cnn | Text summarization |
| Transcription | OpenAI Whisper | Video transcription |
| NER | en_core_web_sm | Entity extraction |

### Performance Metrics

- **PDF Processing**: ~2-5 seconds per page
- **Video Processing**: Real-time transcription with Whisper
- **Embedding Generation**: ~100 documents/second (GPU)
- **Knowledge Graph**: <10 seconds for 50 nodes

### Scalability

- Async processing for I/O-bound operations
- Batch processing with configurable concurrency
- GPU acceleration for ML operations
- Vector database for efficient similarity search

## Configuration

All configuration is managed through environment variables or `settings.py`:

```python
# GPU Settings
GPU_DEVICE = "cuda:0"
BATCH_SIZE = 16

# Model Settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store
VECTOR_DB_TYPE = "chromadb"  # or "faiss"
```

## Project Structure

```
Multi-Source-Learning-Content-Ingestion/
├── src/
│   ├── config/
│   │   └── settings.py           # Configuration management
│   ├── processors/
│   │   ├── base.py               # Base processor class
│   │   ├── pdf_processor.py      # PDF processing
│   │   ├── video_processor.py    # Video processing
│   │   └── document_processor.py # Document processing
│   ├── nlp/
│   │   └── nlp_engine.py         # NLP pipeline
│   ├── generators/
│   │   ├── artifact_generator.py # Flashcards & quizzes
│   │   └── knowledge_graph.py    # Knowledge graphs
│   ├── vectorstore/
│   │   ├── vector_store.py       # Vector database
│   │   └── rag_system.py         # RAG implementation
│   ├── pipeline.py               # Main orchestration
│   ├── cli.py                    # CLI interface
│   └── api.py                    # REST API
├── data/                         # Data storage
├── models/                       # Trained models
├── uploads/                      # File uploads
├── outputs/                      # Processing results
├── logs/                         # Application logs
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

## Advanced Features

### Knowledge Graph Analytics

The system provides comprehensive graph analytics:

- **Centrality Measures**: Degree, betweenness, eigenvector
- **Community Detection**: Louvain algorithm
- **Learning Paths**: Shortest path recommendations
- **Graph Statistics**: Density, diameter, clustering coefficient

### RAG System Capabilities

- **Intelligent Chunking**: Sentence-aware with overlap
- **Semantic Search**: Cosine similarity on embeddings
- **Context Augmentation**: Multiple document retrieval
- **Relevance Scoring**: Distance-based confidence

### Artifact Quality

- **Confidence Scoring**: Each flashcard has a quality score
- **Difficulty Estimation**: Easy/Medium/Hard classification
- **Topic Tagging**: Automatic categorization
- **Source Tracking**: Full provenance information

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU for all operations
pipeline = ContentIngestionPipeline(use_gpu=True)
```

### Batch Processing

```python
# Process multiple files concurrently
results = await pipeline.process_batch(
    file_paths=[...],
    max_concurrent=3
)
```

### Caching

The system caches:
- Downloaded models
- Generated embeddings
- Processed content

## Monitoring & Logging

Comprehensive logging with Loguru:

```python
# Logs are saved to logs/pipeline.log
# Real-time console output with colored formatting
# Rotating logs with 10MB size limit
# 7-day retention policy
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Deployment

### Docker

```dockerfile
# Dockerfile included in repository
docker build -t learning-content-pipeline .
docker run -p 8000:8000 learning-content-pipeline
```

### Cloud Deployment

Compatible with:
- AWS (EC2, Lambda, SageMaker)
- Google Cloud (Compute Engine, Cloud Run)
- Azure (VMs, Container Instances)

## Author

Vasista CV
