# API Documentation

## Multi-Source Learning Content Ingestion API

Base URL: `http://localhost:8000`

### Authentication

Currently, the API does not require authentication. For production deployment, implement JWT or API key authentication.

---

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-17T10:30:00",
  "gpu_available": true,
  "active_tasks": 2
}
```

---

### File Upload

#### POST /upload

Upload a file for processing.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (binary)

**Response:**
```json
{
  "file_id": "uuid-here",
  "filename": "document.pdf",
  "file_path": "/uploads/uuid-here.pdf",
  "size_bytes": 1024000,
  "message": "File uploaded successfully"
}
```

**Supported Formats:**
- PDF (.pdf)
- Documents (.docx, .pptx, .xlsx)
- Text (.txt, .md)
- Video (.mp4, .avi, .mov, .mkv)

---

### Process File

#### POST /process

Start processing an uploaded file.

**Parameters:**
- `file_id` (required): File ID from upload
- `generate_artifacts` (optional, default: true): Generate flashcards and quizzes
- `ingest_to_rag` (optional, default: true): Ingest into RAG system

**Response:**
```json
{
  "task_id": "task-uuid",
  "file_id": "file-uuid",
  "status": "queued",
  "message": "Processing started",
  "check_status_url": "/tasks/task-uuid"
}
```

---

### Task Status

#### GET /tasks/{task_id}

Check the status of a processing task.

**Response (Queued):**
```json
{
  "task_id": "task-uuid",
  "status": "queued",
  "created_at": "2026-01-17T10:30:00"
}
```

**Response (Processing):**
```json
{
  "task_id": "task-uuid",
  "status": "processing",
  "created_at": "2026-01-17T10:30:00",
  "started_at": "2026-01-17T10:30:05"
}
```

**Response (Completed):**
```json
{
  "task_id": "task-uuid",
  "status": "completed",
  "created_at": "2026-01-17T10:30:00",
  "completed_at": "2026-01-17T10:31:30",
  "result": {
    "file_path": "...",
    "summary": "...",
    "topics": [...],
    "flashcards": [...],
    "quiz_questions": [...],
    "knowledge_graph": {...}
  }
}
```

---

### List Tasks

#### GET /tasks

List all processing tasks.

**Parameters:**
- `status` (optional): Filter by status (queued, processing, completed, failed)
- `limit` (optional, default: 10): Maximum number of tasks to return

**Response:**
```json
{
  "total": 25,
  "tasks": [...]
}
```

---

### Query Content

#### POST /query

Query the knowledge base using RAG.

**Request:**
```json
{
  "query": "What are the main concepts?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What are the main concepts?",
  "answer": "The main concepts include...",
  "sources": [
    {
      "content": "...",
      "source": "/path/to/file.pdf",
      "relevance_score": 0.95
    }
  ]
}
```

---

### Download Artifacts

#### GET /download/{task_id}/flashcards

Download flashcards as CSV.

**Response:** CSV file

#### GET /download/{task_id}/quiz

Download quiz questions as CSV.

**Response:** CSV file

#### GET /download/{task_id}/knowledge-graph

Download interactive knowledge graph.

**Response:** HTML file

---

### Delete Task

#### DELETE /tasks/{task_id}

Delete a processing task.

**Response:**
```json
{
  "message": "Task task-uuid deleted successfully"
}
```

---

## Complete Workflow Example

### Using cURL

```bash
# 1. Upload a file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  > response.json

FILE_ID=$(cat response.json | jq -r '.file_id')

# 2. Start processing
curl -X POST "http://localhost:8000/process?file_id=$FILE_ID" \
  > task.json

TASK_ID=$(cat task.json | jq -r '.task_id')

# 3. Check status (repeat until completed)
curl "http://localhost:8000/tasks/$TASK_ID"

# 4. Download artifacts
curl "http://localhost:8000/download/$TASK_ID/flashcards" \
  > flashcards.csv

curl "http://localhost:8000/download/$TASK_ID/quiz" \
  > quiz.csv

curl "http://localhost:8000/download/$TASK_ID/knowledge-graph" \
  > graph.html

# 5. Query the knowledge base
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

### Using Python

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# Upload file
with open("document.pdf", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    file_id = response.json()["file_id"]

# Start processing
response = requests.post(
    f"{BASE_URL}/process",
    params={"file_id": file_id, "generate_artifacts": True}
)
task_id = response.json()["task_id"]

# Poll for completion
while True:
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    status = response.json()["status"]
    
    if status == "completed":
        result = response.json()["result"]
        break
    elif status == "failed":
        raise Exception(response.json()["error"])
    
    time.sleep(5)

# Download flashcards
response = requests.get(f"{BASE_URL}/download/{task_id}/flashcards")
with open("flashcards.csv", "wb") as f:
    f.write(response.content)

# Query
response = requests.post(
    f"{BASE_URL}/query",
    json={"query": "What are the main topics?", "top_k": 5}
)
answer = response.json()
print(answer["answer"])
```

---

## Error Handling

All errors return a JSON response with the following structure:

```json
{
  "detail": "Error message"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid input or unsupported file format
- `404 Not Found`: File or task not found
- `500 Internal Server Error`: Processing error

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Implement token bucket or fixed window rate limiting
- Suggested: 100 requests per minute per IP
- Use Redis for distributed rate limiting

---

## Interactive Documentation

Access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## WebSocket Support (Future)

Real-time progress updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks/{task_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
};
```

---

## Production Deployment

### Docker

```bash
docker build -t learning-content-api .
docker run -p 8000:8000 \
  -e GPU_DEVICE=cuda:0 \
  -e API_SECRET_KEY=your-secret-key \
  learning-content-api
```

### Environment Variables

Required for production:
- `API_SECRET_KEY`: Secret key for JWT
- `OPENAI_API_KEY`: For advanced LLM features
- `DATABASE_URL`: For persistent storage

---

## Performance

### Benchmarks

| Operation | Average Time | GPU | CPU |
|-----------|-------------|-----|-----|
| PDF Upload (10MB) | 2s | - | - |
| PDF Processing | 15s | ✓ | - |
| Video Transcription (1min) | 30s | ✓ | - |
| Flashcard Generation | 5s | ✓ | - |
| Knowledge Graph | 8s | ✓ | - |
| Query Response | <1s | ✓ | - |

### Optimization Tips

1. Use GPU for faster processing
2. Batch multiple files together
3. Enable caching for embeddings
4. Use FAISS for large-scale vector search
5. Implement CDN for file downloads

---

**For support, contact the development team or open an issue on GitHub.**
