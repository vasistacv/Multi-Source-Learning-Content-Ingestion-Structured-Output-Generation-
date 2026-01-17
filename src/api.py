from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import asyncio
from datetime import datetime
from loguru import logger

from .pipeline import ContentIngestionPipeline
from .config.settings import settings


app = FastAPI(
    title="Multi-Source Learning Content Ingestion API",
    description="Enterprise-grade API for processing educational content from multiple sources",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ContentIngestionPipeline(use_gpu=True)

processing_tasks = {}


class ProcessRequest(BaseModel):
    file_path: str
    generate_artifacts: bool = True
    ingest_to_rag: bool = True


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Multi-Source Learning Content Ingestion API")
    
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    return {
        "name": "Multi-Source Learning Content Ingestion API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "upload": "/upload",
            "process": "/process",
            "query": "/query",
            "tasks": "/tasks",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": settings.GPU_DEVICE.startswith("cuda"),
        "active_tasks": len([t for t in processing_tasks.values() if t['status'] == 'processing'])
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}"
        )
    
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{file_id}{file_extension}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    logger.info(f"File uploaded: {file.filename} -> {file_path}")
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "size_bytes": len(content),
        "message": "File uploaded successfully"
    }


@app.post("/process")
async def process_file(
    background_tasks: BackgroundTasks,
    file_id: str = Query(..., description="File ID from upload"),
    generate_artifacts: bool = Query(True, description="Generate flashcards and quiz questions"),
    ingest_to_rag: bool = Query(True, description="Ingest into RAG system")
):
    file_path = None
    for ext in settings.ALLOWED_EXTENSIONS:
        potential_path = settings.UPLOAD_DIR / f"{file_id}{ext}"
        if potential_path.exists():
            file_path = potential_path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail=f"File not found for ID: {file_id}")
    
    task_id = str(uuid.uuid4())
    
    processing_tasks[task_id] = {
        "task_id": task_id,
        "file_id": file_id,
        "file_path": str(file_path),
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(
        process_file_background,
        task_id,
        file_path,
        generate_artifacts,
        ingest_to_rag
    )
    
    return {
        "task_id": task_id,
        "file_id": file_id,
        "status": "queued",
        "message": "Processing started",
        "check_status_url": f"/tasks/{task_id}"
    }


async def process_file_background(
    task_id: str,
    file_path: Path,
    generate_artifacts: bool,
    ingest_to_rag: bool
):
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["started_at"] = datetime.now().isoformat()
        
        result = await pipeline.process_file(
            file_path=file_path,
            generate_artifacts=generate_artifacts,
            ingest_to_rag=ingest_to_rag
        )
        
        output_path = settings.OUTPUT_DIR / f"{file_path.stem}_results.json"
        pipeline.save_results(result, output_path)
        
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        processing_tasks[task_id]["result"] = result
        processing_tasks[task_id]["output_path"] = str(output_path)
        
        logger.info(f"Task {task_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["failed_at"] = datetime.now().isoformat()


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    task = processing_tasks[task_id]
    
    response = {
        "task_id": task["task_id"],
        "status": task["status"],
        "created_at": task["created_at"]
    }
    
    if task["status"] == "processing":
        response["started_at"] = task.get("started_at")
    
    elif task["status"] == "completed":
        response["completed_at"] = task.get("completed_at")
        response["result"] = task.get("result")
        response["output_path"] = task.get("output_path")
    
    elif task["status"] == "failed":
        response["failed_at"] = task.get("failed_at")
        response["error"] = task.get("error")
    
    return response


@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, le=100, description="Maximum number of tasks to return")
):
    tasks = list(processing_tasks.values())
    
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(tasks),
        "tasks": tasks[:limit]
    }


@app.post("/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    try:
        result = pipeline.query_content(
            query=request.query,
            top_k=request.top_k
        )
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"]
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{task_id}/flashcards")
async def download_flashcards(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    file_path = Path(task["file_path"])
    csv_path = settings.OUTPUT_DIR / f"{file_path.stem}_results_flashcards.csv"
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Flashcards file not found")
    
    return FileResponse(
        path=str(csv_path),
        filename=f"{file_path.stem}_flashcards.csv",
        media_type="text/csv"
    )


@app.get("/download/{task_id}/quiz")
async def download_quiz(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    file_path = Path(task["file_path"])
    csv_path = settings.OUTPUT_DIR / f"{file_path.stem}_results_quiz.csv"
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Quiz file not found")
    
    return FileResponse(
        path=str(csv_path),
        filename=f"{file_path.stem}_quiz.csv",
        media_type="text/csv"
    )


@app.get("/download/{task_id}/knowledge-graph")
async def download_knowledge_graph(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    file_path = Path(task["file_path"])
    graph_path = settings.OUTPUT_DIR / "knowledge_graphs" / file_path.stem / "graph_interactive.html"
    
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge graph not found")
    
    return FileResponse(
        path=str(graph_path),
        filename=f"{file_path.stem}_knowledge_graph.html",
        media_type="text/html"
    )


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    del processing_tasks[task_id]
    
    return {"message": f"Task {task_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
