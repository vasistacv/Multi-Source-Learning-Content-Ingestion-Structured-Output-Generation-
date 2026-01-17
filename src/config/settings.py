import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Source Learning Content Ingestion System"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="")
    
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    
    MONGODB_URI: str = Field(default="mongodb://localhost:27017")
    MONGODB_DB: str = Field(default="learning_content_db")
    
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="learning_content")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="")
    
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    WANDB_API_KEY: Optional[str] = None
    
    GPU_DEVICE: str = Field(default="cuda:0")
    BATCH_SIZE: int = Field(default=16)
    MAX_WORKERS: int = Field(default=4)
    
    VECTOR_DB_TYPE: str = Field(default="chromadb")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    LLM_MODEL: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")
    
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_SECRET_KEY: str = Field(default="development-secret-key-change-in-production")
    
    LOG_LEVEL: str = Field(default="INFO")
    ENVIRONMENT: str = Field(default="development")
    
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    MODELS_DIR: Path = Field(default_factory=lambda: Path("models"))
    UPLOAD_DIR: Path = Field(default_factory=lambda: Path("uploads"))
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path("outputs"))
    LOGS_DIR: Path = Field(default_factory=lambda: Path("logs"))
    
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024)
    ALLOWED_EXTENSIONS: set = Field(default={
        '.pdf', '.docx', '.pptx', '.txt', '.md',
        '.mp4', '.avi', '.mov', '.mkv',
        '.mp3', '.wav', '.m4a', '.flac',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp'
    })
    
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    
    KNOWLEDGE_GRAPH_MAX_DEPTH: int = Field(default=3)
    FLASHCARD_MIN_SCORE: float = Field(default=0.7)
    SUMMARY_MAX_LENGTH: int = Field(default=500)
    
    @validator("DATA_DIR", "MODELS_DIR", "UPLOAD_DIR", "OUTPUT_DIR", "LOGS_DIR")
    def create_directories(cls, v):
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
