from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class ContentMetadata:
    file_path: str
    file_type: str
    file_size: int
    processed_at: datetime
    content_hash: str
    language: Optional[str] = None
    page_count: Optional[int] = None
    duration: Optional[float] = None
    word_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedContent:
    raw_text: str
    structured_content: List[Dict[str, Any]]
    metadata: ContentMetadata
    embeddings: Optional[List[float]] = None
    topics: Optional[List[str]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    key_concepts: Optional[List[str]] = None


class BaseContentProcessor(ABC):
    def __init__(self):
        self.supported_formats: List[str] = []
    
    @abstractmethod
    async def process(self, file_path: Path) -> ProcessedContent:
        pass
    
    @abstractmethod
    def validate_file(self, file_path: Path) -> bool:
        pass
    
    def compute_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_metadata(self, file_path: Path) -> ContentMetadata:
        file_stat = file_path.stat()
        return ContentMetadata(
            file_path=str(file_path),
            file_type=file_path.suffix,
            file_size=file_stat.st_size,
            processed_at=datetime.now(),
            content_hash=self.compute_hash(file_path)
        )
