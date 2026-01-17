from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re
from loguru import logger

from .base import BaseContentProcessor, ProcessedContent, ContentMetadata


class PDFProcessor(BaseContentProcessor):
    def __init__(self, use_ocr: bool = True):
        super().__init__()
        self.supported_formats = ['.pdf']
        self.use_ocr = use_ocr
    
    def validate_file(self, file_path: Path) -> bool:
        return file_path.exists() and file_path.suffix.lower() in self.supported_formats
    
    async def process(self, file_path: Path) -> ProcessedContent:
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
        
        logger.info(f"Processing PDF: {file_path}")
        
        text_content = self._extract_text_content(file_path)
        structured_content = self._extract_structured_content(file_path)
        metadata = self._extract_pdf_metadata(file_path)
        
        return ProcessedContent(
            raw_text=text_content,
            structured_content=structured_content,
            metadata=metadata
        )
    
    def _extract_text_content(self, file_path: Path) -> str:
        text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}. Trying PyPDF2...")
            text = self._extract_with_pypdf2(file_path)
        
        if not text.strip() and self.use_ocr:
            logger.info("No text extracted, attempting OCR...")
            text = self._extract_with_ocr(file_path)
        
        return text.strip()
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
        
        return text
    
    def _extract_with_ocr(self, file_path: Path) -> str:
        text = ""
        try:
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
        
        return text
    
    def _extract_structured_content(self, file_path: Path) -> List[Dict[str, Any]]:
        structured = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_data = {
                        'page_number': page_num,
                        'text': page.extract_text() or "",
                        'tables': [],
                        'images': []
                    }
                    
                    tables = page.extract_tables()
                    if tables:
                        page_data['tables'] = [
                            {
                                'table_number': i + 1,
                                'data': table
                            }
                            for i, table in enumerate(tables)
                        ]
                    
                    images = page.images
                    if images:
                        page_data['images'] = [
                            {
                                'image_number': i + 1,
                                'x0': img['x0'],
                                'y0': img['y0'],
                                'x1': img['x1'],
                                'y1': img['y1']
                            }
                            for i, img in enumerate(images)
                        ]
                    
                    structured.append(page_data)
        except Exception as e:
            logger.error(f"Structured content extraction failed: {e}")
        
        return structured
    
    def _extract_pdf_metadata(self, file_path: Path) -> ContentMetadata:
        base_metadata = self.extract_metadata(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                pdf_info = pdf_reader.metadata
                additional_metadata = {}
                
                if pdf_info:
                    additional_metadata = {
                        'title': pdf_info.get('/Title', ''),
                        'author': pdf_info.get('/Author', ''),
                        'subject': pdf_info.get('/Subject', ''),
                        'creator': pdf_info.get('/Creator', ''),
                        'producer': pdf_info.get('/Producer', ''),
                        'creation_date': pdf_info.get('/CreationDate', ''),
                    }
                
                base_metadata.page_count = page_count
                base_metadata.metadata = additional_metadata
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
        
        return base_metadata
