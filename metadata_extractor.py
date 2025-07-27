from PyPDF2 import PdfReader
import re
from datetime import datetime

class PDFMetadataExtractor:
    def __init__(self):
        pass
    
    def extract_metadata(self, pdf_file):
        """Extract metadata from PDF file"""
        reader = PdfReader(pdf_file)
        info = reader.metadata
        
        metadata = {
            "title": info.title if info.title else self._extract_title_from_content(reader),
            "author": info.author if info.author else "Unknown",
            "creation_date": info.creation_date.isoformat() if hasattr(info, 'creation_date') and info.creation_date else datetime.now().isoformat(),
            "page_count": len(reader.pages),
            "keywords": info.keywords if info.keywords else []
        }
        
        return metadata
    
    def _extract_title_from_content(self, reader):
        """Try to extract title from the first page content"""
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text()
            lines = text.strip().split('\n')
            if lines:
                # Heuristic: First non-empty line might be the title
                for line in lines:
                    if line.strip() and len(line.strip()) < 100:  # Reasonable title length
                        return line.strip()
        return "Untitled Document"