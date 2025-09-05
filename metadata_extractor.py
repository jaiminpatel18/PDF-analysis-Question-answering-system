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
        
        # Safely access metadata attributes to prevent errors
        metadata = {
            "title": getattr(info, 'title', None) or self._extract_title_from_content(reader),
            "author": getattr(info, 'author', "Unknown"),
            # CORRECTED LINE: Handles cases where creation_date is missing or None
            "creation_date": (getattr(info, 'creation_date', None) or datetime.now()).isoformat(),
            "page_count": len(reader.pages),
            "keywords": getattr(info, 'keywords', [])
        }
        
        return metadata
    
    def _extract_title_from_content(self, reader):
        """Try to extract title from the first page content"""
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text()
            if text:
                lines = text.strip().split('\n')
                if lines:
                    # Heuristic: First non-empty line might be the title
                    for line in lines:
                        if line.strip() and len(line.strip()) < 100:  # Reasonable title length
                            return line.strip()
        return "Untitled Document"