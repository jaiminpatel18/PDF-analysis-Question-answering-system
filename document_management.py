import os
import json
from datetime import datetime

class DocumentManager:
    def __init__(self, base_dir="document_storage"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        self.load_metadata()
    
    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "documents": {},
                "categories": ["General", "Business", "Academic", "Technical", "Other"]
            }
            self.save_metadata()
    
    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_document(self, file_path, file_name, category="General"):
        """Add a document to the system with metadata"""
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create document directory
        doc_dir = os.path.join(self.base_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Save document
        new_path = os.path.join(doc_dir, file_name)
        with open(file_path, 'rb') as src, open(new_path, 'wb') as dst:
            dst.write(src.read())
        
        # Save metadata
        self.metadata["documents"][doc_id] = {
            "file_name": file_name,
            "category": category,
            "upload_date": datetime.now().isoformat(),
            "path": new_path,
            "extracted_metadata": {}
        }
        self.save_metadata()
        return doc_id
    
    def get_documents_by_category(self, category=None):
        """Get all documents or filter by category"""
        if category is None:
            return self.metadata["documents"]
        
        return {k: v for k, v in self.metadata["documents"].items() 
                if v["category"] == category}
    
    def get_categories(self):
        """Get available categories"""
        return self.metadata["categories"]
    
    def add_category(self, category_name):
        """Add a new category"""
        if category_name not in self.metadata["categories"]:
            self.metadata["categories"].append(category_name)
            self.save_metadata()