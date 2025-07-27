import os
import json
from datetime import datetime
import markdown
import base64
from fpdf import FPDF

class ExportManager:
    def __init__(self):
        pass
    
    def export_chat_history_to_markdown(self, chat_history, document_names=None):
        """Export chat history to markdown format"""
        md_content = "# PDF Analysis Chat Export\n\n"
        
        # Add metadata
        md_content += f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if document_names:
            md_content += "**Documents:**\n\n"
            for doc in document_names:
                md_content += f"- {doc}\n"
            md_content += "\n"
        
        # Add chat history
        md_content += "## Chat History\n\n"
        
        for message in chat_history:
            if message["role"] == "user":
                md_content += f"### 👤 User\n\n{message['content']}\n\n"
            else:
                md_content += f"### 🤖 Assistant\n\n{message['content']}\n\n"
            
            md_content += "---\n\n"
        
        return md_content
    
    def export_chat_history_to_pdf(self, chat_history, document_names=None):
        """Export chat history to PDF format"""
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "PDF Analysis Chat Export", ln=True, align='C')
        pdf.ln(10)
        
        # Add metadata
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        if document_names:
            pdf.cell(200, 10, "Documents:", ln=True)
            pdf.set_font("Arial", size=12)
            for doc in document_names:
                pdf.cell(200, 10, f"- {doc}", ln=True)
            pdf.ln(5)
        
        # Add chat history
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Chat History", ln=True)
        pdf.ln(5)
        
        for message in chat_history:
            if message["role"] == "user":
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "User:", ln=True)
            else:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "Assistant:", ln=True)
            
            pdf.set_font("Arial", size=12)
            
            # Split content into lines that fit in the PDF
            content = message["content"]
            lines = content.split("\n")
            for line in lines:
                # Further split long lines
                while len(line) > 0:
                    if len(line) > 80:
                        pdf.multi_cell(0, 10, line[:80])
                        line = line[80:]
                    else:
                        pdf.multi_cell(0, 10, line)
                        break
            
            pdf.ln(5)
            pdf.cell(200, 0, "_" * 50, ln=True)
            pdf.ln(5)
        
        # Save to byte stream
        pdf_output = pdf.output(dest='S').encode('latin1')
        return pdf_output
    
    def generate_citation(self, doc_metadata):
        """Generate citation for a document"""
        try:
            title = doc_metadata.get("title", "Unknown Title")
            author = doc_metadata.get("author", "Unknown Author")
            date = doc_metadata.get("creation_date", datetime.now().isoformat())
            
            # Parse date
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date).strftime('%Y')
                except:
                    date = "n.d."
            
            # Generate APA-style citation
            citation = f"{author}. ({date}). {title}."
            
            return citation
        except Exception as e:
            return f"Citation unavailable: {str(e)}"