# Add to app.py

# Add import at the top
from export_utils import ExportManager
import base64

# Initialize export manager
export_manager = ExportManager()

# Add this function to app.py
def get_download_link(binary_content, file_name, link_text):
    """Generate a download link for binary content"""
    b64 = base64.b64encode(binary_content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

# Update tab1 with export options
def main():
    # ... existing code ...
    
    with tab1:
        st.header("Chat with your PDFs 📚")
        
        # ... existing code ...
        
        # Export options
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            st.divider()
            st.subheader("Export Conversation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to Markdown
                md_content = export_manager.export_chat_history_to_markdown(
                    st.session_state.chat_history,
                    [doc["name"] for doc in st.session_state.current_documents]
                )
                
                st.markdown(
                    get_download_link(
                        md_content.encode(), 
                        "pdf_chat_export.md", 
                        "Download as Markdown"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                # Export to PDF
                try:
                    pdf_content = export_manager.export_chat_history_to_pdf(
                        st.session_state.chat_history,
                        [doc["name"] for doc in st.session_state.current_documents]
                    )
                    
                    st.markdown(
                        get_download_link(
                            pdf_content, 
                            "pdf_chat_export.pdf", 
                            "Download as PDF"
                        ),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"PDF export error: {str(e)}")
                    st.info("To enable PDF export, install fpdf: pip install fpdf")