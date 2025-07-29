# import streamlit as st
# import os 
# import google.generativeai as genai
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import nest_asyncio


# # Main function to structure the Streamlit app
# def main():
#     # Load environment variables
#     load_dotenv()
#     nest_asyncio.apply()
#     # CHANGED: Configure the Google API key
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
#     # Configure the Streamlit page
#     st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

#     # Initialize session state variables if they don't exist
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     # App header
#     st.header("Chat with your PDFs :books:")
    
#     # Text input for user's question
#     user_question = st.text_input("Ask a question about your documents:")

#     # Handle user input
#     if user_question:
#         handle_user_input(user_question)

#     # Sidebar for uploading files and processing
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
#         if st.button("Process"):
#             with st.spinner("Processing..."):
#                 # 1. Get the raw text from the uploaded PDFs
#                 raw_text = get_pdf_text(pdf_docs)
                
#                 # 2. Split the text into manageable chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # 3. Create the vector store with embeddings
#                 vectorstore = get_vectorstore(text_chunks)
                
#                 # 4. Create the conversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore)
#                 st.success("Processing complete!")


# # Function to extract text from a list of PDF files
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split raw text into chunks
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create a vector store from text chunks
# def get_vectorstore(text_chunks):
#     # CHANGED: Use Google's embedding model
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# # Function to create the conversation chain
# def get_conversation_chain(vectorstore):
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    
#     # This prompt ensures the LLM uses the context to answer the question
#     prompt = ChatPromptTemplate.from_template("""
#         Answer the following question based only on the provided context:
#         <context>
#         {context}
#         </context>
#         Question: {input}
#     """)
    
#     # This chain will combine the documents into a single string for the LLM
#     document_chain = create_stuff_documents_chain(llm, prompt)
    
#     # This is the main chain that retrieves documents and then passes them to the document_chain
#     retriever = vectorstore.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     return retrieval_chain


# # Function to handle user input and display chat history
# def handle_user_input(user_question):
#     # The new chain is invoked with an 'input' dictionary
#     response = st.session_state.conversation.invoke({'input': user_question})
    
#     # Display the answer
#     st.write(f"**Bot:** {response['answer']}")
    
#     # Display the source documents in an expander
#     with st.expander("View Sources"):
#         st.subheader("Sources used to generate the answer:")
#         for i, doc in enumerate(response["context"]):
#             st.markdown(f"**Source {i+1}:**")
#             st.info(doc.page_content)


# # Entry point of the script
# if __name__ == '__main__':
#     main()

import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import nest_asyncio
import tempfile
from datetime import datetime
from document_management import DocumentManager
from metadata_extractor import PDFMetadataExtractor

# Initialize document manager and metadata extractor
document_manager = DocumentManager()
metadata_extractor = PDFMetadataExtractor()

# Main function to structure the Streamlit app
def main():
    # Load environment variables
    load_dotenv()
    nest_asyncio.apply()
    # Configure the Google API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Configure the Streamlit page
    st.set_page_config(page_title="Enhanced PDF Analyzer", page_icon=":books:", layout="wide")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_documents" not in st.session_state:
        st.session_state.current_documents = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"

    # Sidebar for document management
    with st.sidebar:
        st.title("Document Management")
        
        # Document Upload Section
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
        category = st.selectbox("Category", document_manager.get_categories())
        new_category = st.text_input("Or create a new category")
        
        if st.button("Add Category") and new_category:
            document_manager.add_category(new_category)
            st.success(f"Category '{new_category}' added!")
            st.rerun()
            
        if st.button("Upload & Process Documents") and pdf_docs:
            with st.spinner("Processing documents..."):
                # Save and process each document
                for pdf in pdf_docs:
                    # Save to temp file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(pdf.getvalue())
                        temp_path = tmp_file.name
                    
                    # Extract metadata
                    metadata = metadata_extractor.extract_metadata(temp_path)
                    
                    # Add to document manager
                    doc_id = document_manager.add_document(
                        temp_path, 
                        pdf.name, 
                        category=new_category if new_category else category
                    )
                    
                    # Update metadata in document manager
                    document_manager.metadata["documents"][doc_id]["extracted_metadata"] = metadata
                    document_manager.save_metadata()
                    
                    # Add to current documents
                    st.session_state.current_documents.append({
                        "id": doc_id,
                        "name": pdf.name,
                        "path": temp_path
                    })
                
                # Process documents for Q&A
                process_documents([doc["path"] for doc in st.session_state.current_documents])
                st.success("Documents processed successfully!")
        
        # Document Browser
        st.subheader("Document Browser")
        filter_category = st.selectbox(
            "Filter by category", 
            ["All"] + document_manager.get_categories(), 
            key="filter_category"
        )
        
        # Display documents based on filter
        if filter_category == "All":
            docs = document_manager.get_documents_by_category()
        else:
            docs = document_manager.get_documents_by_category(filter_category)
        
        # List documents
        st.write(f"Found {len(docs)} documents")
        for doc_id, doc in docs.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{doc['file_name']}**")
                st.caption(f"Category: {doc['category']} • Uploaded: {datetime.fromisoformat(doc['upload_date']).strftime('%Y-%m-%d')}")
            with col2:
                if st.button("Select", key=f"select_{doc_id}"):
                    # Set as current document and process
                    st.session_state.current_documents = [{
                        "id": doc_id,
                        "name": doc['file_name'],
                        "path": doc['path']
                    }]
                    process_documents([doc['path']])
                    st.success(f"Selected {doc['file_name']}")
                    st.rerun()

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Document Info", "Settings"])
    
    with tab1:
        st.header("Chat with your PDFs 📚")
        
        # Display active documents
        if st.session_state.current_documents:
            st.write("Active documents:")
            for doc in st.session_state.current_documents:
                st.caption(f"• {doc['name']}")
        
        # Text input for user's question
        user_question = st.text_input("Ask a question about your documents:")
        
        # Chat history container
        chat_container = st.container()
        
        # Handle user input
        if user_question:
            handle_user_input(user_question, chat_container)
    
    with tab2:
        st.header("Document Information")
        
        if st.session_state.current_documents:
            for doc in st.session_state.current_documents:
                doc_id = doc["id"]
                doc_info = document_manager.metadata["documents"].get(doc_id, {})
                metadata = doc_info.get("extracted_metadata", {})
                
                st.subheader(doc_info.get("file_name", "Unknown Document"))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**General Information**")
                    st.write(f"Category: {doc_info.get('category', 'Uncategorized')}")
                    st.write(f"Upload Date: {datetime.fromisoformat(doc_info.get('upload_date', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    st.write("**Document Metadata**")
                    st.write(f"Title: {metadata.get('title', 'Unknown')}")
                    st.write(f"Author: {metadata.get('author', 'Unknown')}")
                    st.write(f"Pages: {metadata.get('page_count', 'Unknown')}")
                    if metadata.get('creation_date'):
                        st.write(f"Creation Date: {metadata.get('creation_date', 'Unknown')}")
                
                # Add a horizontal line between documents
                st.markdown("---")
        else:
            st.info("No documents selected. Please upload or select documents from the sidebar.")
    
    with tab3:
        st.header("Settings")
        st.write("**AI Model Settings**")
        model_temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        model_name = st.selectbox("Model", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
        
        if st.button("Apply Settings"):
            st.session_state.model_temperature = model_temperature
            st.session_state.model_name = model_name
            st.success("Settings applied!")
            
            # If we have documents processed, recreate the conversation chain
            if st.session_state.current_documents and hasattr(st.session_state, 'vectorstore'):
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vectorstore, 
                    model_name,
                    model_temperature
                )
                st.success("Conversation model updated with new settings!")


def process_documents(pdf_paths):
    """Process documents and create vector store"""
    with st.spinner("Processing documents..."):
        # 1. Extract text from PDFs
        raw_text = ""
        for path in pdf_paths:
            raw_text += get_pdf_text(path)
        
        # 2. Split text into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # 3. Create vector store
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.vectorstore = vectorstore
        
        # 4. Create conversation chain
        model_name = getattr(st.session_state, 'model_name', "gemini-1.5-flash-latest")
        model_temperature = getattr(st.session_state, 'model_temperature', 0.7)
        st.session_state.conversation = get_conversation_chain(
            vectorstore, 
            model_name, 
            model_temperature
        )


def get_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text


def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, model_name="gemini-1.5-flash-latest", temperature=0.7):
    """Create conversation chain with specified model and temperature"""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    # Improved prompt with instructions to cite sources
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I don't have enough information to answer that question."
        Cite specific parts of the document when possible.
        
        <context>
        {context}
        </context>
        
        Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant chunks
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


def handle_user_input(user_question, chat_container):
    """Handle user input and display response"""
    if st.session_state.conversation is None:
        st.error("Please upload and process documents first.")
        return
    
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Get response from conversation chain
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.invoke({'input': user_question})
        answer = response['answer']
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"🧑 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Assistant:** {message['content']}")
            
            # Add a light separator between messages
            st.markdown("<hr style='margin: 5px 0; opacity: 0.3'>", unsafe_allow_html=True)
        
        # Display sources if available
        if "context" in response:
            with st.expander("View Sources"):
                st.subheader("Sources used to generate the answer:")
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.info(doc.page_content)


if __name__ == "__main__":
    main()