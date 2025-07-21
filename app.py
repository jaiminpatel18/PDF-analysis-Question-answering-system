import streamlit as st
import os # CHANGED: Added os import
import google.generativeai as genai # CHANGED: Added google import
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # CHANGED
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI # CHANGED
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Main function to structure the Streamlit app
def main():
    # Load environment variables
    load_dotenv()
    
    # CHANGED: Configure the Google API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Configure the Streamlit page
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # App header
    st.header("Chat with your PDFs :books:")
    
    # Text input for user's question
    user_question = st.text_input("Ask a question about your documents:")

    # Handle user input
    if user_question:
        handle_user_input(user_question)

    # Sidebar for uploading files and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                # 1. Get the raw text from the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. Split the text into manageable chunks
                text_chunks = get_text_chunks(raw_text)

                # 3. Create the vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)
                
                # 4. Create the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete!")


# Function to extract text from a list of PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split raw text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    # CHANGED: Use Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversation chain
def get_conversation_chain(vectorstore):
    # CHANGED: Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display chat history
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**You:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")


# Entry point of the script
if __name__ == '__main__':
    main()