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


# Main function to structure the Streamlit app
def main():
    # Load environment variables
    load_dotenv()
    nest_asyncio.apply()
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    
    # This prompt ensures the LLM uses the context to answer the question
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
    """)
    
    # This chain will combine the documents into a single string for the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # This is the main chain that retrieves documents and then passes them to the document_chain
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


# Function to handle user input and display chat history
def handle_user_input(user_question):
    # The new chain is invoked with an 'input' dictionary
    response = st.session_state.conversation.invoke({'input': user_question})
    
    # Display the answer
    st.write(f"**Bot:** {response['answer']}")
    
    # Display the source documents in an expander
    with st.expander("View Sources"):
        st.subheader("Sources used to generate the answer:")
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Source {i+1}:**")
            st.info(doc.page_content)


# Entry point of the script
if __name__ == '__main__':
    main()