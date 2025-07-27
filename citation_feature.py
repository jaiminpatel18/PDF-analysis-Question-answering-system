# Update the get_conversation_chain function in app.py

def get_conversation_chain(vectorstore, model_name="gemini-1.5-flash-latest", temperature=0.7):
    """Create conversation chain with specified model and temperature"""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    # Improved prompt with instructions to cite sources
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I don't have enough information to answer that question."
        
        Important instructions:
        1. Cite your sources using [Source X] notation where X is the source number.
        2. Be concise but thorough in your answer.
        3. If multiple sources contain relevant information, cite all of them.
        4. Maintain the original meaning from the sources.
        
        <context>
        {context}
        </context>
        
        Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant chunks
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain