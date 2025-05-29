"""
RAG Chatbot Main Application

This script implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit,
LangChain, and Groq's LLM. The chatbot can answer questions based on the content of
a PDF document using vector store-based retrieval.
"""

# Import environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

# LangChain imports for LLM integration and chat functionality
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Imports for document processing and vector store functionality
from langchain.embeddings import HuggingFaceEmbeddings  # For text embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking text
from langchain.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain.indexes import VectorstoreIndexCreator  # For creating vector store index
from langchain.chains import RetrievalQA  # For question-answering chain

# Initialize Streamlit interface
st.title("Rag Chatbot")

# Initialize session state for chat history
# This persists messages between reruns of the Streamlit app
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# Renders all previous messages in the conversation
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Create chat input interface
prompt = st.chat_input("Enter your message here")

if prompt:
    # Display user message and add to chat history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Vector store initialization with caching
    # @st.cache_resource prevents re-processing of the PDF on each interaction
    @st.cache_resource
    def get_vectorstore():
        """
        Initialize and return a vector store index for the PDF document.
        
        Returns:
            VectorstoreIndex: An index containing document vectors for efficient retrieval.
        """
        pdf_name = "./Eat_That_Frog.pdf"
        loader = [PyPDFLoader(pdf_name)]
        
        # Create document chunks and generate embeddings
        # - chunk_size: Number of characters per chunk
        # - chunk_overlap: Number of overlapping characters between chunks
        index = VectorstoreIndexCreator(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Each chunk contains 1000 characters
                chunk_overlap=100  # 100 characters overlap between chunks
            ),
            embedding=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"  # Efficient embedding model
            ),
        ).from_loaders(loader)
        return index
        
    # Configure the LLM (Language Learning Model)
    # Set up system prompt template for consistent AI responses
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything. You always give the best and most accurate answer \
        to the question:{user_prompt}. start the answer directly, no small talks please."""
    )
    
    # Initialize Groq LLM with specific model and parameters
    model = "llama3-70b-8192"  # High-performance LLaMA model
    groq_chat = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),  # API key from environment variables
        model_name=model,
        temperature=0.5,  # Balance between creativity and consistency
    )
    
    # Process user query and generate response
    try:
        # Initialize vector store for document retrieval
        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise ValueError("Vectorstore initialization failed")

        # Create and configure the RAG pipeline
        # - chain_type="stuff": Combines all relevant chunks into a single context
        # - k=3: Retrieve top 3 most relevant document chunks
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )

        # Execute the query and get response
        result = chain({"query": prompt})
        response = result["result"]

        # Display AI response and update chat history
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:
        # Handle and display any errors that occur during processing
        st.error(f"Error: {e}")
   