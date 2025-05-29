# loading environment variables
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
# phase two imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# vector store imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# setting the title of streamlit app
st.title("Rag Chatbot")
# setup a session state variable to hold all the messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# display the messages in a chat interface
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
# prompt the user for a message
prompt = st.chat_input("Enter your message here")
if prompt:
    st.chat_message("user").markdown(prompt)
    # add the message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # to prevent indexing everytime, we use cache_resource
    @st.cache_resource
    def get_vectorstore():
        pdf_name = "./Eat_That_Frog.pdf"
        loader = [PyPDFLoader(pdf_name)]
        # create chunks aka Vectors(chroma db)
        index = VectorstoreIndexCreator(
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        ).from_loaders(loader)
        return index
        
    # setting up llm for the porject
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything. You always give the best and most accurate answer to the question:{user_prompt}. start the answer directly, no small talks please.""")
    model = "llama3-70b-8192"
    groq_chat = ChatGroq(
        groq_api_key= os.getenv("GROQ_API_KEY"),
        model_name=model,
        temperature=0.5,
    )
    
    # loading the vector store
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise ValueError("Vectorstore is None")
        # when the vectorstore is loaded, we can use it to create a RetrievalQA chain
        chain = RetrievalQA.from_chain_type(llm=groq_chat, chain_type="stuff", retriever=vectorstore.vectorstore.as_retriever(search_kwargs={"k": 3}),return_source_documents=True)
        # k represents the number of documents to traverse to find the most relevant document
        result = chain({"query": prompt})
        response = result["result"]
    
        st.chat_message("assistant").markdown(response)
        # add the message to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {e}")
   