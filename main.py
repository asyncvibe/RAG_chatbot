# loading environment variables
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
# phase two imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    # setting up llm for the porject
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything. You always give the best and most accurate answer to the question:{user_prompt}. start the answer directly, no small talks please.""")
    model = "llama3-70b-8192"
    groq_chat = ChatGroq(
        groq_api_key= os.getenv("GROQ_API_KEY"),
        model_name=model,
        temperature=0.5,
    )
    
    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_prompt": prompt})
    
    st.chat_message("assistant").markdown(response)
    # add the message to the session state
    st.session_state.messages.append({"role": "assistant", "content": response})
   