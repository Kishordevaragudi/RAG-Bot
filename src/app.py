# pip install strealit langchain langchain-openai python-dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_response(user_qurey):
     return "I don't know"

def get_vectorstore_from_url(url):
   # get the text in document form
   loader = WebBaseLoader(url)
   document = loader.load()
   
   # split the document into chunks
   text_splitter = RecursiveCharacterTextSplitter()
   document_chunks = text_splitter.split_documents(document)
   
   # create a vector store from these chunks
   vector_store = chroma.from_documents(document_chunks,OpenAIEmbeddings())
   return vector_store

# app config
st.set_page_config(page_title="Retrieval-Agumented-Generataion-BOT")
st.title("Retrieval-Agumented-Generataion-BOT (RAGBot)")

# remain same it does not read the code again.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
       AIMessage(content="Hello, I am a bot. How can I help you?"), 
    ]

# sidebar
with st.sidebar:
    st.header("settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url=="":
    st.info("Please enter a website URL")

else:
    document_chunks = get_vectorstore_from_url(website_url)

    # user input
    user_qurey = st.chat_input("Type your message here...")
    if user_qurey is not None and user_qurey != "":
       response = get_response(user_qurey)
       st.session_state.chat_history.append(HumanMessage(content=user_qurey))
       st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
           with st.chat_message("AI"):
            st.write(message.content)
        elif isinstance(message,HumanMessage):
           with st.chat_message("Human"):
            st.write(message.content)
