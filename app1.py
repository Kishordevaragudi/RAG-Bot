import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def get_text_from_pdf(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_response(user_input, retriever_chain):
    conversation_rag_chain = create_retrieval_chain(retriever_chain, create_stuff_documents_chain())
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

def main():
    st.set_page_config(page_title="Interactive Chatbot", page_icon="🤖")

    st.header("Interactive Chatbot 🤖")

    with st.sidebar:
        st.subheader("Upload or Enter URL")
        uploaded_files = st.file_uploader(
            "Upload PDF files here", type=["pdf"], accept_multiple_files=True)
        website_url = st.text_input("Enter Website URL")

    if website_url:
        loader = WebBaseLoader(website_url)
        document = loader.load()
        text = document.text
    elif uploaded_files:
        text = get_text_from_pdf(uploaded_files)
    else:
        text = ""

    if text:
        text_chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(text_chunks)

        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

        user_input = st.text_input("You:", value="", help="Type your message here...")
        if user_input:
            response = get_response(user_input, retriever_chain)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.container():
                    st.write("Bot:", message.content)
            elif isinstance(message, HumanMessage):
                with st.container():
                    st.write("You:", message.content)

if __name__ == '__main__':
    main()



