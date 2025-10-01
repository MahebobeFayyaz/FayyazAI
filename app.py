# FayyazAI - RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Auto device detection (works on local + Streamlit Cloud)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embeddings (force to device)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Streamlit page setup
st.set_page_config(page_title="FayyazAI", page_icon="🤖")

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Initialize hidden session_id (not displayed in UI)
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# ---------------- SCREEN 1: ENTER API KEY ----------------
if not st.session_state.api_key:
    st.title("🔑 Welcome to FayyazAI")
    st.write("Enter your **Groq API key** to continue.")
    api_key_input = st.text_input("Groq API Key:", type="password")

    if st.button("Submit"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.rerun()  # ✅ updated
        else:
            st.warning("Please enter a valid API key.")

# ---------------- SCREEN 2: MAIN APP ----------------
else:
    st.title("🤖 FayyazAI")
    st.write("Chat with your PDFs intelligently.")

    # Initialize LLM
    llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="Gemma2-9b-It")

    # Internal session ID
    session_id = st.session_state.session_id

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Chat prompt at the top
    user_input = st.text_input("💬 Ask FayyazAI something:")

    # PDF upload below
    uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextualize prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            st.write("🤖 **FayyazAI:**", response['answer'])
    
