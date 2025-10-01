# FayyazAI - Chat with PDF or Search (Hybrid RAG + Agent)
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

# Tools for search mode
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
import torch
from dotenv import load_dotenv

# ------------------- Environment Setup -------------------
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Auto device detection (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embeddings (force to detected device)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="FayyazAI", page_icon="🧠")

# Session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm FayyazAI 🧠. Upload a PDF or just ask me anything!"}
    ]

# ------------------- SCREEN 1: API KEY -------------------
if not st.session_state.api_key:
    st.title("🔑 Welcome to FayyazAI")
    st.write("Enter your **Groq API key** to continue.")
    api_key_input = st.text_input("Groq API Key:", type="password")

    if st.button("Submit"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.rerun()
        else:
            st.warning("Please enter a valid API key.")

# ------------------- SCREEN 2: MAIN APP -------------------
else:
    st.title("🧠 FayyazAI")
    st.write("Chat with your **PDFs** or search the **web** if no PDF is uploaded.")

    # Initialize LLM
    llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="Gemma2-9b-It")

    # Sidebar PDF uploader
    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

    # Show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if prompt := st.chat_input("💬 Ask FayyazAI something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # ------------------- If PDFs Uploaded → Use RAG -------------------
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = f"./temp.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)

            # Split and embed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Contextualize prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question. Do NOT answer it."
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
                "Use the following retrieved context to answer. "
                "If you don't know, say so. Keep it concise (max 3 sentences)."
                "\n\n{context}"
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
                if st.session_state.session_id not in st.session_state:
                    st.session_state[st.session_state.session_id] = ChatMessageHistory()
                return st.session_state[st.session_state.session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            answer = response["answer"]

        # ------------------- If No PDF → Use Search Agent -------------------
        else:
            arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
            search = DuckDuckGoSearchRun(name="Search")

            tools = [search, arxiv, wiki]
            search_agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                answer = search_agent.run(prompt, callbacks=[st_cb])

        # ------------------- Show Assistant Answer -------------------
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
