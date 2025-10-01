# FayyazAI - Chat with PDF or Search or Translate (Hybrid RAG + Agent + Memory)
import streamlit as st
from dotenv import load_dotenv
import os
import torch

# LangChain / tools
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
from langchain.schema import HumanMessage, AIMessage

# Tools for search mode
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# ---------------- Environment & embeddings ----------------
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Auto device detection (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embeddings (force to detected device)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="FayyazAI", page_icon="🧠")

# session store for RunnableWithMessageHistory (separate from st.session_state)
_store = {}

def get_session_history_store(session_id: str) -> BaseChatMessageHistory:
    """Return/chat history object for RunnableWithMessageHistory (stored in _store)."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# ---------------- session_state initialization ----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# hidden session id (internal)
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# messages for UI chat display (list of dicts {role,content})
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I'm FayyazAI. How can I help you today?"},
        {"role": "assistant", "content": "You can: 1) Upload a PDF  2) Search the web  3) Translate English to another language"},
    ]

# ui state: track current chosen mode
if "mode" not in st.session_state:
    st.session_state.mode = None  # values: "pdf", "search", "translate", None

# temporary storage for loaded vectorstore per session to avoid reprocessing repeatedly (simple caching)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

# ---------------- SCREEN 1: API KEY ----------------
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
    st.stop()  # stop further rendering until key provided

# ---------------- MAIN APP ----------------
st.title("🧠 FayyazAI")
st.write("Choose an action below or just type — I'll route the request appropriately.")

# Initialize LLM (single LLM used for agent and RAG responses)
llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="Gemma2-9b-It")

# Show chat history UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Offer quick-action buttons (assistant prompt)
col1, col2, col3 = st.columns(3)
if col1.button("📄 Upload PDF"):
    st.session_state.mode = "pdf"
if col2.button("🌐 Search web"):
    st.session_state.mode = "search"
if col3.button("🔁 Translate"):
    st.session_state.mode = "translate"

st.sidebar.markdown("### Options")
st.sidebar.markdown("Mode is auto-detected (or use buttons above).")

# Sidebar file uploader (so it's accessible anytime)
uploaded_files = st.sidebar.file_uploader("📂 Upload PDF files (optional)", type="pdf", accept_multiple_files=True)

# If user selected mode explicitly, show a small note
if st.session_state.mode:
    st.info(f"Selected mode: **{st.session_state.mode}**")

# If PDFs are uploaded, set mode to pdf automatically
if uploaded_files:
    st.session_state.mode = "pdf"

# Input area depends on mode
if st.session_state.mode == "translate":
    text_to_translate = st.text_area("Enter English text to translate:", "")
    target_lang = st.text_input("Target language (e.g. French, Hindi, Spanish):", "French")

    if st.button("Translate"):
        if not text_to_translate.strip():
            st.warning("Please enter text to translate.")
        else:
            # system + user + memory
            translate_system = (
                f"You are a helpful translator. Translate the user's English text into {target_lang}. "
                "Keep meaning and tone."
            )
            translate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", translate_system),
                    MessagesPlaceholder("chat_history"),   # <- real memory
                    ("human", "{input}")
                ]
            )

            translate_chain = translate_prompt | llm

            # Wrap in RunnableWithMessageHistory
            translate_with_history = RunnableWithMessageHistory(
                translate_chain,
                get_session_history_store,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            # Add user message to UI chat
            st.session_state.messages.append(
                {"role": "user", "content": f"Translate to {target_lang}: {text_to_translate}"}
            )
            st.chat_message("user").write(text_to_translate)

            # Run the chain with memory
            response = translate_with_history.invoke(
                {"input": text_to_translate},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )

            # Extract final text
            translated = response.content if hasattr(response, "content") else str(response)

            st.session_state.messages.append({"role": "assistant", "content": translated})
            st.chat_message("assistant").write(translated)


elif st.session_state.mode == "search":
    # show search input
    user_query = st.text_input("Search query or question:", "")
    if st.button("Search"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            # prepare agent tools and agent
            arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            search = DuckDuckGoSearchRun(name="Search")

            tools = [search, arxiv, wiki]
            # initialize agent with llm
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                answer = agent.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)

else:
    # Default / PDF mode UI: show prompt at top and PDF uploader below
    user_input = st.text_input("Ask FayyazAI something (or choose a mode above):", "")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # If pdfs uploaded => RAG flow
        if uploaded_files:
            # Check if we've already processed this set of uploaded filenames in this session to avoid re-embedding every query
            filenames = [f.name for f in uploaded_files]
            need_process = filenames != st.session_state.uploaded_filenames or st.session_state.vectorstore is None

            documents = []
            for uploaded_file in uploaded_files:
                temppdf = "./temp.pdf"
                with open(temppdf, "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)

            if need_process:
                # Split and embed once per upload set
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.uploaded_filenames = filenames
            else:
                vectorstore = st.session_state.vectorstore

            retriever = vectorstore.as_retriever()

            # create contextualize prompt for history-aware retrieval
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history, "
                "formulate a standalone question which can be understood without the chat history. "
                "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # history-aware retriever (uses llm)
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # QA prompt for final answer
            system_prompt = (
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you don't know. Keep answers concise (max 3 sentences)."
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

            # Wrap rag_chain with runnable history so it will keep / read conversation history per session
            rag_with_history = RunnableWithMessageHistory(
                rag_chain, get_session_history_store,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # invoke RAG runnable
            response = rag_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            answer = response["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

        # No PDFs => fallback to search agent (auto)
        else:
            arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            search = DuckDuckGoSearchRun(name="Search")

            tools = [search, arxiv, wiki]
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                answer = agent.run(user_input, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)
