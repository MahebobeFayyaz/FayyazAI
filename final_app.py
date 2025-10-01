# FayyazAI - Chat with PDF, Search, Translate, Math Solver, or Summarize URL
import streamlit as st
from dotenv import load_dotenv
import os
import torch
import validators

# LangChain / tools
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
from langchain.chains.summarize import load_summarize_chain

# Tools for search & agents
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, LLMChain

# ---------------- Environment & embeddings ----------------
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
else:
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Auto device detection (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embeddings (force to detected device)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)


# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="FayyazAI", page_icon="🧠")

# session store for RunnableWithMessageHistory
_store = {}
def get_session_history_store(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

# ---------------- session_state initialization ----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I'm FayyazAI. How can I help you today?"}         
    ]
if "mode" not in st.session_state:
    st.session_state.mode = None
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
    st.stop()

# ---------------- MAIN APP ----------------
st.title("🧠 FayyazAI")
st.write("Choose an action below or just type — I'll route the request appropriately.")

# Initialize LLM
llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="Gemma2-9b-It")

# Show chat history UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Quick-action buttons
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("📄 Upload PDF"):
    st.session_state.mode = "pdf"
if col2.button("🌐 Search web"):
    st.session_state.mode = "search"
if col3.button("🔁 Translate"):
    st.session_state.mode = "translate"
if col4.button("🧮 Math Solver"):
    st.session_state.mode = "math"
if col5.button("🦜 Summarize URL"):
    st.session_state.mode = "summarize"

# Sidebar
st.sidebar.markdown("### Options")
uploaded_files = st.sidebar.file_uploader("📂 Upload PDF files (optional)", type="pdf", accept_multiple_files=True)

# Set mode auto
if uploaded_files:
    st.session_state.mode = "pdf"

# If user selected mode explicitly, show a small note
if st.session_state.mode:
    st.info(f"Selected mode: **{st.session_state.mode}**")

# ---------------- MODE HANDLERS ----------------

# TRANSLATE MODE
if st.session_state.mode == "translate":
    text_to_translate = st.text_area("Enter English text to translate:", "")
    target_lang = st.text_input("Target language (e.g. French, Hindi, Spanish):", "French")

    if st.button("Translate"):
        if not text_to_translate.strip():
            st.warning("Please enter text to translate.")
        else:
            translate_system = f"You are a helpful translator. Translate into {target_lang}. Keep meaning and tone."
            translate_prompt = ChatPromptTemplate.from_messages([
                ("system", translate_system),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            translate_chain = translate_prompt | llm
            translate_with_history = RunnableWithMessageHistory(
                translate_chain, get_session_history_store,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            st.session_state.messages.append({"role": "user", "content": f"Translate to {target_lang}: {text_to_translate}"})
            st.chat_message("user").write(text_to_translate)

            response = translate_with_history.invoke(
                {"input": text_to_translate},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            translated = response.content if hasattr(response, "content") else str(response)
            st.session_state.messages.append({"role": "assistant", "content": translated})
            st.chat_message("assistant").write(translated)

# SEARCH MODE
elif st.session_state.mode == "search":
    user_query = st.text_input("Search query or question:", "")
    if st.button("Search"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            search = DuckDuckGoSearchRun(name="Search")

            tools = [search, arxiv, wiki]
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                answer = agent.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)

# MATH MODE
elif st.session_state.mode == "math":
    st.subheader("🧮 Math Problem Solver & Reasoning Assistant")
    question = st.text_area(
        "Enter your math or logic question:",
        "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
        "Then I buy a dozen apples and 2 packs of blueberries. Each pack has 25 berries. "
        "How many total pieces of fruit do I have at the end?"
    )

    if st.button("Solve"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            wikipedia_wrapper = WikipediaAPIWrapper()
            wikipedia_tool = Tool(name="Wikipedia", func=wikipedia_wrapper.run, description="Search Wikipedia for facts")

            math_chain = LLMMathChain.from_llm(llm=llm)
            calculator = Tool(name="Calculator", func=math_chain.run, description="Answer math expressions")

            prompt = """
            You are an agent for solving mathematical or reasoning problems.
            Arrive at the solution step by step, explain clearly, and present the final answer.
            Question: {question}
            Answer:
            """
            reasoning_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["question"], template=prompt))
            reasoning_tool = Tool(name="Reasoning Tool", func=reasoning_chain.run, description="Answer logic questions")

            agent = initialize_agent([wikipedia_tool, calculator, reasoning_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent.run(question, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write("### Response:")
                st.success(response)

# SUMMARIZE MODE (only URL, no YouTube)
elif st.session_state.mode == "summarize":
    st.subheader("🦜 Summarize Website URL")
    generic_url = st.text_input("Enter a Website URL", "")

    if st.button("Summarize"):
        if not st.session_state.api_key.strip() or not generic_url.strip():
            st.error("Please provide your Groq API key and a URL")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL")
        else:
            try:
                with st.spinner("Fetching and summarizing..."):
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                    prompt_template = """
                    Provide a summary of the following content in about 300 words:
                    Content: {text}
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.session_state.messages.append({"role": "assistant", "content": output_summary})
                    st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception: {e}")

# DEFAULT / PDF MODE
else:
    user_input = st.text_input("Ask FayyazAI something (or choose a mode above):", "")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        if uploaded_files:  # PDF mode
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
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.uploaded_filenames = filenames
            else:
                vectorstore = st.session_state.vectorstore

            retriever = vectorstore.as_retriever()
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Rephrase the user query to be standalone using chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer based on retrieved context. Be concise (max 3 sentences).\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            rag_with_history = RunnableWithMessageHistory(
                rag_chain, get_session_history_store,
                input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
            )

            response = rag_with_history.invoke({"input": user_input}, config={"configurable": {"session_id": st.session_state.session_id}})
            answer = response["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

        else:  # fallback to search
            arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
            search = DuckDuckGoSearchRun(name="Search")
            tools = [search, arxiv, wiki]
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                answer = agent.run(user_input, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)
