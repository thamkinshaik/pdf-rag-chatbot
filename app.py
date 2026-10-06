"""PDF Question-Answering Chatbot using RAG.

Stack: Python, LangChain, Hugging Face embeddings (sentence-transformers),
FAISS vector store, Google Gemini LLM API, Streamlit UI.
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.8-flash"

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot (RAG)")


@st.cache_resource
def get_embeddings():
    # Hugging Face model, runs locally on CPU, no API key needed
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index(uploaded_files):
    """Load PDFs -> split into chunks -> embed -> store in FAISS."""
    docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            path = tmp.name
        docs.extend(PyPDFLoader(path).load())
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings()), len(chunks)


with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google API key (Gemini)",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
    )
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Chunks retrieved (k)", 1, 8, 4)
    if st.button("Process documents") and files:
        with st.spinner("Building vector index..."):
            st.session_state.index, n = build_index(files)
            st.session_state.messages = []
        st.success(f"Indexed {n} chunks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your document")
if question:
    if "index" not in st.session_state:
        st.warning("Upload PDFs and click 'Process documents' first.")
    elif not api_key:
        st.warning("Enter your Google API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieval: top-k most similar chunks
        hits = st.session_state.index.similarity_search(question, k=top_k)
        context = "\n\n".join(h.page_content for h in hits)

        # Generation: LLM answers from retrieved context
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key)
        raw = llm.invoke(PROMPT.format(context=context, question=question)).content
        answer = raw if isinstance(raw, str) else "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Sources"):
                for h in hits:
                    page = h.metadata.get("page", "?")
                    st.caption(f"Page {page + 1 if isinstance(page, int) else page}")
                    st.write(h.page_content[:300] + "...")
        st.session_state.messages.append({"role": "assistant", "content": answer})
