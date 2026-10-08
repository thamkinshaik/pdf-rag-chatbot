"""FastAPI backend for the PDF RAG chatbot.

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI)
Needs: GOOGLE_API_KEY set as an environment variable.
"""

import os
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.8-flash"  # same model name as app.py

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""

app = FastAPI(title="PDF RAG API", version="1.0")

_state = {"index": None, "embeddings": None}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=8)


class Source(BaseModel):
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


def get_embeddings():
    if _state["embeddings"] is None:
        _state["embeddings"] = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _state["embeddings"]


def generate_answer(context: str, question: str) -> str:
    """Call Gemini. Kept separate so tests can mock it."""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    raw = llm.invoke(PROMPT.format(context=context, question=question)).content
    if isinstance(raw, str):
        return raw
    return "".join(b.get("text", "") for b in raw if isinstance(b, dict))


@app.get("/health")
def health():
    return {"status": "ok", "indexed": _state["index"] is not None}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Load PDFs -> chunk (800/100) -> embed -> FAISS index."""
    docs = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read())
            path = tmp.name
        try:
            docs.extend(PyPDFLoader(path).load())
        finally:
            os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in the PDFs.")
    _state["index"] = FAISS.from_documents(chunks, get_embeddings())
    return {"files": len(files), "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _state["index"] is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first via /upload.")
    hits = _state["index"].similarity_search(req.question, k=req.top_k)
    context = "\n\n".join(h.page_content for h in hits)
    answer = generate_answer(context, req.question)
    sources = [
        Source(page=int(h.metadata.get("page", 0)) + 1, text=h.page_content[:300])
        for h in hits
    ]
    return AskResponse(answer=answer, sources=sources)
