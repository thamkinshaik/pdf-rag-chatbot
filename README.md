# PDF Question-Answering Chatbot (RAG)

A chatbot that answers questions from your own PDF documents using Retrieval-Augmented Generation (RAG).

## How it works
1. **Load**: PDFs are read with LangChain's `PyPDFLoader`.
2. **Chunk**: Text is split into overlapping chunks (800 chars, 100 overlap).
3. **Embed**: Chunks are converted to vectors with a Hugging Face model (`all-MiniLM-L6-v2`).
4. **Store**: Vectors are stored in a **FAISS** index.
5. **Retrieve**: For each question, the top-k most similar chunks are fetched.
6. **Generate**: The chunks + question are sent to the **Gemini LLM API**, which answers only from that context.
7. Sources (page numbers and text) are shown under each answer.

## Tech stack
Python, LangChain, FAISS, Hugging Face Sentence Transformers, Google Gemini API, Streamlit

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Get a free API key from https://aistudio.google.com/app/apikey, then:

```bash
streamlit run app.py
```

Paste the API key in the sidebar, upload a PDF, click **Process documents**, and start asking questions.

## Ideas to extend
- Add conversation memory (follow-up questions)
- Swap FAISS for Pinecone or Chroma

- ## REST API (FastAPI)

The same RAG pipeline is exposed as a backend API.

```
pip install fastapi uvicorn python-multipart pytest httpx
set GOOGLE_API_KEY=your_key
uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Service status and whether a PDF is indexed |
| `/upload` | POST | Upload PDF(s): chunk (800/100), embed, build FAISS index |
| `/ask` | POST | `{"question": "...", "top_k": 4}` returns answer + source pages |

## Tests

```
pytest -q
```

Gemini and embeddings are mocked, so tests run without an API key.
- Try OpenAI / Anthropic / Hugging Face models instead of Gemini
- Add evaluation of answer quality
