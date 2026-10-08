"""Tests for the FastAPI backend. Run: pytest -q
Gemini and the embedding model are mocked, so no API key or download is needed.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def reset_state():
    api._state["index"] = None
    yield
    api._state["index"] = None


def test_health_reports_not_indexed():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "indexed": False}


def test_ask_without_upload_returns_400():
    r = client.post("/ask", json={"question": "What is this?"})
    assert r.status_code == 400


def test_ask_validates_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_upload_rejects_non_pdf():
    r = client.post("/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


def test_ask_returns_answer_and_sources(monkeypatch):
    fake_hit = SimpleNamespace(page_content="Siemens makes turbines.", metadata={"page": 2})
    fake_index = SimpleNamespace(similarity_search=lambda q, k: [fake_hit])
    api._state["index"] = fake_index
    monkeypatch.setattr(api, "generate_answer", lambda ctx, q: "Turbines.")

    r = client.post("/ask", json={"question": "What does Siemens make?", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Turbines."
    assert body["sources"][0]["page"] == 3  # 0-indexed page 2 -> page 3
