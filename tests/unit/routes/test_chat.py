from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Message
from app.ollama_client import ChatResult
from app.routes.chat import get_chat_service
from app.services.chat import ChatService


def _stub_service(content: str = "Hello!", model: str = "test-model") -> MagicMock:
    service = MagicMock(spec=ChatService)
    service.send.return_value = ChatResult(
        model=model,
        message=Message(role="assistant", content=content),
    )
    return service


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _client(service: MagicMock) -> TestClient:
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app)


def test_chat_returns_200():
    response = _client(_stub_service()).post(
        "/chat", json={"session_id": "s1", "message": "Hello"}
    )
    assert response.status_code == 200


def test_chat_response_shape():
    response = _client(_stub_service(content="Hi!", model="test-model")).post(
        "/chat", json={"session_id": "s1", "message": "Hello"}
    )
    data = response.json()
    assert data["session_id"] == "s1"
    assert data["model"] == "test-model"
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "Hi!"


def test_chat_502_when_service_raises():
    service = MagicMock(spec=ChatService)
    service.send.side_effect = RuntimeError("ollama down")
    response = _client(service).post(
        "/chat", json={"session_id": "s1", "message": "Hello"}
    )
    assert response.status_code == 502
    assert "ollama down" in response.json()["detail"]


def test_chat_422_on_empty_message():
    response = _client(_stub_service()).post(
        "/chat", json={"session_id": "s1", "message": ""}
    )
    assert response.status_code == 422


def test_chat_422_on_empty_session_id():
    response = _client(_stub_service()).post(
        "/chat", json={"session_id": "", "message": "hello"}
    )
    assert response.status_code == 422
