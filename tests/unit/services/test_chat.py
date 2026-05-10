from unittest.mock import MagicMock

import pytest

from app.models import Message
from app.ollama_client import ChatResult
from app.services.chat import ChatService


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def service(mock_client: MagicMock) -> ChatService:
    return ChatService(mock_client)


def _stub_result(model: str = "m", content: str = "reply") -> ChatResult:
    return ChatResult(model=model, message=Message(role="assistant", content=content))


def test_send_returns_chat_result(service: ChatService, mock_client: MagicMock):
    mock_client.chat.return_value = _stub_result()
    result = service.send(session_id="s1", message="hello")
    assert isinstance(result, ChatResult)


def test_send_passes_user_message(service: ChatService, mock_client: MagicMock):
    mock_client.chat.return_value = _stub_result()
    service.send(session_id="s1", message="hello")
    messages = mock_client.chat.call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "hello"


def test_send_passes_model_override(service: ChatService, mock_client: MagicMock):
    mock_client.chat.return_value = _stub_result(model="override")
    service.send(session_id="s1", message="hello", model="override")
    assert mock_client.chat.call_args.kwargs["model"] == "override"


def test_send_passes_none_model_by_default(service: ChatService, mock_client: MagicMock):
    mock_client.chat.return_value = _stub_result()
    service.send(session_id="s1", message="hello")
    assert mock_client.chat.call_args.kwargs["model"] is None
