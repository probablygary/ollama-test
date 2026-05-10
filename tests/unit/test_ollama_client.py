from unittest.mock import MagicMock, patch

import pytest

from app.models import Message
from app.ollama_client import ChatResult, OllamaClient


def _mock_response(role: str = "assistant", content: str = "Hi") -> MagicMock:
    response = MagicMock()
    response.message.role = role
    response.message.content = content
    return response


@pytest.fixture
def raw_client() -> MagicMock:
    """Patched ollama.Client instance injected into OllamaClient."""
    with patch("app.ollama_client.Client") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


def test_chat_returns_chat_result(raw_client: MagicMock):
    raw_client.chat.return_value = _mock_response(content="Hello!")
    client = OllamaClient(model="test-model")
    result = client.chat(messages=[Message(role="user", content="Hi")])
    assert isinstance(result, ChatResult)
    assert result.model == "test-model"
    assert result.message.role == "assistant"
    assert result.message.content == "Hello!"


def test_chat_passes_serialised_messages(raw_client: MagicMock):
    raw_client.chat.return_value = _mock_response()
    client = OllamaClient(model="test-model")
    client.chat(messages=[Message(role="user", content="Hi")])
    raw_client.chat.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )


def test_chat_model_override_used(raw_client: MagicMock):
    raw_client.chat.return_value = _mock_response()
    client = OllamaClient(model="default-model")
    result = client.chat(
        messages=[Message(role="user", content="Hi")],
        model="override-model",
    )
    assert result.model == "override-model"
    raw_client.chat.assert_called_once_with(
        model="override-model",
        messages=[{"role": "user", "content": "Hi"}],
    )


def test_chat_propagates_upstream_exception(raw_client: MagicMock):
    raw_client.chat.side_effect = RuntimeError("connection refused")
    client = OllamaClient(model="test-model")
    with pytest.raises(RuntimeError, match="connection refused"):
        client.chat(messages=[Message(role="user", content="Hi")])
