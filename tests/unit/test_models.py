import pytest
from pydantic import ValidationError

from app.models import ChatRequest, ChatResponse, Message


def test_chat_request_valid():
    req = ChatRequest(session_id="s1", message="hello")
    assert req.session_id == "s1"
    assert req.message == "hello"
    assert req.model is None


def test_chat_request_with_model_override():
    req = ChatRequest(session_id="s1", message="hello", model="llama3")
    assert req.model == "llama3"


def test_chat_request_rejects_empty_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="", message="hello")


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_message_accepts_valid_roles(role: str):
    msg = Message(role=role, content="test")
    assert msg.role == role


def test_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="invalid", content="test")


def test_chat_response_serialises():
    msg = Message(role="assistant", content="hi")
    resp = ChatResponse(session_id="s1", model="m1", message=msg)
    data = resp.model_dump()
    assert data["session_id"] == "s1"
    assert data["model"] == "m1"
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "hi"
