from app.models import Message
from app.ollama_client import ChatResult, OllamaClient


class ChatService:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    def send(
        self,
        session_id: str,
        message: str,
        model: str | None = None,
    ) -> ChatResult:
        messages = [Message(role="user", content=message)]
        return self._client.chat(messages=messages, model=model)
