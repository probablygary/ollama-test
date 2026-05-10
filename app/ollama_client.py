from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ollama import Client

from app.config import settings
from app.models import Message


@dataclass(frozen=True)
class ChatResult:
    model: str
    message: Message


class OllamaClient:
    """Thin wrapper around the Ollama Python client.

    Keeps transport details (host, model defaults) out of the route layer so
    callers only deal with `Message` objects.
    """

    def __init__(
        self,
        model: str = settings.model,
        host: str | None = settings.host,
    ) -> None:
        self._model = model
        self._client = Client(host=host) if host else Client()

    @property
    def default_model(self) -> str:
        return self._model

    def chat(
        self,
        messages: Iterable[Message],
        model: str | None = None,
    ) -> ChatResult:
        target = model or self._model
        payload = [m.model_dump() for m in messages]
        response = self._client.chat(model=target, messages=payload)
        reply = Message(
            role=response.message.role,
            content=response.message.content,
        )
        return ChatResult(model=target, message=reply)
