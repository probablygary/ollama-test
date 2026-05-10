from fastapi import APIRouter, Depends, HTTPException, status

from app.models import ChatRequest, ChatResponse
from app.ollama_client import OllamaClient
from app.services.chat import ChatService

router = APIRouter()

_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def get_chat_service(client: OllamaClient = Depends(get_client)) -> ChatService:
    return ChatService(client)


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        result = service.send(
            session_id=req.session_id,
            message=req.message,
            model=req.model,
        )
    except Exception as exc:  # noqa: BLE001 - surface upstream failures as 502
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ollama call failed: {exc}",
        ) from exc

    return ChatResponse(
        session_id=req.session_id,
        model=result.model,
        message=result.message,
    )
