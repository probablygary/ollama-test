from fastapi import FastAPI

from app.routes.chat import router as chat_router


def create_app() -> FastAPI:
    app = FastAPI(title="Local LLM Chat Agent", version="0.1.0")
    app.include_router(chat_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
