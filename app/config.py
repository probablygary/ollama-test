from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    model: str = "huihui_ai/qwen3-abliterated:4b"
    host: str | None = None


settings = Settings()
