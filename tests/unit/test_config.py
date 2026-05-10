from app.config import Settings


def test_has_default_model():
    s = Settings()
    assert s.model is not None


def test_default_host_is_none():
    s = Settings()
    assert s.host is None


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    s = Settings()
    assert s.model == "llama3"


def test_host_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    s = Settings()
    assert s.host == "http://localhost:11434"
