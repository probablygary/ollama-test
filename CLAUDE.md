## Project
Local LLM chat agent with persistent conversation history.

## Stack
- FastAPI + Pydantic for the API layer
- Ollama Python client for local model inference - model should be configurable
- SQLite for conversation history persistence
- Custom FastAPI middleware for input/output guardrails
- pytest for unit and integration tests

## Architecture
- POST /chat — accepts user message, returns model response
- Conversation history stored per session_id
- Guardrails layer validates input before hitting Ollama, validates output before returning
- No LangChain — direct Ollama API calls only

## Conventions
- Python 3.14.*
- Always apply Black formatting
- Always maintain SWE best practices, e.g.
    - maximise coherence
    - minimise coupling
    - optimise for maintainability and readability
    - use modularisation
    - minimise duplication
- Adhere to FastAPI best practices, recommended structure, and utilise its features where possible
- Type hints everywhere, Pydantic models for all request/response schemas
- Tests in /tests, split into unit/ and integration/
- Integration tests mock the Ollama client, don't require a live model
- Use test-driven development