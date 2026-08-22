"""Integration tests package.

These tests use a real PostgreSQL (pgvector) from docker-compose-dev.yml
but mock the LLM (FakeListChatModel) and mock external services (Go gRPC,
Java BFF). They test the full agent flow: FastAPI → LangGraph → RAG →
metering → HITL.
"""
