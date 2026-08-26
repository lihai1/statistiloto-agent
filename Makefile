.PHONY: dev test proto lint clean install

# ── Install dependencies (local venv) ────────────────────────
install:
	uv venv && uv pip install -e ".[dev]"

# ── Generate Python gRPC stubs from shared proto ─────────────
proto:
	@mkdir -p app/gen
	python -m grpc_tools.protoc \
		-I../proto \
		-I../proto/third_party \
		--python_out=app/gen \
		--grpc_python_out=app/gen \
		--pyi_out=app/gen \
		../proto/lottery.proto
	@touch app/gen/__init__.py
	@echo "Generated gRPC stubs in app/gen/"

# ── Dev: bring up docker-compose-dev.yml (Ollama + pgvector DB) ──
dev:
	docker compose -f docker-compose-dev.yml up -d
	@echo "Dev stack up. Agent at http://localhost:8000"

dev-stop:
	docker compose -f docker-compose-dev.yml down

# ── Run tests ────────────────────────────────────────────────
test:
	pytest -v

test-integration:
	pytest -v -m integration

test-unit:
	pytest -v -m "not integration"

# ── E2E chat flow tests (real Ollama LLM) ─────────────────────
# Requires Ollama reachable at OLLAMA_BASE_URL (default: localhost:11434).
# The model is auto-discovered from the provider's /api/tags endpoint.
# If Ollama isn't on localhost but the root stack container is running,
# a temporary socat proxy is started and cleaned up afterward.
# The DB on port 5433 must already be running (root stack or dev compose).
OLLAMA_PROXY_NAME ?= ollama-test-proxy
OLLAMA_NETWORK ?= statistiloto-new_statistiloto-net
OLLAMA_CONTAINER ?= statistiloto-ollama

test-chat-flows:
	@URL="http://localhost:11434"; \
	if curl -s --connect-timeout 2 $$URL/api/tags >/dev/null 2>&1; then \
		echo "[test-chat-flows] Ollama reachable at $$URL"; \
	elif docker ps --format '{{.Names}}' | grep -q '^$(OLLAMA_CONTAINER)$$'; then \
		echo "[test-chat-flows] Proxying to $(OLLAMA_CONTAINER) via socat"; \
		docker rm -f $(OLLAMA_PROXY_NAME) 2>/dev/null; \
		docker run -d --name $(OLLAMA_PROXY_NAME) --network $(OLLAMA_NETWORK) \
			-p 11434:11434 alpine/socat \
			TCP-LISTEN:11434,fork,reuseaddr TCP:$(OLLAMA_CONTAINER):11434; \
		for i in $$(seq 1 10); do curl -s --connect-timeout 1 $$URL/api/tags >/dev/null 2>&1 && break; sleep 1; done; \
		if ! curl -s --connect-timeout 2 $$URL/api/tags >/dev/null 2>&1; then \
			echo "[test-chat-flows] ERROR: Ollama not reachable"; docker rm -f $(OLLAMA_PROXY_NAME); exit 1; fi; \
		trap 'docker rm -f $(OLLAMA_PROXY_NAME) 2>/dev/null' EXIT; \
	else \
		echo "[test-chat-flows] ERROR: Ollama not found. Start it with: docker compose -f docker-compose-dev.yml up -d ollama"; exit 1; \
	fi; \
	OLLAMA_BASE_URL=$$URL .venv/bin/python -m pytest tests/integration/test_chat_flows.py -v --tb=short

# ── Clean ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .uv

# ── Ingest product docs into RAG ─────────────────────────────
ingest-docs:
	.venv/bin/python -m app.rag.ingest

ingest-docs-force:
	.venv/bin/python -m app.rag.ingest --force
