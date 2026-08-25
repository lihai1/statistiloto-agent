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

# ── Clean ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .uv

# ── Ingest product docs into RAG ─────────────────────────────
ingest-docs:
	.venv/bin/python -m app.rag.ingest

ingest-docs-force:
	.venv/bin/python -m app.rag.ingest --force
