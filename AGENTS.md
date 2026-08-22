# Agent Service — Statistiloto

Python LangGraph agent worker with hierarchical multi-agent orchestration (Option C).

## Architecture

- **Supervisor graph** routes by tier + intent to three worker subgraphs:
  - `nl_assistant` — NL lottery assistant (free/paid/admin)
  - `analyst` — multi-step analysis with tool calling + HITL on write tools (paid/admin)
  - `admin_ops` — admin operations with HITL on write tools (admin only)
- **Single global LLM** for all tiers — Ollama in dev, configurable at runtime by admin
- **Admin is the owner/developer super-user**, NOT a paid tier — no budget limit, sees all users' data
- **RAG** with pgvector — role-scoped corpora, per-tenant `user_data` filtering (admin bypasses user_sub filter)
- **HITL** triggers automatically on ANY write tool call (save_numbers, trigger_scraper) — not configurable per tier, not based on confidence
- **Token metering** — every LLM call logged to `agent.token_usage`; only admin can read token consumption
- **Recursion limit** — per-tier cap on graph super-steps (free=6, paid=25, admin=50) wired into `graph.invoke()`
- **gRPC** to Go lottery-stats-server via generated stubs from `proto/lottery.proto`

## Tool classification (HITL gating)

Write tools (require human approval before executing):
- `save_numbers` — POST to Java BFF, writes saved numbers to DB
- `trigger_scraper` — triggers Go scraper, writes lottery draw data

Read-only tools (execute without HITL):
- `generate_form` — Go gRPC, computes forms
- `get_statistics` — Go gRPC, reads statistics
- `analyze` — Go gRPC, reads historical matches
- `list_saved_numbers` — Java BFF GET, reads saved numbers
- `query_audit_log` — DB SELECT, reads audit log
- `read_token_usage` — DB SELECT, reads token stats

Classification is in `app/tools/registry.py` — `WRITE_TOOLS` and `READ_TOOLS` frozensets.

## Tier capabilities

| Capability | Free | Paid | Admin (owner) |
|---|---|---|---|
| Workers | nl_assistant only | nl_assistant, analyst | all three |
| RAG corpora | docs | docs, lottery_history | docs, lottery_history, user_data (all users), ops_logs |
| Write tools | none | save_numbers | save_numbers, trigger_scraper |
| Read tools | generate_form, get_statistics, analyze | + list_saved_numbers | + query_audit_log, read_token_usage |
| HITL | never (no write tools) | on save_numbers | on save_numbers, trigger_scraper |
| Daily budget | $0 (no limit) | $5.00 | $0 (no limit — owner) |
| Recursion limit | 6 | 25 | 50 |

## Development

### Prerequisites

- Python 3.12+ (use `uv` for venv management)
- Docker + Docker Compose (for pgvector DB)

### Setup

```bash
make install          # create venv + install deps
make proto            # generate gRPC stubs from shared proto
```

### Run dev stack

```bash
make dev              # docker-compose-dev.yml: pgvector DB + Ollama + agent
```

### Tests

Integration tests use a real pgvector PostgreSQL (from docker-compose-dev.yml)
with a mock LLM (FakeListChatModel), mock embeddings, and mock tool clients.
The DB must be running:

```bash
docker compose -f docker-compose-dev.yml up -d db
make test             # all tests (unit + integration) — 70 tests
make test-unit        # unit tests only (no DB needed) — 34 tests
make test-integration # integration tests only (needs DB) — 36 tests
```

### Key environment variables

- `LLM_PROVIDER` — ollama | gemini (default: ollama)
- `LLM_MOCK` — true to use FakeListChatModel for tests
- `OLLAMA_BASE_URL` — Ollama API URL
- `OLLAMA_MODEL` — Ollama model name
- `DB_URI` — PostgreSQL connection string
- `JWT_VERIFY` — false to skip JWT signature verification (dev/test)
- `LOTTERY_GRPC_HOST` — Go lottery service host (empty = tools return empty)
- `BFF_BASE_URL` — Java BFF base URL (empty = saved_numbers tools return empty)

### Test conventions

- Unit tests: `tests/unit/` — no DB, no external services
- Integration tests: `tests/integration/` — real pgvector DB, mock LLM + mock tools
- Mock LLM is injected via `set_llm_store()` in conftest.py
- Mock embeddings injected via `set_embeddings_model()` in conftest.py
- Mock tool clients (lottery gRPC + saved_numbers) injected via `set_mock_client()` in conftest.py
- JWT tokens created via `create_test_jwt(sub=..., tier=...)` in `app/security.py`
