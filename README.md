# Statistiloto Agent

> Python LangGraph agent worker — hierarchical multi-agent orchestration with RAG, HITL, token metering, and gRPC to the Go lottery service.

## Tech Stack

- **Python 3.11–3.13** (3.12 recommended)
- **FastAPI** — HTTP API server (uvicorn)
- **LangGraph** — hierarchical multi-agent graph orchestration with PostgresSaver checkpointer
- **LangChain** — LLM abstraction (Ollama + Google Gemini providers)
- **pgvector** — vector similarity search for RAG (PostgreSQL extension)
- **gRPC** — generated stubs from `proto/lottery.proto` for the Go lottery service
- **Pydantic / pydantic-settings** — request/response models, config validation
- **structlog** — structured logging
- **SSE (sse-starlette)** — server-sent events for streaming chat responses

## Architecture

```mermaid
flowchart LR
    BFF["Java BFF"] -->|POST /chat| Agent["Agent Service<br/>FastAPI"]
    Agent -->|validate JWT| Auth["JWT/JWKS"]
    Agent --> Supervisor["Supervisor Graph"]
    Supervisor -->|route by tier + intent| NL["nl_assistant<br/>(free/paid/admin)"]
    Supervisor -->|route by tier + intent| Analyst["analyst<br/>(paid/admin)"]
    Supervisor -->|route by tier + intent| AdminOps["admin_ops<br/>(admin only)"]
    NL --> Tools["Tools"]
    Analyst --> Tools
    AdminOps --> Tools
    Tools -->|gRPC| Go["Go Lottery Service"]
    Tools -->|HTTP| BFF
    Tools -->|embeddings| Ollama["Ollama LLM"]
    NL --> RAG["pgvector RAG"]
    Analyst --> RAG
    AdminOps --> RAG
    RAG --> DB[("PostgreSQL<br/>pgvector")]
    Agent -->|metering| DB
    Agent -->|SSE events| BFF
```

The **supervisor graph** is the single choke point for tier gating. It routes by `tier` + `intent` to one of three worker subgraphs. Disallowed intents are downgraded to `nl_assistant`.

## Tier Capabilities

| Capability | Free | Paid | Admin (owner) |
|---|---|---|---|
| Workers | `nl_assistant` only | `nl_assistant`, `analyst` | all three |
| RAG corpora | `docs` | `docs`, `lottery_history` | `docs`, `lottery_history`, `user_data` (all users), `ops_logs` |
| Write tools | none | `save_numbers` | `save_numbers`, `trigger_scraper` |
| Read tools | `generate_form`, `get_statistics`, `analyze` | + `list_saved_numbers` | + `query_audit_log`, `read_token_usage` |
| HITL | never (no write tools) | on `save_numbers` | on `save_numbers`, `trigger_scraper` |
| Daily budget | $0 (no limit) | $5.00 | $0 (no limit — owner) |
| Recursion limit | 6 | 25 | 50 |

> **Admin** is the owner/developer super-user — NOT a paid tier. Admin has no budget limit and sees all users' data (the `user_data` RAG filter is bypassed for admin).

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | any authenticated user | Process a chat message. Returns `{response, thread_id}` or `{paused: true, thread_id}` for HITL. Uses SSE for streaming. |
| `POST` | `/approve` | any authenticated user | Resume a paused HITL thread with a human decision (`approved: bool`, optional `edited` value). |
| `GET` | `/healthz` | none | Health check — returns `{"status": "ok"}`. |
| `GET` | `/llm-config` | any authenticated user | Read the current global LLM config (provider, model, base_url, etc.). |
| `PUT` | `/llm-config` | admin only | Update the global LLM config. Hot-reloaded within ~10s via poller (or immediately via `force_refresh()`). |
| `GET` | `/token-usage` | admin only | Read aggregated token usage stats from `agent.token_usage`. |
| `GET` | `/audit-log` | admin only | Read recent audit log entries from the DB. |

## Tool Classification

Tools are classified in `app/tools/registry.py` as `WRITE_TOOLS` and `READ_TOOLS` frozensets. HITL approval is required before executing ANY write tool — this is automatic and not configurable per tier.

### Write Tools (require HITL)

| Tool | Backend | Description |
|---|---|---|
| `save_numbers` | Java BFF (POST) | Writes saved numbers to DB |
| `trigger_scraper` | Go service | Triggers Go scraper, writes lottery draw data |

### Read Tools (no HITL)

| Tool | Backend | Description |
|---|---|---|
| `generate_form` | Go gRPC | Computes lottery forms |
| `get_statistics` | Go gRPC | Reads statistics |
| `analyze` | Go gRPC | Reads historical matches |
| `list_saved_numbers` | Java BFF (GET) | Reads saved numbers |
| `query_audit_log` | DB SELECT | Reads audit log |
| `read_token_usage` | DB SELECT | Reads token stats |

## Project Structure

```
agent/
├── AGENTS.md                  # Architecture reference for AI agents
├── Dockerfile                 # Multi-stage build (Python 3.12-slim)
├── Makefile                   # install, proto, dev, test targets
├── docker-compose-dev.yml     # Ollama + pgvector DB + agent
├── pyproject.toml             # Dependencies + pytest config
├── proto/                     # Shared protobuf (symlinked from ../proto)
├── db/
│   └── init-agent.sql         # DB schema init (agent schema, pgvector)
├── app/
│   ├── main.py                # FastAPI app — endpoints, graph singleton
│   ├── security.py            # JWT validation (JWKS), tier extraction, admin guard
│   ├── metering.py            # Token metering decorator + daily budget check
│   ├── hitl.py                # Human-in-the-loop interrupt helpers
│   ├── checkpointer.py        # PostgresSaver checkpointer management
│   ├── config/
│   │   ├── settings.py        # YAML + env config, tier configs
│   │   └── agent.yaml         # Default config values
│   ├── graphs/
│   │   ├── supervisor.py      # Top-level router (tier + intent gating)
│   │   ├── nl_assistant.py    # NL lottery assistant subgraph
│   │   ├── analyst.py         # Multi-step analysis subgraph (RAG + tools)
│   │   └── admin_ops.py       # Admin operations subgraph (HITL on writes)
│   ├── tools/
│   │   ├── registry.py        # WRITE_TOOLS / READ_TOOLS classification
│   │   ├── lottery_grpc.py    # Go gRPC tool clients (generate_form, etc.)
│   │   ├── saved_numbers.py   # Java BFF saved numbers tools
│   │   └── admin_ops.py       # Audit log + token usage read tools
│   ├── rag/
│   │   ├── store.py           # psycopg3 connection pool (singleton)
│   │   ├── retriever.py       # Role-scoped, per-tenant pgvector retrieval
│   │   └── indexers.py        # Document indexing into pgvector
│   ├── llm/
│   │   ├── config_store.py    # Global LLM config store + hot-reload poller
│   │   └── router.py          # LLM provider routing (Ollama / Gemini / mock)
│   └── gen/                   # Generated gRPC stubs (from make proto)
│       ├── lottery_pb2.py
│       └── lottery_pb2_grpc.py
└── tests/
    ├── unit/                  # 34 unit tests (no DB, no external services)
    │   ├── test_config.py
    │   ├── test_security.py
    │   └── test_supervisor.py
    └── integration/           # 36 integration tests (real pgvector DB, mock LLM)
        ├── conftest.py        # Mock LLM, embeddings, tool clients, test JWT
        ├── test_free_user.py
        ├── test_paid_user.py
        ├── test_admin_user.py
        ├── test_health.py
        ├── test_llm_config.py
        ├── test_metering.py
        └── test_rag.py
```

## Quick Start

### Prerequisites

- Python 3.12+ (use [`uv`](https://github.com/astral-sh/uv) for venv management)
- Docker + Docker Compose (for pgvector DB + Ollama)

### Setup

```bash
make install          # create venv + install deps
make proto            # generate gRPC stubs from shared proto
```

### Run Dev Stack

```bash
make dev              # docker-compose-dev.yml: pgvector DB + Ollama + agent
# Agent at http://localhost:8000
```

To stop the dev stack:

```bash
make dev-stop
```

## Environment Configuration

Settings are loaded from `app/config/agent.yaml` with environment variable overrides applied on top (see `app/config/settings.py`).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama` \| `gemini` \| `mock` |
| `LLM_MOCK` | `false` | Set to `true` to use `FakeListChatModel` for tests |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `GEMINI_API_KEY` | _(empty)_ | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `DB_URI` | `postgresql://postgres:postgres@db:5432/statistiloto` | PostgreSQL connection string |
| `JWT_VERIFY` | `true` | Set to `false` to skip JWT signature verification (dev/test) |
| `JWKS_URL` | _(empty)_ | Keycloak JWKS URL for JWT verification |
| `ISSUER` | _(empty)_ | Expected JWT issuer |
| `AUDIENCE` | _(empty)_ | Expected JWT audience |
| `LOTTERY_GRPC_HOST` | `lottery` | Go lottery service host (empty = tools return empty) |
| `LOTTERY_GRPC_PORT` | `9090` | Go lottery service gRPC port |
| `BFF_BASE_URL` | `http://server:8082` | Java BFF base URL (empty = saved_numbers tools return empty) |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `300` | LLM request timeout in seconds |
| `AGENT_LOG_LEVEL` | `INFO` | Logging level |

## Testing

Integration tests use a real pgvector PostgreSQL (from `docker-compose-dev.yml`) with a mock LLM (`FakeListChatModel`), mock embeddings, and mock tool clients. The DB must be running.

```bash
# Start the DB first
docker compose -f docker-compose-dev.yml up -d db

make test             # all tests (unit + integration) — 70 tests
make test-unit        # unit tests only (no DB needed) — 34 tests
make test-integration # integration tests only (needs DB) — 36 tests
```

### Test Conventions

- **Unit tests** (`tests/unit/`) — no DB, no external services
- **Integration tests** (`tests/integration/`) — real pgvector DB, mock LLM + mock tools
- Mock LLM is injected via `set_llm_store()` in `conftest.py`
- Mock embeddings injected via `set_embeddings_model()` in `conftest.py`
- Mock tool clients (lottery gRPC + saved_numbers) injected via `set_mock_client()` in `conftest.py`
- JWT tokens created via `create_test_jwt(sub=..., tier=...)` in `app/security.py`

## Docker

### Dev (docker-compose-dev.yml)

Brings up three services:

| Service | Image | Port | Description |
|---|---|---|---|
| `ollama` | `ollama/ollama:latest` | `11434` | Local LLM inference (CPU-only dev) |
| `db` | `pgvector/pgvector:pg16` | `5433` | PostgreSQL with pgvector extension |
| `agent` | built from `Dockerfile` | `8000` | Agent service (hot-reload via volume mount) |

```bash
make dev      # up -d
make dev-stop # down
```

### Production Dockerfile

Multi-stage build on `python:3.12-slim`:

1. **Builder stage** — installs dependencies via `uv pip install --system`
2. **Runtime stage** — copies site-packages + `app/` directory, exposes port 8000, healthcheck on `/healthz`

```bash
docker build -t statistiloto-agent .
docker run -p 8000:8000 statistiloto-agent
```

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture reference for AI coding agents
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — Functional and non-functional requirements
- [docs/FLOWS.md](docs/FLOWS.md) — Mermaid sequence/flow diagrams for key operations
