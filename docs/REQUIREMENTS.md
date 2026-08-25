# Requirements — Statistiloto Agent

## Functional Requirements

### Chat (SSE)

- **FR-1**: The agent shall expose `POST /chat` accepting `{session_id, message, intent}` and returning either a `{response, thread_id}` JSON body or a `{paused: true, thread_id}` body when HITL is triggered.
- **FR-2**: Every chat request shall be authenticated via a JWT Bearer token in the `Authorization` header.
- **FR-3**: The agent shall persist conversation state per `thread_id` (`{user_sub}:{session_id}`) using a PostgresSaver checkpointer, enabling multi-turn memory within a session.
- **FR-4**: The agent shall apply a per-tier recursion limit to `graph.invoke()` to cap graph super-steps and prevent infinite loops.

### Human-in-the-Loop (HITL)

- **FR-10**: The agent shall automatically interrupt (`interrupt()`) when ANY write tool is called, regardless of tier or confidence.
- **FR-11**: On interrupt, the agent shall return `{paused: true, thread_id}` (HTTP 200) so the UI can present an approval dialog.
- **FR-12**: The agent shall expose `POST /approve` accepting `{session_id, approved, edited}` to resume a paused thread via `Command(resume={approved, edited})`.
- **FR-13**: HITL is NOT configurable per tier — it always triggers on write tools. Free tier never sees HITL because free tier has no write tools.

### RAG (Retrieval-Augmented Generation)

- **FR-20**: The agent shall use pgvector for vector similarity search (cosine distance `<=>`) over the `agent.embeddings` table.
- **FR-21**: RAG retrieval shall be role-scoped — only corpora allowed for the user's tier are searched.
- **FR-22**: The `user_data` corpus shall ALWAYS be filtered by `user_sub` from the JWT — never trusted to the LLM to self-scope.
- **FR-23**: Admin (owner) shall bypass the `user_sub` filter on `user_data` and see all users' data for support/debugging.
- **FR-24**: Embeddings shall use Ollama `nomic-embed-text` by default, with an injection point for mock embeddings in tests.

### Admin Operations

- **FR-30**: Admin shall be able to read the current global LLM config via `GET /llm-config` (any authenticated user can read).
- **FR-31**: Admin shall be able to update the global LLM config via `PUT /llm-config` (admin only) — provider, model, base_url, api_key, request_timeout_seconds.
- **FR-32**: LLM config updates shall be hot-reloaded — no restart needed. The config store poller refreshes every ~10s, and `PUT /llm-config` forces an immediate refresh.
- **FR-33**: Admin shall be able to trigger the Go scraper (`trigger_scraper` write tool, requires HITL).
- **FR-34**: Admin shall be able to read the audit log via `GET /audit-log` (optional `limit` query param, default 50).
- **FR-35**: Admin shall be able to read token usage stats via `GET /token-usage`.
- **FR-36**: The agent shall expose `GET /sessions` (any authenticated user) to list the caller's chat sessions (newest first) with the tier's session limit, `GET /sessions/{session_id}` to load a session's full message history from the checkpointer, `DELETE /sessions/{session_id}` to delete one session, and `DELETE /sessions` to delete all of the caller's sessions.
- **FR-37**: Chat sessions shall be indexed in `agent.chat_sessions` (title, preview, timestamps); full message history is reconstructed from the LangGraph checkpointer state — messages are never duplicated in `chat_sessions`.
- **FR-38**: Tier-based session retention limits shall be enforced: free=1, paid=15, admin=unlimited. When a user exceeds their limit, the oldest sessions (by `updated_at`) are pruned automatically — both the `chat_sessions` row and the checkpointer state for that thread are deleted.
- **FR-39**: The agent shall expose `GET /llm-configs` (admin only) to list all stored LLM configurations and `POST /llm-configs` (admin only) to create a new stored configuration.
- **FR-40**: The agent shall expose `PUT /llm-configs/{config_id}/activate` (admin only) to activate a stored configuration, `POST /llm-configs/{config_id}/test` (admin only) to smoke-test a stored configuration, and `DELETE /llm-configs/{config_id}` (admin only) to delete a stored configuration.
- **FR-41**: The agent shall expose `GET /llm-models?provider=...` (admin only) to list models available from a given provider. Ollama is queried live via its `/api/tags` endpoint; Gemini, OpenAI, and Anthropic return static model lists.
- **FR-42**: The agent shall expose `POST /reindex` (admin only) to rebuild the `docs` RAG corpus by ingesting markdown from `app/rag/docs_source/` into `agent.embeddings` with content-hash dedup (unchanged chunks are skipped).
- **FR-43**: The supervisor shall accept an optional structured `context` dict (page, selected numbers, groupSize, etc.) from the `/chat` request and forward it to worker subgraphs for grounding.

## Tier Requirements

### Free Tier

- Workers: `nl_assistant` only
- RAG corpora: `docs`
- Write tools: none
- Read tools: `generate_form`, `get_statistics`, `analyze`
- HITL: never (no write tools available)
- Daily budget: $0 (no limit)
- Recursion limit: 6 super-steps

### Paid Tier

- Workers: `nl_assistant`, `analyst`
- RAG corpora: `docs`, `lottery_history`
- Write tools: `save_numbers` (requires HITL)
- Read tools: `generate_form`, `get_statistics`, `analyze`, `list_saved_numbers`
- HITL: on `save_numbers`
- Daily budget: $5.00
- Recursion limit: 25 super-steps

### Admin Tier (Owner/Developer)

- Workers: all three (`nl_assistant`, `analyst`, `admin_ops`)
- RAG corpora: `docs`, `lottery_history`, `user_data` (all users), `ops_logs`
- Write tools: `save_numbers`, `trigger_scraper`, `edit_file` (all require HITL)
- Read tools: all read tools including `query_audit_log`, `read_token_usage`, `search_web`, `read_code`, `list_files`
- HITL: on `save_numbers`, `trigger_scraper`, `edit_file`
- Daily budget: $0 (no limit — owner)
- Recursion limit: 50 super-steps
- Saved sessions: unlimited
- Admin bypasses `user_data` RAG filter — sees all users' data

## Tool Requirements

- **TR-1**: All tools shall be classified as either `WRITE_TOOLS` or `READ_TOOLS` in `app/tools/registry.py` (frozensets).
- **TR-2**: `WRITE_TOOLS` (`save_numbers`, `trigger_scraper`, `edit_file`) shall require HITL approval before executing — the graph interrupts automatically.
- **TR-3**: `READ_TOOLS` (`generate_form`, `get_statistics`, `analyze`, `list_saved_numbers`, `query_audit_log`, `read_token_usage`, `search_web`, `read_code`, `list_files`) shall execute without HITL.
- **TR-4**: Tool availability shall be gated by tier — the supervisor downgrades disallowed intents to `nl_assistant`.
- **TR-5**: Tools shall gracefully degrade when backend services are unavailable (empty `LOTTERY_GRPC_HOST` → tools return empty; empty `BFF_BASE_URL` → saved_numbers tools return empty).

## LLM Requirements

- **LR-1**: There shall be a single global LLM for all tiers — no per-tier model selection.
- **LR-2**: Default provider shall be Ollama (`qwen3:8b`) for local development.
- **LR-3**: The LLM shall be configurable at runtime by admin via `PUT /llm-config` — provider, model, base_url, api_key, timeout.
- **LR-4**: LLM config changes shall hot-reload without restart via a poller (every ~10s) and immediate `force_refresh()` on update.
- **LR-5**: Supported providers: `ollama`, `gemini`, `mock` (for tests).
- **LR-6**: `LLM_MOCK=true` shall use `FakeListChatModel` for testing without a real LLM.
- **LR-7**: LLM request timeout shall be configurable via `LLM_REQUEST_TIMEOUT_SECONDS` (default 300s).

## RAG Requirements

- **RR-1**: The agent shall use pgvector (PostgreSQL extension) for vector storage and cosine similarity search.
- **RR-2**: Embeddings shall be stored in `agent.embeddings` table with columns: `content`, `metadata` (JSONB), `embedding` (vector), `corpus`.
- **RR-3**: Retrieval shall be role-scoped — only corpora allowed for the user's tier are searched.
- **RR-4**: Per-tenant `user_data` filtering: `metadata->>'user_sub' = JWT user_sub` — ALWAYS enforced for non-admin users.
- **RR-5**: Admin bypasses the `user_sub` filter on `user_data` — sees all users' data.
- **RR-6**: Default embedding model: `nomic-embed-text` (Ollama).
- **RR-7**: Default `top_k`: 6 results.

## Security Requirements

- **SR-1**: All endpoints (except `/healthz`) shall require a JWT Bearer token in the `Authorization` header.
- **SR-2**: JWT validation shall verify the signature against Keycloak JWKS (`RS256` algorithm) when `JWT_VERIFY=true`.
- **SR-3**: In dev/test mode (`JWT_VERIFY=false`), the agent shall decode the JWT without signature verification but still extract claims.
- **SR-4**: Tier shall be extracted from JWT claims with priority: explicit `tier` claim > group membership (`/admins`, `/paid`, `/users`) > realm roles > `free`.
- **SR-5**: Admin-only endpoints (`PUT /llm-config`, `GET /llm-configs`, `POST /llm-configs`, `PUT /llm-configs/{config_id}/activate`, `POST /llm-configs/{config_id}/test`, `DELETE /llm-configs/{config_id}`, `GET /llm-models`, `GET /token-usage`, `GET /audit-log`, `POST /reindex`) shall require admin — `require_admin()` raises `JWTError` for non-admin tiers.
- **SR-6**: The raw JWT token shall be passed through to tools (e.g., `save_numbers` forwards it to the Java BFF for backend authorization).

## Metering Requirements

- **MR-1**: Every LLM call shall be logged to `agent.token_usage` table with: `thread_id`, `user_sub`, `tier`, `provider`, `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `ts`.
- **MR-2**: Token counts shall be extracted from LangChain `usage_metadata` (primary) or raw provider `response_metadata` (fallback: Ollama `prompt_eval_count`/`eval_count`).
- **MR-3**: Cost shall be estimated per provider: Ollama = $0 (local/free), Gemini ≈ $0.000075/1K prompt + $0.0003/1K completion.
- **MR-4**: Daily budget shall be enforced per user — `check_daily_budget()` sums `cost_usd` over the last 24 hours and compares to the tier's `daily_budget_usd`.
- **MR-5**: Budget check shall fail open (allow the request) if the DB query fails.
- **MR-6**: Only admin shall be able to read token consumption (`GET /token-usage`).

## Recursion Limits

| Tier | Recursion Limit | Rationale |
|---|---|---|
| Free | 6 | Simple NL assistant — minimal tool chaining |
| Paid | 25 | Multi-step analysis with RAG + tool calls |
| Admin | 50 | Full admin operations with complex workflows |

The recursion limit is wired into `graph.invoke()` via `config["recursion_limit"]` and caps the number of graph super-steps to prevent infinite loops.

## Testing Requirements

- **TR-1**: The test suite shall contain 70 tests total.
- **TR-2**: 34 unit tests (`tests/unit/`) shall run without a DB or any external services.
- **TR-3**: 36 integration tests (`tests/integration/`) shall run against a real pgvector PostgreSQL (from `docker-compose-dev.yml`).
- **TR-4**: Integration tests shall use a mock LLM (`FakeListChatModel`), mock embeddings, and mock tool clients — no real LLM or external service calls.
- **TR-5**: Mock LLM shall be injected via `set_llm_store()` in `conftest.py`.
- **TR-6**: Mock embeddings shall be injected via `set_embeddings_model()` in `conftest.py`.
- **TR-7**: Mock tool clients (lottery gRPC + saved_numbers) shall be injected via `set_mock_client()` in `conftest.py`.
- **TR-8**: Test JWT tokens shall be created via `create_test_jwt(sub=..., tier=...)` in `app/security.py` (HS256, not for production).
- **TR-9**: `make test` runs all tests; `make test-unit` runs unit only; `make test-integration` runs integration only (needs DB).
