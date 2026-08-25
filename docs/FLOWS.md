# Flows — Statistiloto Agent

Mermaid diagrams for the key operational flows in the agent service.

## Chat Request Flow

```mermaid
sequenceDiagram
    participant BFF as Java BFF
    participant Agent as Agent (FastAPI)
    participant Auth as JWT/JWKS
    participant Sup as Supervisor Graph
    participant Worker as Worker Subgraph
    participant Tools as Tools
    participant DB as PostgreSQL

    BFF->>Agent: POST /chat {session_id, message, intent, context}
    Agent->>Auth: validate_jwt(Authorization header)
    Auth-->>Agent: TokenClaims {sub, tier, roles}

    Agent->>DB: graph.get_state(config) — load prior checkpoint
    DB-->>Agent: history (conversation state)

    Agent->>Agent: get_tier_config(tier) → recursion_limit
    Agent->>Sup: graph.invoke(state, config)
    Sup->>Sup: classify_intent(tier, intent)
    Note over Sup: Tier gating: admin_ops→admin only,<br/>analyst→paid/admin, else→nl_assistant
    Note over Sup: Optional `context` dict (page, numbers,<br/>groupSize) forwarded to workers for grounding

    Sup->>Worker: route to worker subgraph
    Worker->>Worker: LLM call (metered)
    Worker->>Tools: call read tools (no HITL)
    Tools-->>Worker: tool results
    Worker->>DB: RAG retrieval (pgvector)
    DB-->>Worker: relevant chunks
    Worker-->>Sup: {response, chunks, ...}
    Sup-->>Agent: result dict

    alt HITL triggered (write tool)
        Agent-->>BFF: {paused: true, thread_id}
    else Normal completion
        Agent-->>BFF: {response, thread_id} (SSE events)
    end
```

## Human-in-the-Loop (HITL) Flow

```mermaid
sequenceDiagram
    participant User as User/UI
    participant BFF as Java BFF
    participant Agent as Agent (FastAPI)
    participant Graph as LangGraph
    participant Tool as Write Tool

    User->>BFF: Send message requiring write
    BFF->>Agent: POST /chat {session_id, message, intent}
    Agent->>Graph: graph.invoke(state, config)
    Graph->>Graph: Worker decides to call write tool
    Graph->>Graph: interrupt() — pause execution
    Graph-->>Agent: result with __interrupt__
    Agent-->>BFF: {paused: true, thread_id}
    BFF-->>User: Show approval dialog (tool name + args)

    User->>BFF: Approve / Reject / Edit
    BFF->>Agent: POST /approve {session_id, approved, edited}
    Agent->>Graph: graph.invoke(Command(resume={approved, edited}), config)
    Graph->>Graph: Resume from interrupt point

    alt approved == true
        Graph->>Tool: Execute write tool (save_numbers / trigger_scraper)
        Tool-->>Graph: tool result
    else approved == false
        Graph->>Graph: Skip write, generate rejection response
    end

    Graph-->>Agent: {response}
    Agent-->>BFF: {response}
    BFF-->>User: Display final response
```

## Supervisor Routing Flow

```mermaid
flowchart TD
    Start([Incoming chat request]) --> Classify{"classify_intent()<br/>tier + intent"}
    Classify -->|"intent = admin_ops<br/>AND tier = admin"| AdminOps["admin_ops subgraph<br/>(admin only)"]
    Classify -->|"intent = admin_ops<br/>AND tier != admin"| Downgrade1["DOWNGRADE → nl_assistant"]
    Classify -->|"intent = analyst<br/>AND tier = paid/admin"| Analyst["analyst subgraph<br/>(paid/admin)"]
    Classify -->|"intent = analyst<br/>AND tier = free"| Downgrade2["DOWNGRADE → nl_assistant"]
    Classify -->|"intent = nl_assistant<br/>(any tier)"| NL["nl_assistant subgraph<br/>(free/paid/admin)"]
    Classify -->|"intent = null"| NLDefault["nl_assistant (default)"]

    AdminOps --> End([END → return response])
    Analyst --> End
    NL --> End
    Downgrade1 --> NL
    Downgrade2 --> NL
    NLDefault --> NL

    style AdminOps fill:#f9c0c0,stroke:#c0392b
    style Analyst fill:#f9e79f,stroke:#f39c12
    style NL fill:#abebc6,stroke:#27ae60
    style Downgrade1 fill:#f5b7b1,stroke:#e74c3c,stroke-dasharray: 5 5
    style Downgrade2 fill:#f5b7b1,stroke:#e74c3c,stroke-dasharray: 5 5
```

**Routing rules** (from `app/graphs/supervisor.py`):

| Intent | Free | Paid | Admin |
|---|---|---|---|
| `nl_assistant` | ✅ | ✅ | ✅ |
| `analyst` | ❌ → `nl_assistant` | ✅ | ✅ |
| `admin_ops` | ❌ → `nl_assistant` | ❌ → `nl_assistant` | ✅ |
| `null` | `nl_assistant` | `nl_assistant` | `nl_assistant` |

## RAG Retrieval Flow

```mermaid
flowchart TD
    Query["User query"] --> Embed["Embed query<br/>Ollama nomic-embed-text"]
    Embed --> Vector["Query embedding vector"]
    Vector --> Search["pgvector cosine search<br/>embedding <=> query::vector"]
    Search --> Scope{"Corpora scope<br/>(role-scoped by tier)"}

    Scope -->|"corpus = user_data<br/>AND tier = admin"| AdminSearch["No user_sub filter<br/>admin sees ALL users"]
    Scope -->|"corpus = user_data<br/>AND tier != admin"| UserSearch["Filter: metadata.user_sub = JWT sub<br/>(per-tenant isolation)"]
    Scope -->|"corpus in docs/lottery_history/ops_logs"| SharedSearch["Filter: user_sub IS NULL<br/>OR user_sub = JWT sub"]

    AdminSearch --> Results["Top-K results<br/>(default k=6)"]
    UserSearch --> Results
    SharedSearch --> Results
    Results --> Context["Inject chunks into LLM context"]
    Context --> LLM["LLM generates response<br/>with RAG context"]
```

**Key security property**: `user_data` is ALWAYS filtered by `user_sub` from the JWT for non-admin users — never trusted to the LLM to self-scope. Admin bypasses this filter to see all users' data for support/debugging.

## LLM Config Update Flow

```mermaid
sequenceDiagram
    participant Admin as Admin User
    participant Agent as Agent (FastAPI)
    participant DB as PostgreSQL
    participant Store as LLMConfigStore
    participant Poller as Config Poller

    Admin->>Agent: PUT /llm-config {provider, model, base_url, api_key, timeout}
    Agent->>Agent: validate_jwt() + require_admin()
    Note over Agent: 403 if not admin

    Agent->>DB: INSERT INTO agent.llm_config (...)
    DB-->>Agent: row inserted

    Agent->>Store: force_refresh()
    Store->>DB: SELECT latest config from agent.llm_config
    DB-->>Store: new config row
    Store->>Store: Rebuild LLM instance (Ollama/Gemini/Mock)
    Store-->>Agent: config applied

    Agent-->>Admin: 200 {status: "accepted", provider, model, note: "Hot-reloaded"}

    Note over Poller: Background poller also runs every ~10s<br/>as a safety net to pick up changes
    Poller->>DB: SELECT latest config
    DB-->>Poller: config row
    Poller->>Store: update if changed
```

**Hot-reload mechanism**: The `PUT /llm-config` endpoint writes to `agent.llm_config` and calls `force_refresh()` for immediate effect. A background poller (started on app startup, every ~10s) acts as a safety net to pick up any changes — no restart needed.

## Chat Session History Flow

```mermaid
sequenceDiagram
    participant BFF as Java BFF
    participant Agent as Agent (FastAPI)
    participant Sessions as app/sessions.py
    participant DB as PostgreSQL (agent schema)
    participant CP as Checkpointer

    Note over BFF,Agent: List sessions
    BFF->>Agent: GET /sessions (Authorization: Bearer <JWT>)
    Agent->>Agent: validate_jwt() → claims {sub, tier}
    Agent->>Sessions: list_sessions(sub, tier)
    Sessions->>DB: SELECT FROM agent.chat_sessions WHERE user_sub=sub ORDER BY updated_at DESC
    DB-->>Sessions: rows
    Sessions-->>Agent: sessions + tier limit (free=1, paid=15, admin=unlimited)
    Agent-->>BFF: 200 { sessions: [...], limit, tier }

    Note over BFF,Agent: Load one session's messages
    BFF->>Agent: GET /sessions/{session_id}
    Agent->>Sessions: get_session_messages(sub, session_id, graph)
    Sessions->>CP: graph.get_state(config={thread_id: "{sub}:{session_id}"})
    CP-->>Sessions: history (message channel)
    Sessions-->>Agent: messages
    Agent-->>BFF: 200 { session_id, messages }

    Note over BFF,Agent: Delete one session
    BFF->>Agent: DELETE /sessions/{session_id}
    Agent->>Sessions: delete_session(sub, session_id)
    Sessions->>DB: DELETE FROM agent.chat_sessions WHERE user_sub=sub AND session_id=...
    Sessions->>CP: delete checkpointer state for thread
    Sessions-->>Agent: deleted (bool)
    Agent-->>BFF: 200 { status: "deleted", session_id } (or 404 if not found)
```

**Retention**: when a chat write creates/updates a session and the user exceeds their tier limit, the oldest sessions (by `updated_at`) are pruned automatically — both the `chat_sessions` row and the checkpointer state for that thread are deleted.

## Docs RAG Reindex Flow

```mermaid
sequenceDiagram
    participant Admin as Admin User
    participant Agent as Agent (FastAPI)
    participant Ingest as app/rag/ingest.py
    participant Embed as Embeddings Model
    participant DB as PostgreSQL (agent.embeddings)

    Admin->>Agent: POST /reindex (Authorization: Bearer <JWT>)
    Agent->>Agent: validate_jwt() + require_admin()
    Agent->>Ingest: ingest_docs()
    Ingest->>Ingest: Read .md files from app/rag/docs_source/
    Ingest->>Ingest: Split into chunks (~500 chars)
    Ingest->>DB: SELECT content_hash FROM agent.embeddings WHERE corpus='docs'
    DB-->>Ingest: existing hashes

    loop Per chunk
        alt content hash unchanged
            Ingest->>Ingest: skip (dedup)
        else new or changed
            Ingest->>Embed: embed chunk text
            Embed-->>Ingest: embedding vector
            Ingest->>DB: INSERT INTO agent.embeddings (content, metadata, embedding, corpus)
        end
    end

    Ingest-->>Agent: { indexed, skipped, total }
    Agent-->>Admin: 200 { status: "ok", indexed, skipped, total }
```

**Idempotent**: content-hash dedup makes re-indexing fast — only changed or new chunks are embedded and inserted. Use `--force` (via `python -m app.rag.ingest --force`) to clear and re-index all docs.
