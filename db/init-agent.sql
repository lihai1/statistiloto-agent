-- Agent schema init — pgvector + metering + audit + llm_config + embeddings.
-- Used by docker-compose-dev.yml and the main docker-compose.yml.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS agent;

-- Token usage metering — every LLM call logs here.
CREATE TABLE IF NOT EXISTS agent.token_usage (
    id                BIGSERIAL PRIMARY KEY,
    thread_id         TEXT NOT NULL,
    user_sub          TEXT NOT NULL,
    tier              TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INT  NOT NULL DEFAULT 0,
    completion_tokens INT  NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(10,4) NOT NULL DEFAULT 0,
    ts                DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_user_ts ON agent.token_usage (user_sub, ts);
CREATE INDEX IF NOT EXISTS idx_token_usage_tier_ts ON agent.token_usage (tier, ts);

-- Audit log — admin actions and HITL decisions.
CREATE TABLE IF NOT EXISTS agent.audit_log (
    id          BIGSERIAL PRIMARY KEY,
    user_sub    TEXT NOT NULL,
    tier        TEXT NOT NULL,
    action      TEXT NOT NULL,
    details     JSONB,
    ts          DOUBLE PRECISION NOT NULL
);

-- Runtime LLM config — admin-reconfigurable via PUT /llm-config.
-- Multiple named configs can be stored; exactly one is active (is_active=true).
-- The config_store polls the active row and hot-reloads the LLM.
CREATE TABLE IF NOT EXISTS agent.llm_config (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',   -- human-readable config name (e.g. "Ollama Llama3.1")
    provider    TEXT NOT NULL,              -- ollama | gemini
    model       TEXT NOT NULL,
    base_url    TEXT,
    api_key     TEXT,                       -- encrypt with pgcrypto in prod
    request_timeout_seconds INT NOT NULL DEFAULT 300,  -- max seconds for a single LLM call
    num_predict INT,                          -- max tokens to generate (NULL = model default)
    context_window_size INT,                  -- context window size in tokens (NULL = model default)
    is_active   BOOLEAN NOT NULL DEFAULT FALSE,  -- only one row should be active at a time
    updated_by  TEXT NOT NULL,              -- admin user_sub
    updated_at  DOUBLE PRECISION NOT NULL
);
-- Add columns to existing tables (idempotent migration for upgrades)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='request_timeout_seconds')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN request_timeout_seconds INT NOT NULL DEFAULT 300;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='is_active')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='name')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN name TEXT NOT NULL DEFAULT '';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='num_predict')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN num_predict INT;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='context_window_size')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN context_window_size INT;
    END IF
END $$;
-- Ensure at most one active config. Uses a partial unique index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_config_one_active
    ON agent.llm_config (is_active) WHERE is_active = TRUE;

-- pgvector embeddings — corpus-scoped, per-tenant for user_data.
CREATE TABLE IF NOT EXISTS agent.embeddings (
    id          BIGSERIAL PRIMARY KEY,
    corpus      TEXT NOT NULL,              -- docs | lottery_history | user_data | ops_logs
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector(768)                 -- nomic-embed-text dim
);
CREATE INDEX IF NOT EXISTS idx_embeddings_corpus ON agent.embeddings (corpus);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON agent.embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Chat session metadata — one row per conversation a user starts.
-- The actual message history lives in the LangGraph checkpointer tables
-- (checkpoints/checkpoint_blobs/checkpoint_writes, keyed by thread_id).
-- This table is the user-facing index: title, last message preview, timestamps,
-- and the session_id used by the UI. Tier-based retention limits are enforced
-- by the agent service (free=1, paid=15, admin=unlimited).
CREATE TABLE IF NOT EXISTS agent.chat_sessions (
    id              BIGSERIAL PRIMARY KEY,
    user_sub        TEXT NOT NULL,
    session_id      TEXT NOT NULL,              -- UI-generated session id (e.g. session-...)
    thread_id       TEXT NOT NULL,              -- checkpointer thread id: "{user_sub}:{session_id}"
    title           TEXT NOT NULL DEFAULT '',   -- derived from the first user message
    last_message    TEXT NOT NULL DEFAULT '',   -- preview of the last user message
    message_count   INT  NOT NULL DEFAULT 0,
    created_at      DOUBLE PRECISION NOT NULL,
    updated_at      DOUBLE PRECISION NOT NULL,
    UNIQUE (user_sub, session_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON agent.chat_sessions (user_sub, updated_at DESC);
