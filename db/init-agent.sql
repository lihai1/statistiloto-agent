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
-- Latest row wins; config_store polls this and hot-reloads the LLM.
CREATE TABLE IF NOT EXISTS agent.llm_config (
    id          BIGSERIAL PRIMARY KEY,
    provider    TEXT NOT NULL,              -- ollama | gemini
    model       TEXT NOT NULL,
    base_url    TEXT,
    api_key     TEXT,                       -- encrypt with pgcrypto in prod
    request_timeout_seconds INT NOT NULL DEFAULT 300,  -- max seconds for a single LLM call
    updated_by  TEXT NOT NULL,              -- admin user_sub
    updated_at  DOUBLE PRECISION NOT NULL
);
-- Add column to existing tables (idempotent migration for upgrades)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='agent' AND table_name='llm_config')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='agent' AND table_name='llm_config' AND column_name='request_timeout_seconds')
    THEN
        ALTER TABLE agent.llm_config ADD COLUMN request_timeout_seconds INT NOT NULL DEFAULT 300;
    END IF;
END $$;

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
