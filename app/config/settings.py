"""Application settings — loads agent.yaml with env var overrides via pydantic-settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings


# ── Dataclasses for typed config ─────────────────────────────

@dataclass
class OllamaConfig:
    base_url: str = "http://ollama:11434"
    model: str = "llama3.1:8b"
    # Trusted local models — pulled on startup when missing from local Ollama.
    models: list[str] = field(default_factory=list)


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-2.0-flash"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    config_poll_seconds: int = 10
    mock: bool = False
    request_timeout_seconds: int = 300


@dataclass
class TierConfig:
    recursion_limit: int = 6
    allowed_tools: list[str] = field(default_factory=list)
    rag_corpora: list[str] = field(default_factory=list)
    daily_budget_usd: float = 0.0


@dataclass
class RAGConfig:
    embedding_model: str = "nomic-embed-text"
    top_k: int = 6
    user_data_scope: str = "by_user_sub"
    auto_ingest: bool = True  # ingest docs + examples on startup (fault-tolerant)


@dataclass
class HITLConfig:
    resume_timeout_hours: int = 168


@dataclass
class LotteryGrpcConfig:
    host: str = "lottery"
    port: str = "9090"


@dataclass
class BffConfig:
    base_url: str = "http://server:8082"


@dataclass
class DatabaseConfig:
    uri: str = "postgresql://postgres:postgres@db:5432/statistiloto"


@dataclass
class SecurityConfig:
    jwt_verify: bool = True
    jwks_url: str = ""
    issuer: str = ""
    audience: str = ""
    # Trusted `iss` claim values. Empty = issuer not checked (signature +
    # audience still verified). Needed because Keycloak's public URL varies
    # by deployment (localhost / ngrok tunnel / prod domain).
    allowed_issuers: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class FreeTierConfig:
    # When False (default), free users receive a generic deterministic response
    # for ambiguous/domain-explanation requests instead of calling the LLM.
    # Toggle at runtime via the admin-only PUT /free-llm endpoint.
    llm_enabled: bool = False


@dataclass
class ToolsConfig:
    cache_ttl_seconds: int = 300   # 5 minutes
    cache_max_size: int = 100


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tiers: dict[str, TierConfig] = field(default_factory=dict)
    rag: RAGConfig = field(default_factory=RAGConfig)
    hitl: HITLConfig = field(default_factory=HITLConfig)
    lottery_grpc: LotteryGrpcConfig = field(default_factory=LotteryGrpcConfig)
    bff: BffConfig = field(default_factory=BffConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    free_tier: FreeTierConfig = field(default_factory=FreeTierConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)


def _load_yaml() -> dict:
    cfg_path = Path(__file__).parent / "agent.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _env_override(data: dict) -> dict:
    """Apply env var overrides on top of YAML values."""
    # LLM
    if v := os.environ.get("LLM_PROVIDER"):
        data.setdefault("llm", {})["provider"] = v
    if v := os.environ.get("OLLAMA_BASE_URL"):
        data.setdefault("llm", {}).setdefault("ollama", {})["base_url"] = v
    if v := os.environ.get("OLLAMA_MODEL"):
        data.setdefault("llm", {}).setdefault("ollama", {})["model"] = v
    if v := os.environ.get("OLLAMA_MODELS"):
        data.setdefault("llm", {}).setdefault("ollama", {})["models"] = [
            s.strip() for s in v.split(",") if s.strip()]
    if v := os.environ.get("GEMINI_API_KEY"):
        data.setdefault("llm", {}).setdefault("gemini", {})["api_key"] = v
    if v := os.environ.get("GEMINI_MODEL"):
        data.setdefault("llm", {}).setdefault("gemini", {})["model"] = v
    if v := os.environ.get("LLM_MOCK"):
        data.setdefault("llm", {})["mock"] = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS"):
        data.setdefault("llm", {})["request_timeout_seconds"] = int(v)
    # DB
    if v := os.environ.get("DB_URI"):
        data.setdefault("database", {})["uri"] = v
    # gRPC
    if v := os.environ.get("LOTTERY_GRPC_HOST"):
        data.setdefault("lottery_grpc", {})["host"] = v
    if v := os.environ.get("LOTTERY_GRPC_PORT"):
        data.setdefault("lottery_grpc", {})["port"] = v
    # BFF
    if v := os.environ.get("BFF_BASE_URL"):
        data.setdefault("bff", {})["base_url"] = v
    # Security
    if v := os.environ.get("JWT_VERIFY"):
        data.setdefault("security", {})["jwt_verify"] = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("JWKS_URL"):
        data.setdefault("security", {})["jwks_url"] = v
    if v := os.environ.get("ISSUER"):
        data.setdefault("security", {})["issuer"] = v
    if v := os.environ.get("AUDIENCE"):
        data.setdefault("security", {})["audience"] = v
    if v := os.environ.get("ALLOWED_ISSUERS"):
        data.setdefault("security", {})["allowed_issuers"] = [s.strip() for s in v.split(",") if s.strip()]
    # Logging
    if v := os.environ.get("AGENT_LOG_LEVEL"):
        data.setdefault("logging", {})["level"] = v
    # Free-tier LLM toggle (default: disabled — free users get generic responses)
    if v := os.environ.get("FREE_LLM_ENABLED"):
        data.setdefault("free_tier", {})["llm_enabled"] = v.lower() in ("1", "true", "yes")
    return data


def _build_settings(data: dict) -> Settings:
    llm_data = data.get("llm", {})
    llm = LLMConfig(
        provider=llm_data.get("provider", "ollama"),
        ollama=OllamaConfig(
            base_url=llm_data.get("ollama", {}).get("base_url", "http://ollama:11434"),
            model=llm_data.get("ollama", {}).get("model", "llama3.1:8b"),
            models=list(llm_data.get("ollama", {}).get("models", [])),
        ),
        gemini=GeminiConfig(
            api_key=llm_data.get("gemini", {}).get("api_key", ""),
            model=llm_data.get("gemini", {}).get("model", "gemini-2.0-flash"),
        ),
        config_poll_seconds=llm_data.get("config_poll_seconds", 10),
        mock=llm_data.get("mock", False),
        request_timeout_seconds=llm_data.get("request_timeout_seconds", 300),
    )

    tiers_data = data.get("tiers", {})
    tiers = {}
    for name, td in tiers_data.items():
        tiers[name] = TierConfig(
            recursion_limit=td.get("recursion_limit", 6),
            allowed_tools=td.get("allowed_tools", []),
            rag_corpora=td.get("rag_corpora", []),
            daily_budget_usd=td.get("daily_budget_usd", 0.0),
        )

    rag_data = data.get("rag", {})
    rag = RAGConfig(
        embedding_model=rag_data.get("embedding_model", "nomic-embed-text"),
        top_k=rag_data.get("top_k", 6),
        user_data_scope=rag_data.get("user_data_scope", "by_user_sub"),
        auto_ingest=rag_data.get("auto_ingest", True),
    )

    hitl_data = data.get("hitl", {})
    hitl = HITLConfig(resume_timeout_hours=hitl_data.get("resume_timeout_hours", 168))

    grpc_data = data.get("lottery_grpc", {})
    lottery_grpc = LotteryGrpcConfig(
        host=grpc_data.get("host", "lottery"),
        port=grpc_data.get("port", "9090"),
    )

    bff_data = data.get("bff", {})
    bff = BffConfig(base_url=bff_data.get("base_url", "http://server:8082"))

    db_data = data.get("database", {})
    database = DatabaseConfig(uri=db_data.get("uri", "postgresql://postgres:postgres@db:5432/statistiloto"))

    sec_data = data.get("security", {})
    _issuers = sec_data.get("allowed_issuers", [])
    if isinstance(_issuers, str):
        _issuers = [s.strip() for s in _issuers.split(",") if s.strip()]
    security = SecurityConfig(
        jwt_verify=sec_data.get("jwt_verify", True),
        jwks_url=sec_data.get("jwks_url", ""),
        issuer=sec_data.get("issuer", ""),
        audience=sec_data.get("audience", ""),
        allowed_issuers=list(_issuers),
    )

    log_data = data.get("logging", {})
    logging_cfg = LoggingConfig(level=log_data.get("level", "INFO"))

    free_tier_data = data.get("free_tier", {})
    free_tier = FreeTierConfig(llm_enabled=free_tier_data.get("llm_enabled", False))

    tools_data = data.get("tools", {})
    tools = ToolsConfig(
        cache_ttl_seconds=tools_data.get("cache_ttl_seconds", 300),
        cache_max_size=tools_data.get("cache_max_size", 100),
    )

    return Settings(
        llm=llm, tiers=tiers, rag=rag, hitl=hitl,
        lottery_grpc=lottery_grpc, bff=bff, database=database,
        security=security, logging=logging_cfg, free_tier=free_tier,
        tools=tools,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings from YAML + env overrides."""
    data = _env_override(_load_yaml())
    return _build_settings(data)


def get_tier_config(tier: str) -> TierConfig:
    """Get capability caps for a tier. Falls back to free if unknown."""
    s = get_settings()
    return s.tiers.get(tier, s.tiers.get("free", TierConfig()))


def reload_settings():
    """Force reload settings (for testing)."""
    get_settings.cache_clear()
