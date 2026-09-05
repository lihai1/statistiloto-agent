"""CapabilityConfig — read-only adapter over agent.yaml.

agent.yaml is the single source of truth for tier → allowed_tools mapping.
This module provides convenience methods to check and render capabilities
without duplicating the registry.

Used by:
  - Supervisor routing (authorization gate)
  - ToolExecutor (defense-in-depth re-authorization)
  - Capability response rendering ("What can you do?")
  - RAG retrieval (corpus filtering via allowed_tools)
  - LLM planner (AUTHORIZED_TOOLS section)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import get_tier_config, get_settings
from app.tools.registry import WRITE_TOOLS, READ_TOOLS

log = logging.getLogger(__name__)

# Bilingual capability descriptions for rendering.
_CAPABILITY_DESCRIPTIONS = {
    "en": {
        "generate_form": "Generate lottery forms with lucky numbers",
        "get_statistics": "Show hot/cold statistics for any group size",
        "analyze": "Analyze your selected numbers across all group sizes",
        "simulate": "Backtest your numbers against historical draws",
        "save_numbers": "Save your favorite number combinations",
        "list_saved_numbers": "View your saved number combinations",
        "trigger_scraper": "Trigger the lottery data scraper",
        "query_audit_log": "Query the admin audit log",
        "read_token_usage": "Read LLM token usage and costs",
        "search_web": "Search the web for lottery-related information",
        "read_code": "Read the agent's source code",
        "list_files": "List files in the agent's source tree",
        "edit_file": "Edit the agent's source code",
        "list_db_tables": "List database tables and columns",
        "query_db": "Run read-only SQL queries against the database",
    },
    "he": {
        "generate_form": "יצירת טפסי לוטו עם מספרי מזל",
        "get_statistics": "הצגת סטטיסטיקת חם/קר לכל גודל קבוצה",
        "analyze": "ניתוח המספרים שבחרת לפי כל גדלי הקבוצות",
        "simulate": "בחינת המספרים שלך מול הגרלות היסטוריות",
        "save_numbers": "שמירת צירופי המספרים המועדפים עליך",
        "list_saved_numbers": "צפייה במספרים ששמרת",
        "trigger_scraper": "הפעלת סקרייפר נתוני הלוטו",
        "query_audit_log": "שאילתת יומן הפעולות",
        "read_token_usage": "קריאת נתוני צריכת טוקנים ועלויות",
        "search_web": "חיפוש מידע באינטרנט",
        "read_code": "קריאת קוד המקור של הסוכן",
        "list_files": "רשימת קבצי קוד המקור",
        "edit_file": "עריכת קוד המקור של הסוכן",
        "list_db_tables": "רשימת טבלאות ועמודות במסד הנתונים",
        "query_db": "הרצת שאילתות SQL לקריאה בלבד",
    },
}


class CapabilityConfig:
    """Read-only adapter over agent.yaml for capability checks.

    This is NOT a separate registry — it reads from agent.yaml via get_tier_config().
    The single source of truth is agent.yaml's tiers.<tier>.allowed_tools.
    """

    @staticmethod
    def is_allowed(tier: str, tool: str) -> bool:
        """Check if a tier is allowed to use a tool.

        Args:
            tier: "free" | "paid" | "admin"
            tool: tool name (e.g. "get_statistics", "save_numbers")

        Returns:
            True if the tool is in the tier's allowed_tools list.
        """
        cfg = get_tier_config(tier)
        return tool in cfg.allowed_tools

    @staticmethod
    def allowed_tools(tier: str) -> list[str]:
        """Return the list of allowed tools for a tier."""
        cfg = get_tier_config(tier)
        return list(cfg.allowed_tools)

    @staticmethod
    def allowed_corpora(tier: str) -> list[str]:
        """Return the list of allowed RAG corpora for a tier."""
        cfg = get_tier_config(tier)
        return list(cfg.rag_corpora)

    @staticmethod
    def is_write_tool(tool: str) -> bool:
        """Check if a tool is a write tool (requires HITL)."""
        return tool in WRITE_TOOLS

    @staticmethod
    def write_tools_for_tier(tier: str) -> list[str]:
        """Return the write tools available to a tier."""
        return [t for t in CapabilityConfig.allowed_tools(tier) if t in WRITE_TOOLS]

    @staticmethod
    def read_tools_for_tier(tier: str) -> list[str]:
        """Return the read-only tools available to a tier."""
        return [t for t in CapabilityConfig.allowed_tools(tier) if t in READ_TOOLS]

    @staticmethod
    def render_user_capabilities(tier: str, lang: str = "en") -> str:
        """Render a human-readable description of what the tier can do.

        Used for the "What can you do?" deterministic response.

        Args:
            tier: "free" | "paid" | "admin"
            lang: "en" | "he"

        Returns:
            A formatted string describing the tier's capabilities.
        """
        descriptions = _CAPABILITY_DESCRIPTIONS.get(lang, _CAPABILITY_DESCRIPTIONS["en"])
        tools = CapabilityConfig.allowed_tools(tier)
        read_tools = [t for t in tools if t in READ_TOOLS]
        write_tools = [t for t in tools if t in WRITE_TOOLS]

        if lang == "he":
            lines = ["הנה מה שאני יכול לעשות עבורך:"]
            if read_tools:
                lines.append("\nכלים לקריאה:")
                for tool in read_tools:
                    desc = descriptions.get(tool, tool)
                    lines.append(f"  • {desc}")
            if write_tools:
                lines.append("\nכלים לכתיבה (דורשים אישור):")
                for tool in write_tools:
                    desc = descriptions.get(tool, tool)
                    lines.append(f"  • {desc}")
            lines.append("\nאלה נתוני עבר בלבד ואינם מעידים על סיכוי גבוה יותר בהגרלה הבאה.")
        else:
            lines = ["Here's what I can do for you:"]
            if read_tools:
                lines.append("\nRead tools:")
                for tool in read_tools:
                    desc = descriptions.get(tool, tool)
                    lines.append(f"  • {desc}")
            if write_tools:
                lines.append("\nWrite tools (require approval):")
                for tool in write_tools:
                    desc = descriptions.get(tool, tool)
                    lines.append(f"  • {desc}")
            lines.append("\nThese are historical observations only and do not imply a higher probability in the next draw.")

        return "\n".join(lines)
