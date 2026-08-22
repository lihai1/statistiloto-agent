"""HITL helpers — interrupt()/Command(resume=) wrappers."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt, Command


def pause_for_approval(payload: dict) -> dict:
    """Pause execution and surface a payload for human review.

    Returns the human's decision when resumed via Command(resume=...).
    """
    return interrupt(payload)


def resume_with_decision(decision: dict, update: dict | None = None, goto: str = ""):
    """Create a Command to resume a paused graph with a decision."""
    return Command(update=update or {}, goto=goto)
