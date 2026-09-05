"""ToolResolver — deterministic operation resolution + argument validation.

The resolver takes a tier-agnostic NormalizedRequest and returns a ToolResolution
containing the target tool, normalized arguments, and whether the request is ready
for direct execution.

Key design points:
  - OperationSpec defines each operation's schema, defaults, and validators.
  - Required parameters must be present for execution_ready=True.
  - Invalid values produce execution_ready=False with validation_errors.
  - Missing known parameters expose a deterministic missing_hint (no LLM).
  - Strength values are normalized via the canonical _STRENGTH_MAP from lottery_grpc.
  - execution_ready is computed here — never stored in NormalizedRequest.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.tools.lottery_grpc import _STRENGTH_MAP

log = logging.getLogger(__name__)


def _normalize_strength(strength: Any) -> int | str | None:
    """Normalize a strength token using the canonical _STRENGTH_MAP.

    Accepts "hot" / "cold" strings or 1/2 integers.
    Returns the canonical integer code (1 for cold, 2 for hot) or None.
    """
    if strength is None:
        return None
    if isinstance(strength, int) and strength in (1, 2):
        return strength
    if isinstance(strength, str):
        s = strength.strip().lower()
        if s.isdigit() and int(s) in (1, 2):
            return int(s)
        if s in _STRENGTH_MAP:
            return _STRENGTH_MAP[s]
        # Accept "hot" / "cold" aliases
        if s == "hot":
            return 2
        if s == "cold":
            return 1
    return None


def _validate_group_size(value: Any) -> tuple[bool, str]:
    if value is None:
        return False, "group_size is required"
    if isinstance(value, str):
        if not value.isdigit():
            return False, f"group_size must be an integer between 1 and 6, got {value!r}"
        value = int(value)
    try:
        value = int(value)
    except (ValueError, TypeError):
        return False, f"group_size must be an integer between 1 and 6, got {value!r}"
    if 1 <= value <= 6:
        return True, ""
    return False, f"group_size {value} is out of range (1-6)"


def _validate_numbers(value: Any, min_count: int = 1) -> tuple[bool, str]:
    if value is None:
        return False, f"numbers are required (at least {min_count})"
    if not isinstance(value, list):
        return False, "numbers must be a list"
    if len(value) < min_count:
        return False, f"expected at least {min_count} numbers, got {len(value)}"
    # Validate all are ints in 1-99
    for n in value:
        if not isinstance(n, int) or not 1 <= n <= 99:
            return False, f"invalid number: {n!r}"
    return True, ""


def _validate_file_path(value: Any) -> tuple[bool, str]:
    if not value:
        return False, "file_path is required"
    if not isinstance(value, str):
        return False, "file_path must be a string"
    return True, ""


def _validate_sql(value: Any) -> tuple[bool, str]:
    if not value:
        return False, "sql is required"
    if not isinstance(value, str):
        return False, "sql must be a string"
    upper = value.strip().upper()
    forbidden = {"INSERT ", "UPDATE ", "DELETE ", "DROP ", "TRUNCATE ", "ALTER "}
    if any(f in upper for f in forbidden):
        return False, "only read-only SELECT queries are allowed"
    if not re.search(r"\bSELECT\b", upper):
        return False, "sql must contain a SELECT statement"
    return True, ""


def _validate_edit_content(content: Any, old_string: Any, new_string: Any) -> tuple[bool, str]:
    """edit_file needs content OR (old_string + new_string)."""
    if content:
        return True, ""
    if old_string and new_string:
        return True, ""
    return False, "edit_file requires 'content' or both 'old_string' and 'new_string'"


@dataclass
class OperationSpec:
    """Canonical definition of an operation and its parameter schema."""

    name: str                          # e.g. "group_frequency"
    tool: str                          # e.g. "get_statistics"
    required: list[str]                # required parameter names
    optional: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    validators: dict[str, Callable[[Any], tuple[bool, str]]] = field(default_factory=dict)
    missing_hint: dict[str, str] = field(default_factory=dict)  # param → hint
    validator: Optional[Callable[[dict], tuple[bool, str]]] = None  # whole-args validator

    def default_args(self) -> dict:
        return dict(self.defaults)


@dataclass
class ToolResolution:
    """Result of resolving a NormalizedRequest."""

    operation: Optional[str]
    tool: Optional[str]
    args: dict[str, Any] = field(default_factory=dict)
    execution_ready: bool = False
    validation_errors: list[str] = field(default_factory=list)
    missing_hint: Optional[str] = None
    confidence: float = 0.0
    requires_llm: bool = False


# ── Canonical operation specs ────────────────────────────────

OPERATION_SPECS: dict[str, OperationSpec] = {
    "group_frequency": OperationSpec(
        name="group_frequency",
        tool="get_statistics",
        required=["group_size", "strength"],
        optional=["how_many", "archive_window"],
        defaults={"how_many": 10},
        validators={
            "group_size": _validate_group_size,
            "strength": lambda v: (True, "") if _normalize_strength(v) is not None else (False, f"strength {v!r} is not valid"),
        },
        missing_hint={
            "group_size": {
                "en": "What group size would you like? E.g. pairs (2), triples (3), singles (1).",
                "he": "באיזה גודל קבוצה תרצה לבדוק? לדוגמה: זוגות (2), שלשות (3), מספרים בודדים (1).",
            },
            "strength": {
                "en": "Do you want hot (most frequent) or cold (least frequent)?",
                "he": "האם תרצה חמים (הכי שכיחים) או קרים (הכי פחות שכיחים)?",
            },
        },
    ),
    "analyze_numbers": OperationSpec(
        name="analyze_numbers",
        tool="analyze",
        required=["form"],
        optional=["archive_window"],
        defaults={},
        validators={
            "form": _validate_numbers,
        },
        missing_hint={
            "form": {
                "en": "Which numbers would you like to analyze?",
                "he": "אילו מספרים תרצה לנתח?",
            },
        },
    ),
    "generate_form": OperationSpec(
        name="generate_form",
        tool="generate_form",
        required=["how_many"],
        optional=["form_type", "strength", "will_be", "will_not_be", "must_strong"],
        defaults={"form_type": 6, "strength": 2},
        validators={
            "form_type": _validate_group_size,
            "how_many": lambda v: (True, "") if v is not None and int(v) > 0 else (False, "how_many must be a positive integer"),
        },
        missing_hint={
            "how_many": {
                "en": "How many forms would you like to generate?",
                "he": "כמה טפסים תרצה ליצור?",
            },
        },
    ),
    "save_numbers": OperationSpec(
        name="save_numbers",
        tool="save_numbers",
        required=["numbers"],
        optional=["category"],
        defaults={"category": "default"},
        validators={
            "numbers": _validate_numbers,
        },
        missing_hint={
            "numbers": {
                "en": "Which numbers would you like to save?",
                "he": "אילו מספרים תרצה לשמור?",
            },
        },
    ),
    "simulate_numbers": OperationSpec(
        name="simulate_numbers",
        tool="simulate",
        required=["form"],
        optional=["strong", "archive_window", "simulate_window"],
        defaults={},
        validators={
            "form": lambda v: _validate_numbers(v, min_count=6),
        },
        missing_hint={
            "form": {
                "en": "Which numbers would you like to backtest? (6, 8, 10, or 12 numbers)",
                "he": "אילו מספרים תרצה לבחון? (6, 8, 10, או 12 מספרים)",
            },
        },
    ),
    "edit_file": OperationSpec(
        name="edit_file",
        tool="edit_file",
        required=["file_path"],
        optional=["content", "old_string", "new_string"],
        defaults={},
        validators={
            "file_path": _validate_file_path,
        },
        missing_hint={
            "file_path": {
                "en": "Which file would you like to edit?",
                "he": "איזה קובץ תרצה לערוך?",
            },
            "content": {
                "en": "What content or change would you like to apply to the file?",
                "he": "מה תוכן או שינוי תרצה להחיל על הקובץ?",
            },
        },
        validator=lambda args: _validate_edit_content(args.get("content"), args.get("old_string"), args.get("new_string")),
    ),
    "read_code": OperationSpec(
        name="read_code",
        tool="read_code",
        required=["file_path"],
        optional=[],
        defaults={},
        validators={
            "file_path": _validate_file_path,
        },
        missing_hint={
            "file_path": {
                "en": "Which file would you like to read?",
                "he": "איזה קובץ תרצה לקרוא?",
            },
        },
    ),
    "query_db": OperationSpec(
        name="query_db",
        tool="query_db",
        required=["sql"],
        optional=[],
        defaults={},
        validators={
            "sql": _validate_sql,
        },
        missing_hint={
            "sql": {
                "en": "What SQL query would you like to run? (read-only)",
                "he": "מהי שאילתת ה-SQL שתרצה להריץ? (לקריאה בלבד)",
            },
        },
    ),
    "query_audit_log": OperationSpec(
        name="query_audit_log",
        tool="query_audit_log",
        required=[],
        optional=["limit", "from_date", "to_date", "action", "user_sub"],
        defaults={"limit": 50},
    ),
    "read_token_usage": OperationSpec(
        name="read_token_usage",
        tool="read_token_usage",
        required=[],
        optional=["limit", "from_date", "to_date"],
        defaults={"limit": 50},
    ),
    "trigger_scraper": OperationSpec(
        name="trigger_scraper",
        tool="trigger_scraper",
        required=[],
        optional=["start_year", "end_year"],
        defaults={},
    ),
    "list_files": OperationSpec(
        name="list_files",
        tool="list_files",
        required=["path"],
        optional=[],
        defaults={},
        validators={
            "path": lambda v: (True, "") if v else (False, "path is required"),
        },
        missing_hint={
            "path": {
                "en": "Which directory would you like to list?",
                "he": "איזו תיקייה תרצה להציג?",
            },
        },
    ),
    "list_db_tables": OperationSpec(
        name="list_db_tables",
        tool="list_db_tables",
        required=[],
        optional=["schema"],
        defaults={"schema": "public"},
    ),
    "list_saved_numbers": OperationSpec(
        name="list_saved_numbers",
        tool="list_saved_numbers",
        required=[],
        optional=[],
        defaults={},
    ),
    "search_web": OperationSpec(
        name="search_web",
        tool="search_web",
        required=["query"],
        optional=[],
        defaults={},
        missing_hint={
            "query": {
                "en": "What would you like to search for?",
                "he": "מה תרצה לחפש?",
            },
        },
    ),
    "out_of_scope": OperationSpec(
        name="out_of_scope",
        tool=None,
        required=[],
        optional=[],
        defaults={},
    ),
    "ambiguous": OperationSpec(
        name="ambiguous",
        tool=None,
        required=[],
        optional=[],
        defaults={},
    ),
    "trivial": OperationSpec(
        name="trivial",
        tool=None,
        required=[],
        optional=[],
        defaults={},
    ),
    "domain_explanation": OperationSpec(
        name="domain_explanation",
        tool=None,
        required=[],
        optional=[],
        defaults={},
    ),
}


def resolve(request: "app.normalizer.NormalizedRequest") -> ToolResolution:
    """Resolve a NormalizedRequest into a ToolResolution.

    The resolution is tier-agnostic: it determines whether the request is
    semantically ready to execute based on the operation's required parameters.
    Tier authorization is applied separately by CapabilityConfig/ToolExecutor.
    """
    from app.normalizer import NormalizedRequest

    if not isinstance(request, NormalizedRequest):
        raise TypeError("resolve() requires a NormalizedRequest")

    operation = request.operation
    if not operation or operation not in OPERATION_SPECS:
        return ToolResolution(
            operation=None,
            tool=None,
            args={},
            execution_ready=False,
            missing_hint=_predefined_out_of_scope_hint(request.language) if request.request_kind == "out_of_scope" else None,
            confidence=request.confidence,
            requires_llm=request.requires_llm,
        )

    spec = OPERATION_SPECS[operation]
    args = spec.default_args()

    # Map normalizer fields to tool argument names.
    field_to_arg: dict[str, str] = {
        "group_size": "group_size",
        "strength": "strength",
        "numbers": "numbers" if operation == "save_numbers" else "form",
        "how_many": "how_many",
        "archive_window": "archive_window",
        "simulate_window": "simulate_window",
        "file_path": "file_path",
        "sql": "sql",
        "content": "content",
        "old_string": "old_string",
        "new_string": "new_string",
        "category": "category",
    }
    if operation == "list_files":
        field_to_arg["file_path"] = "path"
    if operation == "search_web":
        field_to_arg["raw_message"] = "query"

    for field_name, arg_name in field_to_arg.items():
        value = getattr(request, field_name, None)
        if value is not None:
            args[arg_name] = value

    # Normalize strength for get_statistics.
    if "strength" in args and operation == "group_frequency":
        normalized = _normalize_strength(args["strength"])
        if normalized is not None:
            args["strength"] = normalized

    # Validate
    errors: list[str] = []
    missing: list[str] = []
    missing_hint: Optional[str] = None
    for param in spec.required:
        if param not in args or args[param] is None or (isinstance(args[param], (list, str)) and not args[param]):
            missing.append(param)
        elif param in spec.validators:
            ok, err = spec.validators[param](args[param])
            if not ok:
                errors.append(err)

    # Whole-args validator (for edit_file)
    if spec.validator and not missing and not errors:
        ok, err = spec.validator(args)
        if not ok:
            errors.append(err)
            # For edit_file, the user gave file_path but no content/change.
            # Override the missing hint to ask for content explicitly.
            if operation == "edit_file" and "content" in spec.missing_hint:
                missing_hint_map = spec.missing_hint["content"]
                if isinstance(missing_hint_map, dict):
                    missing_hint = missing_hint_map.get(request.language, missing_hint_map.get("en"))
                else:
                    missing_hint = missing_hint_map

    # Determine execution readiness and a deterministic missing hint.
    # missing_hint may already be set by the whole-args validator for cases
    # like edit_file where file_path is present but content is missing.
    execution_ready = not missing and not errors
    if missing:
        first_missing = missing[0]
        hint_map = spec.missing_hint.get(first_missing)
        if isinstance(hint_map, dict):
            missing_hint = hint_map.get(request.language, hint_map.get("en"))
        elif isinstance(hint_map, str):
            missing_hint = hint_map
        elif request.missing_hint:
            missing_hint = request.missing_hint
        else:
            missing_hint = (
                _hebrew_clarify(first_missing) if request.language == "he" else _english_clarify(first_missing)
            )

    if errors:
        execution_ready = False

    return ToolResolution(
        operation=operation,
        tool=spec.tool,
        args=args,
        execution_ready=execution_ready,
        validation_errors=errors,
        missing_hint=missing_hint,
        confidence=request.confidence,
        requires_llm=(not execution_ready and request.request_kind not in ("trivial", "out_of_scope")) and not missing_hint,
    )


def _predefined_out_of_scope_hint(language: str) -> str:
    if language == "he":
        return "אני עוזר ללוטו בלבד. אני יכול להסביר על מספרים חמים וקרים, זוגות, שלשות, וגדלי קבוצות. האם יש משהו קשור ללוטו שאוכל לעזור בו?"
    return "I'm a lottery intelligence assistant. I can explain hot/cold numbers, pairs, triples, group sizes, and the archive window. Is there a lottery-related question I can help with?"


def _english_clarify(param: str) -> str:
    return f"I need the '{param}' parameter to proceed. What would you like it to be?"


def _hebrew_clarify(param: str) -> str:
    return f"אני צריך את הפרמטר '{param}' כדי להמשיך. מה תרצה שיהיה?"
