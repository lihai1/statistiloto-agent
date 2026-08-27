"""Semantic normalizer — converts raw user message into a tier-agnostic NormalizedRequest.

Design goals:
  - Language detection from message content (Hebrew chars → 'he', else 'en').
  - request_kind classification independent of tier.
  - Entity extraction for lottery operations, admin operations, and trivial requests.
  - confidence measures semantic interpretation certainty ONLY (not arg completeness).
  - param_provenance tracks where each extracted value came from.
  - Follow-up detection with conservative state inheritance (Phase 6).
  - Does NOT store execution_ready — that is computed by ToolResolver.

The normalizer is intentionally stateless except for the optional ConversationState.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.tools.lottery_grpc import _STRENGTH_MAP

# ── Language detection ───────────────────────────────────────
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

# ── Request-kind classification regexes ─────────────────────
GREETING_RE = re.compile(
    r"\b(hi|hello|hey|greetings|good (morning|evening|afternoon)|what'?s? up|שלום|היי|הי|בוקר טוב|ערב טוב)\b",
    re.IGNORECASE,
)
GOODBYE_RE = re.compile(r"\b(bye|goodbye|thank|thanks|תודה|להתראות)\b", re.IGNORECASE)
CAPABILITIES_RE = re.compile(
    r"\b(what can you do|what are you able to do|what can i ask|help me|capabilities|\?+)\b",
    re.IGNORECASE,
)
HEBREW_CAPABILITIES_RE = re.compile(
    r"(מה אתה יכול|מה אתה יודע|איך אתה יכול לעזור|עזור לי|מה יש לך)",
    re.IGNORECASE,
)
OUT_OF_SCOPE_RE = re.compile(
    r"\b(weather|joke|news|stock|stocks|crypto|bitcoin|politics|sports|football)\b",
    re.IGNORECASE,
)
HEBREW_OUT_OF_SCOPE_RE = re.compile(
    r"(מזג האוויר|בדיחה|חדשות|מניות|ביטקוין|פוליטיקה|ספורט|כדורגל)",
    re.IGNORECASE,
)

# ── Lottery operation keyword regexes ────────────────────────
STATISTICS_KEYWORDS = re.compile(
    r"\b(hot|cold|warm|statistics|stats|frequent|frequency|most|least|pairs|triples|quads|quints|singles|numbers)\b",
    re.IGNORECASE,
)
HEBREW_STATISTICS_KEYWORDS = re.compile(
    r"(חמים|חמות|קרים|קרות|קר|חם|סטטיסטיקה|סטטיסטיקות|שכיחות|הופעות|זוגות|שלשות|רביעיות|חמישיות|שישיות|מספרים)",
    re.IGNORECASE,
)
FORM_GENERATION_RE = re.compile(
    r"\b(generate|create|make).{0,30}(form|forms|combination|ticket|lucky)\b",
    re.IGNORECASE,
)
HEBREW_FORM_GENERATION_RE = re.compile(
    r"(צור|צרי|תיצור|תיצרי|יצירת|טופס|טפסים|מזל|צירוף)",
    re.IGNORECASE,
)
ANALYZE_RE = re.compile(
    r"\b(analyze|analyse|analysis|what stands out|check)(?:.{0,60}(?:numbers?|my|these|those|the following|\d{1,2}))?",
    re.IGNORECASE,
)
HEBREW_ANALYZE_RE = re.compile(
    r"(נתח|ניתוח|מה בולט|בדוק)(?:.{0,60}(?:מספרים|המספרים|האלה|האלו|\d{1,2}))?",
    re.IGNORECASE,
)
HEBREW_ANALYZE_RE = re.compile(
    r"(נתח|ניתוח|מה בולט|בדוק|מספרים|המספרים|האלה|האלו)",
    re.IGNORECASE,
)

# ── Admin operation keyword regexes ──────────────────────────
ADMIN_SAVE_RE = re.compile(
    r"\b(save|keep|store).{0,50}(numbers?|form|combination)\b",
    re.IGNORECASE,
)
HEBREW_ADMIN_SAVE_RE = re.compile(
    r"(שמור|שמרי|תשמור|תשמרי|שמירת|שמירה).{0,50}(מספרים|טופס|צירוף)",
    re.IGNORECASE,
)
ADMIN_AUDIT_RE = re.compile(
    r"\b(audit|log|history|token usage|usage)\b",
    re.IGNORECASE,
)
HEBREW_ADMIN_AUDIT_RE = re.compile(
    r"(יומן|יומני|ביקורת|אבטחה|שימוש|טוקנים)",
    re.IGNORECASE,
)
ADMIN_CODE_RE = re.compile(
    r"\b(read (?:code|file)|edit(?:\s+(?:file|code))?|list files|list db tables|query db|search web|scraper)\b",
    re.IGNORECASE,
)
HEBREW_ADMIN_CODE_RE = re.compile(
    r"(קרא (?:קוד|קובץ)|ערוך(?:\s+(?:קובץ|קוד))?|רשימת קבצים|רשימת טבלאות|שאילתת db|חפש באינטרנט|סקרייפר)",
    re.IGNORECASE,
)

# ── Group-size detection (language-agnostic) ─────────────────
GROUP_SIZE_PATTERNS = [
    # Hebrew words
    ("he", re.compile(r"מספרים|מספר(?:\s|$)", re.IGNORECASE), 1),
    ("he", re.compile(r"זוגות|זוג(?:\s|$)", re.IGNORECASE), 2),
    ("he", re.compile(r"שלשות|שלישיות|שלשה|שלישית", re.IGNORECASE), 3),
    ("he", re.compile(r"רביעיות|רביעייה|רביעי", re.IGNORECASE), 4),
    ("he", re.compile(r"חמישיות|חמישייה|חמישי", re.IGNORECASE), 5),
    ("he", re.compile(r"שישיות|שישייה|שישי|שש(?:\s|$)", re.IGNORECASE), 6),
    # English words
    ("en", re.compile(r"\bsingles?\b|\bnumbers?\b(?!\s+pairs)", re.IGNORECASE), 1),
    ("en", re.compile(r"\bpairs?\b|\bdoub(?:le|les)?\b", re.IGNORECASE), 2),
    ("en", re.compile(r"\btriples?\b|\btriplets?\b|\bthrees?\b", re.IGNORECASE), 3),
    ("en", re.compile(r"\bquads?\b|\bquadruples?\b|\bfours?\b", re.IGNORECASE), 4),
    ("en", re.compile(r"\bquints?\b|\bquintuples?\b|\bfives?\b", re.IGNORECASE), 5),
    ("en", re.compile(r"\bsix(?:es|es|s)?\b|\bsix-number\b|\bfull combination", re.IGNORECASE), 6),
]

# ── Strength detection ───────────────────────────────────────
STRENGTH_HOT_RE = re.compile(r"\b(hot|warm|most frequent|highest|strong)\b", re.IGNORECASE)
STRENGTH_COLD_RE = re.compile(r"\b(cold|least frequent|lowest|weak)\b", re.IGNORECASE)
HEBREW_STRENGTH_HOT_RE = re.compile(r"חמים|חמות|חם|נפוצים|הכי שכיח", re.IGNORECASE)
HEBREW_STRENGTH_COLD_RE = re.compile(r"קרים|קרות|קר|הכי פחות", re.IGNORECASE)

# ── Number extraction ────────────────────────────────────────
NUMBER_RE = re.compile(r"\b(\d{1,2})\b")

# ── How many / top N ─────────────────────────────────────────
HOW_MANY_RE = re.compile(r"(?:top|show|give|get|first|generate|create|make)\s+(\d+)", re.IGNORECASE)
HEBREW_HOW_MANY_RE = re.compile(r"(?:ה\s*\d+|\d+)\s+(ראשונים|ראשונות|ראשון|ראשונה)", re.IGNORECASE)

# ── File path extraction ─────────────────────────────────────
FILE_PATH_RE = re.compile(r"\b([\w\-/.]+\.(?:py|yaml|yml|json|md|txt|java|go|ts|js|proto|sql))\b")


@dataclass
class ConversationState:
    """Conservative conversation-state for follow-up detection.

    Tracks the last NormalizedRequest so follow-ups can inherit compatible
    parameters. State is only used when a follow-up indicator is present.

    Clarification answers: when the last request had a `missing_hint` (the agent
    asked a clarification question), a short reply is treated as a follow-up
    answer even without conversational markers like "and" / "also".
    """
    last_request: Optional["NormalizedRequest"] = None

    def is_followup(self, message: str) -> bool:
        """Detect follow-up indicators in the message.

        A message is a follow-up if:
        1. It starts with or contains a conversational marker (and, also, גם, ...), OR
        2. The last request had a missing_hint and the reply is short (≤ 6 words)
           — this is a clarification answer.
        """
        msg = message.strip()
        followup_markers = [
            "and ", "also ", "what about ", "how about ", "and the ",
            "ו", "גם ", "מה לגבי ", "מה עם ", "ו",
        ]
        has_marker = any(
            msg.lower().startswith(m.strip()) or (" " + m.strip()) in msg.lower()
            for m in followup_markers
        ) or msg.lower() in {"and?", "?", "what about?", "and them?"}

        if has_marker:
            return True

        # Clarification answer: last request asked a question, reply is short.
        last = self.last_request
        if last and last.missing_hint:
            word_count = len(msg.split())
            if word_count <= 6:
                return True

        return False

    def compatible_params(self, current: "NormalizedRequest") -> dict:
        """Return compatible parameters from last_request that can be inherited.

        Normal case: only inherit when request kinds are compatible (e.g.
        statistics→statistics). Do not inherit across incompatible kinds.

        Clarification answer: when the last request had a missing_hint, the
        current message is answering the clarification, not starting a new
        request. In this case, inherit request_kind and operation from the
        last request, plus any previously-extracted params that are still
        missing in current.
        """
        last = self.last_request
        if not last:
            return {}

        # Clarification answer path: last request had a missing_hint.
        if last.missing_hint:
            compat: dict = {}
            # Inherit request_kind and operation so the tool resolver can
            # complete execution with the newly provided value.
            compat["request_kind"] = last.request_kind
            compat["operation"] = last.operation
            # Inherit previously-extracted params that are still missing.
            for field in ["group_size", "strength", "numbers", "how_many", "archive_window", "file_path", "category"]:
                current_val = getattr(current, field)
                last_val = getattr(last, field)
                if current_val is None and last_val is not None:
                    compat[field] = last_val
            return compat

        # Normal follow-up path: only inherit if kinds match.
        if current.request_kind != last.request_kind:
            return {}
        compat = {}
        for field in ["group_size", "strength", "numbers", "how_many", "archive_window", "file_path"]:
            current_val = getattr(current, field)
            last_val = getattr(last, field)
            if current_val is None and last_val is not None:
                compat[field] = last_val
        return compat


@dataclass
class NormalizedRequest:
    """Tier-agnostic semantic interpretation of the user request."""

    language: str = "en"                    # detected from message content
    request_kind: str = "out_of_scope"      # see REQUEST_KINDS
    operation: Optional[str] = None         # resolved tool/operation name (if any)
    group_size: Optional[int] = None        # 1-6
    strength: Optional[str] = None          # "hot" | "cold" | int
    numbers: Optional[list[int]] = None     # for analyze / save_numbers
    how_many: Optional[int] = None          # for get_statistics / generate_form
    archive_window: Optional[dict] = None   # {window_from, window_to}
    file_path: Optional[str] = None         # for read_code / edit_file / list_files
    sql: Optional[str] = None               # for query_db
    content: Optional[str] = None           # for edit_file content
    old_string: Optional[str] = None        # for edit_file patch
    new_string: Optional[str] = None        # for edit_file patch
    category: Optional[str] = None          # for save_numbers
    confidence: float = 0.0                 # semantic interpretation certainty [0,1]
    trivial_kind: Optional[str] = None      # "greeting" | "capabilities" | "goodbye"
    is_followup: bool = False               # follow-up indicator present?
    param_provenance: dict = field(default_factory=dict)  # field -> source
    client_intent_hint: Optional[str] = None  # untrusted client-provided intent
    missing_hint: Optional[str] = None      # predefined clarification text
    raw_message: str = ""                   # original message

    # Backwards compatibility: whether this request is a tool or a domain explanation
    requires_llm: bool = False              # true if a planner fallback is needed


# Valid request kinds
REQUEST_KINDS = {
    "trivial",
    "domain_explanation",
    "statistics",
    "number_analysis",
    "form_generation",
    "admin_operation",
    "out_of_scope",
    "ambiguous",
}


def _detect_language(message: str) -> str:
    """Detect language from message content. Hebrew chars → 'he', else 'en'."""
    if HEBREW_RE.search(message):
        return "he"
    return "en"


def _extract_numbers(message: str) -> list[int]:
    """Extract all 1-2 digit numbers from the message."""
    return [int(n) for n in NUMBER_RE.findall(message) if 1 <= int(n) <= 99]


def _extract_group_size(message: str, lang: str) -> Optional[int]:
    """Extract group size from explicit keywords (1-6)."""
    for l, pattern, size in GROUP_SIZE_PATTERNS:
        if l == lang and pattern.search(message):
            return size
    return None


def _extract_strength(message: str, lang: str) -> Optional[str]:
    """Extract hot/cold preference from the message."""
    if lang == "he":
        if HEBREW_STRENGTH_HOT_RE.search(message):
            return "hot"
        if HEBREW_STRENGTH_COLD_RE.search(message):
            return "cold"
    else:
        if STRENGTH_HOT_RE.search(message):
            return "hot"
        if STRENGTH_COLD_RE.search(message):
            return "cold"
    return None


def _extract_how_many(message: str, lang: str) -> Optional[int]:
    """Extract 'how many' / 'top N' from the message."""
    m = HOW_MANY_RE.search(message)
    if m:
        return int(m.group(1))
    if lang == "he":
        m = re.search(r"\b(\d+)\s+(ראשונים|ראשונות|ראשון|ראשונה)", message)
        if m:
            return int(m.group(1))
    return None


def _extract_file_path(message: str) -> Optional[str]:
    """Extract a file path from the message."""
    m = FILE_PATH_RE.search(message)
    return m.group(1) if m else None


def _extract_archive_window(message: str, context: Optional[dict]) -> Optional[dict]:
    """Extract archive window from message or context."""
    # TODO: support explicit date ranges in message text
    # For now, use the context if it contains archive/window keys.
    if context:
        window = {}
        for k, v in context.items():
            if k in ("window_from", "window_to"):
                window[k] = v
        return window if window else None
    return None


def _classify_request_kind(message: str, lang: str) -> tuple[str, Optional[str]]:
    """Classify the request into a tier-agnostic request_kind.

    Returns (request_kind, trivial_kind).
    """
    msg = message.strip()

    # 1. Trivial / small talk
    if GREETING_RE.search(message) or msg.lower() in {"hi", "hello", "שלום", "היי", "הי"}:
        return "trivial", "greeting"
    if GOODBYE_RE.search(message):
        return "trivial", "goodbye"
    if CAPABILITIES_RE.search(message) or HEBREW_CAPABILITIES_RE.search(message):
        return "trivial", "capabilities"

    # 2. Out of scope (deterministic)
    if OUT_OF_SCOPE_RE.search(message) or HEBREW_OUT_OF_SCOPE_RE.search(message):
        return "out_of_scope", None

    # 3. Admin operations
    if ADMIN_SAVE_RE.search(message) or HEBREW_ADMIN_SAVE_RE.search(message):
        return "admin_operation", None
    if ADMIN_AUDIT_RE.search(message) or HEBREW_ADMIN_AUDIT_RE.search(message):
        return "admin_operation", None
    if ADMIN_CODE_RE.search(message) or HEBREW_ADMIN_CODE_RE.search(message):
        return "admin_operation", None

    # 4. Form generation
    if FORM_GENERATION_RE.search(message) or HEBREW_FORM_GENERATION_RE.search(message):
        return "form_generation", None

    # 5. Number analysis
    if ANALYZE_RE.search(message) or HEBREW_ANALYZE_RE.search(message):
        return "number_analysis", None

    # 6. Statistics
    if STATISTICS_KEYWORDS.search(message) or HEBREW_STATISTICS_KEYWORDS.search(message):
        return "statistics", None

    # 7. Domain explanation (what is X)
    if re.search(r"\b(what is|what are|explain|how does|what does|מה זה|מה זאת|מהי|מהם)\b", message, re.IGNORECASE):
        return "domain_explanation", None

    # 8. Ambiguous
    if "?" in message or len(message.split()) < 4:
        return "ambiguous", None

    return "out_of_scope", None


def _resolve_operation(request_kind: str) -> Optional[str]:
    """Map request kind to the most likely operation name (preliminary).

    Operation names align with app.tool_resolver.OPERATION_SPECS keys.
    """
    mapping = {
        "statistics": "group_frequency",
        "number_analysis": "analyze_numbers",
        "form_generation": "generate_form",
        "admin_operation": None,  # determined by further admin keyword matching
    }
    return mapping.get(request_kind)


def _resolve_admin_operation(message: str, lang: str, file_path: Optional[str] = None) -> Optional[str]:
    """Determine the specific admin operation from the message.""

    Args:
        message: raw user message.
        lang: detected language.
        file_path: extracted file path, if any.
    """
    if ADMIN_SAVE_RE.search(message) or HEBREW_ADMIN_SAVE_RE.search(message):
        return "save_numbers"
    if re.search(r"\b(audit log|audit)\b", message, re.IGNORECASE) or \
       re.search(r"(יומן הפעולות|יומן)", message, re.IGNORECASE):
        return "query_audit_log"
    if re.search(r"\b(token usage|usage)\b", message, re.IGNORECASE) or \
       re.search(r"(שימוש טוקנים|טוקנים|שימוש)", message, re.IGNORECASE):
        return "read_token_usage"
    if re.search(r"\b(scraper)\b", message, re.IGNORECASE) or \
       re.search(r"(סקרייפר|סרוק|לסרוק)", message, re.IGNORECASE):
        return "trigger_scraper"
    if re.search(r"\b(search web)\b", message, re.IGNORECASE) or \
       re.search(r"(חפש באינטרנט|חיפוש ברשת)", message, re.IGNORECASE):
        return "search_web"
    if re.search(r"\bread (?:code|file)\b", message, re.IGNORECASE) or \
       re.search(r"(קרא קוד|קרא קובץ)", message, re.IGNORECASE):
        return "read_code"
    if re.search(r"\bedit (?:file|code)\b", message, re.IGNORECASE) or \
       re.search(r"(ערוך קובץ|ערוך קוד)", message, re.IGNORECASE):
        return "edit_file"
    if re.search(r"\blist files\b", message, re.IGNORECASE) or \
       re.search(r"(רשימת קבצים|הצג קבצים)", message, re.IGNORECASE):
        return "list_files"
    if re.search(r"\blist db tables\b", message, re.IGNORECASE) or \
       re.search(r"(רשימת טבלאות|הצג טבלאות)", message, re.IGNORECASE):
        return "list_db_tables"
    if re.search(r"\bquery db\b", message, re.IGNORECASE) or \
       re.search(r"(שאילתת db|שאילתת מסד)", message, re.IGNORECASE):
        return "query_db"

    # Fallback: if a file path is present and the user said edit/read, resolve.
    if file_path and (re.search(r"\b(edit|change)\b", message, re.IGNORECASE) or \
                     re.search(r"(ערוך|שנה|החלף)", message, re.IGNORECASE)):
        return "edit_file"
    if file_path and (re.search(r"\b(read|show)\b", message, re.IGNORECASE) or \
                     re.search(r"(קרא|הצג)", message, re.IGNORECASE)):
        return "read_code"
    return None


def _compute_confidence(request_kind: str, message: str, has_required: bool) -> float:
    """Compute semantic interpretation confidence (NOT argument completeness)."""
    if request_kind == "trivial":
        return 1.0
    if request_kind == "out_of_scope" and len(message.strip()) > 0:
        return 1.0
    if request_kind in ("statistics", "number_analysis", "form_generation", "admin_operation"):
        # Confidence is high if we identified the operation intent clearly.
        # Missing required parameters does NOT lower confidence.
        return 0.9 if has_required else 0.85
    if request_kind == "domain_explanation":
        return 0.9
    if request_kind == "ambiguous":
        return 0.4
    return 0.7


def _predefined_clarification(request_kind: str, operation: Optional[str], language: str) -> Optional[str]:
    """Return a deterministic missing-parameter clarification if known."""
    if request_kind == "statistics" and operation == "group_frequency":
        if language == "he":
            return "באיזה גודל קבוצה תרצה לבדוק? לדוגמה: זוגות (2), שלשות (3), מספרים בודדים (1)."
        return "What group size would you like? E.g. pairs (2), triples (3), singles (1)."

    if request_kind == "admin_operation" and operation == "edit_file":
        if language == "he":
            return "מה התוכן או השינוי שתרצה לבצע בקובץ?"
        return "What content or change would you like to apply to the file?"

    if request_kind == "admin_operation" and operation == "query_db":
        if language == "he":
            return "מהי שאילתת ה-SQL שתרצה להריץ? (לקריאה בלבד)"
        return "What SQL query would you like to run? (read-only)"

    if request_kind == "form_generation" and operation == "generate_form":
        if language == "he":
            return "כמה טפסים תרצה ליצור?"
        return "How many forms would you like to generate?"

    if request_kind == "number_analysis" and operation == "analyze_numbers":
        if language == "he":
            return "אילו מספרים תרצה לנתח?"
        return "Which numbers would you like to analyze?"

    return None


def _extract_clarification_value(
    message: str,
    last_request: NormalizedRequest,
) -> dict:
    """Extract a clarification answer based on what was asked.

    When the last request had a missing_hint, the user's reply is an answer
    to that specific question. This helper extracts the value based on the
    operation and the missing parameter.

    Returns a dict of {field: value} for the extracted clarification value.
    """
    if not last_request or not last_request.missing_hint or not last_request.operation:
        return {}

    op = last_request.operation
    msg = message.strip()

    # generate_form: "how_many" was asked → bare number = how_many
    if op == "generate_form":
        m = re.match(r"^(\d+)$", msg)
        if m:
            return {"how_many": int(m.group(1))}
        # "10 forms" / "10 טפסים"
        m = re.search(r"(\d+)", msg)
        if m:
            return {"how_many": int(m.group(1))}

    # group_frequency: "group_size" was asked → bare number (1-6) or group keyword
    if op == "group_frequency":
        # Try group-size keywords first (handles "שלשות", "pairs", etc.)
        lang = last_request.language
        gs = _extract_group_size(msg, lang)
        if gs is not None:
            return {"group_size": gs}
        # Bare number 1-6
        m = re.match(r"^(\d)$", msg)
        if m and 1 <= int(m.group(1)) <= 6:
            return {"group_size": int(m.group(1))}

    # group_frequency: "strength" was asked → hot/cold
    if op == "group_frequency":
        lang = last_request.language
        strength = _extract_strength(msg, lang)
        if strength is not None:
            return {"strength": strength}

    # analyze_numbers: "form" (numbers) was asked → extract numbers
    if op == "analyze_numbers":
        nums = _extract_numbers(msg)
        if nums:
            return {"numbers": nums}

    # save_numbers: "numbers" was asked → extract numbers
    if op == "save_numbers":
        nums = _extract_numbers(msg)
        if nums:
            return {"numbers": nums}

    # search_web: "query" was asked → use the raw message as the query
    if op == "search_web":
        if msg:
            return {"raw_message": msg}

    return {}


def _apply_state_inheritance(
    request: NormalizedRequest,
    conversation: Optional[ConversationState],
) -> NormalizedRequest:
    """Conservatively inherit compatible parameters from conversation state.

    Clarification answers: when the last request had a missing_hint, the
    current message is treated as an answer. We:
    1. Extract the clarification value from the message.
    2. Inherit request_kind, operation, and previously-extracted params.
    3. Merge the clarification value into the request.
    4. Re-compute missing_hint and confidence based on the merged state.
    """
    if not conversation or not conversation.is_followup(request.raw_message):
        request.is_followup = False
        return request

    last = conversation.last_request
    if not last:
        request.is_followup = False
        return request

    request.is_followup = True

    # If this is a clarification answer, extract the value first.
    clarification_values: dict = {}
    if last.missing_hint:
        clarification_values = _extract_clarification_value(request.raw_message, last)

    compat = conversation.compatible_params(request)
    for field, value in compat.items():
        setattr(request, field, value)
        request.param_provenance[field] = "state"

    # Merge clarification values (these take priority over inherited values).
    for field, value in clarification_values.items():
        setattr(request, field, value)
        request.param_provenance[field] = "clarification"

    # Re-compute missing_hint and confidence if we inherited request_kind/operation.
    if last.missing_hint and ("request_kind" in compat or "operation" in compat):
        request.missing_hint = _predefined_clarification(request.request_kind, request.operation, request.language)
        has_required = request.operation is not None
        request.confidence = _compute_confidence(request.request_kind, request.raw_message, has_required)

    return request


def normalize(
    message: str,
    lang_hint: Optional[str] = None,
    context: Optional[dict] = None,
    conversation: Optional[ConversationState] = None,
    client_intent_hint: Optional[str] = None,
) -> NormalizedRequest:
    """Normalize a raw user message into a tier-agnostic NormalizedRequest.

    Args:
        message: raw user message text.
        lang_hint: optional client-provided language hint (not authoritative).
        context: structured UI context (page, groupSize, numbers, etc.).
        conversation: optional conversation state for follow-up detection.
        client_intent_hint: untrusted client-provided intent string.

    Returns:
        NormalizedRequest with language, request_kind, operation, entities,
        confidence, and provenance. execution_ready is intentionally NOT set.
    """
    language = _detect_language(message)
    # Client lang hint is kept only for backwards compatibility/filtering,
    # but content-derived language is authoritative.
    raw_message = message

    request_kind, trivial_kind = _classify_request_kind(message, language)
    operation = _resolve_operation(request_kind)

    provenance: dict[str, str] = {"language": "message", "request_kind": "message"}

    # Extract common entities
    numbers = _extract_numbers(message)
    if numbers:
        provenance["numbers"] = "message"

    group_size = _extract_group_size(message, language)
    if group_size is not None:
        provenance["group_size"] = "message"
    elif context and context.get("groupSize") is not None:
        group_size = int(context["groupSize"]) if 1 <= int(context["groupSize"]) <= 6 else None
        if group_size is not None:
            provenance["group_size"] = "context"

    strength = _extract_strength(message, language)
    if strength is not None:
        provenance["strength"] = "message"
    elif context and context.get("ordering") in ("hot", "cold"):
        strength = context["ordering"]
        provenance["strength"] = "context"

    how_many = _extract_how_many(message, language)
    if how_many is not None:
        provenance["how_many"] = "message"
    elif context and context.get("howMany") is not None:
        how_many = int(context["howMany"])
        provenance["how_many"] = "context"

    archive_window = _extract_archive_window(message, context)
    if archive_window:
        provenance["archive_window"] = "context"

    file_path = _extract_file_path(message)
    if file_path:
        provenance["file_path"] = "message"

    # Admin-specific operation refinement
    if request_kind == "admin_operation":
        admin_op = _resolve_admin_operation(message, language, file_path=file_path)
        if admin_op:
            operation = admin_op
            provenance["operation"] = "message"
        # Extract category for save_numbers
        if operation == "save_numbers":
            category = "default"
            m = re.search(r"\b(?:as|under|category)\s+(\w+)", message, re.IGNORECASE)
            if m:
                category = m.group(1)
            provenance["category"] = "default"
        else:
            category = None
    else:
        category = None

    # Compute confidence: clear operation identification with some entities.
    has_required = operation is not None
    confidence = _compute_confidence(request_kind, message, has_required)

    # Only set missing_hint when a required parameter is actually missing.
    # This prevents false follow-up detection when the request is complete.
    missing_hint = _predefined_clarification(request_kind, operation, language)
    if missing_hint:
        # Check if the operation's required params are all present.
        if operation == "group_frequency" and group_size is not None and strength is not None:
            missing_hint = None
        elif operation == "generate_form" and how_many is not None:
            missing_hint = None
        elif operation == "analyze_numbers" and numbers:
            missing_hint = None
        elif operation == "save_numbers" and numbers:
            missing_hint = None

    request = NormalizedRequest(
        language=language,
        request_kind=request_kind,
        operation=operation,
        group_size=group_size,
        strength=strength,
        numbers=numbers if numbers else None,
        how_many=how_many,
        archive_window=archive_window,
        file_path=file_path,
        category=category,
        confidence=confidence,
        trivial_kind=trivial_kind,
        is_followup=False,
        param_provenance=provenance,
        client_intent_hint=client_intent_hint,
        missing_hint=missing_hint,
        raw_message=raw_message,
        requires_llm=False,
    )

    # Apply conservative state inheritance
    request = _apply_state_inheritance(request, conversation)

    return request
