"""Deterministic response renderer — zero-LLM responses for well-defined paths.

Used by the supervisor for:
  - greetings / goodbyes
  - capabilities
  - unauthorized requests
  - missing known parameters
  - clear out-of-scope requests
  - simple structured results
  - empty results
  - tool errors

All responses are language-aware and include grounding disclaimers where relevant.
"""

from __future__ import annotations

from app.capability import CapabilityConfig
from app.domain_registry import get_disclaimer


def render_greeting(language: str = "en") -> str:
    if language == "he":
        return "שלום! אני עוזר ללוטו. במה אוכל לעזור?"
    return "Hello! I'm your lottery intelligence assistant. What would you like to know?"


def render_goodbye(language: str = "en") -> str:
    if language == "he":
        return "תודה! אם תזדקק לעזרה נוספת, אני כאן."
    return "You're welcome! Feel free to ask if you need anything else."


def render_capabilities(tier: str, language: str = "en") -> str:
    return CapabilityConfig.render_user_capabilities(tier, language)


def render_unauthorized(tool: str, tier: str, language: str = "en") -> str:
    if language == "he":
        return f"הפעולה '{tool}' אינה זמינה בדרגה הנוכחית שלך. אפשר לשדרג כדי לקבל גישה, או לשאול על יכולות אחרות שזמינות לך."
    return f"The action '{tool}' is not available on your current tier. You can upgrade to access it, or ask about other available features."


def render_missing(missing_hint: str | None, language: str = "en") -> str:
    if missing_hint:
        return missing_hint
    if language == "he":
        return "אני צריך עוד פרטים כדי להמשיך. תוכל לפרט?"
    return "I need a bit more information to proceed. Can you clarify?"


def render_out_of_scope(language: str = "en") -> str:
    if language == "he":
        return "אני עוזר ללוטו בלבד. אני יכול להסביר על מספרים חמים וקרים, זוגות, שלשות, וגדלי קבוצות. האם יש משהו קשור ללוטו שאוכל לעזור בו?"
    return "I'm a lottery intelligence assistant. I can explain hot/cold numbers, pairs, triples, group sizes, and the archive window. Is there a lottery-related question I can help with?"


def render_free_generic(language: str = "en") -> str:
    """Generic deterministic response for free-tier ambiguous/domain requests.

    Returned when free-tier LLM is disabled (the default). Tells the user
    what they CAN do deterministically without invoking the LLM.
    """
    if language == "he":
        return (
            "אני יכול לעזור עם פעולות ספציפיות בחינם: לייצר טפסים, להציג סטטיסטיקות "
            "(מספרים חמים/קרים, זוגות, שלשות), ולנתח מספרים. לשאלות כלליות או הסברים "
            "מפורטים, נדרשת דרגה בתשלום. אנא נסח מחדש את הבקשה עם פעולה ספציפית."
        )
    return (
        "I can help with specific actions on the free tier: generate forms, "
        "show statistics (hot/cold numbers, pairs, triples), and analyze numbers. "
        "For general questions or detailed explanations, a paid plan is required. "
        "Please rephrase your request with a specific action."
    )


def render_empty_statistics(language: str = "en") -> str:
    if language == "he":
        return "לא נמצאו נתונים סטטיסטיים. ייתכן שהטווח שנבחר ריק או שלא הוגדרו מספיק הגרלות."
    return "No statistical data is available. The selected archive window may be empty or too narrow."


def render_empty_audit_log(language: str = "en") -> str:
    if language == "he":
        return "לא נמצאו רשומות ביומן הפעולות לטווח המבוקש."
    return "No audit entries were found for the requested range."


def render_tool_error(language: str = "en") -> str:
    if language == "he":
        return "לא הצלחתי לאחזר את התוצאה כרגע. אנא נסה שוב מאוחר יותר."
    return "I couldn't retrieve the result right now. Please try again later."


def render_multi_request(requests: list[str], language: str = "en") -> str:
    """Render a message asking the user to pick one request at a time.

    Used by the supervisor's ``direct_multi_request`` node when the user
    message contains multiple distinct operations (e.g. "generate 5 forms
    and analyze 1,2,3").
    """
    if language == "he":
        header = "אני יכול לטפל בבקשה אחת בכל פעם. אנא בחר אחת:"
        lines = [header]
        for i, req in enumerate(requests, 1):
            lines.append(f"{i}. {req}")
        return "\n".join(lines)
    header = "I can handle one request at a time. Please pick one:"
    lines = [header]
    for i, req in enumerate(requests, 1):
        lines.append(f"{i}. {req}")
    return "\n".join(lines)


def render_statistics_result(groups: list, how_many: int, group_size: int, strength: str, language: str = "en") -> str:
    """Render a get_statistics result as readable text without an LLM."""
    if not groups:
        return render_empty_statistics(language)
    size_words = {
        1: "singles" if language == "en" else "מספרים בודדים",
        2: "pairs" if language == "en" else "זוגות",
        3: "triples" if language == "en" else "שלשות",
        4: "quads" if language == "en" else "רביעיות",
        5: "quints" if language == "en" else "חמישיות",
        6: "six-number groups" if language == "en" else "שישיות",
    }
    size_word = size_words.get(group_size, "groups" if language == "en" else "קבוצות")
    strength_word = "hot" if strength in ("hot", 2) else "cold" if strength in ("cold", 1) else str(strength)

    if language == "he":
        lines = [f"ה-{len(groups)} {size_word} ה{strength_word}ים:"]
        for i, g in enumerate(groups, 1):
            nums = g.get("numbers", [])
            count = g.get("count", 0)
            nums_str = " + ".join(str(n) for n in nums)
            lines.append(f"{i}. {nums_str} — הופיע {count} פעמים")
        lines.append(get_disclaimer(language))
    else:
        lines = [f"The top {len(groups)} {strength_word} {size_word}:\n"]
        for i, g in enumerate(groups, 1):
            nums = g.get("numbers", [])
            count = g.get("count", 0)
            nums_str = " + ".join(str(n) for n in nums)
            lines.append(f"{i}. {nums_str} — {count} appearances")
        lines.append(get_disclaimer(language))

    return "\n".join(lines)


def render_generate_form_result(forms: list, how_many: int, language: str = "en") -> str:
    """Render a generate_form result as readable text without an LLM."""
    if not forms:
        if language == "he":
            return "לא ניתן היה ליצור טפסים. נסה עם פרמטרים שונים."
        return "No forms could be generated. Try with different parameters."

    if language == "he":
        lines = [f"הטפסים שנוצרו ({len(forms)}):"]
        for i, form in enumerate(forms, 1):
            nums = form.get("numbers", [])
            nums_str = " + ".join(str(n) for n in nums)
            strong = form.get("strong")
            if strong is not None:
                lines.append(f"{i}. {nums_str} | חזק: {strong}")
            else:
                lines.append(f"{i}. {nums_str}")
        lines.append(get_disclaimer(language))
    else:
        lines = [f"Generated forms ({len(forms)}):"]
        for i, form in enumerate(forms, 1):
            nums = form.get("numbers", [])
            nums_str = " + ".join(str(n) for n in nums)
            strong = form.get("strong")
            if strong is not None:
                lines.append(f"{i}. {nums_str} | Strong: {strong}")
            else:
                lines.append(f"{i}. {nums_str}")
        lines.append(get_disclaimer(language))

    return "\n".join(lines)


def render_analyze_result(frequency_groups: list, archive_size: int, language: str = "en") -> str:
    """Render an analyze result as readable text without an LLM.

    Shows the top entries per group size (singles, pairs, triples, etc.)
    found in the user's selected numbers against historical draws.
    """
    if not frequency_groups:
        if language == "he":
            return "לא נמצאו תדירויות עבור המספרים שנבחרו."
        return "No frequency data found for the selected numbers."

    size_words = {
        1: "singles" if language == "en" else "מספרים בודדים",
        2: "pairs" if language == "en" else "זוגות",
        3: "triples" if language == "en" else "שלשות",
        4: "quads" if language == "en" else "רביעיות",
        5: "quints" if language == "en" else "חמישיות",
        6: "six-number groups" if language == "en" else "שישיות",
    }

    if language == "he":
        lines = [f"ניתוח המספרים שלך (מתוך {archive_size} הגרלות):"]
        for group in frequency_groups:
            size = group.get("size", 0)
            entries = group.get("entries", [])
            if not entries:
                continue
            size_word = size_words.get(size, f"גודל {size}")
            lines.append(f"\n{size_word}:")
            for entry in entries:
                nums = entry.get("numbers", [])
                count = entry.get("count", 0)
                nums_str = " + ".join(str(n) for n in nums)
                lines.append(f"  {nums_str} — הופיע {count} פעמים")
        lines.append(get_disclaimer(language))
    else:
        lines = [f"Analysis of your numbers (from {archive_size} draws):"]
        for group in frequency_groups:
            size = group.get("size", 0)
            entries = group.get("entries", [])
            if not entries:
                continue
            size_word = size_words.get(size, f"size {size}")
            lines.append(f"\n{size_word}:")
            for entry in entries:
                nums = entry.get("numbers", [])
                count = entry.get("count", 0)
                nums_str = " + ".join(str(n) for n in nums)
                lines.append(f"  {nums_str} — {count} appearances")
        lines.append(get_disclaimer(language))

    return "\n".join(lines)


def render_structured_result(tool: str, result: dict, language: str = "en") -> str:
    """Render a tool result for simple cases (zero LLM calls)."""
    if tool == "get_statistics":
        return render_statistics_result(
            result.get("groups", []),
            result.get("how_many", 10),
            result.get("group_size", 2),
            result.get("strength", "hot"),
            language,
        )
    if tool == "generate_form":
        return render_generate_form_result(
            result.get("forms", []),
            result.get("how_many", 1),
            language,
        )
    if tool == "analyze":
        return render_analyze_result(
            result.get("frequency_groups", []),
            result.get("archive_size", 0),
            language,
        )
    if "error" in result:
        return render_tool_error(language)
    # Generic: avoid raw JSON, return a concise summary.
    if language == "he":
        return "הפעולה הושלמה. התוצאה זמינה במערכת."
    return "The action completed successfully. The result is available in the system."
