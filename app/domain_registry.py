"""Domain registry — small canonical definitions for deterministic responses.

Used for quick, deterministic domain explanations where an LLM call is not needed.
All explanations are translated and include the grounding disclaimer.
"""

from __future__ import annotations


_DOMAIN_DEFINITIONS = {
    "en": {
        "hot": "Hot numbers or groups appeared more frequently in past draws within the chosen archive window.",
        "cold": "Cold numbers or groups appeared less frequently in past draws within the chosen archive window.",
        "pair": "A pair is two numbers that appeared together in the same draw (group size 2).",
        "triple": "A triple is three numbers that appeared together in the same draw (group size 3).",
        "quad": "A quad is four numbers that appeared together in the same draw (group size 4).",
        "quint": "A quint is five numbers that appeared together in the same draw (group size 5).",
        "six": "A six-number group is the complete regular lottery combination, excluding the strong number (group size 6).",
        "archive_window": "The archive window controls which historical draws are included in the analysis. A narrower window (e.g. last 100 draws) may show different patterns than the full archive.",
        "group_size": "Group size (1-6) is the number of numbers analyzed together: 1 singles, 2 pairs, 3 triples, 4 quads, 5 quints, 6 full six-number groups.",
        "strong_number": "The strong number is always separate from the six-number group and is never part of it.",
        "probability": "Each 6-number combination has a 1 in 2,324,784 chance; including the strong number, the first-prize odds are 1 in 16,273,488. Every draw is independent and random — historical frequency does not increase the chance of a number being drawn next.",
        "lucky_numbers": "Lucky numbers are your chosen numbers that will appear in every generated form. They are pinned, not statistically selected.",
        "luck": "In a fair lottery every combination is equally likely — no numbers are inherently luckier than others, and lucky charms or rituals do not change the odds.",
        "saved_numbers": "Saved numbers are sets you previously bookmarked in your wallet. You can reuse them as lucky numbers or for analysis.",
    },
    "he": {
        "hot": "מספרים או קבוצות חמים הופיעו בתדירות גבוהה יותר בהגרלות עבר בטווח הנבחר.",
        "cold": "מספרים או קבוצות קרים הופיעו בתדירות נמוכה יותר בהגרלות עבר בטווח הנבחר.",
        "pair": "זוג הוא שני מספרים שהופיעו יחד באותה הגרלה (גודל קבוצה 2).",
        "triple": "שלשה היא שלושה מספרים שהופיעו יחד באותה הגרלה (גודל קבוצה 3).",
        "quad": "רביעייה היא ארבעה מספרים שהופיעו יחד באותה הגרלה (גודל קבוצה 4).",
        "quint": "חמישייה היא חמישה מספרים שהופיעו יחד באותה הגרלה (גודל קבוצה 5).",
        "six": "שישייה היא צירוף מספרי הלוטו המלא, לא כולל המספר החזק (גודל קבוצה 6).",
        "archive_window": "טווח הגרלות קובע אילו הגרלות היסטוריות ייכללו בניתוח. טווח צר יותר (למשל 100 ההגרלות האחרונות) עשוי להראות דפוסים שונים מהארכיון המלא.",
        "group_size": "גודל קבוצה (1-6) הוא מספר המספרים שנבדקים יחד: 1 מספרים בודדים, 2 זוגות, 3 שלשות, 4 רביעיות, 5 חמישיות, 6 שישיות מלאות.",
        "strong_number": "המספר החזק תמיד נפרד מקבוצת השישה ולעולם לא חלק ממנה.",
        "probability": "לכל צירוף של 6 מספרים סיכוי של 1 ל-2,324,784; עם המספר החזק, הסיכוי לזכות בפרס הראשון הוא 1 ל-16,273,488. כל הגרלה היא עצמאית ואקראית — תדירות היסטורית אינה מגדילה את הסיכוי שמספר יוגרל.",
        "lucky_numbers": "מספרי מזל הם מספרים שבחרת שיופיעו בכל טופס שייווצר. הם קבועים, לא נבחרים סטטיסטית.",
        "luck": "בלוטו הוגן לכל צירוף סיכוי זהה — אין מספרים 'ברי מזל' מלידה, וקמעות או טקסים אינם משנים את הסיכויים.",
        "saved_numbers": "מספרים שמורים הם סטים ששמרת בעבר בארנק שלך. תוכל לעשות בהם שימוש חוזר כמספרי מזל או לניתוח.",
    },
}

_GROUNDING_DISCLAIMER = {
    "en": "These are historical observations only and do not imply a higher probability in the next draw.",
    "he": "אלה נתוני עבר בלבד; הם לא מעידים על סיכוי גבוה יותר בהגרלה הבאה.",
}


def get_domain_definition(term: str, language: str = "en") -> str | None:
    """Return a deterministic explanation for a known domain term, or None."""
    defs = _DOMAIN_DEFINITIONS.get(language, _DOMAIN_DEFINITIONS["en"])
    return defs.get(term)


def get_disclaimer(language: str = "en") -> str:
    """Return the grounding disclaimer in the requested language."""
    return _GROUNDING_DISCLAIMER.get(language, _GROUNDING_DISCLAIMER["en"])


def explain(term: str, language: str = "en") -> str | None:
    """Return a full deterministic explanation with disclaimer, or None."""
    definition = get_domain_definition(term, language)
    if not definition:
        return None
    return f"{definition} {get_disclaimer(language)}"


def detect_topic(message: str) -> str | None:
    """Detect a known domain topic in a message for deterministic explanation."""
    msg = message.lower()
    # Hot
    if re.search(r"\b(hot|warm)\b", msg) or any(w in msg for w in ["חם", "חמים", "חמות"]):
        return "hot"
    # Cold
    if re.search(r"\b(cold)\b", msg) or any(w in msg for w in ["קר", "קרים", "קרות"]):
        return "cold"
    # Pair
    if re.search(r"\bpairs?\b", msg) or "זוג" in msg or "זוגות" in msg:
        return "pair"
    # Triple
    if re.search(r"\btriples?\b", msg) or "שלש" in msg or "שלשות" in msg:
        return "triple"
    # Quad
    if re.search(r"\bquads?\b", msg) or "רביע" in msg:
        return "quad"
    # Quint
    if re.search(r"\bquints?\b", msg) or "חמיש" in msg:
        return "quint"
    # Six
    if re.search(r"\bsix(?:es|s)?\b", msg) or "שיש" in msg or "שש" in msg:
        return "six"
    # Archive window
    if re.search(r"\barchive window\b", msg) or "טווח" in msg or "חלון" in msg:
        return "archive_window"
    # Group size
    if re.search(r"\bgroup size\b", msg) or "גודל קבוצה" in msg:
        return "group_size"
    # Strong number
    if re.search(r"\bstrong number\b", msg) or "מספר חזק" in msg:
        return "strong_number"
    # Probability
    if re.search(r"\b(probability|odds|chance)\b", msg) or any(w in msg for w in ["הסתברות", "סיכוי", "סיכויים"]):
        return "probability"
    # Lucky numbers
    if re.search(r"\b(lucky numbers?|pinned numbers?)\b", msg) or "מספרי מזל" in msg:
        return "lucky_numbers"
    # Saved numbers
    if re.search(r"\b(saved numbers?|bookmarked numbers?)\b", msg) or "מספרים שמורים" in msg:
        return "saved_numbers"
    # Luck (checked last so "מספרי מזל" still hits lucky_numbers above)
    if re.search(r"\b(luck|lucky|superstition)\b", msg) or "מזל" in msg:
        return "luck"
    return None


import re  # noqa: E402
