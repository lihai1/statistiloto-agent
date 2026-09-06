"""Unit tests for language-neutral message inheritance (bug 7).

A message that contains only digits/whitespace/punctuation carries no language
signal. When such a message arrives within an ongoing conversation, the
normalizer inherits the language from the prior turn instead of falling back to
the default ("en").
"""

from app.normalizer import (
    normalize,
    ConversationState,
    NormalizedRequest,
    _is_language_neutral,
)


def _conv_with_language(lang: str) -> ConversationState:
    """Build a ConversationState whose last_request has the given language."""
    last = NormalizedRequest(language=lang)
    return ConversationState(last_request=last)


class TestIsLanguageNeutral:
    def test_pure_number_is_neutral(self):
        assert _is_language_neutral("1") is True

    def test_pure_numbers_with_punctuation_is_neutral(self):
        assert _is_language_neutral("1, 2, 3") is True

    def test_empty_string_is_neutral(self):
        assert _is_language_neutral("") is True

    def test_whitespace_only_is_neutral(self):
        assert _is_language_neutral("   ") is True

    def test_latin_letters_not_neutral(self):
        assert _is_language_neutral("hello 1") is False

    def test_hebrew_not_neutral(self):
        assert _is_language_neutral("שלום") is False

    def test_mixed_latin_hebrew_not_neutral(self):
        assert _is_language_neutral("hot זוגות") is False


class TestLanguageInheritance:
    def test_pure_number_inherits_hebrew_from_conversation(self):
        """'1' with conversation language='he' → inherits 'he'."""
        conv = _conv_with_language("he")
        req = normalize(message="1", conversation=conv)
        assert req.language == "he"
        assert req.param_provenance["language"] == "conversation"

    def test_pure_number_inherits_english_from_conversation(self):
        """'1' with conversation language='en' → inherits 'en'."""
        conv = _conv_with_language("en")
        req = normalize(message="1", conversation=conv)
        assert req.language == "en"
        assert req.param_provenance["language"] == "conversation"

    def test_pure_number_no_conversation_falls_back_to_en(self):
        """'1' with no conversation state → falls back to default 'en'."""
        req = normalize(message="1")
        assert req.language == "en"
        assert req.param_provenance["language"] == "message"

    def test_pure_number_no_last_request_falls_back_to_en(self):
        """'1' with conversation but no last_request → falls back to 'en'."""
        conv = ConversationState(last_request=None)
        req = normalize(message="1", conversation=conv)
        assert req.language == "en"
        assert req.param_provenance["language"] == "message"

    def test_mixed_language_message_does_not_inherit(self):
        """'hello 1' has Latin letters → uses detected language, no inheritance."""
        conv = _conv_with_language("he")
        req = normalize(message="hello 1", conversation=conv)
        assert req.language == "en"  # detected from Latin letters
        assert req.param_provenance["language"] == "message"

    def test_hebrew_message_does_not_inherit(self):
        """'שלום' is Hebrew → uses 'he' regardless of conversation state."""
        conv = _conv_with_language("en")
        req = normalize(message="שלום", conversation=conv)
        assert req.language == "he"
        assert req.param_provenance["language"] == "message"

    def test_empty_message_no_conversation_falls_back_to_en(self):
        """Empty message with no conversation → falls back to 'en'."""
        req = normalize(message="")
        assert req.language == "en"
        assert req.param_provenance["language"] == "message"

    def test_whitespace_only_message_inherits_from_conversation(self):
        """Whitespace-only message with conversation → inherits language."""
        conv = _conv_with_language("he")
        req = normalize(message="   ", conversation=conv)
        assert req.language == "he"
        assert req.param_provenance["language"] == "conversation"

    def test_punctuation_only_message_inherits_from_conversation(self):
        """Punctuation-only message with conversation → inherits language."""
        conv = _conv_with_language("he")
        req = normalize(message="???", conversation=conv)
        assert req.language == "he"
        assert req.param_provenance["language"] == "conversation"
