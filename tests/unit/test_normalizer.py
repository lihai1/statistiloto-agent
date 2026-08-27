"""Unit tests for the semantic normalizer."""
import pytest

from app.normalizer import (
    normalize,
    ConversationState,
    NormalizedRequest,
    _detect_language,
    _extract_group_size,
    _extract_strength,
    _extract_numbers,
)


class TestLanguageDetection:
    def test_hebrew_message(self):
        assert _detect_language("מה הזוגות החמים?") == "he"

    def test_english_message(self):
        assert _detect_language("What are the hot pairs?") == "en"

    def test_mixed_message_prefers_hebrew(self):
        assert _detect_language("hot pairs זוגות") == "he"


class TestEntityExtraction:
    def test_extract_numbers(self):
        assert _extract_numbers("Analyze 7, 11, 17, 24, 31, 36") == [7, 11, 17, 24, 31, 36]

    def test_extract_group_size_pairs_en(self):
        assert _extract_group_size("What are the hot pairs?", "en") == 2

    def test_extract_group_size_triples_en(self):
        assert _extract_group_size("Show me cold triples", "en") == 3

    def test_extract_group_size_singles_he(self):
        assert _extract_group_size("מה המספרים החמים?", "he") == 1

    def test_extract_group_size_pairs_he(self):
        assert _extract_group_size("מה הזוגות החמים?", "he") == 2

    def test_extract_strength_hot(self):
        assert _extract_strength("hot pairs", "en") == "hot"

    def test_extract_strength_cold(self):
        assert _extract_strength("cold triples", "en") == "cold"

    def test_extract_strength_hot_he(self):
        assert _extract_strength("זוגות חמים", "he") == "hot"

    def test_extract_strength_cold_he(self):
        assert _extract_strength("שלשות קרות", "he") == "cold"


class TestNormalize:
    def test_greeting_en(self):
        req = normalize("Hi")
        assert req.request_kind == "trivial"
        assert req.trivial_kind == "greeting"
        assert req.language == "en"
        assert req.confidence == 1.0

    def test_greeting_he(self):
        req = normalize("שלום")
        assert req.request_kind == "trivial"
        assert req.trivial_kind == "greeting"
        assert req.language == "he"

    def test_capabilities_en(self):
        req = normalize("What can you do?")
        assert req.request_kind == "trivial"
        assert req.trivial_kind == "capabilities"

    def test_capabilities_he(self):
        req = normalize("מה אתה יכול לעשות?")
        assert req.request_kind == "trivial"
        assert req.trivial_kind == "capabilities"

    def test_out_of_scope_weather(self):
        req = normalize("What's the weather?")
        assert req.request_kind == "out_of_scope"
        assert req.confidence == 1.0

    def test_out_of_scope_joke(self):
        req = normalize("ספר בדיחה")
        assert req.request_kind == "out_of_scope"

    def test_statistics_hot_pairs_en(self):
        req = normalize("What are the hot pairs?")
        assert req.request_kind == "statistics"
        assert req.operation == "group_frequency"
        assert req.group_size == 2
        assert req.strength == "hot"
        assert req.language == "en"

    def test_statistics_cold_triples_he(self):
        req = normalize("מה השלשות הקרות?")
        assert req.request_kind == "statistics"
        assert req.operation == "group_frequency"
        assert req.group_size == 3
        assert req.strength == "cold"
        assert req.language == "he"

    def test_statistics_hot_only_needs_group_size(self):
        """"hot" alone → request is statistics, but missing group_size."""
        req = normalize("hot")
        assert req.request_kind == "statistics"
        assert req.operation == "group_frequency"
        assert req.strength == "hot"
        assert req.group_size is None
        assert req.confidence > 0.8
        assert req.missing_hint is not None

    def test_analyze_en(self):
        req = normalize("Analyze my numbers 7, 11, 17, 24, 31, 36")
        assert req.request_kind == "number_analysis"
        assert req.operation == "analyze_numbers"
        assert req.numbers == [7, 11, 17, 24, 31, 36]

    def test_analyze_he(self):
        req = normalize("נתח את המספרים 7, 11, 17, 24, 31, 36")
        assert req.request_kind == "number_analysis"
        assert req.operation == "analyze_numbers"

    def test_generate_form_en(self):
        req = normalize("Generate 3 forms")
        assert req.request_kind == "form_generation"
        assert req.operation == "generate_form"
        assert req.how_many == 3

    def test_generate_form_he(self):
        req = normalize("צור 3 טפסים")
        assert req.request_kind == "form_generation"
        assert req.operation == "generate_form"

    def test_save_numbers_admin_en(self):
        req = normalize("Save my numbers 1, 2, 3, 4, 5, 6")
        assert req.request_kind == "admin_operation"
        assert req.operation == "save_numbers"
        assert req.numbers == [1, 2, 3, 4, 5, 6]

    def test_query_audit_log_admin_en(self):
        req = normalize("Show me the audit log")
        assert req.request_kind == "admin_operation"
        assert req.operation == "query_audit_log"

    def test_edit_file_admin_en(self):
        req = normalize("Edit app/main.py")
        assert req.request_kind == "admin_operation"
        assert req.operation == "edit_file"
        assert req.file_path == "app/main.py"


class TestFollowUpInheritance:
    def test_inherit_group_size(self):
        """Follow-up "and the cold ones?" inherits group_size=2 from previous stats request."""
        first = normalize("What are the hot pairs?")
        conversation = ConversationState(last_request=first)
        second = normalize("And the cold ones?", conversation=conversation)
        assert second.is_followup is True
        assert second.group_size == 2  # inherited
        assert second.strength == "cold"  # new, from message

    def test_no_inherit_on_request_kind_change(self):
        first = normalize("What are the hot pairs?")
        conversation = ConversationState(last_request=first)
        second = normalize("Analyze 7, 11, 17", conversation=conversation)
        assert second.request_kind == "number_analysis"
        assert second.group_size is None  # not inherited from statistics

    def test_explicit_conflict_wins(self):
        first = normalize("What are the hot pairs?")  # group_size=2
        conversation = ConversationState(last_request=first)
        second = normalize("And triples?", conversation=conversation)
        assert second.group_size == 3  # explicit in message wins over inherited 2

    def test_no_inherit_without_followup_indicator(self):
        first = normalize("What are the hot pairs?")
        conversation = ConversationState(last_request=first)
        second = normalize("What are the cold triples?", conversation=conversation)
        assert second.is_followup is False
        assert second.group_size == 3  # from message, not inherited


class TestClarificationFollowUp:
    """Tests for clarification-answer follow-up detection and inheritance.

    When the agent asks a clarification question (missing_hint is set) and the
    user replies with a short answer, the reply should be treated as a
    follow-up that inherits the previous request_kind/operation and fills in
    the missing parameter.
    """

    def test_bare_number_answers_how_many_for_generate_form_en(self):
        """"10" after "Generate forms" → how_many=10, kind=form_generation."""
        first = normalize("Generate forms")
        assert first.request_kind == "form_generation"
        assert first.missing_hint is not None
        conversation = ConversationState(last_request=first)
        second = normalize("10", conversation=conversation)
        assert second.is_followup is True
        assert second.request_kind == "form_generation"
        assert second.operation == "generate_form"
        assert second.how_many == 10

    def test_bare_number_answers_how_many_for_generate_form_he(self):
        """"10" after "תייצר לי טופס" → how_many=10, kind=form_generation."""
        first = normalize("תייצר לי טופס")
        assert first.request_kind == "form_generation"
        assert first.missing_hint is not None
        conversation = ConversationState(last_request=first)
        second = normalize("10", conversation=conversation)
        assert second.is_followup is True
        assert second.request_kind == "form_generation"
        assert second.operation == "generate_form"
        assert second.how_many == 10

    def test_bare_number_answers_group_size_for_statistics_en(self):
        """"3" after "hot" (missing group_size) → group_size=3, kind=statistics."""
        first = normalize("hot")
        assert first.request_kind == "statistics"
        assert first.missing_hint is not None
        conversation = ConversationState(last_request=first)
        second = normalize("3", conversation=conversation)
        assert second.is_followup is True
        assert second.request_kind == "statistics"
        assert second.operation == "group_frequency"
        assert second.group_size == 3
        assert second.strength == "hot"  # inherited from first

    def test_group_keyword_answers_group_size_for_statistics_he(self):
        """"שלשות" after "חם" (missing group_size) → group_size=3."""
        first = normalize("חם")
        assert first.request_kind == "statistics"
        assert first.missing_hint is not None
        conversation = ConversationState(last_request=first)
        second = normalize("שלשות", conversation=conversation)
        assert second.is_followup is True
        assert second.request_kind == "statistics"
        assert second.group_size == 3
        assert second.strength == "hot"  # inherited

    def test_numbers_answer_analyze_clarification(self):
        """"7 11 17" after "analyze" (missing numbers) → numbers=[7,11,17]."""
        first = normalize("analyze")
        # "analyze" alone may not trigger number_analysis without numbers;
        # use a message that triggers the kind but leaves numbers missing.
        first = normalize("Analyze my numbers")
        if first.missing_hint is None:
            # If "Analyze my numbers" doesn't produce a missing_hint, craft one
            first = normalize("Analyze")
        conversation = ConversationState(last_request=first)
        second = normalize("7 11 17 24 31 36", conversation=conversation)
        assert second.is_followup is True
        assert second.request_kind == "number_analysis"
        assert second.numbers == [7, 11, 17, 24, 31, 36]

    def test_long_message_not_treated_as_clarification_answer(self):
        """A long message (> 6 words) is not a clarification answer."""
        first = normalize("Generate forms")
        assert first.missing_hint is not None
        conversation = ConversationState(last_request=first)
        second = normalize("What are the hot pairs for the last year?", conversation=conversation)
        # 8 words → not a clarification answer, but may still be a normal request
        assert second.is_followup is False
        assert second.request_kind == "statistics"

    def test_no_clarification_inheritance_without_missing_hint(self):
        """If last request had no missing_hint, short reply is not a follow-up."""
        first = normalize("What are the hot pairs?")
        assert first.missing_hint is None  # fully specified
        conversation = ConversationState(last_request=first)
        second = normalize("10", conversation=conversation)
        # "10" is not a follow-up because first had no missing_hint
        assert second.is_followup is False


class TestConfidenceSemantics:
    def test_confidence_high_despite_missing_args(self):
        req = normalize("hot")
        assert req.confidence > 0.8
        assert req.group_size is None
        assert req.missing_hint is not None

    def test_confidence_for_clear_request(self):
        req = normalize("What are the hot pairs?")
        assert req.confidence >= 0.9


class TestClientLangHintNotAuthoritative:
    def test_hebrew_detected_despite_en_hint(self):
        req = normalize("מה הזוגות החמים?", lang_hint="en")
        assert req.language == "he"
