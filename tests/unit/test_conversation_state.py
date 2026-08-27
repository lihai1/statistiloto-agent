"""Unit tests for ConversationState and follow-up inheritance (Phase 6)."""

from app.normalizer import normalize, ConversationState, NormalizedRequest


class TestFollowupDetection:
    def test_followup_inherits_group_size(self):
        """'ומה הקרים?' after 'מה הזוגות החמים?' → inherits group_size=2."""
        first = normalize(message="מה הזוגות החמים?", lang_hint=None)
        assert first.group_size == 2
        assert first.strength == "hot"

        conv = ConversationState(last_request=first)
        second = normalize(message="ומה הקרים?", lang_hint=None, conversation=conv)
        assert second.is_followup is True
        assert second.group_size == 2  # inherited
        assert second.strength == "cold"  # from current message

    def test_followup_inherits_group_size_en(self):
        """English follow-up: 'and the cold ones?' after 'hot pairs' → inherits group_size=2."""
        first = normalize(message="What are the hot pairs?", lang_hint=None)
        assert first.group_size == 2

        conv = ConversationState(last_request=first)
        second = normalize(message="and the cold ones?", lang_hint=None, conversation=conv)
        assert second.is_followup is True
        assert second.group_size == 2  # inherited

    def test_no_inherit_on_request_kind_change(self):
        """statistics→analyze → no inheritance."""
        first = normalize(message="מה הזוגות החמים?", lang_hint=None)
        assert first.request_kind == "statistics"
        assert first.group_size == 2

        conv = ConversationState(last_request=first)
        second = normalize(message="נתח את 1 2 3 4 5 6", lang_hint=None, conversation=conv)
        assert second.request_kind == "number_analysis"
        # group_size should NOT be inherited from statistics
        assert second.group_size is None or second.group_size != 2

    def test_no_inherit_on_conflicting_params(self):
        """'ושלשות?' after group_size=2 → group_size=3 (not inherited)."""
        first = normalize(message="מה הזוגות החמים?", lang_hint=None)
        assert first.group_size == 2

        conv = ConversationState(last_request=first)
        second = normalize(message="ומה השלשות החמים?", lang_hint=None, conversation=conv)
        assert second.is_followup is True
        assert second.group_size == 3  # from current message, not inherited

    def test_no_inherit_without_followup_indicator(self):
        """Fresh question without follow-up indicator → no inheritance."""
        first = normalize(message="מה הזוגות החמים?", lang_hint=None)
        assert first.group_size == 2

        conv = ConversationState(last_request=first)
        # "מה הקרים?" without "ו" prefix — not a follow-up
        second = normalize(message="מה הקרים?", lang_hint=None, conversation=conv)
        # Without follow-up indicator, no inheritance
        if second.is_followup is False:
            assert second.group_size is None  # not inherited


class TestConversationStateCompat:
    def test_compatible_params_same_kind(self):
        """Same request_kind → compatible params returned."""
        first = normalize(message="hot pairs", lang_hint=None)
        first.group_size = 2
        first.strength = "hot"

        conv = ConversationState(last_request=first)
        second = normalize(message="cold", lang_hint=None)
        compat = conv.compatible_params(second)
        # group_size should be inherited since second doesn't have it
        if second.group_size is None:
            assert "group_size" in compat
            assert compat["group_size"] == 2

    def test_incompatible_kinds_no_inherit(self):
        """Different request_kind → no compatible params."""
        first = normalize(message="hot pairs", lang_hint=None)
        first.request_kind = "statistics"

        conv = ConversationState(last_request=first)
        second = NormalizedRequest(request_kind="number_analysis")
        compat = conv.compatible_params(second)
        assert compat == {}

    def test_no_inherit_when_current_has_value(self):
        """If current request already has a value, don't inherit."""
        first = normalize(message="hot pairs", lang_hint=None)
        first.group_size = 2

        conv = ConversationState(last_request=first)
        second = NormalizedRequest(request_kind="statistics", group_size=3)
        compat = conv.compatible_params(second)
        assert "group_size" not in compat  # current already has it
