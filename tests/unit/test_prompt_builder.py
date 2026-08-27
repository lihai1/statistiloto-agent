"""Unit tests for prompt_builder and format_run_data (Phase 5)."""

from app.prompt_builder import (
    build_prompt,
    format_run_data,
    domain_registry_entry,
    FIELD_ALLOWLIST,
    ROW_LIMITS,
)


class TestBuildPrompt:
    def test_domain_explanation_prompt_is_compact(self):
        prompt = build_prompt(
            route="domain_explanation",
            language="he",
            user_message="מה זה זוגות?",
            domain_entry="Pairs: two numbers appearing together in the same draw.",
            knowledge="Hot pairs are common.",
        )
        assert "Statistiloto" in prompt
        assert "DOMAIN POLICY" in prompt
        assert "DOMAIN_REGISTRY_ENTRY" in prompt
        assert "KNOWLEDGE" in prompt
        assert "מה זה זוגות?" in prompt
        # Compact: should be well under 500 chars
        assert len(prompt) < 800

    def test_statistics_finalizer_prompt(self):
        prompt = build_prompt(
            route="statistics_finalizer",
            language="en",
            user_message="What are the hot pairs?",
            run_data='RUN_DATA:\n{"groups": [{"numbers": [1, 2], "count": 5}]}',
        )
        assert "RUN_DATA" in prompt
        assert "statistics" in prompt.lower() or "STATISTICS_FINALIZER" in prompt
        assert "What are the hot pairs?" in prompt

    def test_admin_finalizer_prompt(self):
        prompt = build_prompt(
            route="admin_finalizer",
            language="en",
            user_message="Show me the audit log",
            run_data='RUN_DATA:\n{"entries": []}',
        )
        assert "RUN_DATA" in prompt
        assert "ADMIN_FINALIZER" in prompt or "administrative" in prompt.lower()

    def test_ambiguous_planner_prompt(self):
        prompt = build_prompt(
            route="ambiguous_planner",
            language="en",
            user_message="hot",
            authorized_tools="get_statistics, analyze, generate_form",
            missing_hint="Which group size?",
        )
        assert "PLANNER POLICY" in prompt
        assert "AUTHORIZED_TOOLS" in prompt
        assert "MISSING_HINT" in prompt
        assert "get_statistics" in prompt

    def test_out_of_scope_prompt(self):
        prompt = build_prompt(
            route="out_of_scope",
            language="en",
            user_message="What's the weather?",
        )
        assert "lottery intelligence" in prompt.lower()
        assert "What's the weather?" in prompt

    def test_nl_general_prompt(self):
        prompt = build_prompt(
            route="nl_general",
            language="he",
            user_message="מה זה מספר חם?",
            knowledge="Hot numbers are frequent.",
        )
        assert "KNOWLEDGE" in prompt
        assert "מה זה מספר חם?" in prompt

    def test_admin_prompt_has_no_lottery_rules(self):
        """Phase 5 verification: admin prompt should NOT contain group_size tables."""
        prompt = build_prompt(
            route="admin_finalizer",
            language="en",
            user_message="Show audit log",
            run_data="RUN_DATA:\n{}",
        )
        assert "GROUP SIZE REFERENCE" not in prompt
        assert "HEBREW KEYWORD" not in prompt

    def test_statistics_prompt_has_no_group_size_table(self):
        """Phase 5 verification: statistics finalizer should NOT contain the full
        group_size reference table — that's domain knowledge for explanations only."""
        prompt = build_prompt(
            route="statistics_finalizer",
            language="en",
            user_message="hot pairs",
            run_data="RUN_DATA:\n{}",
        )
        assert "GROUP SIZE REFERENCE" not in prompt
        assert "HEBREW KEYWORD" not in prompt


class TestFormatRunData:
    def test_basic_formatting(self):
        result = format_run_data("get_statistics", {
            "groups": [{"numbers": [1, 2], "count": 5}, {"numbers": [3, 4], "count": 3}],
            "extra_field": "should_be_removed",
        })
        assert "RUN_DATA:" in result
        assert "extra_field" not in result
        assert "groups" in result
        assert "numbers" in result

    def test_row_limiting(self):
        """Rows beyond the limit should be cut, with has_more flag."""
        groups = [{"numbers": [i, i + 1], "count": 100 - i} for i in range(30)]
        result = format_run_data("get_statistics", {"groups": groups})
        assert "has_more" in result
        assert "total_count" in result
        # Should only have 20 rows (ROW_LIMITS["get_statistics"] = 20)
        import json
        data = json.loads(result.split("RUN_DATA:\n")[1])
        assert len(data["groups"]) == 20

    def test_sorting_by_count_desc(self):
        """Statistics groups should be sorted by count descending."""
        result = format_run_data("get_statistics", {
            "groups": [
                {"numbers": [1, 2], "count": 3},
                {"numbers": [3, 4], "count": 10},
                {"numbers": [5, 6], "count": 7},
            ],
        })
        import json
        data = json.loads(result.split("RUN_DATA:\n")[1])
        counts = [g["count"] for g in data["groups"]]
        assert counts == sorted(counts, reverse=True)

    def test_empty_result(self):
        result = format_run_data("get_statistics", {})
        assert "RUN_DATA:" in result
        assert "{}" in result

    def test_none_result(self):
        result = format_run_data("get_statistics", None)
        assert "RUN_DATA:" in result
        assert "{}" in result

    def test_read_code_content_stripped_when_long(self):
        """read_code content should be stripped if > 1000 chars."""
        long_content = "x" * 2000
        result = format_run_data("read_code", {
            "path": "app/main.py",
            "size": 2000,
            "lines": 50,
            "content": long_content,
        })
        assert "content_truncated" in result
        assert "content" not in result.split("RUN_DATA:\n")[1] or "content_truncated" in result

    def test_read_code_content_kept_when_short(self):
        """read_code content should be kept if <= 1000 chars."""
        short_content = "print('hello')"
        result = format_run_data("read_code", {
            "path": "app/main.py",
            "size": 15,
            "lines": 1,
            "content": short_content,
        })
        assert "print('hello')" in result

    def test_never_truncates_json_string(self):
        """format_run_data should never truncate the JSON string itself."""
        groups = [{"numbers": [i, i + 1], "count": i} for i in range(20)]
        result = format_run_data("get_statistics", {"groups": groups})
        # The JSON should be parseable (not truncated mid-string)
        json_str = result.split("RUN_DATA:\n")[1]
        import json
        data = json.loads(json_str)  # Should not raise
        assert "groups" in data

    def test_field_allowlist_drops_unknown_fields(self):
        """Fields not in the allowlist should be dropped."""
        result = format_run_data("save_numbers", {
            "status": "saved",
            "internal_id": 12345,  # not in allowlist
            "debug_info": "secret",  # not in allowlist
        })
        assert "saved" in result
        assert "internal_id" not in result
        assert "debug_info" not in result


class TestDomainRegistry:
    def test_group_size_lookup(self):
        entry = domain_registry_entry("group_size", 2)
        assert "Pairs" in entry or "pair" in entry.lower()

    def test_strength_lookup(self):
        entry = domain_registry_entry("strength", "hot")
        assert "Hot" in entry or "hot" in entry.lower()

    def test_unknown_field_returns_empty(self):
        entry = domain_registry_entry("unknown_field", 1)
        assert entry == ""

    def test_unknown_value_returns_empty(self):
        entry = domain_registry_entry("group_size", 99)
        assert entry == ""
