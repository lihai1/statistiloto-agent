"""Unit tests for deterministic renderers — generate_form, analyze, statistics.

Verifies that render_structured_result() produces meaningful, data-rich output
for all three tool types, in both English and Hebrew, including edge cases
(empty results, missing fields). Regression guard for the existing statistics
renderer.
"""

from app.renderer import (
    render_structured_result,
    render_statistics_result,
    render_generate_form_result,
    render_analyze_result,
)


# ── generate_form ─────────────────────────────────────────────


class TestRenderGenerateForm:
    def test_render_generate_form_en(self):
        result = render_structured_result(
            "generate_form",
            {"forms": [{"numbers": [1, 2, 3, 4, 5, 6], "strong": 7}]},
            "en",
        )
        assert "1 + 2 + 3 + 4 + 5 + 6" in result
        assert "7" in result  # strong number
        assert "action completed" not in result.lower()
        # Should include the grounding disclaimer
        assert "historical" in result.lower() or "probability" in result.lower()

    def test_render_generate_form_he(self):
        result = render_structured_result(
            "generate_form",
            {"forms": [{"numbers": [1, 2, 3, 4, 5, 6], "strong": 7}]},
            "he",
        )
        # Should contain Hebrew characters
        assert any("\u0590" <= c <= "\u05FF" for c in result)
        assert "1 + 2 + 3 + 4 + 5 + 6" in result
        assert "action completed" not in result.lower()

    def test_render_generate_form_multiple(self):
        result = render_structured_result(
            "generate_form",
            {"forms": [
                {"numbers": [1, 2, 3, 4, 5, 6], "strong": 7},
                {"numbers": [10, 20, 30, 40, 50, 60], "strong": 5},
            ]},
            "en",
        )
        assert "1 + 2 + 3 + 4 + 5 + 6" in result
        assert "10 + 20 + 30 + 40 + 50 + 60" in result
        assert "7" in result
        assert "5" in result

    def test_render_generate_form_no_strong(self):
        """Forms without a strong number should still render."""
        result = render_structured_result(
            "generate_form",
            {"forms": [{"numbers": [1, 2, 3, 4, 5, 6]}]},
            "en",
        )
        assert "1 + 2 + 3 + 4 + 5 + 6" in result
        assert "action completed" not in result.lower()

    def test_render_generate_form_empty(self):
        result = render_structured_result("generate_form", {"forms": []}, "en")
        assert "action completed" not in result.lower()
        assert len(result) > 0  # some meaningful message

    def test_render_generate_form_direct(self):
        """Direct function call (not via dispatch) should also work."""
        result = render_generate_form_result(
            [{"numbers": [1, 2, 3, 4, 5, 6], "strong": 7}], 1, "en",
        )
        assert "1 + 2 + 3 + 4 + 5 + 6" in result
        assert "7" in result


# ── analyze ───────────────────────────────────────────────────


class TestRenderAnalyze:
    def test_render_analyze_en(self):
        result = render_structured_result(
            "analyze",
            {
                "frequency_groups": [
                    {"size": 2, "entries": [{"numbers": [1, 2], "count": 5}]},
                    {"size": 1, "entries": [{"numbers": [3], "count": 10}]},
                ],
                "archive_size": 100,
            },
            "en",
        )
        assert "1 + 2" in result
        assert "5" in result  # count
        assert "100" in result  # archive_size
        assert "action completed" not in result.lower()

    def test_render_analyze_he(self):
        result = render_structured_result(
            "analyze",
            {
                "frequency_groups": [
                    {"size": 2, "entries": [{"numbers": [1, 2], "count": 5}]},
                ],
                "archive_size": 100,
            },
            "he",
        )
        assert any("\u0590" <= c <= "\u05FF" for c in result)
        assert "1 + 2" in result
        assert "action completed" not in result.lower()

    def test_render_analyze_empty(self):
        result = render_structured_result(
            "analyze",
            {"frequency_groups": [], "archive_size": 0},
            "en",
        )
        assert "action completed" not in result.lower()
        assert len(result) > 0

    def test_render_analyze_direct(self):
        """Direct function call should also work."""
        result = render_analyze_result(
            [{"size": 2, "entries": [{"numbers": [1, 2], "count": 5}]}],
            100,
            "en",
        )
        assert "1 + 2" in result
        assert "5" in result
        assert "100" in result


# ── statistics (regression guard) ─────────────────────────────


class TestRenderStatisticsRegression:
    def test_render_statistics_still_works(self):
        result = render_structured_result(
            "get_statistics",
            {"groups": [{"numbers": [1, 2], "count": 5}], "how_many": 10,
             "group_size": 2, "strength": "hot"},
            "en",
        )
        assert "1 + 2" in result
        assert "5" in result
        assert "hot" in result.lower() or "pair" in result.lower()

    def test_render_statistics_direct_still_works(self):
        result = render_statistics_result(
            [{"numbers": [1, 2], "count": 5}], 10, 2, "hot", "en",
        )
        assert "1 + 2" in result
        assert "5" in result
