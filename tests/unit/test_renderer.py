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
    render_simulate_result,
    render_multi_request,
    render_tool_error,
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


# ── multi_request (#15) ────────────────────────────────────────


class TestRenderMultiRequest:
    def test_render_multi_request_en(self):
        result = render_multi_request(
            ["generate 5 forms", "analyze 1,2,3"], "en",
        )
        assert "one request at a time" in result.lower()
        assert "1." in result
        assert "2." in result
        assert "generate 5 forms" in result
        assert "analyze 1,2,3" in result

    def test_render_multi_request_he(self):
        result = render_multi_request(
            ["צור 5 טפסים", "נתח 1,2,3"], "he",
        )
        assert any("\u0590" <= c <= "\u05FF" for c in result)
        assert "1." in result
        assert "2." in result

    def test_render_multi_request_single(self):
        """Even a single request renders the pick-one message."""
        result = render_multi_request(["generate 5 forms"], "en")
        assert "one request at a time" in result.lower()
        assert "generate 5 forms" in result


# ── error precedence (#bug-4) ─────────────────────────────────


class TestRenderErrorPrecedence:
    """Tool error results must render as errors, not empty tool-specific output."""

    def test_generate_form_error_not_masked(self):
        """Error in generate_form result → render_tool_error, not 'No forms'."""
        result = render_structured_result(
            "generate_form", {"error": "No module named 'lottery_pb2'", "tool": "generate_form"}, "en",
        )
        assert result == render_tool_error("en")
        assert "No forms" not in result

    def test_analyze_error_not_masked(self):
        """Error in analyze result → render_tool_error, not 'No frequency data'."""
        result = render_structured_result(
            "analyze", {"error": "Service unavailable", "tool": "analyze"}, "en",
        )
        assert result == render_tool_error("en")
        assert "No frequency data" not in result

    def test_get_statistics_error_not_masked(self):
        """Error in get_statistics result → render_tool_error, not empty stats."""
        result = render_structured_result(
            "get_statistics", {"error": "timeout", "tool": "get_statistics"}, "he",
        )
        assert result == render_tool_error("he")


# ── simulate result ───────────────────────────────────────────


class TestRenderSimulateResult:
    def test_render_simulate_with_summary_en(self):
        draws = [
            {"draw_number": 100, "winning_numbers": [1, 2, 3, 4, 5, 6],
             "winning_strong": 7, "prize_won": 100.0, "tier_hits": [], "ticket_cost": 3.0,
             "used_real_prizes": True},
        ]
        summary = {
            "total_draws": 100, "total_combinations": 100,
            "total_spent": 300.0, "total_won": 100.0, "net": -200.0,
            "tier_summaries": [
                {"tier": 1, "label": "6+strong", "total_hits": 0, "total_amount": 0},
                {"tier": 8, "label": "3", "total_hits": 5, "total_amount": 15.0},
            ],
            "draws_with_real_prizes": 100,
        }
        result = render_simulate_result(draws, summary, "en")
        assert "100 draws" in result
        assert "₪300" in result
        assert "Loss" in result
        assert "3: 5 hits" in result

    def test_render_simulate_empty_en(self):
        result = render_simulate_result([], None, "en")
        assert "No simulation results" in result

    def test_render_simulate_empty_he(self):
        result = render_simulate_result([], None, "he")
        assert "לא נמצאו תוצאות סימולציה" in result

    def test_render_simulate_via_structured_result(self):
        """render_structured_result dispatches to render_simulate_result for 'simulate'."""
        result = render_structured_result(
            "simulate",
            {"draws": [], "summary": None},
            "en",
        )
        assert "No simulation results" in result
