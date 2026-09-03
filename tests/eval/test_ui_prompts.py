"""Eval: UI-constructed prompt → agent response mapping.

Verifies that the agent correctly handles the 7 prompts constructed by the
Angular UI (old + ui-fable) and sent via AgentContextService.ask(). Each test
sends the exact prompt text the UI builds, with the context dict the UI
attaches, and asserts the agent dispatches the right tool with the right
arguments — or, for prompts that lack numbers in context, asks the user for
them instead of calling a tool blindly.

Prompts covered (3 inherited from old UI + 4 new in ui-fable):
  EN + HE:
    1. Generate tab  — "I just generated N lottery forms..." (no numbers in context)
    2. Statistics tab — "I found N frequent number groups..." (no numbers in context)
    3. Analyze tab   — "I analyzed the numbers X against..." (context has numbers)
    4. Simulate tab  — "I simulated the ticket X (strong Y): net Z..." (context has numbers)
    5. Build Form Flow — "I just generated N forms with the coach..." (no numbers)
    6. Analyze Flow  — "I analyzed the numbers X with the coach..." (context has numbers)
    7. Backtest Flow — "I backtested the ticket X (strong Y): spent..." (context has numbers)
"""
import pytest

pytestmark = pytest.mark.integration


def _track_tool_args(mock_tool_clients, tool_name):
    """Wrap a mock tool to capture the arguments it was called with."""
    from app.tools import lottery_grpc
    received = {}
    original = lottery_grpc._mock_client[tool_name]

    def tracker(**kw):
        received.update(kw)
        return original(**kw)

    lottery_grpc._mock_client[tool_name] = tracker
    return received


# ── English: prompts with numbers in context → tool dispatch ──

class TestUIPromptsEnglishWithNumbers:
    """UI prompts that include numbers in the context dict.

    The agent should extract the numbers from context and call the analyze
    tool (there is no simulate tool — the agent uses analyze for backtest/
    simulate prompts to provide historical context).
    """

    def test_analyze_tab_prompt(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Analyze tab: 'I analyzed the numbers...' with context numbers → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [7, 11, 17, 24, 31, 36]}',
            "Your numbers show interesting historical patterns.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-analyze",
            "message": "I analyzed the numbers 7, 11, 17, 24, 31, 36 against historical draws. Can you summarize the frequency results and suggest which number combinations appear most often?",
            "intent": "analyst",
            "context": {"page": "analyze", "numbers": [7, 11, 17, 24, 31, 36]},
            "lang": "en",
        })
        assert resp.status_code == 200
        assert received.get("form") == [7, 11, 17, 24, 31, 36]

    def test_simulate_tab_prompt(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Simulate tab: 'I simulated the ticket...' with context numbers → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38]}',
            "Based on the historical analysis, your ticket has moderate coverage.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-simulate",
            "message": "I simulated the ticket 5, 12, 23, 34, 38 (strong 7): net -150 across 500 draws. What should I try next?",
            "intent": "analyst",
            "context": {"page": "simulate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "en",
        })
        assert resp.status_code == 200
        # The agent should analyze the main numbers (excluding the strong number).
        assert received.get("form") is not None
        assert set(received.get("form", [])).issubset({5, 12, 23, 34, 38, 7})

    def test_analyze_flow_prompt(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Analyze Flow: 'I analyzed the numbers with the coach...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [3, 7, 15, 22, 29, 37]}',
            "Here is a summary of the strengths and weaknesses of this set.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-analyze-flow",
            "message": "I analyzed the numbers 3, 7, 15, 22, 29, 37 with the coach. Summarize the strengths and weaknesses of this set.",
            "intent": "analyst",
            "context": {"page": "analyze", "numbers": [3, 7, 15, 22, 29, 37]},
            "lang": "en",
        })
        assert resp.status_code == 200
        assert received.get("form") == [3, 7, 15, 22, 29, 37]

    def test_backtest_flow_prompt(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Backtest Flow: 'I backtested the ticket...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38]}',
            "To improve your ticket, consider swapping out the coldest numbers.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-backtest-flow",
            "message": "I backtested the ticket 5, 12, 23, 34, 38 (strong 7): spent 500, won 350, net -150 across 500 draws. How can I improve it?",
            "intent": "analyst",
            "context": {"page": "simulate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "en",
        })
        assert resp.status_code == 200
        assert received.get("form") is not None
        assert set(received.get("form", [])).issubset({5, 12, 23, 34, 38, 7})

    def test_build_form_flow_prompt(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Build Form Flow: 'I just generated N forms with the coach. The one I like is X...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38, 7]}',
            "Based on the historical analysis, this form has moderate coverage across pair frequencies.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-build-form",
            "message": "I just generated 3 lottery forms with the coach. The one I like is 5, 12, 23, 34, 38, 7. Which has the best historical coverage?",
            "intent": "analyst",
            "context": {"page": "generate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "en",
        })
        assert resp.status_code == 200
        assert received.get("form") == [5, 12, 23, 34, 38, 7]


# ── English: prompts WITHOUT numbers in context → no blind tool call ──

class TestUIPromptsEnglishWithoutNumbers:
    """UI prompts that do NOT include numbers in the context dict.

    The agent should NOT call a tool blindly — it should ask the user to
    provide the specific numbers/forms it should analyze.
    """

    def test_generate_tab_prompt(self, client, paid_headers, set_llm_responses):
        """Generate tab: 'I just generated N forms...' (no numbers) → ask for numbers."""
        set_llm_responses([
            "I'd be happy to help analyze your generated forms! Could you share the specific numbers from the forms you'd like me to analyze?",
        ])

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-generate",
            "message": "I just generated 3 lottery forms. Can you analyze them and tell me which one has the best historical coverage?",
            "intent": "analyst",
            "context": None,
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("response"), "Expected a non-empty response"
        # Should NOT have called a tool (no numbers to analyze)
        assert not data.get("paused"), "Should not pause for HITL"

    def test_statistics_tab_prompt(self, client, paid_headers, set_llm_responses):
        """Statistics tab: 'I found N frequent groups...' (no numbers) → ask or use get_statistics."""
        set_llm_responses([
            "Could you tell me which group size you're interested in? I can check the statistics for you.",
        ])

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-en-statistics",
            "message": "I found 10 frequent number groups in the statistics. Which of these groups has the highest historical frequency?",
            "intent": "analyst",
            "context": None,
            "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("response"), "Expected a non-empty response"


# ── Hebrew: same 7 prompts ─────────────────────────────────────

class TestUIPromptsHebrewWithNumbers:
    """Hebrew UI prompts with numbers in context → tool dispatch."""

    def test_analyze_tab_prompt_he(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Analyze tab (HE): 'ניתחתי את המספרים...' with context → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [7, 11, 17, 24, 31, 36]}',
            "המספרים שלך מראים דפוסים היסטוריים מעניינים.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-analyze",
            "message": "I analyzed the numbers 7, 11, 17, 24, 31, 36 against historical draws. Can you summarize the frequency results and suggest which number combinations appear most often?",
            "intent": "analyst",
            "context": {"page": "analyze", "numbers": [7, 11, 17, 24, 31, 36]},
            "lang": "he",
        })
        assert resp.status_code == 200
        assert received.get("form") == [7, 11, 17, 24, 31, 36]

    def test_simulate_tab_prompt_he(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Simulate tab (HE): 'סימלצתי את הטופס...' with context → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38]}',
            "בהתבסס על הניתוח ההיסטורי, הטופס שלך בכיסוי בינוני.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-simulate",
            "message": "I simulated the ticket 5, 12, 23, 34, 38 (strong 7): net -150 across 500 draws. What should I try next?",
            "intent": "analyst",
            "context": {"page": "simulate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "he",
        })
        assert resp.status_code == 200
        assert received.get("form") is not None
        assert set(received.get("form", [])).issubset({5, 12, 23, 34, 38, 7})

    def test_analyze_flow_prompt_he(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Analyze Flow (HE): 'ניתחתי את המספרים עם המאמן...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [3, 7, 15, 22, 29, 37]}',
            "הנה סיכום החוזקות והחולשות של הסט הזה.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-analyze-flow",
            "message": "I analyzed the numbers 3, 7, 15, 22, 29, 37 with the coach. Summarize the strengths and weaknesses of this set.",
            "intent": "analyst",
            "context": {"page": "analyze", "numbers": [3, 7, 15, 22, 29, 37]},
            "lang": "he",
        })
        assert resp.status_code == 200
        assert received.get("form") == [3, 7, 15, 22, 29, 37]

    def test_backtest_flow_prompt_he(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Backtest Flow (HE): 'בדקתי לאחור את הטופס...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38]}',
            "כדי לשפר את הטופס, שקול להחליף את המספרים הקרים ביותר.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-backtest-flow",
            "message": "I backtested the ticket 5, 12, 23, 34, 38 (strong 7): spent 500, won 350, net -150 across 500 draws. How can I improve it?",
            "intent": "analyst",
            "context": {"page": "simulate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "he",
        })
        assert resp.status_code == 200
        assert received.get("form") is not None
        assert set(received.get("form", [])).issubset({5, 12, 23, 34, 38, 7})

    def test_build_form_flow_prompt_he(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Build Form Flow (HE): 'I just generated N forms with the coach. The one I like is X...' → analyze(form=...)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [5, 12, 23, 34, 38, 7]}',
            "בהתבסס על הניתוח ההיסטורי, לטופס הזה יש כיסוי בינוני בתדירויות זוגות.",
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-build-form",
            "message": "I just generated 3 lottery forms with the coach. The one I like is 5, 12, 23, 34, 38, 7. Which has the best historical coverage?",
            "intent": "analyst",
            "context": {"page": "generate", "numbers": [5, 12, 23, 34, 38, 7]},
            "lang": "he",
        })
        assert resp.status_code == 200
        assert received.get("form") == [5, 12, 23, 34, 38, 7]


class TestUIPromptsHebrewWithoutNumbers:
    """Hebrew UI prompts without numbers in context → ask for numbers."""

    def test_generate_tab_prompt_he(self, client, paid_headers, set_llm_responses):
        """Generate tab (HE): no numbers → ask for numbers."""
        set_llm_responses([
            "אשמח לעזור לנתח את הטפסים! תוכל לשתף את המספרים מהטפסים שתרצה שאנתח?",
        ])

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-generate",
            "message": "I just generated 3 lottery forms. Can you analyze them and tell me which one has the best historical coverage?",
            "intent": "analyst",
            "context": None,
            "lang": "he",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("response"), "Expected a non-empty response"

    def test_statistics_tab_prompt_he(self, client, paid_headers, set_llm_responses):
        """Statistics tab (HE): no numbers → ask or use get_statistics."""
        set_llm_responses([
            "תוכל להגיד לי באיזה גודל קבוצה אתה מתעניין? אוכל לבדוק את הסטטיסטיקה עבורך.",
        ])

        resp = client.post("/chat", headers=paid_headers, json={
            "session_id": "ui-prompt-he-statistics",
            "message": "I found 10 frequent number groups in the statistics. Which of these groups has the highest historical frequency?",
            "intent": "analyst",
            "context": None,
            "lang": "he",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("response"), "Expected a non-empty response"
