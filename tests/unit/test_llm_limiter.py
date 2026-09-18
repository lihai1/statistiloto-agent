"""Unit tests for LLM call serializer (app.llm.limiter) — no DB / no real LLM needed.

Verifies that concurrent calls to ``llm_invoke`` are serialized so only one
LLM inference is in-flight at a time, and that the second call completes
successfully after the first.
"""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

import threading
import time
from unittest.mock import MagicMock

import pytest

from app.llm.limiter import llm_invoke, _llm_lock


class TestLlmInvokeSerialization:
    """Tests that llm_invoke serializes concurrent LLM calls."""

    def test_single_call_returns_response(self):
        """A single call should return the LLM response."""
        llm = MagicMock()
        llm.invoke.return_value = "mock-response"

        result = llm_invoke(llm, "test prompt")

        assert result == "mock-response"
        llm.invoke.assert_called_once_with("test prompt")

    def test_concurrent_calls_serialize_to_one_in_flight(self):
        """Two concurrent calls should never have more than 1 in-flight invoke."""
        in_flight = 0
        max_in_flight = 0
        counter_lock = threading.Lock()

        def slow_invoke(prompt):
            nonlocal in_flight, max_in_flight
            with counter_lock:
                in_flight += 1
                if in_flight > max_in_flight:
                    max_in_flight = in_flight
            time.sleep(0.05)  # simulate inference latency
            with counter_lock:
                in_flight -= 1
            return f"response-{prompt}"

        llm = MagicMock()
        llm.invoke.side_effect = slow_invoke

        results = [None, None]
        errors = [None, None]

        def worker(idx):
            try:
                results[idx] = llm_invoke(llm, f"prompt-{idx}")
            except Exception as e:
                errors[idx] = e

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=5)
        t1.join(timeout=5)

        assert errors == [None, None], f"Unexpected errors: {errors}"
        assert max_in_flight == 1, f"Expected max 1 in-flight, got {max_in_flight}"
        assert results[0] == "response-prompt-0"
        assert results[1] == "response-prompt-1"

    def test_second_call_completes_after_first(self):
        """The second call should complete successfully after the first releases the lock."""
        call_order = []
        order_lock = threading.Lock()

        def track_invoke(prompt):
            with order_lock:
                call_order.append(prompt)
            time.sleep(0.03)
            return f"done-{prompt}"

        llm = MagicMock()
        llm.invoke.side_effect = track_invoke

        results = [None, None]

        def worker(idx):
            results[idx] = llm_invoke(llm, f"call-{idx}")

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=5)
        t1.join(timeout=5)

        assert results[0] == "done-call-0"
        assert results[1] == "done-call-1"
        assert len(call_order) == 2
        assert set(call_order) == {"call-0", "call-1"}

    def test_lock_is_released_after_exception(self):
        """If llm.invoke raises, the lock should be released for the next caller."""
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("inference failed")

        with pytest.raises(RuntimeError, match="inference failed"):
            llm_invoke(llm, "failing-prompt")

        # Lock should be free now — a subsequent call should succeed.
        llm2 = MagicMock()
        llm2.invoke.return_value = "recovery-response"
        result = llm_invoke(llm2, "recovery-prompt")
        assert result == "recovery-response"

    def test_forwards_kwargs_to_invoke(self):
        """Extra kwargs should be forwarded to llm.invoke()."""
        llm = MagicMock()
        llm.invoke.return_value = "ok"

        llm_invoke(llm, "prompt", stop=["\n"], temperature=0.7)

        llm.invoke.assert_called_once_with("prompt", stop=["\n"], temperature=0.7)
