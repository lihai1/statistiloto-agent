"""Unit tests for RAG examples ingestion helpers.

Tests the language-separated corpus layout, optional metadata fields
(lang, context, approval, final), and chunk formatting — without DB.
"""
import yaml
import pytest

from app.rag.ingest import (
    _format_example_chunk,
    _discover_example_files,
    _validate_example_pair,
    _detect_tool_name,
    EXAMPLES_DIR,
)


class TestFormatExampleChunk:
    def test_minimal_user_assistant(self):
        pair = {"user": "Hi", "assistant": "Hello"}
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "User: Hi" in chunk
        assert "Assistant: Hello" in chunk
        assert "Language:" not in chunk
        assert "Context:" not in chunk
        assert "Approval:" not in chunk
        assert "Final:" not in chunk

    def test_lang_field_included(self):
        pair = {"lang": "he", "user": "שלום", "assistant": "שלום!"}
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Language: he" in chunk
        assert "User: שלום" in chunk
        assert "Assistant: שלום!" in chunk

    def test_context_field_included(self):
        pair = {
            "user": "show me hot pairs",
            "assistant": "TOOL: get_statistics ARGS: {}",
            "context": {"page": "statistics", "groupSize": 2, "ordering": "hot"},
        }
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Context:" in chunk
        assert "page=statistics" in chunk
        assert "groupSize=2" in chunk
        assert "ordering=hot" in chunk

    def test_approval_flow_fields(self):
        pair = {
            "lang": "en",
            "user": "Save 1,2,3,4,5,6",
            "assistant": "TOOL: save_numbers ARGS: {\"numbers\": [1,2,3,4,5,6]}",
            "approval": "approved",
            "final": "Saved your numbers: 1,2,3,4,5,6.",
        }
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Language: en" in chunk
        assert "Approval: approved" in chunk
        assert "User: Save 1,2,3,4,5,6" in chunk
        assert "Assistant: TOOL: save_numbers" in chunk
        assert "Final: Saved your numbers" in chunk

    def test_rejected_approval(self):
        pair = {
            "lang": "he",
            "user": "שמור 1,2,3",
            "assistant": "TOOL: save_numbers ARGS: {}",
            "approval": "rejected",
            "final": "השמירה בוטלה.",
        }
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Approval: rejected" in chunk
        assert "Final: השמירה בוטלה." in chunk

    def test_missing_user_returns_none(self):
        assert _format_example_chunk({"assistant": "x"}) is None

    def test_missing_assistant_returns_none(self):
        assert _format_example_chunk({"user": "x"}) is None

    def test_empty_user_returns_none(self):
        assert _format_example_chunk({"user": "   ", "assistant": "x"}) is None

    def test_empty_context_dict_omitted(self):
        pair = {"user": "Hi", "assistant": "Hello", "context": {}}
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Context:" not in chunk

    def test_non_dict_context_omitted(self):
        pair = {"user": "Hi", "assistant": "Hello", "context": "not a dict"}
        chunk = _format_example_chunk(pair)
        assert chunk is not None
        assert "Context:" not in chunk


class TestDiscoverExampleFiles:
    def test_discovers_en_and_he_subdirs(self):
        """The shipped corpus has en/ and he/ subdirectories."""
        if not EXAMPLES_DIR.exists():
            pytest.skip("examples_source/ not present in this checkout")
        files = _discover_example_files(EXAMPLES_DIR)
        rel = [str(f.relative_to(EXAMPLES_DIR)) for f in files]
        assert any(p.startswith("en/") for p in rel), f"no en/ files: {rel}"
        assert any(p.startswith("he/") for p in rel), f"no he/ files: {rel}"

    def test_discovers_recursive_temp_dir(self, tmp_path):
        """A nested layout is discovered recursively."""
        (tmp_path / "en").mkdir()
        (tmp_path / "he").mkdir()
        (tmp_path / "en" / "free.yaml").write_text("[]", encoding="utf-8")
        (tmp_path / "en" / "approvals.yaml").write_text("[]", encoding="utf-8")
        (tmp_path / "he" / "free.yaml").write_text("[]", encoding="utf-8")
        # Legacy flat file at root should also be discovered.
        (tmp_path / "legacy.yaml").write_text("[]", encoding="utf-8")

        files = _discover_example_files(tmp_path)
        rel = sorted(str(f.relative_to(tmp_path)) for f in files)
        assert rel == [
            "en/approvals.yaml",
            "en/free.yaml",
            "he/free.yaml",
            "legacy.yaml",
        ]

    def test_returns_sorted_for_determinism(self, tmp_path):
        (tmp_path / "z.yaml").write_text("[]", encoding="utf-8")
        (tmp_path / "a.yaml").write_text("[]", encoding="utf-8")
        (tmp_path / "m.yaml").write_text("[]", encoding="utf-8")
        files = _discover_example_files(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_empty_dir_returns_empty(self, tmp_path):
        assert _discover_example_files(tmp_path) == []


class TestShippedCorpusStructure:
    """Validate the shipped YAML files are well-formed and have lang metadata."""

    @classmethod
    @pytest.fixture(scope="class")
    def shipped_files(cls):
        if not EXAMPLES_DIR.exists():
            pytest.skip("examples_source/ not present")
        return _discover_example_files(EXAMPLES_DIR)

    def test_all_files_parse_as_yaml_lists(self, shipped_files):
        for f in shipped_files:
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            assert isinstance(data, list), f"{f.name} is not a YAML list"

    def test_all_examples_have_lang(self, shipped_files):
        """Every shipped example must declare a language."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assert isinstance(pair, dict), f"{rel}[{i}] not a dict"
                lang = pair.get("lang")
                assert lang in {"en", "he"}, f"{rel}[{i}] missing/invalid lang: {lang!r}"

    def test_all_examples_have_user_and_assistant(self, shipped_files):
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assert pair.get("user"), f"{rel}[{i}] missing user"
                assert pair.get("assistant"), f"{rel}[{i}] missing assistant"

    def test_lang_matches_folder(self, shipped_files):
        """Examples in en/ must have lang=en; examples in he/ must have lang=he."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            parts = rel.parts
            if len(parts) < 2:
                continue  # legacy flat file
            folder_lang = parts[0]
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assert pair.get("lang") == folder_lang, (
                    f"{rel}[{i}] lang={pair.get('lang')!r} but folder={folder_lang}"
                )

    def test_approval_examples_have_final(self, shipped_files):
        """Examples with an `approval` field must also have a `final` response."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                if pair.get("approval"):
                    assert pair.get("final"), (
                        f"{rel}[{i}] has approval={pair['approval']!r} but no final"
                    )
                    assert pair["approval"] in {"approved", "rejected"}, (
                        f"{rel}[{i}] invalid approval: {pair['approval']!r}"
                    )


class TestShippedCorpusSecurityMetadata:
    """Validate the shipped YAML files have audience + required_capability metadata."""

    @classmethod
    @pytest.fixture(scope="class")
    def shipped_files(cls):
        if not EXAMPLES_DIR.exists():
            pytest.skip("examples_source/ not present")
        return _discover_example_files(EXAMPLES_DIR)

    def test_all_examples_have_audience(self, shipped_files):
        """Every shipped example must declare an audience (public or admin)."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                audience = pair.get("audience")
                assert audience in {"public", "admin"}, (
                    f"{rel}[{i}] missing/invalid audience: {audience!r}"
                )

    def test_tool_examples_have_required_capability(self, shipped_files):
        """Every example with a TOOL: line in assistant must have required_capability set."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assistant = pair.get("assistant", "")
                tool = _detect_tool_name(assistant)
                if tool:
                    cap = pair.get("required_capability")
                    assert cap == tool, (
                        f"{rel}[{i}] has TOOL:{tool} but required_capability={cap!r}"
                    )

    def test_non_tool_examples_have_null_or_missing_capability(self, shipped_files):
        """Non-TOOL examples should have required_capability=null or missing."""
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assistant = pair.get("assistant", "")
                tool = _detect_tool_name(assistant)
                if not tool:
                    cap = pair.get("required_capability")
                    # null or missing is fine; a tool name would be wrong
                    if cap is not None:
                        assert cap not in {
                            "generate_form", "get_statistics", "analyze",
                            "save_numbers", "list_saved_numbers", "trigger_scraper",
                            "query_audit_log", "read_token_usage", "search_web",
                            "read_code", "list_files", "edit_file",
                            "list_db_tables", "query_db",
                        }, (
                            f"{rel}[{i}] is non-TOOL but has required_capability={cap!r}"
                        )

    def test_admin_tool_examples_have_admin_audience(self, shipped_files):
        """Examples for admin-only tools must have audience=admin."""
        admin_tools = {
            "trigger_scraper", "edit_file", "query_audit_log", "read_token_usage",
            "search_web", "read_code", "list_files", "list_db_tables", "query_db",
        }
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                cap = pair.get("required_capability")
                if cap in admin_tools:
                    assert pair.get("audience") == "admin", (
                        f"{rel}[{i}] has admin-only required_capability={cap} "
                        f"but audience={pair.get('audience')!r}"
                    )

    def test_free_yaml_has_no_stale_refusals(self, shipped_files):
        """free.yaml must NOT contain stale refusal examples for generate/analyze.

        Free tier CAN use generate_form, get_statistics, and analyze per agent.yaml.
        The old stale examples incorrectly refused these operations.
        """
        for f in shipped_files:
            rel = f.relative_to(EXAMPLES_DIR)
            if not rel.name == "free.yaml":
                continue
            with open(f, encoding="utf-8") as fh:
                pairs = yaml.safe_load(fh)
            for i, pair in enumerate(pairs):
                assistant = (pair.get("assistant") or "").lower()
                # Must NOT contain stale refusal phrases
                stale_phrases = [
                    "i can't generate forms",
                    "i can't run analysis",
                    "form generation requires a paid subscription",
                    "number analysis requires a paid subscription",
                    "i can't retrieve live statistics",
                ]
                for phrase in stale_phrases:
                    assert phrase not in assistant, (
                        f"{rel}[{i}] contains stale refusal phrase: {phrase!r}"
                    )


class TestValidateExamplePair:
    """Test the fail-closed validation for example pairs."""

    def test_missing_audience_raises(self):
        """Ingestion must FAIL if audience is missing."""
        pair = {"user": "Hi", "assistant": "Hello"}
        with pytest.raises(ValueError, match="audience"):
            _validate_example_pair(pair, "en/test.yaml", 0)

    def test_invalid_audience_raises(self):
        """Ingestion must FAIL if audience is not 'public' or 'admin'."""
        pair = {"user": "Hi", "assistant": "Hello", "audience": "internal"}
        with pytest.raises(ValueError, match="invalid audience"):
            _validate_example_pair(pair, "en/test.yaml", 0)

    def test_tool_without_required_capability_raises(self):
        """Ingestion must FAIL if TOOL: appears but required_capability is missing."""
        pair = {
            "user": "Show me stats",
            "assistant": 'TOOL: get_statistics ARGS: {"group_size": 2}',
            "audience": "public",
        }
        with pytest.raises(ValueError, match="required_capability"):
            _validate_example_pair(pair, "en/test.yaml", 0)

    def test_tool_with_matching_required_capability_passes(self):
        """TOOL example with correct required_capability passes validation."""
        pair = {
            "user": "Show me stats",
            "assistant": 'TOOL: get_statistics ARGS: {"group_size": 2}',
            "audience": "public",
            "required_capability": "get_statistics",
        }
        _validate_example_pair(pair, "en/test.yaml", 0)  # should not raise

    def test_non_tool_without_required_capability_passes(self):
        """Non-TOOL example without required_capability passes validation."""
        pair = {
            "user": "What are hot numbers?",
            "assistant": "Hot numbers are...",
            "audience": "public",
        }
        _validate_example_pair(pair, "en/test.yaml", 0)  # should not raise

    def test_non_tool_with_null_required_capability_passes(self):
        """Non-TOOL example with required_capability=null passes validation."""
        pair = {
            "user": "What are hot numbers?",
            "assistant": "Hot numbers are...",
            "audience": "public",
            "required_capability": None,
        }
        _validate_example_pair(pair, "en/test.yaml", 0)  # should not raise


class TestDetectToolName:
    """Test the TOOL: line detection helper."""

    def test_detects_tool_with_args(self):
        assert _detect_tool_name('TOOL: get_statistics ARGS: {"group_size": 2}') == "get_statistics"

    def test_detects_tool_without_args(self):
        assert _detect_tool_name("TOOL: save_numbers") == "save_numbers"

    def test_detects_tool_case_insensitive(self):
        assert _detect_tool_name("tool: analyze ARGS: {}") == "analyze"

    def test_returns_none_for_plain_text(self):
        assert _detect_tool_name("Hot numbers are...") is None

    def test_returns_none_for_empty(self):
        assert _detect_tool_name("") is None

    def test_returns_none_for_none(self):
        assert _detect_tool_name(None) is None
