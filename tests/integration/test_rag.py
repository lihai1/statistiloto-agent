"""Integration test: RAG retrieval with pgvector.

Tests that embeddings can be indexed and retrieved with role-scoped filtering.
Uses real pgvector DB but mock embeddings (pre-computed vectors, no Ollama needed).
"""
import json
import pytest

pytestmark = pytest.mark.integration


class TestRAGRetrieval:
    def test_index_and_retrieve_docs(self, db_pool):
        """Index a doc and retrieve it via vector similarity."""
        from app.rag.indexers import index_document
        from app.rag.retriever import retrieve

        # Create a simple embedding (768-dim, all zeros except first element).
        embedding = [0.1] + [0.0] * 767
        query_embedding = [0.1] + [0.0] * 767

        index_document("docs", "Lottery analysis documentation", {"source": "test"}, embedding)

        results = retrieve(
            query="lottery",
            corpora=["docs"],
            user_sub="test-user",
            tier="free",
            embedding=query_embedding,
        )
        assert len(results) >= 1
        assert "Lottery analysis" in results[0]["text"]

    def test_user_data_isolated_by_user_sub(self, db_pool):
        """user_data corpus is always filtered by user_sub — no cross-tenant leakage."""
        from app.rag.indexers import index_user_data
        from app.rag.retriever import retrieve

        embedding = [0.5] + [0.0] * 767

        # Index data for user A and user B.
        index_user_data("user-a", "User A saved numbers: [1,2,3]", {"type": "saved"}, embedding)
        index_user_data("user-b", "User B saved numbers: [4,5,6]", {"type": "saved"}, embedding)

        # User A retrieves — should only see their data.
        results_a = retrieve(
            query="my numbers",
            corpora=["user_data"],
            user_sub="user-a",
            tier="paid",
            embedding=embedding,
        )
        assert all(r["metadata"].get("user_sub") == "user-a" for r in results_a)
        assert any("User A" in r["text"] for r in results_a)
        assert not any("User B" in r["text"] for r in results_a)

        # User B retrieves — should only see their data.
        results_b = retrieve(
            query="my numbers",
            corpora=["user_data"],
            user_sub="user-b",
            tier="paid",
            embedding=embedding,
        )
        assert all(r["metadata"].get("user_sub") == "user-b" for r in results_b)
        assert any("User B" in r["text"] for r in results_b)
        assert not any("User A" in r["text"] for r in results_b)

    def test_admin_sees_all_users_data(self, db_pool):
        """Admin (owner/developer) can see ALL users' user_data — user_sub filter bypassed."""
        from app.rag.indexers import index_user_data
        from app.rag.retriever import retrieve

        embedding = [0.5] + [0.0] * 767

        index_user_data("user-a", "User A saved numbers: [1,2,3]", {"type": "saved"}, embedding)
        index_user_data("user-b", "User B saved numbers: [4,5,6]", {"type": "saved"}, embedding)

        # Admin retrieves — should see BOTH users' data.
        results = retrieve(
            query="my numbers",
            corpora=["user_data"],
            user_sub="admin-user",
            tier="admin",
            embedding=embedding,
        )
        assert any("User A" in r["text"] for r in results)
        assert any("User B" in r["text"] for r in results)

    def test_role_scoped_corpora(self, db_pool):
        """Free tier only searches 'docs', not 'lottery_history'."""
        from app.rag.indexers import index_document
        from app.rag.retriever import retrieve

        embedding = [0.9] + [0.0] * 767

        index_document("docs", "Product documentation", {}, embedding)
        index_document("lottery_history", "Draw 1234: numbers [1,2,3,4,5,6]", {}, embedding)

        # Free tier: only docs
        free_results = retrieve(
            query="lottery",
            corpora=["docs"],  # free tier only gets docs
            user_sub="free-user",
            tier="free",
            embedding=embedding,
        )
        # Should not contain lottery_history content
        assert not any("Draw 1234" in r["text"] for r in free_results)

        # Paid tier: docs + lottery_history
        paid_results = retrieve(
            query="lottery",
            corpora=["docs", "lottery_history"],
            user_sub="paid-user",
            tier="paid",
            embedding=embedding,
        )
        assert any("Draw 1234" in r["text"] for r in paid_results)

    def test_examples_filtered_by_lang(self, db_pool):
        """Examples corpus is filtered by metadata->>'lang' when lang is set.

        Indexes one English and one Hebrew example with the same embedding,
        then verifies that lang='he' only returns the Hebrew example and
        lang='en' only returns the English one. Without lang, both are
        returned (back-compat).
        """
        from app.rag.indexers import index_document, clear_corpus
        from app.rag.retriever import retrieve

        embedding = [0.7] + [0.0] * 767
        clear_corpus("examples")
        try:
            index_document(
                "examples",
                "Language: en\nUser: Show me hot pairs\nAssistant: TOOL: get_statistics ARGS: {}",
                {"source": "en/paid.yaml", "lang": "en", "type": "qa_example",
                 "audience": "public", "required_capability": "get_statistics"},
                embedding,
            )
            index_document(
                "examples",
                "Language: he\nUser: הצג לי זוגות חמים\nAssistant: TOOL: get_statistics ARGS: {}",
                {"source": "he/paid.yaml", "lang": "he", "type": "qa_example",
                 "audience": "public", "required_capability": "get_statistics"},
                embedding,
            )

            # Hebrew user — only Hebrew examples.
            he_results = retrieve(
                query="זוגות חמים",
                corpora=["examples"],
                user_sub="test-user",
                tier="paid",
                embedding=embedding,
                lang="he",
            )
            assert len(he_results) >= 1
            assert all(r["metadata"].get("lang") == "he" for r in he_results), (
                f"Hebrew query returned non-he examples: {[r['metadata'].get('lang') for r in he_results]}"
            )
            assert any("he" in r["text"].lower() or "זוגות" in r["text"] for r in he_results)

            # English user — only English examples.
            en_results = retrieve(
                query="hot pairs",
                corpora=["examples"],
                user_sub="test-user",
                tier="paid",
                embedding=embedding,
                lang="en",
            )
            assert len(en_results) >= 1
            assert all(r["metadata"].get("lang") == "en" for r in en_results), (
                f"English query returned non-en examples: {[r['metadata'].get('lang') for r in en_results]}"
            )
            assert any("hot pairs" in r["text"] for r in en_results)

            # No lang filter — both returned (back-compat).
            all_results = retrieve(
                query="pairs",
                corpora=["examples"],
                user_sub="test-user",
                tier="paid",
                embedding=embedding,
            )
            langs = {r["metadata"].get("lang") for r in all_results}
            assert "en" in langs and "he" in langs, (
                f"Without lang filter, expected both en and he, got: {langs}"
            )
        finally:
            clear_corpus("examples")

    def test_lang_filter_does_not_affect_docs(self, db_pool):
        """The lang filter only applies to the 'examples' corpus — docs
        remain language-agnostic and are returned regardless of lang."""
        from app.rag.indexers import index_document, clear_corpus
        from app.rag.retriever import retrieve

        embedding = [0.6] + [0.0] * 767
        clear_corpus("examples")
        try:
            index_document(
                "examples",
                "Language: en\nUser: hot pairs\nAssistant: TOOL: get_statistics ARGS: {}",
                {"source": "en/paid.yaml", "lang": "en", "type": "qa_example",
                 "audience": "public", "required_capability": "get_statistics"},
                embedding,
            )
            index_document("docs", "Hot and cold numbers documentation", {}, embedding)

            # Hebrew user querying docs + examples — should get the doc
            # (no lang filter on docs) but NOT the English example.
            results = retrieve(
                query="hot numbers",
                corpora=["docs", "examples"],
                user_sub="test-user",
                tier="paid",
                embedding=embedding,
                lang="he",
            )
            # Docs returned (lang-agnostic).
            assert any("Hot and cold" in r["text"] for r in results), "docs should be returned regardless of lang"
            # English example NOT returned (lang=he filters it out).
            example_results = [r for r in results if r["metadata"].get("type") == "qa_example"]
            assert len(example_results) == 0, (
                f"Hebrew user should not get English examples, got: {example_results}"
            )
        finally:
            clear_corpus("examples")

    def test_examples_filtered_by_audience_and_capability(self, db_pool):
        """Examples corpus is filtered by audience + required_capability.

        Free users must not retrieve admin-only examples or examples for tools
        they cannot use. Admin can retrieve both public and admin examples.
        """
        from app.rag.indexers import index_document, clear_corpus
        from app.rag.retriever import retrieve

        embedding = [0.8] + [0.0] * 767
        clear_corpus("examples")
        try:
            index_document(
                "examples",
                "Language: en\nUser: Show hot pairs\nAssistant: TOOL: get_statistics ARGS: {}",
                {"source": "en/paid.yaml", "lang": "en", "type": "qa_example",
                 "audience": "public", "required_capability": "get_statistics"},
                embedding,
            )
            index_document(
                "examples",
                "Language: en\nUser: Save numbers\nAssistant: TOOL: save_numbers ARGS: {}",
                {"source": "en/approvals.yaml", "lang": "en", "type": "qa_example",
                 "audience": "public", "required_capability": "save_numbers"},
                embedding,
            )
            index_document(
                "examples",
                "Language: en\nUser: Query audit log\nAssistant: TOOL: query_audit_log ARGS: {}",
                {"source": "en/admin.yaml", "lang": "en", "type": "qa_example",
                 "audience": "admin", "required_capability": "query_audit_log"},
                embedding,
            )

            # Free user: can get the get_statistics example, NOT save_numbers (unauthorized),
            # NOT query_audit_log (admin audience).
            free_results = retrieve(
                query="show me",
                corpora=["examples"],
                user_sub="free-user",
                tier="free",
                embedding=embedding,
                lang="en",
            )
            free_caps = {r["metadata"].get("required_capability") for r in free_results}
            assert "get_statistics" in free_caps
            assert "save_numbers" not in free_caps
            assert "query_audit_log" not in free_caps

            # Paid user: can get get_statistics and save_numbers, NOT query_audit_log.
            paid_results = retrieve(
                query="show me",
                corpora=["examples"],
                user_sub="paid-user",
                tier="paid",
                embedding=embedding,
                lang="en",
            )
            paid_caps = {r["metadata"].get("required_capability") for r in paid_results}
            assert "get_statistics" in paid_caps
            assert "save_numbers" in paid_caps
            assert "query_audit_log" not in paid_caps

            # Admin user: can get all three.
            admin_results = retrieve(
                query="show me",
                corpora=["examples"],
                user_sub="admin-user",
                tier="admin",
                embedding=embedding,
                lang="en",
            )
            admin_caps = {r["metadata"].get("required_capability") for r in admin_results}
            assert "get_statistics" in admin_caps
            assert "save_numbers" in admin_caps
            assert "query_audit_log" in admin_caps
        finally:
            clear_corpus("examples")


class TestIntentAwareRetrieval:
    """Phase 4: The retrieve node skips RAG for deterministic paths."""

    def test_retrieve_skips_rag_for_statistics(self, db_pool):
        """Statistics requests with high confidence → 0 RAG chunks."""
        from app.graphs.common import make_retrieve_node

        retrieve_docs = make_retrieve_node("test")
        state = {
            "user_sub": "test-user",
            "tier": "free",
            "session_id": "test-sess",
            "message": "What are the hot pairs?",
            "lang": "en",
            "context": None,
        }
        result = retrieve_docs(state)
        assert result["chunks"] == [], f"Expected 0 chunks for statistics, got {result['chunks']}"

    def test_retrieve_skips_rag_for_trivial(self, db_pool):
        """Trivial requests (greetings) → 0 RAG chunks."""
        from app.graphs.common import make_retrieve_node

        retrieve_docs = make_retrieve_node("test")
        state = {
            "user_sub": "test-user",
            "tier": "free",
            "session_id": "test-sess",
            "message": "Hi",
            "lang": "en",
            "context": None,
        }
        result = retrieve_docs(state)
        assert result["chunks"] == []

    def test_retrieve_skips_rag_for_out_of_scope(self, db_pool):
        """Out-of-scope requests → 0 RAG chunks."""
        from app.graphs.common import make_retrieve_node

        retrieve_docs = make_retrieve_node("test")
        state = {
            "user_sub": "test-user",
            "tier": "free",
            "session_id": "test-sess",
            "message": "What's the weather today?",
            "lang": "en",
            "context": None,
        }
        result = retrieve_docs(state)
        assert result["chunks"] == []

    def test_retrieve_skips_rag_for_admin_operation(self, db_pool):
        """Admin operations with high confidence → 0 RAG chunks."""
        from app.graphs.common import make_retrieve_node

        retrieve_docs = make_retrieve_node("test")
        state = {
            "user_sub": "admin-user",
            "tier": "admin",
            "session_id": "test-sess",
            "message": "Show me the audit log",
            "lang": "en",
            "context": None,
        }
        result = retrieve_docs(state)
        assert result["chunks"] == []

    def test_retrieve_fetches_for_domain_explanation(self, db_pool, mock_embeddings):
        """Domain explanation → fetches from docs corpus."""
        from app.graphs.common import make_retrieve_node
        from app.rag.indexers import index_document, clear_corpus

        embedding = [0.3] + [0.0] * 767
        clear_corpus("docs")
        try:
            index_document("docs", "Hot numbers appear more frequently in past draws.", {}, embedding)
            retrieve_docs = make_retrieve_node("test")
            state = {
                "user_sub": "test-user",
                "tier": "free",
                "session_id": "test-sess",
                "message": "What does the archive window mean?",
                "lang": "en",
                "context": None,
            }
            result = retrieve_docs(state)
            assert len(result["chunks"]) > 0, "Domain explanation should fetch docs"
        finally:
            clear_corpus("docs")
