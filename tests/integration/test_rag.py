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
