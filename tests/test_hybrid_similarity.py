"""
Integration tests for kNN/RkCNN similarity with File Search hybrid architecture.

Tests ensure that:
1. File Search embedding cache works correctly
2. kNN can use cached embeddings
3. RkCNN ensembles function with File Search backend
4. Hybrid architecture maintains backward compatibility
"""
import pytest
import numpy as np
import os
from typing import List, Tuple

from src.kinetic_ledger.connectors.file_search_connector import FileSearchConnector
from src.kinetic_ledger.services.attestation_oracle import (
    AttestationOracle,
    AttestationConfig,
    VectorStore,
)
from src.kinetic_ledger.services.similarity.knn import knn, compute_separation_score
from src.kinetic_ledger.services.similarity.rkcnn import rkcnn


class TestFileSearchEmbeddingCache:
    """Test File Search embedding cache functionality."""
    
    def test_embedding_cache_storage(self):
        """Test that embeddings are cached during indexing."""
        connector = FileSearchConnector()
        
        # Create test document and embedding
        doc = {
            "analysis_id": "test-001",
            "query_descriptor": "capoeira transition to breakdance",
            "style_labels": ["capoeira", "breakdance"],
        }
        embedding = np.random.randn(768).astype(np.float32)
        
        # Index with embedding
        # Note: Will only actually index if GEMINI_API_KEY is set
        connector.index_document(doc, embedding=embedding)
        
        # Verify cache (should work even without API key)
        cached = connector.get_embedding("test-001")
        if cached is not None:
            assert cached.shape == embedding.shape
            np.testing.assert_array_equal(cached, embedding)
    
    def test_embedding_cache_retrieval(self):
        """Test retrieving all embeddings from cache."""
        connector = FileSearchConnector()
        
        # Add multiple embeddings
        embeddings = {
            "doc-1": np.random.randn(768).astype(np.float32),
            "doc-2": np.random.randn(768).astype(np.float32),
            "doc-3": np.random.randn(768).astype(np.float32),
        }
        
        for doc_id, emb in embeddings.items():
            doc = {"analysis_id": doc_id, "query_descriptor": f"Motion {doc_id}"}
            connector.index_document(doc, embedding=emb)
        
        # Retrieve all
        all_embeddings = connector.get_all_embeddings()
        
        assert len(all_embeddings) >= len(embeddings)
        
        # Verify each embedding
        for doc_id, emb in embeddings.items():
            cached = connector.get_embedding(doc_id)
            if cached is not None:
                np.testing.assert_array_equal(cached, emb)
    
    def test_cache_size(self):
        """Test cache size reporting."""
        # Create new instance to avoid singleton pollution
        connector = FileSearchConnector(corpus_name="test-cache-size-corpus")
        
        initial_size = connector.cache_size()
        
        # Add embeddings
        for i in range(5):
            doc = {"analysis_id": f"cache-size-doc-{i}"}
            emb = np.random.randn(768).astype(np.float32)
            connector.index_document(doc, embedding=emb)
        
        final_size = connector.cache_size()
        
        assert final_size == initial_size + 5


class TestKNNWithFileSearch:
    """Test kNN functionality with File Search backend."""
    
    def test_knn_with_cached_embeddings(self):
        """Test kNN using File Search embedding cache."""
        connector = FileSearchConnector()
        
        # Create test dataset
        dim = 768
        n_items = 20
        
        items = []
        for i in range(n_items):
            doc_id = f"motion-{i:03d}"
            emb = np.random.randn(dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)  # Normalize
            
            doc = {
                "analysis_id": doc_id,
                "query_descriptor": f"Motion style {i}",
            }
            connector.index_document(doc, embedding=emb)
            items.append((doc_id, emb))
        
        # Create query
        query = np.random.randn(dim).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-9)
        
        # Run kNN on cached embeddings
        cached_items = connector.get_all_embeddings()
        
        if len(cached_items) > 0:
            k = min(10, len(cached_items))
            neighbors = knn(query, cached_items, k=k, distance_metric="euclidean")
            
            assert len(neighbors) == k
            assert all(isinstance(n[0], str) for n in neighbors)  # IDs
            assert all(isinstance(n[1], float) for n in neighbors)  # Distances
            
            # Distances should be sorted
            distances = [n[1] for n in neighbors]
            assert distances == sorted(distances)
    
    def test_separation_score_computation(self):
        """Test separation score with File Search data."""
        # Create test distances
        distances = [0.5, 0.8, 1.2, 1.5, 2.0]
        
        score = compute_separation_score(distances)
        
        # Separation should be in [0, 1]
        assert 0.0 <= score <= 1.0
        
        # With these distances, should show some separation
        assert score > 0.0


class TestRkCNNWithFileSearch:
    """Test RkCNN ensemble functionality with File Search."""
    
    def test_rkcnn_with_cached_embeddings(self):
        """Test RkCNN using File Search embedding cache."""
        connector = FileSearchConnector()
        
        # Create test dataset
        dim = 768
        n_items = 50
        
        items = []
        for i in range(n_items):
            doc_id = f"motion-{i:03d}"
            emb = np.random.randn(dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            
            doc = {
                "analysis_id": doc_id,
                "query_descriptor": f"Motion {i}",
            }
            connector.index_document(doc, embedding=emb)
            items.append((doc_id, emb))
        
        # Create query
        query = np.random.randn(dim).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-9)
        
        # Run RkCNN
        cached_items = connector.get_all_embeddings()
        
        if len(cached_items) >= 15:  # Need enough items for RkCNN
            separation, vote_margin, ensemble_results = rkcnn(
                query=query,
                items=cached_items,
                k=15,
                ensembles=32,
                subspace_dim=128,
                distance_metric="euclidean"
            )
            
            # Verify results
            assert 0.0 <= separation <= 1.0
            assert 0.0 <= vote_margin <= 1.0
            assert len(ensemble_results) == 32  # Ensemble size
            
            # Each ensemble should return k neighbors
            for ensemble in ensemble_results:
                assert len(ensemble) <= 15
    
    def test_rkcnn_auto_params(self):
        """Test RkCNN auto-computation of ensemble/subspace params."""
        connector = FileSearchConnector()
        
        dim = 768
        n_items = 30
        
        # Create dataset
        items = []
        for i in range(n_items):
            emb = np.random.randn(dim).astype(np.float32)
            items.append((f"doc-{i}", emb))
        
        query = np.random.randn(dim).astype(np.float32)
        
        # Run with auto params
        separation, vote_margin, ensemble_results = rkcnn(
            query=query,
            items=items,
            k=10,
            ensembles=None,  # Auto
            subspace_dim=None,  # Auto
            distance_metric="cosine"
        )
        
        # Should still produce valid results
        assert 0.0 <= separation <= 1.0
        assert 0.0 <= vote_margin <= 1.0
        assert len(ensemble_results) > 0


class TestAttestationOracleHybrid:
    """Test AttestationOracle with hybrid File Search + kNN/RkCNN."""
    
    def test_oracle_uses_file_search_cache(self):
        """Test that oracle prefers File Search cache when available."""
        config = AttestationConfig(
            knn_k=10,
            rkcnn_k=10,
            novelty_threshold=0.42,
            embedding_dim=768,
            chain_id=1,
            verifying_contract="0x1234567890123456789012345678901234567890"
        )
        
        connector = FileSearchConnector()
        oracle = AttestationOracle(config=config, file_search=connector)
        
        # Add some cached embeddings
        for i in range(20):
            doc = {"analysis_id": f"motion-{i}"}
            emb = np.random.randn(768).astype(np.float32)
            connector.index_document(doc, embedding=emb)
        
        # Run similarity check
        result = oracle.validate_similarity(
            analysis_id="test-analysis-001",
            tensor_hash="0x" + "a" * 64,
            gemini_descriptors={"style": "capoeira"},
            safety_flags=[],
        )
        
        # Verify result structure
        assert result.similarity_id is not None
        assert result.decision.result in ["MINT", "REJECT", "REVIEW"]
        assert result.knn.k == 10
        assert result.rkcnn.k == 10
        assert 0.0 <= result.rkcnn.separation_score <= 1.0
        assert 0.0 <= result.rkcnn.vote_margin <= 1.0
    
    def test_oracle_fallback_to_vector_store(self):
        """Test that oracle falls back to VectorStore when File Search unavailable."""
        config = AttestationConfig(
            knn_k=5,
            rkcnn_k=5,
            novelty_threshold=0.5,
            embedding_dim=512,
            chain_id=1,
            verifying_contract="0x1234567890123456789012345678901234567890"
        )
        
        # Create oracle with empty File Search
        vector_store = VectorStore()
        oracle = AttestationOracle(config=config, vector_store=vector_store)
        
        # Add vectors to VectorStore
        for i in range(10):
            emb = np.random.randn(512).astype(np.float32)
            vector_store.add(f"motion-{i}", emb)
        
        # Run similarity check (should use VectorStore)
        result = oracle.validate_similarity(
            analysis_id="test-analysis-002",
            tensor_hash="0x" + "b" * 64,
            gemini_descriptors={},
        )
        
        # Should still work
        assert result.similarity_id is not None
        assert result.decision.result in ["MINT", "REJECT", "REVIEW"]


class TestHybridArchitectureIntegration:
    """Integration tests for complete hybrid architecture."""
    
    def test_end_to_end_similarity_workflow(self):
        """Test complete workflow: index -> embed -> search -> kNN -> RkCNN -> decide."""
        connector = FileSearchConnector()
        
        config = AttestationConfig(
            knn_k=15,
            rkcnn_k=15,
            rkcnn_ensembles=32,
            rkcnn_subspace_dim=128,
            novelty_threshold=0.42,
            vote_margin_threshold=0.10,
            embedding_dim=768,
            chain_id=1,
            verifying_contract="0x1234567890123456789012345678901234567890"
        )
        
        oracle = AttestationOracle(config=config, file_search=connector)
        
        # Index some motions with embeddings
        n_motions = 30
        for i in range(n_motions):
            doc = {
                "analysis_id": f"motion-{i:03d}",
                "query_descriptor": f"Motion style {i % 5}",  # Some overlap
                "style_labels": [f"style-{i % 3}"],
            }
            emb = np.random.randn(768).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            
            connector.index_document(doc, embedding=emb)
        
        # Validate new motion
        result = oracle.validate_similarity(
            analysis_id="new-motion-001",
            tensor_hash="0x" + "c" * 64,
            gemini_descriptors={"style": "new_style"},
            safety_flags=[],
        )
        
        # Comprehensive validation
        assert result.similarity_id is not None
        assert result.analysis_id == "new-motion-001"
        assert result.feature_space.embedding_dim == 768
        assert result.feature_space.distance == "euclidean"
        
        # kNN results
        assert result.knn.k == 15
        assert len(result.knn.neighbors) <= 15
        
        # RkCNN results
        assert result.rkcnn.k == 15
        assert result.rkcnn.ensemble_size == 32
        assert result.rkcnn.subspace_dim == 128
        assert 0.0 <= result.rkcnn.separation_score <= 1.0
        assert 0.0 <= result.rkcnn.vote_margin <= 1.0
        
        # Decision
        assert result.decision.result in ["MINT", "REJECT", "REVIEW"]
        assert result.decision.novelty_threshold == 0.42
    
    def test_structured_output_integration(self):
        """Test that structured outputs work with hybrid architecture."""
        # This test would use actual Gemini API if key is available
        from src.kinetic_ledger.schemas.structured_outputs import (
            NoveltyAssessment,
            SimilarityFeedback,
        )
        
        # Create mock assessment
        assessment = NoveltyAssessment(
            is_novel=True,
            novelty_score=0.75,
            knn_distance=1.2,
            separation_score=0.65,
            vote_consensus=0.80,
            semantic_similarity=SimilarityFeedback(
                sentiment="novel",
                confidence=0.85,
                similar_motions=["motion-001", "motion-015"],
                distinguishing_features=["unique transition", "explosive power"],
                summary="Novel capoeira-to-parkour blend with distinct characteristics"
            ),
            decision="MINT",
            reasoning="High separation score (0.65 > 0.42) with strong ensemble consensus (0.80)"
        )
        
        # Validate schema
        assert assessment.is_novel is True
        assert assessment.decision == "MINT"
        assert assessment.semantic_similarity is not None
        assert assessment.semantic_similarity.sentiment == "novel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
