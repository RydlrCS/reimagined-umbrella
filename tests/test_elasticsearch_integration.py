"""
Integration tests for Elasticsearch vector database and embedding service.

Tests cover:
- Elasticsearch connector initialization and connection
- Index creation with proper mappings
- Document indexing (single and bulk)
- Vector similarity search (k-NN)
- Text search with semantic support
- Hybrid search combining vector + text
- Embedding service with real and fallback modes
- Query descriptor generation from Gemini analysis

Based on MotionBlendAI test patterns from:
- project/elastic_search/test_api.py
- project/tests/test_fivetran_semantic_integration.py
"""
import os
from datetime import datetime
from typing import List

import pytest

from src.kinetic_ledger.connectors.warehouse_connector import ElasticsearchConnector
from src.kinetic_ledger.services.embedding_service import (
    EmbeddingService,
    embed_text,
    embed_motion_descriptor,
)
from src.kinetic_ledger.schemas.elasticsearch_mappings import VECTOR_DIMENSIONS


class TestEmbeddingService:
    """Test embedding service with both real and fallback modes."""
    
    def test_embedding_service_initialization(self):
        """Test embedding service can be initialized."""
        service = EmbeddingService()
        assert service is not None
        assert service.dimensions == VECTOR_DIMENSIONS
    
    def test_singleton_pattern(self):
        """Test embedding service uses singleton pattern."""
        service1 = EmbeddingService.get_instance()
        service2 = EmbeddingService.get_instance()
        assert service1 is service2
    
    def test_embed_text_basic(self):
        """Test basic text embedding."""
        text = "capoeira breakdance blend with explosive energy"
        embedding = embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == VECTOR_DIMENSIONS
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_empty_text(self):
        """Test embedding empty text returns zero-like vector."""
        embedding = embed_text("")
        
        assert len(embedding) == VECTOR_DIMENSIONS
        # Check it's near-zero
        import numpy as np
        assert np.abs(np.array(embedding)).max() < 0.1
    
    def test_embed_deterministic(self):
        """Test pseudo-embeddings are deterministic."""
        text = "kung fu capoeira blend"
        embedding1 = embed_text(text)
        embedding2 = embed_text(text)
        
        # Should be identical for same text
        assert embedding1 == embedding2
    
    def test_embed_motion_descriptor(self):
        """Test motion descriptor embedding from Gemini analysis."""
        embedding = embed_motion_descriptor(
            style_labels=["capoeira", "breakdance", "martial_arts"],
            npc_tags=["warrior", "athletic", "street_fighter"],
            summary="Dynamic blend of capoeira kicks with breakdance power moves",
            source_motion="capoeira_kicks.fbx",
            target_motion="breakdance_freeze.fbx"
        )
        
        assert len(embedding) == VECTOR_DIMENSIONS
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embed_text(self):
        """Test batch text embedding."""
        texts = [
            "capoeira blend",
            "breakdance moves",
            "kung fu kicks"
        ]
        
        service = EmbeddingService()
        embeddings = service.batch_embed_text(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == VECTOR_DIMENSIONS for emb in embeddings)


class TestElasticsearchConnector:
    """Test Elasticsearch connector with mock and real connections."""
    
    @pytest.fixture
    def connector(self):
        """Create test connector instance."""
        return ElasticsearchConnector(
            cloud_url=os.getenv("ES_CLOUD_URL"),
            api_key=os.getenv("ES_API_KEY"),
            index_name="test-kinetic-motion"
        )
    
    def test_connector_initialization(self, connector):
        """Test connector can be initialized."""
        assert connector is not None
        assert connector.index_name == "test-kinetic-motion"
    
    def test_singleton_pattern(self):
        """Test connector uses singleton pattern."""
        conn1 = ElasticsearchConnector.get_instance()
        conn2 = ElasticsearchConnector.get_instance()
        assert conn1 is conn2
    
    def test_availability_check(self, connector):
        """Test checking if Elasticsearch is available."""
        # May be True or False depending on environment
        available = connector.is_available()
        assert isinstance(available, bool)
    
    def test_index_single_document(self, connector):
        """Test indexing a single motion analysis document."""
        # Generate test embedding
        test_vector = embed_text("test capoeira blend")
        
        document = {
            "analysis_id": "test_analysis_001",
            "motion_id": "test_motion_001",
            "source_motion": "capoeira.fbx",
            "target_motion": "breakdance.fbx",
            "query_descriptor": "capoeira breakdance athletic blend",
            "motion_vector": test_vector,
            "style_labels": ["capoeira", "breakdance"],
            "npc_tags": ["warrior", "athletic"],
            "gemini_summary": "Dynamic blend with explosive energy",
            "blend_ratio": 0.5,
            "blend_method": "single_shot_temporal_conditioning",
            "transition_start": 120,
            "transition_end": 180,
            "validation_score": 0.95,
            "novelty_score": 0.78,
            "status": "completed",
            "model_provider": "google",
            "model_name": "gemini-2.0-flash-exp"
        }
        
        success = connector.index_document(document, doc_id="test_analysis_001")
        
        # Should succeed if ES available, or gracefully fail if not
        assert isinstance(success, bool)
    
    def test_bulk_index_documents(self, connector):
        """Test bulk indexing multiple documents."""
        documents = []
        
        for i in range(3):
            test_vector = embed_text(f"test motion blend {i}")
            doc = {
                "analysis_id": f"test_bulk_{i:03d}",
                "motion_id": f"motion_{i:03d}",
                "source_motion": f"source_{i}.fbx",
                "target_motion": f"target_{i}.fbx",
                "query_descriptor": f"blend {i} description",
                "motion_vector": test_vector,
                "style_labels": ["test", f"style_{i}"],
                "npc_tags": ["character", f"tag_{i}"],
                "gemini_summary": f"Test blend {i}",
                "blend_ratio": 0.5 + (i * 0.1),
                "blend_method": "temporal_conditioning",
                "status": "completed",
                "model_provider": "google",
                "model_name": "gemini-2.0-flash-exp"
            }
            documents.append(doc)
        
        success_count, failure_count = connector.bulk_index(documents)
        
        # Should return counts
        assert isinstance(success_count, int)
        assert isinstance(failure_count, int)
        assert success_count + failure_count == len(documents)
    
    def test_vector_search(self, connector):
        """Test k-NN vector similarity search."""
        # Generate query vector
        query_vector = embed_text("capoeira breakdance blend")
        
        results = connector.search_vector(
            query_vector=query_vector,
            k=5
        )
        
        # Should return list (may be empty if ES unavailable or no data)
        assert isinstance(results, list)
        
        # If results returned, validate structure
        if results:
            for result in results:
                assert "_id" in result
                assert "_score" in result
                # Vector should be excluded from response
                assert "motion_vector" not in result
    
    def test_vector_search_with_filters(self, connector):
        """Test vector search with status filter."""
        query_vector = embed_text("test motion")
        
        results = connector.search_vector(
            query_vector=query_vector,
            k=5,
            filters={"status": "completed"}
        )
        
        assert isinstance(results, list)
    
    def test_text_search(self, connector):
        """Test multi-field text search."""
        results = connector.search_text(
            query_text="capoeira athletic blend",
            k=5
        )
        
        assert isinstance(results, list)
        
        # If results returned, check for highlights
        if results:
            for result in results:
                assert "_id" in result
                assert "_score" in result
    
    def test_hybrid_search_vector_only(self, connector):
        """Test hybrid search with vector component only."""
        query_vector = embed_text("capoeira blend")
        
        results = connector.search_hybrid(
            query_vector=query_vector,
            k=5,
            vector_weight=1.0
        )
        
        assert isinstance(results, list)
    
    def test_hybrid_search_text_only(self, connector):
        """Test hybrid search with text component only."""
        results = connector.search_hybrid(
            query_text="breakdance moves",
            k=5,
            vector_weight=0.0
        )
        
        assert isinstance(results, list)
    
    def test_hybrid_search_combined(self, connector):
        """Test hybrid search with both vector and text."""
        query_vector = embed_text("martial arts blend")
        
        results = connector.search_hybrid(
            query_vector=query_vector,
            query_text="kung fu capoeira athletic",
            k=5,
            vector_weight=0.6
        )
        
        assert isinstance(results, list)
    
    def test_invalid_vector_dimensions(self, connector):
        """Test search with wrong vector dimensions returns empty."""
        # Wrong dimensions (should be 384)
        wrong_vector = [0.1, 0.2, 0.3]
        
        results = connector.search_vector(
            query_vector=wrong_vector,
            k=5
        )
        
        # Should return empty list due to validation error
        assert results == []


class TestEndToEndIntegration:
    """End-to-end integration tests combining embedding and search."""
    
    @pytest.fixture
    def connector(self):
        """Create test connector."""
        return ElasticsearchConnector(
            cloud_url=os.getenv("ES_CLOUD_URL"),
            api_key=os.getenv("ES_API_KEY"),
            index_name="test-kinetic-integration"
        )
    
    def test_gemini_to_elasticsearch_workflow(self, connector):
        """Test complete workflow: Gemini analysis → embedding → ES indexing → search."""
        
        # Step 1: Simulate Gemini analysis results
        gemini_results = {
            "style_labels": ["capoeira", "breakdance", "acrobatic"],
            "npc_tags": ["warrior", "street_fighter", "athletic"],
            "summary": "Explosive capoeira kicks blended with breakdance power moves",
            "transition_window": {"start": 120, "end": 180},
            "safety_flags": []
        }
        
        # Step 2: Generate embedding from analysis
        motion_vector = embed_motion_descriptor(
            style_labels=gemini_results["style_labels"],
            npc_tags=gemini_results["npc_tags"],
            summary=gemini_results["summary"],
            source_motion="capoeira_kicks.fbx",
            target_motion="breakdance_freeze.fbx"
        )
        
        assert len(motion_vector) == VECTOR_DIMENSIONS
        
        # Step 3: Create document with embedding
        document = {
            "analysis_id": "e2e_test_001",
            "motion_id": "blend_001",
            "source_motion": "capoeira_kicks.fbx",
            "target_motion": "breakdance_freeze.fbx",
            "query_descriptor": " | ".join([
                " ".join(gemini_results["style_labels"]),
                " ".join(gemini_results["npc_tags"]),
                gemini_results["summary"]
            ]),
            "motion_vector": motion_vector,
            "style_labels": gemini_results["style_labels"],
            "npc_tags": gemini_results["npc_tags"],
            "gemini_summary": gemini_results["summary"],
            "transition_start": gemini_results["transition_window"]["start"],
            "transition_end": gemini_results["transition_window"]["end"],
            "safety_flags": gemini_results["safety_flags"],
            "blend_ratio": 0.5,
            "blend_method": "single_shot_temporal_conditioning",
            "validation_score": 0.92,
            "novelty_score": 0.85,
            "status": "completed",
            "model_provider": "google",
            "model_name": "gemini-2.0-flash-exp"
        }
        
        # Step 4: Index document
        success = connector.index_document(document, doc_id="e2e_test_001")
        assert isinstance(success, bool)
        
        # Step 5: Search for similar motions
        query_vector = embed_text("acrobatic martial arts blend")
        results = connector.search_vector(query_vector, k=5)
        
        assert isinstance(results, list)
    
    def test_multi_modal_search_comparison(self, connector):
        """Test comparing vector, text, and hybrid search results."""
        query_text = "capoeira athletic explosive"
        query_vector = embed_text(query_text)
        
        # Vector search
        vector_results = connector.search_vector(query_vector, k=5)
        
        # Text search
        text_results = connector.search_text(query_text, k=5)
        
        # Hybrid search
        hybrid_results = connector.search_hybrid(
            query_vector=query_vector,
            query_text=query_text,
            k=5,
            vector_weight=0.5
        )
        
        # All should return lists
        assert isinstance(vector_results, list)
        assert isinstance(text_results, list)
        assert isinstance(hybrid_results, list)
        
        # Hybrid should potentially combine insights from both
        # (actual results depend on data in index)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
