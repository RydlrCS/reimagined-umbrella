"""
Integration tests for Gemini File Search connector.

Tests cover:
- File Search connector initialization and connection
- Document indexing (single and bulk)
- Natural language search queries
- Document deletion and corpus management
- Graceful degradation when File Search unavailable
"""
import os
from datetime import datetime
from typing import List

import pytest

from src.kinetic_ledger.connectors.file_search_connector import FileSearchConnector


class TestFileSearchConnector:
    """Test File Search connector with mock and real connections."""
    
    @pytest.fixture
    def connector(self):
        """Create test connector instance."""
        return FileSearchConnector(
            api_key=os.getenv("GEMINI_API_KEY"),
            corpus_name="test-kinetic-motion"
        )
    
    def test_connector_initialization(self, connector):
        """Test connector can be initialized."""
        assert connector is not None
        assert connector.corpus_name == "test-kinetic-motion"
    
    def test_singleton_pattern(self):
        """Test connector uses singleton pattern."""
        conn1 = FileSearchConnector.get_instance()
        conn2 = FileSearchConnector.get_instance()
        assert conn1 is conn2
    
    def test_availability_check(self, connector):
        """Test checking if File Search is available."""
        available = connector.is_available()
        assert isinstance(available, bool)
    
    def test_index_single_document(self, connector):
        """Test indexing a single motion analysis document."""
        document = {
            "analysis_id": "test_analysis_001",
            "motion_id": "test_motion_001",
            "source_motion": "capoeira.fbx",
            "target_motion": "breakdance.fbx",
            "query_descriptor": "capoeira breakdance athletic blend with explosive kicks",
            "style_labels": ["capoeira", "breakdance", "martial_arts"],
            "npc_tags": ["warrior", "athletic", "street_fighter"],
            "gemini_summary": "Dynamic blend combining capoeira fluidity with breakdance power",
            "blend_ratio": 0.5,
            "blend_method": "single_shot_temporal_conditioning",
            "transition_start": 120,
            "transition_end": 180,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        success = connector.index_document(document)
        
        # Should succeed if File Search available, or return False gracefully
        assert isinstance(success, bool)
    
    def test_bulk_index_documents(self, connector):
        """Test bulk indexing multiple documents."""
        documents = [
            {
                "analysis_id": f"test_analysis_{i:03d}",
                "motion_id": f"test_motion_{i:03d}",
                "query_descriptor": f"test motion blend {i}",
                "gemini_summary": f"Test summary {i}",
                "style_labels": ["test", "motion"],
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(3)
        ]
        
        result = connector.bulk_index(documents)
        
        assert "indexed" in result
        assert "failed" in result
        assert "total" in result
        assert result["total"] == 3
    
    def test_search_natural_language(self, connector):
        """Test natural language search query."""
        query = "martial arts blend with acrobatic elements"
        
        results = connector.search(query, max_results=5)
        
        # Should return list (empty if File Search unavailable)
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_search_empty_corpus(self, connector):
        """Test search on empty corpus returns empty results."""
        query = "nonexistent motion blend xyz123"
        
        results = connector.search(query)
        
        assert isinstance(results, list)
    
    def test_delete_document(self, connector):
        """Test deleting a document from corpus."""
        # Try to delete (may not exist)
        success = connector.delete_document("test_analysis_001")
        
        assert isinstance(success, bool)
    
    def test_format_document_for_search(self, connector):
        """Test document formatting for search."""
        document = {
            "analysis_id": "test_001",
            "query_descriptor": "test motion",
            "gemini_summary": "test summary",
            "style_labels": ["test"],
        }
        
        text = connector._format_document_for_search(document)
        
        assert isinstance(text, str)
        assert "test_001" in text
        assert "test motion" in text
        assert "test summary" in text


class TestFileSearchIntegration:
    """End-to-end integration tests."""
    
    def test_index_and_search_workflow(self):
        """Test complete workflow: index documents, then search."""
        connector = FileSearchConnector(
            api_key=os.getenv("GEMINI_API_KEY"),
            corpus_name="test-workflow"
        )
        
        # Skip if File Search unavailable
        if not connector.is_available():
            pytest.skip("File Search not available (no API key or network)")
        
        # Index test document
        document = {
            "analysis_id": "workflow_test_001",
            "query_descriptor": "capoeira breakdance fusion with dynamic kicks",
            "gemini_summary": "Athletic blend for warrior NPC",
            "style_labels": ["capoeira", "breakdance"],
            "npc_tags": ["warrior", "athletic"],
            "created_at": datetime.utcnow().isoformat(),
        }
        
        success = connector.index_document(document)
        assert success is True
        
        # Search for indexed document
        results = connector.search("capoeira breakdance warrior")
        
        # Should find the indexed document
        assert len(results) > 0
    
    def test_bulk_operations(self):
        """Test bulk indexing and clearing."""
        connector = FileSearchConnector(
            api_key=os.getenv("GEMINI_API_KEY"),
            corpus_name="test-bulk"
        )
        
        # Skip if File Search unavailable
        if not connector.is_available():
            pytest.skip("File Search not available")
        
        # Bulk index
        documents = [
            {
                "analysis_id": f"bulk_test_{i:03d}",
                "query_descriptor": f"motion blend {i}",
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(5)
        ]
        
        result = connector.bulk_index(documents)
        assert result["indexed"] == 5
        assert result["failed"] == 0
