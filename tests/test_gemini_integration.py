#!/usr/bin/env python3
"""
Integration test for Gemini API with real API key.
Tests the GeminiClient and GeminiAnalyzerService with live API calls.

SECURITY NOTE: These tests require GEMINI_API_KEY environment variable.
Tests are skipped if API key is not set to prevent exposing credentials.
"""
import os
import pytest
from kinetic_ledger.services.gemini_analyzer import (
    GeminiClient,
    GeminiAnalyzerService,
    GeminiAnalysisRequest,
)


# SECURITY: Only load API key from environment, never hardcode
# Tests will be skipped if not set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HAS_API_KEY = bool(GEMINI_API_KEY)


@pytest.mark.skipif(not HAS_API_KEY, reason="GEMINI_API_KEY not set - skipping integration tests")
class TestGeminiIntegration:
    """Test suite for Gemini API integration."""
    
    def test_gemini_client_initialization(self):
        """Test that GeminiClient initializes correctly."""
        client = GeminiClient(
            api_key=GEMINI_API_KEY,
            model="gemini-2.0-flash-exp"
        )
        
        # SECURITY: Never assert or print the actual API key value
        assert client.api_key is not None  # Changed from == to verify existence only
        assert len(client.api_key) > 0
        assert client.model == "gemini-2.0-flash-exp"
        assert client.provider == "google"
        assert client._client is not None
    
    def test_gemini_client_motion_analysis(self):
        """Test GeminiClient analyze_motion_preview with real API."""
        client = GeminiClient(
            api_key=GEMINI_API_KEY,
            model="gemini-2.0-flash-exp"
        )
        
        # Sample motion data
        segments = [
            {"label": "capoeira", "start_frame": 0, "end_frame": 124},
            {"label": "breakdance_freezes", "start_frame": 125, "end_frame": 249},
        ]
        
        npc_context = {
            "game": "biomimicry_multi_agent_sim",
            "intent": ["de_escalation", "triage"],
            "environment": "chaotic_crowd_scene",
        }
        
        result = client.analyze_motion_preview(
            preview_uri="demo://preview/capoeira_to_breakdance",
            segments=segments,
            npc_context=npc_context,
        )
        
        # Validate response structure
        assert "style_labels" in result
        assert "transition_window" in result
        assert "npc_tags" in result
        assert "safety_flags" in result
        assert "summary" in result
        
        # Validate types
        assert isinstance(result["style_labels"], list)
        assert isinstance(result["transition_window"], dict)
        assert isinstance(result["npc_tags"], list)
        assert isinstance(result["safety_flags"], list)
        assert isinstance(result["summary"], str)
        
        # Validate transition window structure
        assert "start_frame" in result["transition_window"]
        assert "end_frame" in result["transition_window"]
    
    def test_gemini_analyzer_service(self):
        """Test GeminiAnalyzerService end-to-end."""
        service = GeminiAnalyzerService(
            gemini_api_key=GEMINI_API_KEY,
            model_version="gemini-2.0-flash-exp"
        )
        
        # Create analysis request
        request = GeminiAnalysisRequest(
            request_id="test-request-001",
            preview_uri="demo://preview/martial_arts_combo",
            metadata_ref="metadata://demo/001",
            blend_segments=[
                {"label": "kung_fu", "start_frame": 0, "end_frame": 100},
                {"label": "capoeira", "start_frame": 101, "end_frame": 200},
            ],
            npc_context={
                "game": "action_rpg",
                "intent": ["combat", "evasive"],
                "environment": "dojo",
            },
        )
        
        # Run analysis
        analysis = service.analyze(request, correlation_id="test-001")
        
        # Validate GeminiAnalysis structure
        assert analysis.analysis_id is not None
        assert analysis.request_id == "test-request-001"
        assert analysis.created_at > 0
        assert analysis.model.provider == "google"
        assert analysis.model.name == "gemini-2.0-flash-exp"
        assert analysis.inputs.preview_uri == "demo://preview/martial_arts_combo"
        assert analysis.outputs.style_labels is not None
        assert analysis.outputs.npc_tags is not None
        assert analysis.outputs.transition_window is not None
    
    def test_gemini_fallback_when_no_api_key(self):
        """Test that fallback analysis works when no API key is provided."""
        client = GeminiClient(api_key=None)
        
        segments = [
            {"label": "capoeira", "start_frame": 0, "end_frame": 124},
            {"label": "breakdance", "start_frame": 125, "end_frame": 249},
        ]
        
        result = client.analyze_motion_preview(
            preview_uri="demo://preview/test",
            segments=segments,
        )
        
        # Should use fallback analysis
        assert "style_labels" in result
        assert "capoeira" in result["style_labels"]
        assert "breakdance" in result["style_labels"]
        assert result["transition_window"]["start_frame"] == 109  # first_end - 15
        assert result["transition_window"]["end_frame"] == 140    # second_start + 15
    
    def test_query_descriptor_building(self):
        """Test building query descriptor from analysis."""
        service = GeminiAnalyzerService(
            gemini_api_key=GEMINI_API_KEY,
            model_version="gemini-2.0-flash-exp"
        )
        
        request = GeminiAnalysisRequest(
            request_id="test-descriptor-001",
            preview_uri="demo://preview/test",
            blend_segments=[
                {"label": "style_a", "start_frame": 0, "end_frame": 100},
            ],
        )
        
        analysis = service.analyze(request)
        descriptor = service.build_query_descriptor(analysis)
        
        # Validate descriptor structure
        assert "style_labels" in descriptor
        assert "npc_tags" in descriptor
        assert "transition_window" in descriptor
        assert "summary" in descriptor


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
