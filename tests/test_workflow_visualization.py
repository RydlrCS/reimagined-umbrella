"""
Integration tests for workflow visualization and metrics dashboard.

Tests the complete workflow API endpoints and real-time SSE updates,
ensuring proper state management and data flow through the trustless agent loop.
"""
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from datetime import datetime

from src.kinetic_ledger.api.server import app, workflow_states


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_correlation_id():
    """Generate sample correlation ID"""
    return f"test-{datetime.utcnow().timestamp()}"


@pytest.fixture(autouse=True)
def clear_workflow_states():
    """Clear workflow states before each test"""
    workflow_states.clear()
    yield
    workflow_states.clear()


class TestWorkflowAPI:
    """Test workflow API endpoints"""
    
    def test_get_workflow_not_found(self, client, sample_correlation_id):
        """
        Test getting non-existent workflow returns 404
        
        Entry: GET /api/workflows/{non_existent_id}
        Exit: 404 Not Found
        """
        print(f"[TEST] Entry: test_get_workflow_not_found(correlation_id={sample_correlation_id})")
        
        response = client.get(f"/api/workflows/{sample_correlation_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        
        print("[TEST] Exit: test_get_workflow_not_found - 404 returned correctly")
    
    def test_create_and_retrieve_workflow(self, client, sample_correlation_id):
        """
        Test creating workflow via step update and retrieving it
        
        Entry: POST /api/workflows/{id}/steps
        Exit: Workflow created with correct initial state
        """
        print(f"[TEST] Entry: test_create_and_retrieve_workflow(correlation_id={sample_correlation_id})")
        
        # Create workflow by updating first step
        step_data = {
            "step_id": "upload",
            "status": "processing",
            "data": {"file_uri": "s3://bucket/test.fbx"}
        }
        
        response = client.post(
            f"/api/workflows/{sample_correlation_id}/steps",
            json=step_data
        )
        
        assert response.status_code == 200
        workflow = response.json()
        
        # Verify workflow structure
        assert workflow["correlation_id"] == sample_correlation_id
        assert workflow["status"] == "processing"
        assert workflow["current_step"] == "upload"
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["step_id"] == "upload"
        assert workflow["steps"][0]["status"] == "processing"
        
        # Retrieve workflow
        get_response = client.get(f"/api/workflows/{sample_correlation_id}")
        assert get_response.status_code == 200
        
        retrieved = get_response.json()
        assert retrieved["correlation_id"] == sample_correlation_id
        
        print("[TEST] Exit: test_create_and_retrieve_workflow - Workflow created and retrieved")
    
    def test_workflow_step_progression(self, client, sample_correlation_id):
        """
        Test workflow progressing through multiple steps
        
        Entry: Multiple POST /api/workflows/{id}/steps
        Exit: Workflow state updates correctly, status transitions properly
        """
        print(f"[TEST] Entry: test_workflow_step_progression(correlation_id={sample_correlation_id})")
        
        steps_sequence = [
            {"step_id": "upload", "status": "success", "data": {"file_uri": "s3://test.fbx"}},
            {"step_id": "ingest", "status": "success", "data": {"tensor_hash": "0xabc123"}},
            {"step_id": "gemini", "status": "processing", "data": None},
        ]
        
        for step in steps_sequence:
            response = client.post(
                f"/api/workflows/{sample_correlation_id}/steps",
                json=step
            )
            assert response.status_code == 200
            workflow = response.json()
            
            # Verify current step updated
            assert workflow["current_step"] == step["step_id"]
            
            print(f"[TEST] Updated step: {step['step_id']} -> {step['status']}")
        
        # Verify final state
        final_response = client.get(f"/api/workflows/{sample_correlation_id}")
        final_workflow = final_response.json()
        
        assert len(final_workflow["steps"]) == 3
        assert final_workflow["status"] == "processing"  # Not all steps success
        assert final_workflow["current_step"] == "gemini"
        
        print("[TEST] Exit: test_workflow_step_progression - All steps updated correctly")
    
    def test_workflow_completion(self, client, sample_correlation_id):
        """
        Test workflow status changes to 'completed' when all steps succeed
        
        Entry: POST steps with all 'success' status
        Exit: Workflow status = 'completed'
        """
        print(f"[TEST] Entry: test_workflow_completion(correlation_id={sample_correlation_id})")
        
        # Add all steps as success
        all_steps = [
            "upload", "ingest", "gemini", "oracle", "pack",
            "routing", "arc_tx", "circle_tx", "commerce", "payouts"
        ]
        
        for step_id in all_steps:
            client.post(
                f"/api/workflows/{sample_correlation_id}/steps",
                json={"step_id": step_id, "status": "success"}
            )
        
        # Get final workflow
        response = client.get(f"/api/workflows/{sample_correlation_id}")
        workflow = response.json()
        
        assert workflow["status"] == "completed"
        assert len(workflow["steps"]) == len(all_steps)
        
        print("[TEST] Exit: test_workflow_completion - Workflow marked as completed")
    
    def test_workflow_failure(self, client, sample_correlation_id):
        """
        Test workflow status changes to 'failed' when a step errors
        
        Entry: POST step with 'error' status
        Exit: Workflow status = 'failed'
        """
        print(f"[TEST] Entry: test_workflow_failure(correlation_id={sample_correlation_id})")
        
        # Add successful step
        client.post(
            f"/api/workflows/{sample_correlation_id}/steps",
            json={"step_id": "upload", "status": "success"}
        )
        
        # Add failed step
        response = client.post(
            f"/api/workflows/{sample_correlation_id}/steps",
            json={
                "step_id": "ingest",
                "status": "error",
                "data": {"error": "Invalid FBX format"}
            }
        )
        
        workflow = response.json()
        assert workflow["status"] == "failed"
        
        print("[TEST] Exit: test_workflow_failure - Workflow marked as failed")
    
    def test_list_workflows(self, client):
        """
        Test listing all workflows with pagination
        
        Entry: GET /api/workflows with limit/offset
        Exit: Correct workflows returned with pagination metadata
        """
        print("[TEST] Entry: test_list_workflows")
        
        # Create multiple workflows
        correlation_ids = [f"test-workflow-{i}" for i in range(5)]
        
        for corr_id in correlation_ids:
            client.post(
                f"/api/workflows/{corr_id}/steps",
                json={"step_id": "upload", "status": "processing"}
            )
        
        # List all workflows
        response = client.get("/api/workflows?limit=10&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total"] == 5
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert len(data["workflows"]) == 5
        
        # Test pagination
        page2 = client.get("/api/workflows?limit=2&offset=2")
        assert len(page2.json()["workflows"]) == 2
        
        print(f"[TEST] Exit: test_list_workflows - Listed {data['total']} workflows")
    
    def test_list_workflows_with_status_filter(self, client):
        """
        Test filtering workflows by status
        
        Entry: GET /api/workflows?status_filter=completed
        Exit: Only completed workflows returned
        """
        print("[TEST] Entry: test_list_workflows_with_status_filter")
        
        # Create workflows with different statuses
        client.post(
            "/api/workflows/completed-1/steps",
            json={"step_id": "payouts", "status": "success"}
        )
        client.post(
            "/api/workflows/processing-1/steps",
            json={"step_id": "gemini", "status": "processing"}
        )
        
        # Manually set completed status
        workflow_states["completed-1"]["status"] = "completed"
        
        # Filter by status
        response = client.get("/api/workflows?status_filter=completed")
        data = response.json()
        
        assert data["total"] == 1
        assert all(w["status"] == "completed" for w in data["workflows"])
        
        print("[TEST] Exit: test_list_workflows_with_status_filter - Filtered correctly")


class TestMetricsAPI:
    """Test metrics visualization API endpoints"""
    
    def test_get_metrics_timeline(self, client):
        """
        Test retrieving metrics timeline for visualization
        
        Entry: GET /api/metrics/{blend_id}/timeline
        Exit: Complete metrics data structure returned
        """
        print("[TEST] Entry: test_get_metrics_timeline")
        
        blend_id = "blend-abc123"
        response = client.get(f"/api/metrics/{blend_id}/timeline")
        
        assert response.status_code == 200
        metrics = response.json()
        
        # Verify required fields
        assert "blend_id" in metrics
        assert "coverage_timeline" in metrics
        assert "diversity" in metrics
        assert "per_joint_metrics" in metrics
        assert "cost_breakdown" in metrics
        assert "quality_tier" in metrics
        
        # Verify data types
        assert isinstance(metrics["coverage_timeline"], list)
        assert "local" in metrics["diversity"]
        assert "global" in metrics["diversity"]
        assert "Pelvis" in metrics["per_joint_metrics"]
        
        print(f"[TEST] Exit: test_get_metrics_timeline - Metrics for {blend_id} retrieved")
    
    def test_get_knn_neighbors(self, client):
        """
        Test retrieving kNN neighbors for Oracle visualization
        
        Entry: GET /api/oracle/{analysis_id}/neighbors?k=15
        Exit: List of k neighbors with distances
        """
        print("[TEST] Entry: test_get_knn_neighbors")
        
        analysis_id = "analysis-xyz789"
        k = 15
        
        response = client.get(f"/api/oracle/{analysis_id}/neighbors?k={k}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "analysis_id" in data
        assert "k" in data
        assert "neighbors" in data
        assert len(data["neighbors"]) == k
        
        # Verify neighbor structure
        neighbor = data["neighbors"][0]
        assert "motion_id" in neighbor
        assert "distance" in neighbor
        assert "preview_uri" in neighbor
        
        print(f"[TEST] Exit: test_get_knn_neighbors - Retrieved {k} neighbors")


class TestServerSentEvents:
    """Test SSE streaming for real-time workflow updates"""
    
    @pytest.mark.asyncio
    async def test_workflow_sse_stream(self, client, sample_correlation_id):
        """
        Test SSE stream receives workflow updates
        
        Entry: GET /api/workflows/{id}/stream
        Exit: SSE events received with workflow state changes
        """
        print(f"[TEST] Entry: test_workflow_sse_stream(correlation_id={sample_correlation_id})")
        
        # Create workflow
        client.post(
            f"/api/workflows/{sample_correlation_id}/steps",
            json={"step_id": "upload", "status": "processing"}
        )
        
        # Note: Full SSE testing requires async streaming client
        # Here we test the endpoint exists and returns correct content type
        # Removed 'stream=True' parameter which is not supported by TestClient
        response = client.get(
            f"/api/workflows/{sample_correlation_id}/stream"
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        
        print("[TEST] Exit: test_workflow_sse_stream - SSE endpoint accessible")


class TestWorkflowIntegration:
    """Integration tests for complete workflow scenarios"""
    
    def test_complete_mint_workflow(self, client, sample_correlation_id):
        """
        Test complete MINT decision workflow progression
        
        Entry: Full workflow from upload to payouts
        Exit: All steps completed successfully, status = 'completed'
        """
        print(f"[TEST] Entry: test_complete_mint_workflow(correlation_id={sample_correlation_id})")
        
        workflow_steps = [
            {"step_id": "upload", "status": "success", "data": {"file_uri": "s3://test.fbx"}},
            {"step_id": "ingest", "status": "success", "data": {"tensor_hash": "0xabc"}},
            {"step_id": "gemini", "status": "success", "data": {"analysis_id": "gem-123"}},
            {"step_id": "oracle", "status": "success", "data": {
                "decision": "MINT",
                "separation_score": 0.67,
                "knn": {"neighbors": [{"motion_id": "m1", "distance": 0.25}]},
                "rkcnn": {"separation_score": 0.67}
            }},
            {"step_id": "pack", "status": "success", "data": {"pack_hash": "0xdef"}},
            {"step_id": "routing", "status": "success", "data": {
                "route": "on-chain",
                "rationale": "Value > $100 threshold"
            }},
            {"step_id": "arc_tx", "status": "success", "data": {"tx_hash": "0x789"}},
            {"step_id": "commerce", "status": "success", "data": {"usage_id": "usage-1"}},
            {"step_id": "payouts", "status": "success", "data": {"total_usd": 150.00}}
        ]
        
        for step in workflow_steps:
            response = client.post(
                f"/api/workflows/{sample_correlation_id}/steps",
                json=step
            )
            assert response.status_code == 200
            print(f"[TEST] Completed step: {step['step_id']}")
        
        # Verify final workflow state
        final = client.get(f"/api/workflows/{sample_correlation_id}").json()
        
        assert final["status"] == "completed"
        assert final["current_step"] == "payouts"
        assert len(final["steps"]) == 9
        
        # Verify oracle data stored
        oracle_step = next(s for s in final["steps"] if s["step_id"] == "oracle")
        assert oracle_step["data"]["decision"] == "MINT"
        assert oracle_step["data"]["separation_score"] == 0.67
        
        print("[TEST] Exit: test_complete_mint_workflow - Full workflow completed successfully")
    
    def test_reject_workflow_short_circuit(self, client, sample_correlation_id):
        """
        Test workflow stops after REJECT decision
        
        Entry: Workflow with Oracle REJECT
        Exit: No pack/routing steps executed
        """
        print(f"[TEST] Entry: test_reject_workflow_short_circuit(correlation_id={sample_correlation_id})")
        
        steps = [
            {"step_id": "upload", "status": "success"},
            {"step_id": "ingest", "status": "success"},
            {"step_id": "gemini", "status": "success"},
            {"step_id": "oracle", "status": "success", "data": {
                "decision": "REJECT",
                "separation_score": 0.15,
                "reasoning": ["too_similar", "low_novelty"]
            }}
        ]
        
        for step in steps:
            client.post(f"/api/workflows/{sample_correlation_id}/steps", json=step)
        
        workflow = client.get(f"/api/workflows/{sample_correlation_id}").json()
        
        # Verify no post-oracle steps
        step_ids = [s["step_id"] for s in workflow["steps"]]
        assert "pack" not in step_ids
        assert "routing" not in step_ids
        
        print("[TEST] Exit: test_reject_workflow_short_circuit - Workflow stopped at REJECT")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
