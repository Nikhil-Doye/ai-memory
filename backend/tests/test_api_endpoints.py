"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi import status
from datetime import datetime
from unittest.mock import AsyncMock


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check returns 200 and valid structure."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "neo4j" in data
        assert "neo4j_vector" in data
        assert "embeddings" in data
        
        # Verify timestamp is ISO format string
        assert isinstance(data["timestamp"], str)
        # Try parsing to verify it's valid ISO format
        datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
    
    def test_health_check_json_serializable(self, client):
        """Test health check response is JSON serializable."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        # Should not raise exception
        data = response.json()
        assert isinstance(data, dict)


class TestMemoryEndpoints:
    """Tests for memory creation and retrieval endpoints."""
    
    def test_create_memory_success(self, client, mock_memory_store, mock_embedding_service):
        """Test creating a memory successfully."""
        mock_memory_store.find_similar_by_hash.return_value = []
        mock_memory_store.vector_search.return_value = []
        mock_memory_store.add_memory = AsyncMock()
        mock_memory_store.get_all_memories.return_value = []
        
        response = client.post(
            "/api/memories",
            json={
                "text": "Test memory content",
                "metadata": {"source": "test"},
                "source": "test"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "id" in data
        assert "content" in data
        assert data["content"] == "Test memory content"
        assert "embedding" in data
        assert "created_at" in data
        assert "metadata" in data
    
    def test_create_memory_duplicate_hash(self, client, mock_memory_store):
        """Test creating duplicate memory (same hash) is rejected."""
        mock_memory_store.find_similar_by_hash.return_value = ["existing-memory-id"]
        
        response = client.post(
            "/api/memories",
            json={
                "text": "Duplicate content",
                "metadata": {}
            }
        )
        
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "duplicate" in response.json()["detail"].lower()
    
    def test_create_memory_semantic_duplicate(self, client, mock_memory_store, mock_embedding_service):
        """Test creating semantic duplicate is rejected."""
        from main import Memory
        
        # Mock finding an existing similar memory
        existing_memory = Memory(
            id="existing",
            content="User prefers morning workouts",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={},
            content_hash="hash123"
        )
        
        mock_memory_store.find_similar_by_hash.return_value = []
        mock_memory_store.vector_search.return_value = [existing_memory]
        mock_embedding_service.cosine_similarity.return_value = 0.96  # Above threshold
        
        response = client.post(
            "/api/memories",
            json={
                "text": "User likes morning exercise",
                "metadata": {}
            }
        )
        
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "duplicate" in response.json()["detail"].lower()
    
    def test_get_memory_not_found(self, client, mock_memory_store):
        """Test getting non-existent memory returns 404."""
        mock_memory_store.get_memory = AsyncMock(return_value=None)
        
        response = client.get("/api/memories/non-existent-id")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_memory_success(self, client, mock_memory_store, sample_memory):
        """Test getting an existing memory."""
        mock_memory_store.get_memory = AsyncMock(return_value=sample_memory)
        
        response = client.get(f"/api/memories/{sample_memory.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == sample_memory.id
        assert data["content"] == sample_memory.content


class TestSearchEndpoint:
    """Tests for search endpoint."""
    
    def test_search_success(self, client, mock_memory_store, mock_embedding_service, sample_memory):
        """Test semantic search returns results."""
        mock_memory_store.vector_search.return_value = [sample_memory]
        
        response = client.post(
            "/api/search",
            json={
                "query": "test query",
                "top_k": 10,
                "include_inactive": False
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        results = response.json()
        assert isinstance(results, list)
    
    def test_search_invalid_request(self, client):
        """Test search with invalid request."""
        response = client.post(
            "/api/search",
            json={
                "query": "",  # Empty query
                "top_k": -1  # Invalid top_k
            }
        )
        
        # Should return validation error
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_get_stats_success(self, client, mock_stats_cache, mock_memory_store):
        """Test getting stats."""
        # Mock cache miss - will compute stats
        mock_stats_cache.get_stats.return_value = None
        
        mock_memory_store.get_all_memories.return_value = []
        mock_memory_store.get_relationships = AsyncMock(return_value=[])
        
        response = client.get("/api/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_memories" in data
        assert "active_memories" in data
        assert "inactive_memories" in data
        assert "total_relationships" in data
        assert "relationship_types" in data
        assert "avg_relationships_per_memory" in data
    
    def test_get_stats_cached(self, client, mock_stats_cache):
        """Test getting stats from cache."""
        cached_stats = {
            "total_memories": 10,
            "active_memories": 8,
            "inactive_memories": 2,
            "total_relationships": 15,
            "relationship_types": {},
            "avg_relationships_per_memory": 1.5
        }
        mock_stats_cache.get_stats.return_value = cached_stats
        
        response = client.get("/api/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_memories"] == 10


class TestGraphEndpoint:
    """Tests for knowledge graph endpoint."""
    
    def test_get_graph_success(self, client, mock_memory_store):
        """Test getting knowledge graph."""
        mock_memory_store.get_all_memories.return_value = []
        mock_memory_store.get_relationships = AsyncMock(return_value=[])
        
        response = client.get("/api/graph?depth=2&active_only=true")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "nodes" in data
        assert "edges" in data
        assert "total_nodes" in data
        assert "total_edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
    
    def test_get_graph_with_memory_ids(self, client, mock_memory_store):
        """Test getting subgraph for specific memories."""
        mock_memory_store.get_subgraph.return_value = ([], [])
        
        response = client.get("/api/graph?memory_ids=id1&memory_ids=id2&depth=2")
        
        assert response.status_code == status.HTTP_200_OK


class TestErrorHandling:
    """Tests for error handling across endpoints."""
    
    def test_invalid_json_returns_422(self, client):
        """Test invalid JSON returns 422."""
        response = client.post(
            "/api/memories",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client):
        """Test missing required fields returns validation error."""
        response = client.post(
            "/api/memories",
            json={"metadata": {}}  # Missing 'text' field
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

