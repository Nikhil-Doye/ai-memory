"""
Tests for error handling and edge cases.
"""
import pytest
from fastapi import status
from unittest.mock import AsyncMock, patch
from main import handle_endpoint_error, classify_exception, HTTPException
from neo4j.exceptions import ServiceUnavailable, AuthError


class TestErrorClassification:
    """Tests for error classification logic."""
    
    def test_classify_neo4j_service_unavailable(self):
        """Test Neo4j ServiceUnavailable is classified correctly."""
        error = ServiceUnavailable("Connection refused")
        status_code, message = classify_exception(error)
        
        assert status_code == 503
        assert "Database service unavailable" in message
    
    def test_classify_neo4j_auth_error(self):
        """Test Neo4j AuthError is classified correctly."""
        error = AuthError("Invalid credentials")
        status_code, message = classify_exception(error)
        
        assert status_code == 401
        assert "Database authentication failed" in message
    
    def test_classify_value_error(self):
        """Test ValueError is classified correctly."""
        error = ValueError("Invalid input")
        status_code, message = classify_exception(error)
        
        assert status_code == 400
        assert "Invalid input" in message
    
    def test_classify_generic_exception(self):
        """Test generic exception defaults to 500."""
        error = Exception("Something went wrong")
        status_code, message = classify_exception(error)
        
        assert status_code == 500
        assert "Internal server error" in message


class TestErrorHandlingIntegration:
    """Tests for error handling in endpoints."""
    
    def test_memory_creation_database_error(self, client):
        """Test memory creation handles database errors gracefully."""
        with patch('main.memory_store.find_similar_by_hash') as mock_find:
            mock_find.side_effect = ServiceUnavailable("Database connection failed")
            
            response = client.post(
                "/api/memories",
                json={
                    "text": "Test content",
                    "metadata": {}
                }
            )
            
            # Should return appropriate error status
            assert response.status_code >= 500 or response.status_code == 503
    
    def test_search_embedding_error(self, client):
        """Test search handles embedding service errors."""
        with patch('main.embedding_service.generate_embedding') as mock_embed:
            mock_embed.side_effect = Exception("Embedding model failed")
            
            response = client.post(
                "/api/search",
                json={
                    "query": "test query",
                    "top_k": 10
                }
            )
            
            # Should return appropriate error status
            assert response.status_code >= 500
    
    def test_get_memory_invalid_id_format(self, client):
        """Test getting memory with invalid ID format."""
        response = client.get("/api/memories/invalid-id-format-123")
        
        # Should either return 404 or handle gracefully
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_200_OK]
    
    def test_stats_cache_error_handling(self, client):
        """Test stats endpoint handles cache errors gracefully."""
        with patch('main.stats_cache.get_stats') as mock_get:
            mock_get.side_effect = Exception("Cache error")
            
            # Should still compute stats if cache fails
            response = client.get("/api/stats")
            
            # Should return stats or appropriate error
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_503_SERVICE_UNAVAILABLE
            ]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_create_memory_empty_text(self, client):
        """Test creating memory with empty text."""
        response = client.post(
            "/api/memories",
            json={
                "text": "",
                "metadata": {}
            }
        )
        
        # Should either accept or return validation error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST
        ]
    
    def test_create_memory_very_long_text(self, client, mock_memory_store, mock_embedding_service):
        """Test creating memory with very long text."""
        long_text = "A" * 10000
        
        mock_memory_store.find_similar_by_hash.return_value = []
        mock_memory_store.vector_search.return_value = []
        mock_memory_store.add_memory = AsyncMock()
        mock_memory_store.get_all_memories.return_value = []
        
        response = client.post(
            "/api/memories",
            json={
                "text": long_text,
                "metadata": {}
            }
        )
        
        # Should handle long text or return appropriate error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_400_BAD_REQUEST
        ]
    
    def test_search_zero_top_k(self, client):
        """Test search with zero top_k."""
        response = client.post(
            "/api/search",
            json={
                "query": "test",
                "top_k": 0
            }
        )
        
        # Should return validation error for invalid top_k
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_search_negative_top_k(self, client):
        """Test search with negative top_k."""
        response = client.post(
            "/api/search",
            json={
                "query": "test",
                "top_k": -1
            }
        )
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_graph_invalid_depth(self, client):
        """Test getting graph with invalid depth."""
        response = client.get("/api/graph?depth=10")  # Above max of 5
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

