"""
Integration tests that test multiple components together.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock


class TestMemoryCreationFlow:
    """Integration tests for complete memory creation flow."""
    
    @pytest.mark.asyncio
    async def test_complete_memory_creation_flow(self, client):
        """Test complete flow: create memory -> search -> get relationships."""
        with patch('main.memory_store') as mock_store, \
             patch('main.embedding_service') as mock_embedding:
            
            from main import Memory, MemoryRelationship, RelationType
            
            # Setup mocks
            new_memory = Memory(
                id="new-memory",
                content="Test memory",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash123"
            )
            
            existing_memory = Memory(
                id="existing-memory",
                content="Related memory",
                embedding=[0.15] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash456"
            )
            
            relationship = MemoryRelationship(
                from_id=existing_memory.id,
                to_id=new_memory.id,
                relation_type=RelationType.EXTEND,
                confidence=0.85,
                created_at=datetime.utcnow()
            )
            
            mock_store.find_similar_by_hash = AsyncMock(return_value=[])
            mock_store.vector_search = AsyncMock(return_value=[])
            mock_store.add_memory = AsyncMock(return_value=new_memory)
            mock_store.get_all_memories = AsyncMock(return_value=[existing_memory])
            mock_store.get_relationships = AsyncMock(return_value=[relationship])
            mock_store.add_relationship = AsyncMock()
            
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_embedding.cosine_similarity.return_value = 0.85
            
            # Step 1: Create memory
            create_response = client.post(
                "/api/memories",
                json={
                    "text": "Test memory",
                    "metadata": {"source": "test"}
                }
            )
            
            assert create_response.status_code == 200
            
            # Step 2: Search for the memory
            search_response = client.post(
                "/api/search",
                json={
                    "query": "test",
                    "top_k": 10
                }
            )
            
            assert search_response.status_code == 200
            
            # Step 3: Get relationships
            mock_store.get_memory = AsyncMock(return_value=new_memory)
            
            rel_response = client.get(f"/api/memories/{new_memory.id}/relationships")
            
            assert rel_response.status_code == 200
            relationships = rel_response.json()
            assert isinstance(relationships, list)


class TestStatsCacheIntegration:
    """Integration tests for stats caching."""
    
    @pytest.mark.asyncio
    async def test_stats_cache_invalidation_on_create(self, client):
        """Test that creating memory invalidates stats cache."""
        with patch('main.memory_store') as mock_store, \
             patch('main.stats_cache') as mock_cache, \
             patch('main.embedding_service') as mock_embedding:
            
            from main import Memory
            
            new_memory = Memory(
                id="new",
                content="Test",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash"
            )
            
            mock_store.find_similar_by_hash = AsyncMock(return_value=[])
            mock_store.vector_search = AsyncMock(return_value=[])
            mock_store.add_memory = AsyncMock()
            mock_store.get_all_memories = AsyncMock(return_value=[])
            mock_store.get_relationships = AsyncMock(return_value=[])
            
            mock_cache.get_stats = AsyncMock(return_value=None)
            mock_cache.set_stats = AsyncMock()
            mock_cache.invalidate = AsyncMock()
            
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            
            # Create memory
            client.post(
                "/api/memories",
                json={"text": "Test", "metadata": {}}
            )
            
            # Verify cache was invalidated
            mock_cache.invalidate.assert_called_once()


class TestGraphUpdateNotifications:
    """Integration tests for real-time graph updates."""
    
    @pytest.mark.asyncio
    async def test_memory_creation_triggers_notification(self, client):
        """Test that creating memory triggers graph update notification."""
        with patch('main.memory_store') as mock_store, \
             patch('main.graph_notifier') as mock_notifier, \
             patch('main.embedding_service') as mock_embedding:
            
            from main import Memory
            
            new_memory = Memory(
                id="new",
                content="Test",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash"
            )
            
            mock_store.find_similar_by_hash = AsyncMock(return_value=[])
            mock_store.vector_search = AsyncMock(return_value=[])
            mock_store.add_memory = AsyncMock(return_value=new_memory)
            mock_store.get_all_memories = AsyncMock(return_value=[])
            mock_store.add_relationship = AsyncMock()
            
            mock_notifier.notify_update = AsyncMock()
            
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            
            # Create memory
            client.post(
                "/api/memories",
                json={"text": "Test", "metadata": {}}
            )
            
            # Verify notification was sent
            assert mock_notifier.notify_update.called
            call_args = mock_notifier.notify_update.call_args
            assert call_args[0][0] == "memory_created"  # Event type

