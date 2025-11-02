"""
Tests for database operations and MemoryStore logic.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from main import Memory, MemoryRelationship, RelationType, MemoryStore


class TestMemoryStore:
    """Tests for MemoryStore class."""
    
    @pytest.fixture
    def memory_store(self):
        """Create a MemoryStore instance for testing."""
        with patch.dict('os.environ', {'USE_NEO4J': 'false'}):
            store = MemoryStore()
            return store
    
    @pytest.fixture
    def sample_memory(self):
        """Sample memory for testing."""
        return Memory(
            id="test-1",
            content="Test memory",
            embedding=[0.1] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={"test": True},
            content_hash="hash123"
        )
    
    @pytest.mark.asyncio
    async def test_add_memory_in_memory(self, memory_store, sample_memory):
        """Test adding memory to in-memory store."""
        result = await memory_store.add_memory(sample_memory)
        
        assert result == sample_memory
        assert sample_memory.id in memory_store.memories
        assert memory_store.memories[sample_memory.id] == sample_memory
        assert sample_memory.content_hash in memory_store.content_index
    
    @pytest.mark.asyncio
    async def test_get_memory_in_memory(self, memory_store, sample_memory):
        """Test retrieving memory from in-memory store."""
        await memory_store.add_memory(sample_memory)
        
        result = await memory_store.get_memory(sample_memory.id)
        
        assert result is not None
        assert result.id == sample_memory.id
        assert result.content == sample_memory.content
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, memory_store):
        """Test retrieving non-existent memory."""
        result = await memory_store.get_memory("non-existent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_similar_by_hash(self, memory_store, sample_memory):
        """Test finding memories by content hash."""
        await memory_store.add_memory(sample_memory)
        
        results = await memory_store.find_similar_by_hash(sample_memory.content_hash)
        
        assert len(results) > 0
        assert sample_memory.id in results
    
    @pytest.mark.asyncio
    async def test_add_relationship(self, memory_store, sample_memory):
        """Test adding relationship."""
        memory2 = Memory(
            id="test-2",
            content="Another memory",
            embedding=[0.2] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={},
            content_hash="hash456"
        )
        
        await memory_store.add_memory(sample_memory)
        await memory_store.add_memory(memory2)
        
        relationship = MemoryRelationship(
            from_id=sample_memory.id,
            to_id=memory2.id,
            relation_type=RelationType.EXTEND,
            confidence=0.85,
            created_at=datetime.utcnow(),
            reasoning="Test relationship"
        )
        
        await memory_store.add_relationship(relationship)
        
        assert len(memory_store.relationships) > 0
        assert relationship in memory_store.relationships
    
    @pytest.mark.asyncio
    async def test_get_relationships(self, memory_store, sample_memory):
        """Test retrieving relationships."""
        memory2 = Memory(
            id="test-2",
            content="Another memory",
            embedding=[0.2] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={},
            content_hash="hash456"
        )
        
        await memory_store.add_memory(sample_memory)
        await memory_store.add_memory(memory2)
        
        relationship = MemoryRelationship(
            from_id=sample_memory.id,
            to_id=memory2.id,
            relation_type=RelationType.EXTEND,
            confidence=0.85,
            created_at=datetime.utcnow()
        )
        
        await memory_store.add_relationship(relationship)
        
        relationships = await memory_store.get_relationships(sample_memory.id)
        
        assert len(relationships) > 0
        assert any(rel.from_id == sample_memory.id for rel in relationships)
    
    @pytest.mark.asyncio
    async def test_vector_search_in_memory(self, memory_store, sample_memory):
        """Test vector search in in-memory store."""
        await memory_store.add_memory(sample_memory)
        
        query_embedding = [0.1] * 768
        
        # Mock embedding service
        with patch('main.embedding_service') as mock_service:
            mock_service.cosine_similarity.return_value = 0.9
            
            results = await memory_store.vector_search(query_embedding, top_k=5)
            
            assert len(results) > 0
            assert sample_memory in results
    
    @pytest.mark.asyncio
    async def test_get_all_memories(self, memory_store, sample_memory):
        """Test retrieving all memories."""
        await memory_store.add_memory(sample_memory)
        
        all_memories = await memory_store.get_all_memories(include_inactive=False)
        
        assert len(all_memories) > 0
        assert sample_memory in all_memories
    
    @pytest.mark.asyncio
    async def test_get_all_memories_exclude_inactive(self, memory_store, sample_memory):
        """Test retrieving only active memories."""
        inactive_memory = Memory(
            id="test-inactive",
            content="Inactive memory",
            embedding=[0.3] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=False,  # Inactive
            metadata={},
            content_hash="hash789"
        )
        
        await memory_store.add_memory(sample_memory)
        await memory_store.add_memory(inactive_memory)
        
        active_memories = await memory_store.get_all_memories(include_inactive=False)
        
        assert sample_memory in active_memories
        assert inactive_memory not in active_memories
    
    @pytest.mark.asyncio
    async def test_relationship_update_marks_old_inactive(self, memory_store, sample_memory):
        """Test UPDATE relationship marks old memory as inactive."""
        memory2 = Memory(
            id="test-2",
            content="Updated memory",
            embedding=[0.2] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={},
            content_hash="hash456"
        )
        
        await memory_store.add_memory(sample_memory)
        await memory_store.add_memory(memory2)
        
        relationship = MemoryRelationship(
            from_id=sample_memory.id,
            to_id=memory2.id,
            relation_type=RelationType.UPDATE,
            confidence=0.95,
            created_at=datetime.utcnow()
        )
        
        await memory_store.add_relationship(relationship)
        
        # Check that old memory is marked inactive
        old_memory = memory_store.memories[sample_memory.id]
        assert old_memory.is_active is False

