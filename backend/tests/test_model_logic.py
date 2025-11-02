"""
Tests for model logic and business rules.
"""
import pytest
from datetime import datetime
from main import MemoryProcessor, RelationType, Memory, MemoryRelationship
from unittest.mock import AsyncMock, patch


class TestMemoryProcessor:
    """Tests for MemoryProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a MemoryProcessor instance."""
        return MemoryProcessor()
    
    def test_compute_content_hash(self, processor):
        """Test content hash computation."""
        text = "Test content"
        hash1 = processor.compute_content_hash(text)
        hash2 = processor.compute_content_hash(text)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 hex digest first 16 chars
    
    def test_compute_content_hash_different_content(self, processor):
        """Test different content produces different hash."""
        hash1 = processor.compute_content_hash("Content 1")
        hash2 = processor.compute_content_hash("Content 2")
        
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_create_memory(self, processor):
        """Test memory creation."""
        with patch('main.embedding_service') as mock_service:
            mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            
            memory = await processor.create_memory(
                content="Test memory",
                metadata={"source": "test"},
                source="test"
            )
            
            assert memory.content == "Test memory"
            assert memory.metadata["source"] == "test"
            assert len(memory.embedding) == 768
            assert memory.is_active is True
            assert memory.version == 1
    
    @pytest.mark.asyncio
    async def test_infer_relationships_high_similarity(self, processor):
        """Test relationship inference for high similarity (UPDATE)."""
        with patch('main.embedding_service') as mock_service:
            mock_service.cosine_similarity.return_value = 0.96  # Above 0.95 threshold
            
            new_memory = Memory(
                id="new",
                content="Updated information",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash1"
            )
            
            existing_memory = Memory(
                id="existing",
                content="Old information",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash2"
            )
            
            relationships = await processor.infer_relationships(
                new_memory,
                [existing_memory]
            )
            
            assert len(relationships) > 0
            assert relationships[0].relation_type == RelationType.UPDATE
            assert relationships[0].confidence == 0.96
    
    @pytest.mark.asyncio
    async def test_infer_relationships_moderate_similarity(self, processor):
        """Test relationship inference for moderate similarity."""
        with patch('main.embedding_service') as mock_service:
            mock_service.cosine_similarity.return_value = 0.80  # Between 0.75 and 0.95
            
            new_memory = Memory(
                id="new",
                content="Extended information with more details",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash1"
            )
            
            existing_memory = Memory(
                id="existing",
                content="Short info",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash2"
            )
            
            relationships = await processor.infer_relationships(
                new_memory,
                [existing_memory]
            )
            
            assert len(relationships) > 0
            # Should be EXTEND since new memory is longer
            assert relationships[0].relation_type == RelationType.EXTEND
    
    @pytest.mark.asyncio
    async def test_infer_relationships_low_similarity(self, processor):
        """Test relationship inference for low similarity."""
        with patch('main.embedding_service') as mock_service:
            mock_service.cosine_similarity.return_value = 0.50  # Below 0.75 threshold
            
            new_memory = Memory(
                id="new",
                content="Completely different content",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash1"
            )
            
            existing_memory = Memory(
                id="existing",
                content="Unrelated content",
                embedding=[0.9] * 768,  # Different embedding
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash2"
            )
            
            relationships = await processor.infer_relationships(
                new_memory,
                [existing_memory]
            )
            
            # Should have no relationships for low similarity
            assert len(relationships) == 0
    
    @pytest.mark.asyncio
    async def test_check_semantic_duplicates_found(self, processor):
        """Test semantic duplicate detection."""
        with patch('main.embedding_service') as mock_service, \
             patch('main.memory_store') as mock_store:
            
            existing_memory = Memory(
                id="existing",
                content="Existing memory",
                embedding=[0.1] * 768,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1,
                is_active=True,
                metadata={},
                content_hash="hash1"
            )
            
            mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_store.vector_search = AsyncMock(return_value=[existing_memory])
            mock_service.cosine_similarity.return_value = 0.96  # Above threshold
            
            result = await processor.check_semantic_duplicates("Similar content")
            
            assert result is not None
            memory, similarity = result
            assert memory.id == existing_memory.id
            assert similarity == 0.96
    
    @pytest.mark.asyncio
    async def test_check_semantic_duplicates_not_found(self, processor):
        """Test semantic duplicate detection when no duplicate exists."""
        with patch('main.embedding_service') as mock_service, \
             patch('main.memory_store') as mock_store:
            
            mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_store.vector_search = AsyncMock(return_value=[])
            
            result = await processor.check_semantic_duplicates("Unique content")
            
            assert result is None


class TestRelationTypes:
    """Tests for RelationType enum."""
    
    def test_relation_type_values(self):
        """Test all relation types are properly defined."""
        assert RelationType.UPDATE == "UPDATE"
        assert RelationType.EXTEND == "EXTEND"
        assert RelationType.DERIVE == "DERIVE"
        assert RelationType.CHUNK_SEQUENCE == "CHUNK_SEQUENCE"
    
    def test_relation_type_from_value(self):
        """Test creating RelationType from string value."""
        assert RelationType("UPDATE") == RelationType.UPDATE
        assert RelationType("EXTEND") == RelationType.EXTEND
        assert RelationType("CHUNK_SEQUENCE") == RelationType.CHUNK_SEQUENCE

