"""
Pytest configuration and shared fixtures for backend tests.
"""
import pytest
import os
import sys
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app, memory_store, stats_cache, graph_notifier


@pytest.fixture(scope="session")
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="function")
def mock_memory_store():
    """Mock memory store for isolated tests."""
    with patch('main.memory_store') as mock:
        mock.use_neo4j = False
        mock.memories = {}
        mock.relationships = []
        mock.content_index = {}
        
        # Mock async methods
        mock.get_memory = AsyncMock(return_value=None)
        mock.add_memory = AsyncMock()
        mock.get_all_memories = AsyncMock(return_value=[])
        mock.find_similar_by_hash = AsyncMock(return_value=[])
        mock.get_relationships = AsyncMock(return_value=[])
        mock.add_relationship = AsyncMock()
        mock.vector_search = AsyncMock(return_value=[])
        mock.get_subgraph = AsyncMock(return_value=([], []))
        
        yield mock


@pytest.fixture(scope="function")
def mock_embedding_service():
    """Mock embedding service."""
    with patch('main.embedding_service') as mock:
        mock.generate_embedding = AsyncMock(return_value=[0.1] * 768)
        mock.cosine_similarity = MagicMock(return_value=0.85)
        yield mock


@pytest.fixture(scope="function")
def mock_stats_cache():
    """Mock stats cache."""
    with patch('main.stats_cache') as mock:
        mock.get_stats = AsyncMock(return_value=None)
        mock.set_stats = AsyncMock()
        mock.invalidate = AsyncMock()
        yield mock


@pytest.fixture(scope="function")
def sample_memory():
    """Sample memory object for testing."""
    from main import Memory
    from datetime import datetime
    
    return Memory(
        id="test-memory-1",
        content="This is a test memory",
        embedding=[0.1] * 768,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        version=1,
        is_active=True,
        metadata={"test": True},
        content_hash="abc123"
    )


@pytest.fixture(scope="function")
def sample_memory_input():
    """Sample memory input for API tests."""
    return {
        "text": "User prefers morning workouts at 6 AM",
        "metadata": {"source": "test"},
        "source": "test"
    }


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset caches before each test."""
    # Clear graph notifier subscribers
    graph_notifier.subscribers.clear()
    
    # Reset stats cache in-memory store if it exists
    if hasattr(stats_cache, 'in_memory_cache'):
        stats_cache.in_memory_cache = None
    
    yield
    
    # Cleanup after test
    graph_notifier.subscribers.clear()
    if hasattr(stats_cache, 'in_memory_cache'):
        stats_cache.in_memory_cache = None

