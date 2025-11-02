# Backend Tests

This directory contains comprehensive unit and integration tests for the AI Memory Platform backend.

## Test Structure

- `test_api_endpoints.py` - Tests for all FastAPI endpoints
- `test_database_operations.py` - Tests for MemoryStore and database operations
- `test_model_logic.py` - Tests for business logic and model classes
- `test_error_handling.py` - Tests for error handling and edge cases
- `test_integration.py` - Integration tests for complete workflows
- `conftest.py` - Shared pytest fixtures and configuration

## Running Tests

### Run all tests:
```bash
cd backend
pytest
```

### Run specific test file:
```bash
pytest tests/test_api_endpoints.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage:
```bash
pytest --cov=main --cov-report=html
```

## Test Coverage

The test suite covers:

- ✅ All API endpoints (GET, POST)
- ✅ Memory creation and retrieval
- ✅ Semantic search functionality
- ✅ Knowledge graph operations
- ✅ Statistics endpoint
- ✅ Error handling and edge cases
- ✅ Database operations (in-memory fallback)
- ✅ Model logic and business rules
- ✅ Cache operations
- ✅ Real-time notifications

## Note

Tests use mocked dependencies to avoid requiring:
- Running Neo4j database
- Running Redis server
- Hugging Face model downloads

This allows fast, reliable CI/CD testing without external dependencies.

