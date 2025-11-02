# API Documentation

Complete reference documentation for the AI Memory Platform REST API.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Data Models](#data-models)
- [Endpoints](#endpoints)
  - [Memory Management](#memory-management)
  - [Search & Discovery](#search--discovery)
  - [Knowledge Graph](#knowledge-graph)
  - [Statistics & Monitoring](#statistics--monitoring)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://yourdomain.com`

All API endpoints are prefixed with `/api` (except `/health`).

## Authentication

Currently, no authentication is required. For production deployments, implement API key authentication or OAuth2.

**Future Authentication (Planned)**:
```http
Authorization: Bearer <api_key>
```

## Data Models

### MemoryInput

Request model for creating a new memory.

```json
{
  "text": "string (required)",
  "metadata": {
    "source": "string",
    "key": "value"
  },
  "source": "string (optional)"
}
```

**Fields**:
- `text` (string, required): Text content to memorize
- `metadata` (object, optional): Key-value metadata (default: `{}`)
- `source` (string, optional): Source identifier (e.g., "manual", "pdf", "api")

### Memory

Response model for memory objects.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "User prefers morning workouts at 6 AM",
  "embedding": [0.1, 0.2, 0.3, ...],  // 768-dimensional vector
  "created_at": "2024-01-01T00:00:00.000000",
  "updated_at": "2024-01-01T00:00:00.000000",
  "version": 1,
  "is_active": true,
  "metadata": {
    "source": "user_input"
  },
  "content_hash": "a1b2c3d4e5f6"
}
```

**Fields**:
- `id` (string): Unique memory identifier (UUID)
- `content` (string): Memory text content
- `embedding` (array[float]): Vector embedding (768 dimensions default)
- `created_at` (datetime): Creation timestamp (ISO 8601)
- `updated_at` (datetime): Last update timestamp (ISO 8601)
- `version` (integer): Memory version number
- `is_active` (boolean): Whether memory is active
- `metadata` (object): Key-value metadata
- `content_hash` (string): SHA256 hash of content (first 16 chars)

### MemoryRelationship

Model for relationships between memories.

```json
{
  "from_id": "550e8400-e29b-41d4-a716-446655440000",
  "to_id": "660e8400-e29b-41d4-a716-446655440001",
  "relation_type": "UPDATE",
  "confidence": 0.95,
  "created_at": "2024-01-01T00:00:00.000000",
  "reasoning": "High content similarity suggests updated information"
}
```

**Fields**:
- `from_id` (string): Source memory ID
- `to_id` (string): Target memory ID
- `relation_type` (enum): Relationship type:
  - `UPDATE`: Supersedes previous information
  - `EXTEND`: Adds context while keeping original valid
  - `DERIVE`: Inferred insight from patterns
  - `CHUNK_SEQUENCE`: Sequential chunks from same document
- `confidence` (float): Confidence score (0.0-1.0)
- `created_at` (datetime): Creation timestamp (ISO 8601)
- `reasoning` (string, optional): Explanation for relationship

### SearchQuery

Request model for semantic search.

```json
{
  "query": "morning exercise routine",
  "top_k": 10,
  "include_inactive": false,
  "relation_depth": 2
}
```

**Fields**:
- `query` (string, required): Search query text
- `top_k` (integer, default: 10, range: 1-100): Number of results to return
- `include_inactive` (boolean, default: false): Include inactive memories
- `relation_depth` (integer, default: 2, range: 0-5): Depth for relationship expansion

### KnowledgeGraphResponse

Response model for knowledge graph.

```json
{
  "nodes": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "User prefers morning workouts at 6 AM...",
      "version": 1,
      "is_active": true,
      "created_at": "2024-01-01T00:00:00.000000",
      "metadata": {}
    }
  ],
  "edges": [
    {
      "from_id": "550e8400-e29b-41d4-a716-446655440000",
      "to_id": "660e8400-e29b-41d4-a716-446655440001",
      "relation_type": "EXTEND",
      "confidence": 0.85
    }
  ],
  "total_nodes": 10,
  "total_edges": 15
}
```

**Fields**:
- `nodes` (array[GraphNode]): Graph nodes (memories)
- `edges` (array[GraphEdge]): Graph edges (relationships)
- `total_nodes` (integer): Total number of nodes
- `total_edges` (integer): Total number of edges

**Note**: Node `content` is truncated to 100 characters for performance.

## Endpoints

### Memory Management

#### Create Memory

Create a new memory from text input.

```http
POST /api/memories
Content-Type: application/json
```

**Request Body**:
```json
{
  "text": "User prefers morning workouts at 6 AM",
  "metadata": {"source": "user_input"},
  "source": "manual"
}
```

**Response**: `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "User prefers morning workouts at 6 AM",
  "embedding": [...],
  "created_at": "2024-01-01T00:00:00.000000",
  "updated_at": "2024-01-01T00:00:00.000000",
  "version": 1,
  "is_active": true,
  "metadata": {"source": "user_input"},
  "content_hash": "a1b2c3d4e5f6"
}
```

**Errors**:
- `409 Conflict`: Exact duplicate or semantic near-duplicate found
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST "http://localhost:8000/api/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers morning workouts at 6 AM",
    "metadata": {"source": "user_input"},
    "source": "manual"
  }'
```

**Notes**:
- Automatically checks for duplicates (exact hash and semantic similarity >= 0.95)
- Generates embeddings using configured model
- Infers relationships with existing memories
- Broadcasts graph update event via SSE

#### Upload PDF

Extract text from PDF and create memories for each chunk.

```http
POST /api/memories/pdf
Content-Type: multipart/form-data
```

**Request**:
- `file` (file, required): PDF file to process

**Response**: `200 OK`
```json
{
  "created": 5,
  "skipped": 2,
  "skipped_details": [
    "Chunk 3: exact duplicate",
    "Chunk 7: semantic duplicate (similarity: 96%)"
  ],
  "memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Chunk 1 content...",
      ...
    }
  ],
  "relationships_created": 8
}
```

**Fields**:
- `created` (integer): Number of memories created
- `skipped` (integer): Number of duplicate chunks skipped
- `skipped_details` (array[string]): Details about skipped chunks
- `memories` (array[Memory]): Created memory objects
- `relationships_created` (integer): Number of relationships created

**Errors**:
- `400 Bad Request`: Invalid file or no text extracted
- `500 Internal Server Error`: PDF processing error

**Example**:
```bash
curl -X POST "http://localhost:8000/api/memories/pdf" \
  -F "file=@document.pdf"
```

**Notes**:
- PDFs are automatically split into chunks (max 2000 chars per chunk)
- Consecutive chunks from same PDF are linked with `CHUNK_SEQUENCE` relationship
- Duplicate chunks (exact or semantic) are skipped
- All chunks from same PDF share a `pdf_source_id` in metadata

#### Get Memory

Retrieve a specific memory by ID.

```http
GET /api/memories/{memory_id}
```

**Path Parameters**:
- `memory_id` (string, required): Memory UUID

**Response**: `200 OK` with `Memory` object

**Errors**:
- `404 Not Found`: Memory not found

**Example**:
```bash
curl "http://localhost:8000/api/memories/550e8400-e29b-41d4-a716-446655440000"
```

#### Get Memory Relationships

Get all relationships for a specific memory.

```http
GET /api/memories/{memory_id}/relationships
```

**Path Parameters**:
- `memory_id` (string, required): Memory UUID

**Response**: `200 OK` with array of `MemoryRelationship` objects

**Example**:
```bash
curl "http://localhost:8000/api/memories/550e8400-e29b-41d4-a716-446655440000/relationships"
```

### Search & Discovery

#### Semantic Search

Search memories using semantic similarity.

```http
POST /api/search
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "morning exercise routine",
  "top_k": 10,
  "include_inactive": false,
  "relation_depth": 2
}
```

**Response**: `200 OK` with array of `Memory` objects (sorted by relevance)

**Example**:
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "morning exercise routine",
    "top_k": 10,
    "include_inactive": false,
    "relation_depth": 2
  }'
```

**Notes**:
- Uses vector similarity search via Neo4j
- Results sorted by cosine similarity score
- Can expand results using relationship depth

### Knowledge Graph

#### Get Knowledge Graph

Retrieve the knowledge graph with optional filtering.

```http
GET /api/graph?memory_ids=id1&memory_ids=id2&depth=2&active_only=true
```

**Query Parameters**:
- `memory_ids` (array[string], optional): Filter to specific memories and their neighbors
- `depth` (integer, default: 2, range: 0-5): Traversal depth for subgraph
- `active_only` (boolean, default: true): Include only active memories

**Response**: `200 OK` with `KnowledgeGraphResponse`

**Examples**:
```bash
# Get entire graph
curl "http://localhost:8000/api/graph?depth=2&active_only=true"

# Get subgraph around specific memories
curl "http://localhost:8000/api/graph?memory_ids=id1&memory_ids=id2&depth=3"
```

**Notes**:
- Uses optimized batch queries (no N+1 problems)
- Node content truncated to 100 chars for performance
- Can filter by specific memory IDs for focused subgraphs

#### Stream Graph Updates (SSE)

Stream real-time graph update notifications via Server-Sent Events.

```http
GET /api/graph/stream
Accept: text/event-stream
```

**Response**: `200 OK` with Server-Sent Events stream

**Event Types**:
- `connected`: Connection confirmation
- `memory_created`: New memory created
- `relationships_added`: New relationships added
- `graph_updated`: General graph update
- `heartbeat`: Keep-alive ping (every 30 seconds)

**Example Event**:
```
data: {"type": "memory_created", "data": {"memory_id": "...", "content_preview": "..."}}

: heartbeat
```

**Example**:
```bash
curl -N "http://localhost:8000/api/graph/stream"
```

**JavaScript Example**:
```javascript
const eventSource = new EventSource('http://localhost:8000/api/graph/stream');
eventSource.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Graph update:', update);
};
```

### Statistics & Monitoring

#### Get Platform Statistics

Get platform statistics and metrics.

```http
GET /api/stats
```

**Response**: `200 OK`
```json
{
  "total_memories": 150,
  "active_memories": 142,
  "inactive_memories": 8,
  "total_relationships": 245,
  "relationship_types": {
    "UPDATE": 12,
    "EXTEND": 156,
    "DERIVE": 67,
    "CHUNK_SEQUENCE": 10
  },
  "avg_relationships_per_memory": 1.63,
  "cached_at": "2024-01-01T00:00:00.000000"
}
```

**Fields**:
- `total_memories` (integer): Total number of memories
- `active_memories` (integer): Number of active memories
- `inactive_memories` (integer): Number of inactive memories
- `total_relationships` (integer): Total number of relationships
- `relationship_types` (object): Count by relationship type
- `avg_relationships_per_memory` (float): Average relationships per memory
- `cached_at` (datetime): Cache timestamp (ISO 8601)

**Example**:
```bash
curl "http://localhost:8000/api/stats"
```

**Notes**:
- Results are cached for 5 minutes for performance
- Cache automatically invalidates on data changes
- Uses Redis if available, falls back to in-memory cache

#### Health Check

Get system health status.

```http
GET /health
```

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000000",
  "neo4j": {
    "connected": true,
    "vector_search_enabled": true
  },
  "embeddings": {
    "model_loaded": true,
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "dimension": 768
  }
}
```

**Fields**:
- `status` (string): Overall health status
- `timestamp` (datetime): Check timestamp (ISO 8601)
- `neo4j` (object): Neo4j connection status
  - `connected` (boolean): Connection status
  - `vector_search_enabled` (boolean): Vector search available
- `embeddings` (object): Embedding model status
  - `model_loaded` (boolean): Model loaded successfully
  - `model_name` (string): Model name
  - `dimension` (integer): Embedding dimension

**Example**:
```bash
curl "http://localhost:8000/health"
```

## Error Handling

All endpoints follow consistent error handling:

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Success |
| 400 | Bad Request | Invalid input or validation error |
| 401 | Unauthorized | Authentication required |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Duplicate content (exact or semantic) |
| 422 | Unprocessable Entity | Request validation failed |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Database or service unavailable |
| 504 | Gateway Timeout | Request timeout |

### Common Error Scenarios

#### 409 Conflict - Duplicate Memory

```json
{
  "detail": "Exact duplicate memory already exists: 550e8400-e29b-41d4-a716-446655440000"
}
```

or

```json
{
  "detail": "Semantic near-duplicate found (similarity: 96%). Existing memory ID: 550e8400-e29b-41d4-a716-446655440000. Content may be too similar to existing memory."
}
```

#### 404 Not Found - Memory Not Found

```json
{
  "detail": "Memory not found: 550e8400-e29b-41d4-a716-446655440000"
}
```

#### 422 Unprocessable Entity - Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider adding rate limiting middleware.

**Recommended Limits**:
- Create Memory: 100 requests/minute
- Search: 200 requests/minute
- Get Graph: 50 requests/minute
- Upload PDF: 10 requests/minute

## OpenAPI Documentation

Interactive API documentation is automatically generated:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

The OpenAPI schema follows OpenAPI 3.0 specification and can be imported into API testing tools like Postman or Insomnia.

## Best Practices

1. **Use appropriate HTTP methods**: GET for retrieval, POST for creation
2. **Handle errors gracefully**: Check status codes and error messages
3. **Use pagination**: For large result sets (not yet implemented)
4. **Cache responses**: Stats endpoint uses caching for performance
5. **Monitor rate limits**: When implemented, respect rate limits
6. **Use SSE for real-time updates**: Subscribe to `/api/graph/stream` for live updates
7. **Validate input**: All inputs are validated via Pydantic models

## Versioning

Current API version: **1.0.0**

API versioning strategy (future):
- URL versioning: `/api/v1/memories`
- Header versioning: `Accept: application/vnd.api+json;version=1`

## Support

For API support:
- Check interactive docs: `http://localhost:8000/docs`
- Review error messages for detailed information
- Check system health: `GET /health`

