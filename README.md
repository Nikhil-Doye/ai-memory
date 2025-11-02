# AI Memory Management Platform

A semantic memory management system that uses graph databases and vector embeddings to store, retrieve, and visualize knowledge relationships. Built with FastAPI backend and React frontend.

## 🚀 Features

- **Semantic Memory Storage**: Store and retrieve memories with vector embeddings using Qwen3-Embedding-0.6B model
- **Knowledge Graph**: Visualize relationships between memories using Neo4j graph database
- **Intelligent Relationships**: Automatically infer relationships between memories:
  - **UPDATE**: Supersedes previous information
  - **EXTEND**: Adds context while keeping original valid
  - **DERIVE**: Inferred insights from patterns
  - **CHUNK_SEQUENCE**: Sequential chunks from the same document/source
- **Vector Search**: Fast semantic search across memories using Neo4j's native vector search
- **PDF Upload**: Extract and process text from PDF documents with automatic chunking
- **Interactive Visualization**: Explore knowledge graph with interactive node graph visualization
- **Version Control**: Track memory versions and manage active/inactive states
- **Real-time Stats**: Monitor platform statistics and memory metrics
- **Real-time Updates**: Server-Sent Events (SSE) for live graph updates
- **Semantic Deduplication**: Prevents near-duplicate memories from being stored
- **Performance Optimized**: Batch queries and Redis caching for sub-400ms latency

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   React     │────▶│   FastAPI    │────▶│   Neo4j     │
│  Frontend   │     │   Backend    │     │   Database  │
│  (Vite)     │◀────│   (Python)   │◀────│  (Graph +   │
└─────────────┘     └──────────────┘     │   Vector)   │
      │                    │              └─────────────┘
      │                    │
      │                    ▼
      │            ┌──────────────┐
      │            │   Redis      │
      │            │   (Cache)    │
      │            └──────────────┘
      │
      └───────────┐
                  ▼
          ┌──────────────┐
          │  Hugging Face│
          │  Embeddings  │
          │  (Qwen3)     │
          └──────────────┘
```

### System Components

#### Backend (FastAPI)
- **Memory Processing**: Handles memory creation, embedding generation, and relationship inference
- **Graph Operations**: Manages Neo4j graph database with optimized batch queries
- **Vector Search**: Semantic similarity search using Neo4j's native vector index
- **Caching Layer**: Redis-based caching for stats and frequently accessed data
- **Real-time Streaming**: Server-Sent Events for live graph updates
- **Error Handling**: Comprehensive error classification and HTTP status codes

#### Frontend (React + Vite)
- **Graph Visualization**: Interactive D3.js/React Flow visualization
- **Memory Management**: Create, view, and manage memories
- **Search Interface**: Semantic search with results visualization
- **Statistics Dashboard**: Real-time platform metrics

#### Database Layer
- **Neo4j**: Graph database storing:
  - Memories as nodes with vector embeddings
  - Relationships as edges with types and confidence scores
  - Vector index for cosine similarity search
- **Redis**: In-memory cache for:
  - Platform statistics
  - Frequently accessed data
  - Cache invalidation on data changes

#### Embedding Service
- **Model**: Qwen3-Embedding-0.6B (768 dimensions default, supports 32-1024)
- **Features**: 
  - Automatic fallback to simulated embeddings if model unavailable
  - Optional instruction prompts for query/document distinction
  - Configurable embedding dimensions

### Data Flow

1. **Memory Creation**:
   ```
   Text Input → Hash Check → Semantic Deduplication → 
   Embedding Generation → Memory Storage → Relationship Inference
   ```

2. **Search Flow**:
   ```
   Query → Embedding Generation → Vector Search (Neo4j) → 
   Relationship Expansion → Results Ranking → Response
   ```

3. **Graph Retrieval**:
   ```
   Request → Batch Relationship Fetch → Node/Edge Construction → 
   Filtering (active_only) → Response
   ```

## 📋 Prerequisites

- **Python 3.11+** (for backend)
- **Node.js 18+** (for frontend)
- **Docker and Docker Compose** (for Neo4j and Redis)
- **(Optional) Hugging Face token** for embedding model access
  - Get token from: https://huggingface.co/settings/tokens
  - Accept model terms: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-memory
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

Create a `.env` file in the root directory or `backend/` directory:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here  # REQUIRED when USE_NEO4J=true
USE_NEO4J=true

# Redis Configuration (optional, defaults to in-memory cache)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# CORS Configuration (comma-separated list)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Embedding Model Configuration
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMENSION=768
USE_HF_EMBEDDING_MODEL=true
USE_EMBEDDING_INSTRUCTION=false
EMBEDDING_INSTRUCTION=query

# Hugging Face Token (optional, required for private models)
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
```

**Security Note**: Never commit `.env` files with real credentials. The `.env` file is already in `.gitignore`.

## 🐳 Docker Setup (Recommended)

The easiest way to run Neo4j and Redis:

```bash
# Start Neo4j and Redis services
docker-compose up -d

# Check services are running
docker-compose ps

# View Neo4j logs
docker-compose logs neo4j

# View Redis logs
docker-compose logs redis

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

This will start:
- **Neo4j** on:
  - Web UI: `http://localhost:7474` (username: `neo4j`, password from `.env`)
  - Bolt protocol: `bolt://localhost:7687`
- **Redis** on `localhost:6379`

## 🚀 Running the Application

### Development Mode

1. **Start Docker services** (Neo4j and Redis):
   ```bash
   docker-compose up -d
   ```

2. **Start Backend**:
   ```bash
   cd backend
   # Activate virtual environment if not already active
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   
   # Run with auto-reload
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Backend will be available at:
   - API: `http://localhost:8000`
   - Interactive API Docs (Swagger): `http://localhost:8000/docs`
   - Alternative API Docs (ReDoc): `http://localhost:8000/redoc`
   - OpenAPI JSON: `http://localhost:8000/openapi.json`

3. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   
   Frontend will be available at `http://localhost:5173` (Vite default port)

### Production Mode

See [Deployment](#-deployment) section below.

## 📚 REST API Documentation

### Base URL

All API endpoints are prefixed with `/api`:
- Development: `http://localhost:8000/api`
- Production: `https://yourdomain.com/api`

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

For detailed API documentation with examples, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Data Models

#### MemoryInput
```json
{
  "text": "string (required)",
  "metadata": {
    "key": "value"
  },
  "source": "string (optional)"
}
```

#### Memory
```json
{
  "id": "string (UUID)",
  "content": "string",
  "embedding": [0.1, 0.2, ...],  // 768-dimensional vector
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "version": 1,
  "is_active": true,
  "metadata": {},
  "content_hash": "string"
}
```

#### MemoryRelationship
```json
{
  "from_id": "string",
  "to_id": "string",
  "relation_type": "UPDATE | EXTEND | DERIVE | CHUNK_SEQUENCE",
  "confidence": 0.95,
  "created_at": "2024-01-01T00:00:00Z",
  "reasoning": "string (optional)"
}
```

#### SearchQuery
```json
{
  "query": "string (required)",
  "top_k": 10,  // 1-100
  "include_inactive": false,
  "relation_depth": 2  // 0-5
}
```

#### KnowledgeGraphResponse
```json
{
  "nodes": [
    {
      "id": "string",
      "content": "string (truncated to 100 chars)",
      "version": 1,
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z",
      "metadata": {}
    }
  ],
  "edges": [
    {
      "from_id": "string",
      "to_id": "string",
      "relation_type": "UPDATE | EXTEND | DERIVE | CHUNK_SEQUENCE",
      "confidence": 0.95
    }
  ],
  "total_nodes": 10,
  "total_edges": 15
}
```

### API Endpoints

#### Memory Management

##### Create Memory
```http
POST /api/memories
Content-Type: application/json

{
  "text": "User prefers morning workouts at 6 AM",
  "metadata": {"source": "user_input"},
  "source": "manual"
}
```

**Response**: `200 OK` with `Memory` object

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

##### Upload PDF
```http
POST /api/memories/pdf
Content-Type: multipart/form-data

file: [PDF file]
```

**Response**: `200 OK` with JSON:
```json
{
  "created": 5,
  "skipped": 2,
  "skipped_details": [
    "Chunk 3: exact duplicate",
    "Chunk 7: semantic duplicate (similarity: 96%)"
  ],
  "memories": [...],
  "relationships_created": 8
}
```

**Errors**:
- `400 Bad Request`: Invalid file or no text extracted
- `500 Internal Server Error`: PDF processing error

**Example**:
```bash
curl -X POST "http://localhost:8000/api/memories/pdf" \
  -F "file=@document.pdf"
```

##### Get Memory
```http
GET /api/memories/{memory_id}
```

**Response**: `200 OK` with `Memory` object

**Errors**:
- `404 Not Found`: Memory not found

**Example**:
```bash
curl "http://localhost:8000/api/memories/550e8400-e29b-41d4-a716-446655440000"
```

##### Get Memory Relationships
```http
GET /api/memories/{memory_id}/relationships
```

**Response**: `200 OK` with array of `MemoryRelationship` objects

**Example**:
```bash
curl "http://localhost:8000/api/memories/550e8400-e29b-41d4-a716-446655440000/relationships"
```

#### Search & Discovery

##### Semantic Search
```http
POST /api/search
Content-Type: application/json

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

##### Get Knowledge Graph
```http
GET /api/graph?memory_ids=id1&memory_ids=id2&depth=2&active_only=true
```

**Query Parameters**:
- `memory_ids` (optional, repeatable): Filter to specific memories and their neighbors
- `depth` (default: 2, range: 0-5): Traversal depth for subgraph
- `active_only` (default: true): Include only active memories

**Response**: `200 OK` with `KnowledgeGraphResponse`

**Example**:
```bash
# Get entire graph
curl "http://localhost:8000/api/graph?depth=2&active_only=true"

# Get subgraph around specific memories
curl "http://localhost:8000/api/graph?memory_ids=id1&memory_ids=id2&depth=3"
```

##### Stream Graph Updates (SSE)
```http
GET /api/graph/stream
Accept: text/event-stream
```

**Response**: Server-Sent Events stream with graph update notifications:
```
data: {"type": "memory_created", "data": {...}}

data: {"type": "relationships_added", "data": {...}}

: heartbeat
```

**Example**:
```bash
curl -N "http://localhost:8000/api/graph/stream"
```

#### Statistics & Monitoring

##### Get Platform Statistics
```http
GET /api/stats
```

**Response**: `200 OK` with statistics:
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
  "cached_at": "2024-01-01T00:00:00Z"
}
```

**Performance**: Cached with 5-minute TTL for sub-400ms response times

**Example**:
```bash
curl "http://localhost:8000/api/stats"
```

##### Health Check
```http
GET /health
```

**Response**: `200 OK` with system status:
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

**Example**:
```bash
curl "http://localhost:8000/health"
```

### Error Responses

All endpoints follow consistent error handling:

```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid input or validation error
- `401 Unauthorized`: Authentication required
- `404 Not Found`: Resource not found
- `409 Conflict`: Duplicate content (exact or semantic)
- `422 Unprocessable Entity`: Request validation failed
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Database or service unavailable
- `504 Gateway Timeout`: Request timeout

### Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider adding rate limiting middleware.

## 🎨 Frontend Features

- **Interactive Graph Visualization**: Drag, zoom, and explore the knowledge graph
- **Memory Management**: Add new memories via text input or PDF upload
- **Semantic Search**: Search memories using natural language queries
- **Memory Details**: View detailed information about selected memories
- **Statistics Dashboard**: Monitor total memories, relationships, and active nodes
- **Timeline View**: Visualize memory evolution over time
- **Real-time Updates**: Auto-refresh graph on memory creation

## 🔧 Development

### Backend Development

```bash
cd backend

# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run tests with coverage
pytest --cov=main --cov-report=html
```

**Code Structure**:
- `main.py`: FastAPI application, models, and endpoints
- `tests/`: Comprehensive test suite
  - `test_api_endpoints.py`: Endpoint tests
  - `test_database_operations.py`: Database operation tests
  - `test_model_logic.py`: Business logic tests
  - `test_error_handling.py`: Error handling tests
  - `test_integration.py`: Integration tests

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run tests (if available)
npm test
```

### Testing

#### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run specific test file
pytest tests/test_api_endpoints.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=main --cov-report=html
```

**Test Coverage**:
- ✅ All API endpoints
- ✅ Database operations (with in-memory fallback)
- ✅ Model logic and business rules
- ✅ Error handling and edge cases
- ✅ Integration workflows
- ✅ Cache operations

#### Frontend Tests

```bash
cd frontend
npm test
```

### Code Quality

**Backend**:
- Type hints throughout
- Pydantic models for validation
- Comprehensive error handling
- Async/await for I/O operations

**Frontend**:
- React functional components
- TypeScript (if enabled)
- ESLint configuration

## 📁 Project Structure

```
ai-memory/
├── backend/
│   ├── main.py              # FastAPI application, models, endpoints
│   ├── requirements.txt     # Python dependencies
│   ├── pytest.ini          # Pytest configuration
│   ├── Dockerfile          # Backend Docker image
│   └── tests/              # Backend tests
│       ├── conftest.py     # Pytest fixtures
│       ├── test_api_endpoints.py
│       ├── test_database_operations.py
│       ├── test_model_logic.py
│       ├── test_error_handling.py
│       └── test_integration.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── main.jsx        # React entry point
│   │   └── index.css       # Styles
│   ├── package.json        # Node.js dependencies
│   ├── vite.config.js      # Vite configuration
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   └── Dockerfile          # Frontend Docker image
├── data/
│   ├── neo4j/              # Neo4j data directory (gitignored)
│   ├── neo4j-logs/         # Neo4j logs (gitignored)
│   └── redis/              # Redis data directory (gitignored)
├── docker-compose.yml      # Docker Compose configuration
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🚀 Deployment

### Production Considerations

1. **Environment Variables**: Use secure secret management (AWS Secrets Manager, HashiCorp Vault, etc.)
2. **Database**: 
   - Use managed Neo4j service (Neo4j Aura) or properly secured instance
   - Enable authentication and use strong passwords
   - Configure backup strategy
3. **CORS**: Configure CORS middleware for production domains only
   ```env
   CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
   ```
4. **HTTPS**: Use reverse proxy (nginx) with SSL certificates
5. **Monitoring**: Set up logging and monitoring services
   - Application logs
   - Database performance metrics
   - API response times
6. **Backup**: Regular backups of Neo4j database
7. **Scaling**: Consider horizontal scaling with load balancer

### Docker Deployment

#### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

#### Run Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Production Docker Setup

For production, consider using separate `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_HOST=redis
    depends_on:
      - neo4j
      - redis
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

### Environment-Specific Configuration

1. **Development**: `.env.development`
2. **Staging**: `.env.staging`
3. **Production**: `.env.production` (never commit to git)

### Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
    }
}
```

### Database Backup

```bash
# Neo4j backup (if using Neo4j Enterprise)
neo4j-admin backup --backup-dir=/backups --name=backup-$(date +%Y%m%d)

# Or use Neo4j dump (community edition)
neo4j-admin dump --database=neo4j --to=/backups/backup-$(date +%Y%m%d).dump
```

## 🔐 Security Notes

- **Default Credentials**: Change all default Neo4j credentials in production
- **Environment Variables**: Store sensitive tokens in `.env` file (already gitignored)
- **Authentication**: Enable authentication for Neo4j in production
- **CORS**: Restrict CORS origins to known frontend URLs
- **HTTPS**: Always use HTTPS in production
- **Input Validation**: All inputs are validated via Pydantic models
- **SQL Injection**: Not applicable (using Neo4j, not SQL)
- **Rate Limiting**: Consider adding rate limiting for production

## 🐛 Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Test Neo4j connection
curl http://localhost:7474

# Reset Neo4j (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

**Common Issues**:
- Neo4j not running: Start with `docker-compose up -d`
- Wrong password: Check `NEO4J_PASSWORD` in `.env`
- Connection refused: Check `NEO4J_URI` matches docker-compose port
- Vector index not created: Ensure Neo4j 5.x+ with vector support

### Embedding Model Issues

**Symptoms**: Embeddings are simulated/random instead of real embeddings

**Solutions**:
- Ensure internet connection for model download
- Verify Hugging Face token is set if required
- Check that model terms are accepted: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Check model loading logs in backend output
- Model will automatically fall back to simulated embeddings if loading fails

### Redis Connection Issues

**Symptoms**: Stats endpoint slow, cache not working

**Solutions**:
- Check Redis is running: `docker-compose ps redis`
- Verify Redis connection: `redis-cli ping` (should return `PONG`)
- System will fall back to in-memory cache if Redis unavailable
- Check `REDIS_HOST` and `REDIS_PORT` in `.env`

### Port Conflicts

**Symptoms**: Services won't start, "port already in use"

**Solutions**:
- Change ports in `docker-compose.yml`
- Update `NEO4J_URI` in `.env` if Neo4j port changes
- Update frontend API base URL if backend port changes
- Kill processes using ports:
  ```bash
  # Windows
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  
  # Linux/Mac
  lsof -ti:8000 | xargs kill
  ```

### Performance Issues

**Symptoms**: Slow API responses, timeout errors

**Solutions**:
- Check Neo4j connection and query performance
- Verify Redis cache is working (check logs)
- Monitor database size and index usage
- Check embedding model loading time
- Review batch query optimizations are working

### PDF Processing Issues

**Symptoms**: PDF upload fails or no text extracted

**Solutions**:
- Ensure PDF is not password-protected
- Check PDF file is not corrupted
- Verify `pdfplumber` library is installed
- Check backend logs for extraction errors
- Try a different PDF file

## 📝 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` | When `USE_NEO4J=true` |
| `NEO4J_USER` | Neo4j username | `neo4j` | When `USE_NEO4J=true` |
| `NEO4J_PASSWORD` | Neo4j password | (none) | **YES** when `USE_NEO4J=true` |
| `USE_NEO4J` | Enable Neo4j (false uses in-memory) | `false` | No |
| `REDIS_HOST` | Redis host | `localhost` | No |
| `REDIS_PORT` | Redis port | `6379` | No |
| `REDIS_DB` | Redis database number | `0` | No |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:5173,...` | No |
| `EMBEDDING_MODEL` | Hugging Face model name | `Qwen/Qwen3-Embedding-0.6B` | No |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `768` | No |
| `USE_HF_EMBEDDING_MODEL` | Use Hugging Face model | `true` | No |
| `USE_EMBEDDING_INSTRUCTION` | Use instruction prompts | `false` | No |
| `EMBEDDING_INSTRUCTION` | Instruction type (`query`/`document`) | `query` | No |
| `HF_TOKEN` | Hugging Face authentication token | (none) | Optional |
| `HUGGINGFACE_HUB_TOKEN` | Alternative HF token env var | (none) | Optional |

## 👥 Developer Onboarding

### Getting Started

1. **Clone the repository**
2. **Set up environment**: Follow [Installation](#-installation) steps
3. **Start services**: Run `docker-compose up -d`
4. **Start backend**: Follow [Backend Setup](#2-backend-setup)
5. **Start frontend**: Follow [Frontend Setup](#3-frontend-setup)
6. **Verify**: Check health endpoint and Swagger docs

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Follow code style and conventions
3. **Write tests**: Add tests for new functionality
4. **Run tests**: `pytest` (backend) or `npm test` (frontend)
5. **Check linting**: Ensure code passes linting
6. **Commit changes**: Write clear commit messages
7. **Push and create PR**: Submit for review

### Code Style

**Backend (Python)**:
- Follow PEP 8 style guide
- Use type hints for all functions
- Document complex functions with docstrings
- Use async/await for I/O operations

**Frontend (JavaScript/React)**:
- Use functional components with hooks
- Follow React best practices
- Use meaningful variable names

### Testing Guidelines

- Write tests for all new endpoints
- Test error cases and edge cases
- Aim for >80% code coverage
- Use mocks for external dependencies

### API Documentation

- Document all new endpoints in this README
- Add docstrings to endpoint functions
- Update OpenAPI schema automatically (via FastAPI)
- Include request/response examples

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

---

**Built with** FastAPI • React • Neo4j • Sentence Transformers • Docker • Redis

**Version**: 1.0.0
