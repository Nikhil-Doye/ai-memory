# AI Memory Management Platform

A semantic memory management system that uses graph databases and vector embeddings to store, retrieve, and visualize knowledge relationships. Built with FastAPI backend and React frontend.

## 🚀 Features

- **Semantic Memory Storage**: Store and retrieve memories with vector embeddings using Qwen3-Embedding-0.6B model
- **Knowledge Graph**: Visualize relationships between memories using Neo4j graph database
- **Intelligent Relationships**: Automatically infer relationships between memories:
  - **UPDATE**: Supersedes previous information
  - **EXTEND**: Adds context while keeping original valid
  - **DERIVE**: Inferred insights from patterns
- **Vector Search**: Fast semantic search across memories using Neo4j's native vector search
- **PDF Upload**: Extract and process text from PDF documents
- **Interactive Visualization**: Explore knowledge graph with interactive node graph visualization
- **Version Control**: Track memory versions and manage active/inactive states
- **Real-time Stats**: Monitor platform statistics and memory metrics

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   React     │────▶│   FastAPI    │────▶│   Neo4j     │
│  Frontend   │     │   Backend    │     │   Database  │
└─────────────┘     └──────────────┘     └─────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │   Redis      │
                    │   (Cache)    │
                    └──────────────┘
```

### Components

- **Backend**: FastAPI application handling memory ingestion, vector search, and graph operations
- **Frontend**: React application with interactive knowledge graph visualization
- **Neo4j**: Graph database storing memories as nodes and relationships as edges, with vector index for similarity search
- **Embedding Model**: Qwen3-Embedding-0.6B from Hugging Face (768 dimensions default)

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- (Optional) Hugging Face token for embedding model access

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
NEO4J_PASSWORD=memoryplatform2024
USE_NEO4J=true

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

**Note**: For the embedding model, you may need to:
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Accept model terms at https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
3. Add the token to your `.env` file

## 🐳 Docker Setup (Recommended)

The easiest way to run the entire stack:

```bash
# Start Neo4j and Redis services
docker-compose up -d

# Check services are running
docker-compose ps

# View Neo4j logs
docker-compose logs neo4j

# View Redis logs
docker-compose logs redis
```

This will start:
- **Neo4j** on `http://localhost:7474` (Web UI) and `bolt://localhost:7687` (Bolt protocol)
- **Redis** on `localhost:6379`

## 🚀 Running the Application

### Start Services

1. **Start Docker services** (if using Docker):
   ```bash
   docker-compose up -d
   ```

2. **Start Backend**:
   ```bash
   cd backend
   python main.py
   # Or use uvicorn directly:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Backend will be available at `http://localhost:8000`
   API docs at `http://localhost:8000/docs`

3. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will be available at `http://localhost:5173` (Vite default port)

## 📚 API Endpoints

### Memories

- `POST /api/memories` - Create a new memory from text
- `POST /api/memories/pdf` - Upload and process PDF file
- `GET /api/memories/{memory_id}` - Get specific memory
- `GET /api/memories/{memory_id}/relationships` - Get memory relationships

### Search & Graph

- `POST /api/search` - Semantic search across memories
- `GET /api/graph` - Get knowledge graph (with optional memory_ids and depth)
- `GET /api/stats` - Get platform statistics

### Health

- `GET /health` - Health check with system status

### Example API Usage

```bash
# Create a memory
curl -X POST "http://localhost:8000/api/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers morning workouts at 6 AM",
    "metadata": {"source": "user_input"},
    "source": "manual"
  }'

# Search memories
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "morning exercise routine",
    "top_k": 10,
    "include_inactive": false,
    "relation_depth": 2
  }'

# Get knowledge graph
curl "http://localhost:8000/api/graph?depth=2&active_only=true"
```

## 🎨 Frontend Features

- **Interactive Graph Visualization**: Drag, zoom, and explore the knowledge graph
- **Memory Management**: Add new memories via text input or PDF upload
- **Semantic Search**: Search memories using natural language queries
- **Memory Details**: View detailed information about selected memories
- **Statistics Dashboard**: Monitor total memories, relationships, and active nodes
- **Timeline View**: Visualize memory evolution over time

## 🔧 Development

### Backend Development

```bash
cd backend

# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

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
```

### Testing

```bash
# Backend tests (if available)
cd backend
pytest

# Frontend tests (if available)
cd frontend
npm test
```

## 📁 Project Structure

```
ai-memory/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Backend Docker image
│   └── tests/              # Backend tests
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

## 🔐 Security Notes

- Default Neo4j credentials are for development only. Change them in production.
- Store sensitive tokens in `.env` file (already gitignored).
- Enable authentication and use strong passwords in production environments.
- Consider using environment-specific configuration files.

## 🐛 Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Reset Neo4j (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Embedding Model Issues

- Ensure you have internet connection for model download
- Verify Hugging Face token is set if required
- Check that model terms are accepted on Hugging Face
- Model will fall back to simulated embeddings if loading fails

### Port Conflicts

- Change ports in `docker-compose.yml` if default ports are in use
- Update `NEO4J_URI` in `.env` if Neo4j port changes
- Update frontend API base URL if backend port changes

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `memoryplatform2024` |
| `USE_NEO4J` | Enable Neo4j (false uses in-memory) | `false` |
| `EMBEDDING_MODEL` | Hugging Face model name | `Qwen/Qwen3-Embedding-0.6B` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `768` |
| `USE_HF_EMBEDDING_MODEL` | Use Hugging Face model | `true` |
| `HF_TOKEN` | Hugging Face authentication token | (none) |

## 🚀 Deployment

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **Database**: Use managed Neo4j service or properly secured instance
3. **CORS**: Configure CORS middleware for production domains
4. **HTTPS**: Use reverse proxy (nginx) with SSL certificates
5. **Monitoring**: Set up logging and monitoring services
6. **Backup**: Regular backups of Neo4j database

### Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

---

**Built with** FastAPI • React • Neo4j • Sentence Transformers • Docker

