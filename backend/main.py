"""
AI Memory Management Platform - FastAPI Backend
Handles semantic memory ingestion, graph relationships, and vector search
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime
from enum import Enum
import asyncio
import hashlib
import uuid
import os
import json
import io
import traceback
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, TransientError
import pdfplumber

# Redis for caching (optional, with in-memory fallback)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        print("⚠️  Redis not available. Stats caching will use in-memory fallback.")

# Load environment variables from .env file
# Check both backend/.env and root/.env
env_path = os.path.join(os.path.dirname(__file__), '.env')
root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
elif os.path.exists(root_env_path):
    load_dotenv(root_env_path)
else:
    load_dotenv()  # Try default location

# ============= Error Handling Utilities =============

def classify_exception(exception: Exception) -> Tuple[int, str]:
    """
    Classify exceptions and return appropriate HTTP status code and message.
    Returns (status_code, error_message) tuple.
    """
    error_type = type(exception).__name__
    error_msg = str(exception)
    
    # Neo4j/Database errors
    if isinstance(exception, (ServiceUnavailable,)):
        return (503, f"Database service unavailable: {error_msg}. Please check if Neo4j is running.")
    
    if isinstance(exception, (AuthError,)):
        return (401, f"Database authentication failed: {error_msg}. Check your Neo4j credentials.")
    
    if isinstance(exception, (TransientError,)):
        return (503, f"Database transient error: {error_msg}. Please try again later.")
    
    # Validation errors
    if isinstance(exception, ValidationError):
        return (422, f"Validation error: {error_msg}")
    
    if isinstance(exception, ValueError):
        # Check for specific value errors
        if "NEO4J_PASSWORD" in error_msg:
            return (500, error_msg)  # Already has good message
        return (400, f"Invalid input: {error_msg}")
    
    if isinstance(exception, (TypeError, AttributeError)):
        return (400, f"Invalid request format: {error_msg}")
    
    # File/Upload errors
    if isinstance(exception, (FileNotFoundError, IOError, OSError)):
        return (404, f"File operation failed: {error_msg}")
    
    # Embedding/Model errors (can be identified by error message patterns)
    error_lower = error_msg.lower()
    if "embedding" in error_lower or "model" in error_lower or "huggingface" in error_lower:
        if "token" in error_lower or "auth" in error_lower or "401" in error_msg:
            return (401, f"Model authentication failed: {error_msg}. Check your Hugging Face token.")
        if "connection" in error_lower or "network" in error_lower:
            return (503, f"Model service unavailable: {error_msg}. Check your internet connection.")
        return (500, f"Embedding/model error: {error_msg}")
    
    # PDF processing errors
    if "pdf" in error_lower or "pdfplumber" in error_lower:
        if "corrupted" in error_lower or "invalid" in error_lower:
            return (400, f"Invalid PDF file: {error_msg}")
        return (422, f"PDF processing error: {error_msg}")
    
    # JSON/Serialization errors
    if isinstance(exception, (json.JSONDecodeError, UnicodeDecodeError)):
        return (400, f"Data format error: {error_msg}")
    
    # Timeout errors
    if isinstance(exception, asyncio.TimeoutError) or "timeout" in error_lower:
        return (504, f"Request timeout: {error_msg}")
    
    # Permission errors
    if isinstance(exception, PermissionError):
        return (403, f"Permission denied: {error_msg}")
    
    # Memory/Resource errors
    if isinstance(exception, (MemoryError,)):
        return (507, f"Insufficient resources: {error_msg}")
    
    # Default: Internal server error
    return (500, f"Internal server error: {error_msg}")

def handle_endpoint_error(exception: Exception, context: str = "operation") -> HTTPException:
    """
    Handle exceptions in endpoints by classifying them and returning appropriate HTTPException.
    Provides better error messages and status codes for UI/UX.
    """
    status_code, error_message = classify_exception(exception)
    
    # Log full traceback for debugging (server-side only)
    if status_code >= 500:
        print(f"⚠️  Error in {context}: {type(exception).__name__}: {error_message}")
        traceback.print_exc()
    
    return HTTPException(status_code=status_code, detail=error_message)

# ============= Real-Time Graph Updates =============

class GraphUpdateNotifier:
    """
    Manages real-time notifications for knowledge graph changes.
    Uses asyncio queues to broadcast updates to connected clients via SSE.
    """
    def __init__(self):
        self.subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
    
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to graph updates. Returns a queue that will receive update events."""
        queue = asyncio.Queue()
        async with self._lock:
            self.subscribers.add(queue)
        print(f"📡 Graph update subscriber connected. Total subscribers: {len(self.subscribers)}")
        return queue
    
    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from graph updates."""
        async with self._lock:
            self.subscribers.discard(queue)
        print(f"📡 Graph update subscriber disconnected. Total subscribers: {len(self.subscribers)}")
    
    async def notify_update(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast a graph update to all subscribers.
        
        Args:
            event_type: Type of event (e.g., 'memory_created', 'memory_updated', 'relationship_added', 'graph_updated')
            data: Event data payload
        """
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Send to all subscribers (non-blocking)
        async with self._lock:
            dead_queues = set()
            for queue in self.subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Queue full, mark for removal
                    dead_queues.add(queue)
                except Exception as e:
                    # Queue error, mark for removal
                    print(f"⚠️  Error sending update to subscriber: {e}")
                    dead_queues.add(queue)
            
            # Clean up dead queues
            self.subscribers -= dead_queues
        
        if len(self.subscribers) > 0:
            print(f"📢 Broadcasted {event_type} to {len(self.subscribers)} subscriber(s)")

# Global notifier instance
graph_notifier = GraphUpdateNotifier()

# ============= Stats Caching =============

class StatsCache:
    """
    Caches platform statistics for fast retrieval.
    Supports Redis (preferred) with in-memory fallback.
    Automatically invalidates when data changes.
    """
    def __init__(self):
        self.redis_client = None
        self.use_redis = False
        self.in_memory_cache: Optional[Dict[str, Any]] = None
        self.cache_key = "platform:stats"
        self._lock = asyncio.Lock()
        
        # Initialize Redis connection if available
        if REDIS_AVAILABLE:
            try:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                
                # Try async Redis
                try:
                    # redis.asyncio.Redis for async operations
                    if hasattr(redis, 'Redis'):
                        # Check if it's async redis module
                        self.redis_client = redis.Redis(
                            host=redis_host,
                            port=redis_port,
                            db=redis_db,
                            decode_responses=True,
                            socket_connect_timeout=2
                        )
                        # Test connection
                        try:
                            # Try a simple ping to verify connection works
                            import asyncio
                            # We'll do an async ping in get_stats if needed
                            self.use_redis = True
                            print(f"✅ Redis client initialized for stats caching ({redis_host}:{redis_port})")
                        except Exception as e:
                            print(f"⚠️  Could not connect to Redis: {e}. Using in-memory cache.")
                            self.use_redis = False
                    else:
                        self.use_redis = False
                except Exception as e:
                    print(f"⚠️  Could not initialize Redis: {e}. Using in-memory cache.")
                    self.use_redis = False
            except Exception as e:
                print(f"⚠️  Redis initialization failed: {e}. Using in-memory cache.")
                self.use_redis = False
    
    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get cached stats if available"""
        async with self._lock:
            if self.use_redis and self.redis_client:
                try:
                    # Handle both async and sync Redis clients
                    if hasattr(self.redis_client, 'get') and asyncio.iscoroutinefunction(self.redis_client.get):
                        cached = await self.redis_client.get(self.cache_key)
                    else:
                        # Sync client - run in executor
                        cached = await asyncio.to_thread(self.redis_client.get, self.cache_key)
                    
                    if cached:
                        return json.loads(cached)
                except Exception as e:
                    print(f"⚠️  Redis get error: {e}. Falling back to in-memory.")
                    self.use_redis = False
            
            # Fallback to in-memory cache
            return self.in_memory_cache
    
    async def set_stats(self, stats: Dict[str, Any], ttl: int = 300):
        """
        Cache stats data with optional TTL (time-to-live in seconds).
        Default TTL is 5 minutes (300s).
        """
        async with self._lock:
            stats_json = json.dumps(stats)
            
            if self.use_redis and self.redis_client:
                try:
                    # Handle both async and sync Redis clients
                    if hasattr(self.redis_client, 'setex') and asyncio.iscoroutinefunction(self.redis_client.setex):
                        await self.redis_client.setex(self.cache_key, ttl, stats_json)
                    else:
                        # Sync client - run in executor
                        await asyncio.to_thread(self.redis_client.setex, self.cache_key, ttl, stats_json)
                    return
                except Exception as e:
                    print(f"⚠️  Redis set error: {e}. Falling back to in-memory.")
                    self.use_redis = False
            
            # Fallback to in-memory cache (no TTL for in-memory)
            self.in_memory_cache = stats
    
    async def invalidate(self):
        """Invalidate cached stats (call when data changes)"""
        async with self._lock:
            if self.use_redis and self.redis_client:
                try:
                    # Handle both async and sync Redis clients
                    if hasattr(self.redis_client, 'delete') and asyncio.iscoroutinefunction(self.redis_client.delete):
                        await self.redis_client.delete(self.cache_key)
                    else:
                        # Sync client - run in executor
                        await asyncio.to_thread(self.redis_client.delete, self.cache_key)
                except Exception as e:
                    print(f"⚠️  Redis delete error: {e}")
            
            self.in_memory_cache = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception:
                pass

# Global stats cache instance
stats_cache = StatsCache()

app = FastAPI(title="AI Memory Platform", version="1.0.0")

# CORS for frontend access
# Read allowed origins from environment variable, with safe defaults for development
# Set CORS_ORIGINS environment variable to comma-separated list of allowed origins
# Example: CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://yourdomain.com
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    # Split by comma and strip whitespace
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    # Default to localhost development ports for safety
    allowed_origins = [
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",   # Common React port
        "http://127.0.0.1:5173",   # Alternative localhost
        "http://127.0.0.1:3000",   # Alternative localhost
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Models =============

class RelationType(str, Enum):
    UPDATE = "UPDATE"  # Supersedes previous information
    EXTEND = "EXTEND"  # Adds context, keeps original valid
    DERIVE = "DERIVE"  # Inferred insight from patterns
    CHUNK_SEQUENCE = "CHUNK_SEQUENCE"  # Sequential chunks from the same document/source

class MemoryInput(BaseModel):
    text: str = Field(..., description="Text content to memorize")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source: Optional[str] = Field(None, description="Source identifier")

class Memory(BaseModel):
    id: str
    content: str
    embedding: List[float]
    created_at: datetime
    updated_at: datetime
    version: int
    is_active: bool
    metadata: Dict[str, Any]
    content_hash: str

class MemoryRelationship(BaseModel):
    from_id: str
    to_id: str
    relation_type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    reasoning: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    include_inactive: bool = False
    relation_depth: int = Field(default=2, ge=0, le=5)

class GraphNode(BaseModel):
    id: str
    content: str
    version: int
    is_active: bool
    created_at: datetime
    metadata: Dict[str, Any]

class GraphEdge(BaseModel):
    from_id: str
    to_id: str
    relation_type: RelationType
    confidence: float

class KnowledgeGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_nodes: int
    total_edges: int

# ============= Neo4j Storage (Graph + Vector Search) =============

class MemoryStore:
    """
    Neo4j-based store for both graph relationships and vector embeddings.
    Uses Neo4j's native vector search capabilities (Neo4j 5.x+).
    """
    def __init__(self):
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"
        
        # Password must be provided via environment variable - never hardcode credentials
        # Only require password if Neo4j is enabled
        if self.use_neo4j:
            self.neo4j_password = os.getenv("NEO4J_PASSWORD")
            if not self.neo4j_password:
                raise ValueError(
                    "NEO4J_PASSWORD environment variable is required when USE_NEO4J=true. "
                    "Please set NEO4J_PASSWORD in your .env file or environment variables. "
                    "Never commit passwords to source code."
                )
        else:
            # Not needed if Neo4j is disabled
            self.neo4j_password = None
        
        self.driver = None
        
        # Fallback in-memory storage
        self.memories: Dict[str, Memory] = {}
        self.relationships: List[MemoryRelationship] = []
        self.content_index: Dict[str, List[str]] = {}  # hash -> memory_ids
    
        if self.use_neo4j:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                print(f"📡 Neo4j driver created (URI: {self.neo4j_uri})")
                print("   Connection will be verified during startup...")
            except Exception as e:
                print(f"⚠️  Warning: Failed to create Neo4j driver: {e}")
                print("⚠️  Falling back to in-memory storage")
                self.use_neo4j = False
                self.driver = None
    
    async def verify_connection(self) -> bool:
        """Verify that Neo4j connection actually works"""
        if not self.driver:
            return False
        
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            print("✅ Neo4j connection verified successfully")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Neo4j connection verification failed: {e}")
            print(f"   URI: {self.neo4j_uri}")
            print("   Make sure Neo4j is running and accessible.")
            print("   If using Docker, run: docker-compose up -d")
            print("   Or start Neo4j manually and ensure it's listening on the configured port.")
            return False
    
    async def initialize(self):
        """Initialize vector index after startup"""
        if self.use_neo4j and self.driver:
            # First verify the connection actually works
            if not await self.verify_connection():
                print("⚠️  Disabling Neo4j usage due to connection failure")
                print("⚠️  Falling back to in-memory storage")
                self.use_neo4j = False
                if self.driver:
                    try:
                        await self.driver.close()
                    except Exception:
                        pass
                    self.driver = None
                return
            
            # Connection verified, now initialize vector index
            await self._initialize_vector_index()
    
    async def _initialize_vector_index(self):
        """Initialize vector index for embeddings if not exists"""
        if not self.driver:
            return
        
        try:
            async with self.driver.session() as session:
                # Check if vector index exists, create if not
                embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                # Neo4j 5.x vector index syntax
                query = """
                CREATE VECTOR INDEX memory_embedding_idx IF NOT EXISTS
                FOR (m:Memory) ON m.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                await session.run(query, dimensions=embedding_dim)
                print(f"✅ Vector index initialized (dimension: {embedding_dim})")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Warning: Could not create vector index: {error_msg}")
            print("⚠️  Vector search may not work. Ensure Neo4j 5.x+ with vector support.")
            
            # Provide more specific guidance based on error type
            if "Connection" in error_msg or "connect" in error_msg.lower():
                print("   Connection error - verify Neo4j is running and accessible.")
            elif "vector" in error_msg.lower() or "index" in error_msg.lower():
                print("   Vector index error - ensure Neo4j version 5.x or higher with vector support.")
                print("   Neo4j 5.11+ includes native vector search capabilities.")
    
    async def add_memory(self, memory: Memory) -> Memory:
        """Add memory to Neo4j or in-memory store"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    query = """
                    MERGE (m:Memory {id: $id})
                    SET m.content = $content,
                        m.embedding = $embedding,
                        m.created_at = $created_at,
                        m.updated_at = $updated_at,
                        m.version = $version,
                        m.is_active = $is_active,
                        m.metadata = $metadata,
                        m.content_hash = $content_hash
                    RETURN m
                    """
                    await session.run(
                        query,
                        id=memory.id,
                        content=memory.content,
                        embedding=memory.embedding,
                        created_at=memory.created_at.isoformat(),
                        updated_at=memory.updated_at.isoformat(),
                        version=memory.version,
                        is_active=memory.is_active,
                        metadata=json.dumps(memory.metadata),
                        content_hash=memory.content_hash
                    )
            except Exception as e:
                print(f"Warning: Failed to add memory to Neo4j: {e}. Using in-memory.")
                self.memories[memory.id] = memory
                if memory.content_hash not in self.content_index:
                    self.content_index[memory.content_hash] = []
                self.content_index[memory.content_hash].append(memory.id)
        else:
            # In-memory fallback
            self.memories[memory.id] = memory
        if memory.content_hash not in self.content_index:
            self.content_index[memory.content_hash] = []
        self.content_index[memory.content_hash].append(memory.id)
        
        return memory
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory from Neo4j or in-memory store"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    query = """
                    MATCH (m:Memory {id: $id})
                    RETURN m
                    """
                    result = await session.run(query, id=memory_id)
                    record = await result.single()
                    if record:
                        m = record["m"]
                        return Memory(
                            id=m["id"],
                            content=m["content"],
                            embedding=m.get("embedding", []),
                            created_at=datetime.fromisoformat(m["created_at"]),
                            updated_at=datetime.fromisoformat(m["updated_at"]),
                            version=m["version"],
                            is_active=m["is_active"],
                            metadata=json.loads(m.get("metadata", "{}")),
                            content_hash=m["content_hash"]
                        )
            except Exception as e:
                print(f"Warning: Failed to get memory from Neo4j: {e}")
        
        return self.memories.get(memory_id)
    
    async def add_relationship(self, rel: MemoryRelationship):
        """Add relationship to Neo4j or in-memory store"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    # Create relationship
                    query = """
                    MATCH (from:Memory {id: $from_id})
                    MATCH (to:Memory {id: $to_id})
                    MERGE (from)-[r:RELATES_TO {
                        type: $relation_type,
                        confidence: $confidence,
                        created_at: $created_at,
                        reasoning: $reasoning
                    }]->(to)
                    """
                    await session.run(
                        query,
                        from_id=rel.from_id,
                        to_id=rel.to_id,
                        relation_type=rel.relation_type.value,
                        confidence=rel.confidence,
                        created_at=rel.created_at.isoformat(),
                        reasoning=rel.reasoning or ""
                    )
        
        # If UPDATE relationship, mark old memory as inactive
                    if rel.relation_type == RelationType.UPDATE:
                        update_query = """
                        MATCH (m:Memory {id: $from_id})
                        SET m.is_active = false
                        """
                        await session.run(update_query, from_id=rel.from_id)
            except Exception as e:
                print(f"Warning: Failed to add relationship to Neo4j: {e}. Using in-memory.")
                self.relationships.append(rel)
                if rel.relation_type == RelationType.UPDATE:
                    old_memory = self.memories.get(rel.from_id)
                    if old_memory:
                        old_memory.is_active = False
        else:
            # In-memory fallback
            self.relationships.append(rel)
        if rel.relation_type == RelationType.UPDATE:
            old_memory = self.memories.get(rel.from_id)
            if old_memory:
                old_memory.is_active = False
    
    async def find_similar_by_hash(self, content_hash: str) -> List[str]:
        """Find memories with same content hash"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    query = """
                    MATCH (m:Memory {content_hash: $content_hash})
                    RETURN m.id as id
                    """
                    result = await session.run(query, content_hash=content_hash)
                    return [record["id"] async for record in result]
            except Exception as e:
                print(f"Warning: Failed to search Neo4j: {e}")
        
        return self.content_index.get(content_hash, [])
    
    async def get_relationships(self, memory_id: str, 
                         relation_types: Optional[List[RelationType]] = None) -> List[MemoryRelationship]:
        """Get relationships for a memory"""
        # Use batch method for single ID (more efficient)
        results = await self.get_relationships_batch([memory_id], relation_types)
        return results.get(memory_id, [])
    
    async def get_relationships_batch(self, memory_ids: List[str],
                                    relation_types: Optional[List[RelationType]] = None) -> Dict[str, List[MemoryRelationship]]:
        """
        Batch fetch relationships for multiple memories in a single query.
        Returns a dictionary mapping memory_id -> list of relationships.
        This prevents N+1 query problems.
        """
        if not memory_ids:
            return {}
        
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    rel_filter = ""
                    if relation_types:
                        rel_types = [rt.value for rt in relation_types]
                        rel_filter = "WHERE r.type IN $relation_types"
                    
                    # Efficient batch query using WHERE ... IN clause
                    # Leverages Neo4j's relationship traversal for optimal performance
                    # Uses UNION to handle both directions of relationships
                    if rel_filter:
                        # When relation_types filter is present
                        query = f"""
                        MATCH (m:Memory)-[r:RELATES_TO]-(related:Memory)
                        WHERE m.id IN $memory_ids AND r.type IN $relation_types
                        RETURN 
                            m.id as memory_id,
                            startNode(r).id as from_id,
                            endNode(r).id as to_id,
                            r.type as relation_type, 
                            r.confidence as confidence,
                            r.created_at as created_at,
                            r.reasoning as reasoning
                        """
                    else:
                        query = """
                        MATCH (m:Memory)-[r:RELATES_TO]-(related:Memory)
                        WHERE m.id IN $memory_ids
                        RETURN 
                            m.id as memory_id,
                            startNode(r).id as from_id,
                            endNode(r).id as to_id,
                            r.type as relation_type, 
                            r.confidence as confidence,
                            r.created_at as created_at,
                            r.reasoning as reasoning
                        """
                    
                    params = {"memory_ids": memory_ids}
                    if relation_types:
                        params["relation_types"] = rel_types
                    
                    result = await session.run(query, **params)
                    
                    # Group relationships by memory_id
                    relationships_by_id: Dict[str, List[MemoryRelationship]] = {mid: [] for mid in memory_ids}
                    
                    async for record in result:
                        memory_id = record["memory_id"]
                        from_id = record["from_id"]
                        to_id = record["to_id"]
                        
                        rel = MemoryRelationship(
                            from_id=from_id,
                            to_id=to_id,
                            relation_type=RelationType(record["relation_type"]),
                            confidence=record["confidence"],
                            created_at=datetime.fromisoformat(record["created_at"]),
                            reasoning=record.get("reasoning")
                        )
                        relationships_by_id[memory_id].append(rel)
                    
                    return relationships_by_id
            except Exception as e:
                print(f"Warning: Failed to get relationships from Neo4j: {e}")
        
        # In-memory fallback - batch operation
        relationships_by_id: Dict[str, List[MemoryRelationship]] = {mid: [] for mid in memory_ids}
        
        for rel in self.relationships:
            if rel.from_id in memory_ids:
                relationships_by_id[rel.from_id].append(rel)
            if rel.to_id in memory_ids:
                relationships_by_id[rel.to_id].append(rel)
        
        # Filter by relation types if specified
        if relation_types:
            for memory_id in relationships_by_id:
                relationships_by_id[memory_id] = [
                    r for r in relationships_by_id[memory_id] 
                    if r.relation_type in relation_types
                ]
        
        return relationships_by_id
    
    async def vector_search(self, query_embedding: List[float], top_k: int = 10, 
                           include_inactive: bool = False) -> List[Memory]:
        """Vector similarity search using Neo4j"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    # Neo4j 5.x vector search syntax
                    # Query more results to account for filtering
                    query_k = top_k * 3 if not include_inactive else top_k
                    query = """
                    CALL db.index.vector.queryNodes('memory_embedding_idx', $query_k, $query_embedding)
                    YIELD node as m, score
                    WHERE m:Memory
                    """
                    if not include_inactive:
                        query += "AND m.is_active = true\n"
                    query += """
                    RETURN m, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
                    result = await session.run(
                        query,
                        query_embedding=query_embedding,
                        query_k=query_k,
                        top_k=top_k
                    )
                    
                    memories = []
                    async for record in result:
                        m = record["m"]
                        memory = Memory(
                            id=m["id"],
                            content=m["content"],
                            embedding=m.get("embedding", []),
                            created_at=datetime.fromisoformat(m["created_at"]),
                            updated_at=datetime.fromisoformat(m["updated_at"]),
                            version=m["version"],
                            is_active=m["is_active"],
                            metadata=json.loads(m.get("metadata", "{}")),
                            content_hash=m["content_hash"]
                        )
                        memories.append(memory)
                        if len(memories) >= top_k:
                            break
                    return memories
            except Exception as e:
                print(f"Warning: Vector search in Neo4j failed: {e}. Falling back to in-memory.")
        
        # In-memory fallback: linear search
        candidates = [m for m in self.memories.values() 
                     if include_inactive or m.is_active]
        scored_memories = []
        for memory in candidates:
            similarity = embedding_service.cosine_similarity(query_embedding, memory.embedding)
            scored_memories.append((similarity, memory))
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_memories[:top_k]]
    
    async def get_subgraph(self, memory_ids: List[str], depth: int = 2) -> tuple[List[Memory], List[MemoryRelationship]]:
        """Get connected subgraph using Neo4j or in-memory"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    query = """
                    MATCH path = (start:Memory)-[*1..$depth]-(related:Memory)
                    WHERE start.id IN $memory_ids
                    UNWIND nodes(path) as n
                    UNWIND relationships(path) as r
                    RETURN DISTINCT n as node, collect(DISTINCT r) as rels
                    """
                    result = await session.run(query, memory_ids=memory_ids, depth=depth)
                    
                    nodes_dict = {}
                    edges = []
                    async for record in result:
                        m = record["node"]
                        memory_id = m["id"]
                        if memory_id not in nodes_dict:
                            nodes_dict[memory_id] = Memory(
                                id=m["id"],
                                content=m["content"],
                                embedding=m.get("embedding", []),
                                created_at=datetime.fromisoformat(m["created_at"]),
                                updated_at=datetime.fromisoformat(m["updated_at"]),
                                version=m["version"],
                                is_active=m["is_active"],
                                metadata=json.loads(m.get("metadata", "{}")),
                                content_hash=m["content_hash"]
                            )
                        
                        for rel in record["rels"]:
                            if rel:
                                edges.append(MemoryRelationship(
                                    from_id=rel.start_node["id"],
                                    to_id=rel.end_node["id"],
                                    relation_type=RelationType(rel["type"]),
                                    confidence=rel["confidence"],
                                    created_at=datetime.fromisoformat(rel["created_at"]),
                                    reasoning=rel.get("reasoning")
                                ))
                    
                    return list(nodes_dict.values()), edges
            except Exception as e:
                print(f"Warning: Failed to get subgraph from Neo4j: {e}. Using in-memory.")
        
        # In-memory fallback - optimized with batch relationship fetching
        visited = set(memory_ids)
        queue = [(mid, 0) for mid in memory_ids]
        edges = []
        all_visited_ids = set(memory_ids)
        
        while queue:
            current_batch = []
            current_depth = queue[0][1] if queue else depth
            
            # Collect all IDs at current depth for batch processing
            while queue and queue[0][1] == current_depth:
                current_id, depth_level = queue.pop(0)
                if depth_level < depth:
                    current_batch.append(current_id)
            
            if not current_batch:
                break
            
            # Batch fetch relationships for all nodes at current depth
            relationships_by_id = await self.get_relationships_batch(current_batch)
            
            for current_id in current_batch:
                rels = relationships_by_id.get(current_id, [])
                for rel in rels:
                    # Check if edge already added (deduplicate)
                    edge_key = (rel.from_id, rel.to_id, rel.relation_type)
                    if not any(e.from_id == edge_key[0] and e.to_id == edge_key[1] and e.relation_type == edge_key[2] for e in edges):
                        edges.append(rel)
                    
                    # Determine next node ID
                    next_id = rel.to_id if rel.from_id == current_id else rel.from_id
                    if next_id not in visited and current_depth + 1 < depth:
                        visited.add(next_id)
                        all_visited_ids.add(next_id)
                        queue.append((next_id, current_depth + 1))
        
        # Batch fetch all visited memories
        nodes = []
        visited_list = list(all_visited_ids)
        if visited_list:
            for mid in visited_list:
                memory = await self.get_memory(mid)
                if memory:
                    nodes.append(memory)
        
        return nodes, edges
    
    async def get_all_memories(self, include_inactive: bool = False) -> List[Memory]:
        """Get all memories (for in-memory fallback compatibility)"""
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    filter_clause = "" if include_inactive else "WHERE m.is_active = true"
                    query = f"""
                    MATCH (m:Memory)
                    {filter_clause}
                    RETURN m
                    """
                    result = await session.run(query)
                    memories = []
                    async for record in result:
                        m = record["m"]
                        memories.append(Memory(
                            id=m["id"],
                            content=m["content"],
                            embedding=m.get("embedding", []),
                            created_at=datetime.fromisoformat(m["created_at"]),
                            updated_at=datetime.fromisoformat(m["updated_at"]),
                            version=m["version"],
                            is_active=m["is_active"],
                            metadata=json.loads(m.get("metadata", "{}")),
                            content_hash=m["content_hash"]
                        ))
                    return memories
            except Exception as e:
                print(f"Warning: Failed to get memories from Neo4j: {e}")
        
        return [m for m in self.memories.values() 
                if include_inactive or m.is_active]
    
    async def get_all_relationships(self, active_only: bool = True) -> List[MemoryRelationship]:
        """
        Efficiently fetch all relationships in the graph in a single query.
        Uses Neo4j's relationship traversal for optimal performance.
        Prevents N+1 queries by fetching all relationships at once.
        """
        if self.use_neo4j and self.driver:
            try:
                async with self.driver.session() as session:
                    # Single optimized query to get all relationships
                    # Use explicit direction matching to avoid duplicates
                    if active_only:
                        query = """
                        MATCH (m:Memory)-[r:RELATES_TO]-(related:Memory)
                        WHERE m.is_active = true AND related.is_active = true
                        RETURN DISTINCT
                            startNode(r).id as from_id,
                            endNode(r).id as to_id,
                            r.type as relation_type, 
                            r.confidence as confidence,
                            r.created_at as created_at,
                            r.reasoning as reasoning
                        """
                    else:
                        query = """
                        MATCH (m:Memory)-[r:RELATES_TO]-(related:Memory)
                        RETURN DISTINCT
                            startNode(r).id as from_id,
                            endNode(r).id as to_id,
                            r.type as relation_type, 
                            r.confidence as confidence,
                            r.created_at as created_at,
                            r.reasoning as reasoning
                        """
                    
                    result = await session.run(query)
                    
                    relationships = []
                    seen_edges = set()
                    async for record in result:
                        from_id = record["from_id"]
                        to_id = record["to_id"]
                        edge_key = (from_id, to_id, record["relation_type"])
                        
                        # Deduplicate in case of bidirectional traversal
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            relationships.append(MemoryRelationship(
                                from_id=from_id,
                                to_id=to_id,
                                relation_type=RelationType(record["relation_type"]),
                                confidence=record["confidence"],
                                created_at=datetime.fromisoformat(record["created_at"]),
                                reasoning=record.get("reasoning")
                            ))
                    
                    return relationships
            except Exception as e:
                print(f"Warning: Failed to get all relationships from Neo4j: {e}")
        
        # In-memory fallback
        rels = self.relationships
        if active_only:
            # Filter to only relationships between active memories
            active_ids = {m.id for m in self.memories.values() if m.is_active}
            rels = [r for r in rels if r.from_id in active_ids and r.to_id in active_ids]
        
        # Deduplicate
        seen = set()
        unique_rels = []
        for rel in rels:
            edge_key = (rel.from_id, rel.to_id, rel.relation_type)
            if edge_key not in seen:
                seen.add(edge_key)
                unique_rels.append(rel)
        
        return unique_rels

memory_store = MemoryStore()

# Initialize Neo4j vector index on startup
@app.on_event("startup")
async def startup_event():
    await memory_store.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    await stats_cache.close()

# ============= Embedding Service =============

class EmbeddingService:
    """
    Hugging Face embeddings service using Qwen3-Embedding-0.6B model.
    Model loaded from: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    Falls back to simulated embeddings if model fails to load.
    """
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # Default 768, supports 32-1024
        self.use_instruction = os.getenv("USE_EMBEDDING_INSTRUCTION", "false").lower() == "true"
        self.instruction_prompt = os.getenv("EMBEDDING_INSTRUCTION", "query")  # "query" for queries, "document" for documents
        self.model = None
        self.use_model = os.getenv("USE_HF_EMBEDDING_MODEL", "true").lower() == "true"
        
        if self.use_model:
            try:
                # Get Hugging Face token from environment (if provided)
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
                
                # Set token in environment if provided (sentence-transformers will pick it up automatically)
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                    print(f"Loading Qwen3-Embedding-0.6B model from Hugging Face (with authentication)...")
                else:
                    print(f"Loading Qwen3-Embedding-0.6B model from Hugging Face...")
                
                # Standard usage as per Hugging Face documentation
                # SentenceTransformer automatically uses HF_TOKEN from environment if set
                self.model = SentenceTransformer(self.model_name)
                print(f"✅ Model loaded successfully: {self.model_name}")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print(f"⚠️  Authentication error: Model requires Hugging Face token")
                    print("⚠️  Please set HF_TOKEN environment variable with your Hugging Face token")
                    print("⚠️  Get your token at: https://huggingface.co/settings/tokens")
                    print("⚠️  Then add to .env: HF_TOKEN=your_token_here")
                    print("⚠️  Note: You may also need to accept model terms at: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B")
                else:
                    print(f"⚠️  Warning: Failed to load embedding model: {e}")
                print("⚠️  Falling back to simulated embeddings")
                self.model = None
                self.use_model = False
    
    async def generate_embedding(self, text: str, prompt_name: Optional[str] = None) -> List[float]:
        """
        Generate embeddings using Qwen3-Embedding-0.6B model from Hugging Face.
        Falls back to simulated embeddings if model is not available.
        
        Qwen3-Embedding-0.6B supports:
        - Embedding dimensions: 32 to 1,024 (default: 768)
        - Context length: 32,768 tokens
        - Multilingual support: 100+ languages
        - Instruction awareness: Can use prompts like "query" or "document"
        
        Args:
            text: Input text to embed
            prompt_name: Optional prompt name ("query" or "document") for instruction-aware embedding
        """
        if self.model:
            try:
                # Run model encoding in thread pool to avoid blocking
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                def encode_text():
                    # Use instruction prompt if specified
                    if prompt_name:
                        return self.model.encode(text, prompt_name=prompt_name, normalize_embeddings=True)
                    elif self.use_instruction:
                        return self.model.encode(text, prompt_name=self.instruction_prompt, normalize_embeddings=True)
                    else:
                        return self.model.encode(text, normalize_embeddings=True)
                
                # Execute in thread pool for async compatibility
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    embedding = await loop.run_in_executor(executor, encode_text)
                
                # Convert to list and ensure correct dimension
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                # Handle dimension adjustment if needed
                # Qwen3-Embedding-0.6B outputs up to 1024 dimensions
                # We can adjust by truncating or using the model's dimension parameter
                if len(embedding_list) != self.embedding_dimension:
                    if len(embedding_list) < self.embedding_dimension:
                        # Pad with zeros if needed (shouldn't happen, but safety check)
                        embedding_list = embedding_list + [0.0] * (self.embedding_dimension - len(embedding_list))
                    else:
                        # Truncate if needed (model supports up to 1024)
                        embedding_list = embedding_list[:self.embedding_dimension]
                
                return embedding_list
            except Exception as e:
                # Log error and fallback to simulated embeddings
                print(f"Warning: Embedding generation failed: {e}. Using simulated embeddings.")
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                return [(hash_val >> i) % 100 / 100.0 for i in range(self.embedding_dimension)]
        else:
            # No model loaded, use simulated embeddings
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 100 / 100.0 for i in range(self.embedding_dimension)]
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        similarity = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
        # Clamp to [0.0, 1.0] to handle floating-point precision errors
        return max(0.0, min(1.0, similarity))

embedding_service = EmbeddingService()

# ============= Memory Processing =============

class MemoryProcessor:
    """Handles memory creation and relationship inference"""
    
    @staticmethod
    def compute_content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    @staticmethod
    async def create_memory(content: str, metadata: Dict[str, Any], 
                           source: Optional[str] = None) -> Memory:
        # Use "document" prompt for documents/memories (or None for general embeddings)
        embedding = await embedding_service.generate_embedding(content, prompt_name="document")
        content_hash = MemoryProcessor.compute_content_hash(content)
        
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
            is_active=True,
            metadata={**metadata, "source": source} if source else metadata,
            content_hash=content_hash
        )
        
        return memory
    
    @staticmethod
    async def check_semantic_duplicates(content: str, similarity_threshold: float = 0.95) -> Optional[Tuple[Memory, float]]:
        """
        Check for semantic near-duplicates before memory creation using embedding similarity.
        Returns (most_similar_memory, similarity_score) if found above threshold, None otherwise.
        This prevents data bloat from reworded duplicates.
        """
        # Generate embedding for the new content
        new_embedding = await embedding_service.generate_embedding(content, prompt_name="document")
        
        # Search for similar memories using vector search
        # Check top 5 most similar to find potential duplicates
        similar_memories = await memory_store.vector_search(
            new_embedding,
            top_k=5,
            include_inactive=False
        )
        
        # Calculate exact similarity scores for each candidate
        best_match = None
        best_similarity = 0.0
        
        for existing_memory in similar_memories:
            similarity = embedding_service.cosine_similarity(
                new_embedding, existing_memory.embedding
            )
            similarity = max(0.0, min(1.0, similarity))  # Clamp to valid range
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_memory
        
        # Return match if similarity exceeds threshold
        if best_match and best_similarity >= similarity_threshold:
            return (best_match, best_similarity)
        
        return None
    
    @staticmethod
    async def infer_relationships(new_memory: Memory, 
                                  existing_memories: List[Memory]) -> List[MemoryRelationship]:
        """Infer relationships based on content similarity and patterns"""
        relationships = []
        
        for existing in existing_memories:
            similarity = embedding_service.cosine_similarity(
                new_memory.embedding, existing.embedding
            )
            
            # Clamp similarity to valid range [0.0, 1.0] to handle floating-point precision errors
            similarity = max(0.0, min(1.0, similarity))
            
            if similarity > 0.95:
                # Very high similarity - likely an UPDATE
                rel = MemoryRelationship(
                    from_id=existing.id,
                    to_id=new_memory.id,
                    relation_type=RelationType.UPDATE,
                    confidence=similarity,
                    created_at=datetime.utcnow(),
                    reasoning="High content similarity suggests updated information"
                )
                relationships.append(rel)
            
            elif similarity > 0.75:
                # Moderate similarity - EXTEND or DERIVE
                # Check if new memory adds detail (EXTEND) or synthesizes (DERIVE)
                if len(new_memory.content) > len(existing.content):
                    rel_type = RelationType.EXTEND
                    reason = "New memory adds context to existing information"
                else:
                    rel_type = RelationType.DERIVE
                    reason = "New memory derives insight from existing information"
                
                rel = MemoryRelationship(
                    from_id=existing.id,
                    to_id=new_memory.id,
                    relation_type=rel_type,
                    confidence=similarity,
                    created_at=datetime.utcnow(),
                    reasoning=reason
                )
                relationships.append(rel)
        
        return relationships

processor = MemoryProcessor()

# ============= API Endpoints =============

@app.post("/api/memories", response_model=Memory)
async def create_memory(input_data: MemoryInput):
    """Create a new memory from text input"""
    try:
        # Check for exact hash duplicates first (fast check)
        content_hash = MemoryProcessor.compute_content_hash(input_data.text)
        similar_hashes = await memory_store.find_similar_by_hash(content_hash)
        if similar_hashes:
            existing_memory = await memory_store.get_memory(similar_hashes[0])
            raise HTTPException(
                status_code=409, 
                detail=f"Exact duplicate memory already exists: {similar_hashes[0]}"
            )
        
        # Check for semantic near-duplicates BEFORE creating memory
        # This prevents reworded duplicates from creating data bloat
        duplicate_check = await processor.check_semantic_duplicates(
            input_data.text, 
            similarity_threshold=0.95
        )
        if duplicate_check:
            existing_memory, similarity = duplicate_check
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Semantic near-duplicate found (similarity: {similarity:.2%}). "
                    f"Existing memory ID: {existing_memory.id}. "
                    f"Content may be too similar to existing memory."
                )
            )
        
        # No duplicates found - create memory object
        memory = await processor.create_memory(
            input_data.text, 
            input_data.metadata,
            input_data.source
        )
        
        # Store memory
        await memory_store.add_memory(memory)
        
        # Infer relationships with existing memories
        existing_memories = await memory_store.get_all_memories(include_inactive=False)
        existing_memories = [m for m in existing_memories if m.id != memory.id]
        relationships = await processor.infer_relationships(memory, existing_memories)
        
        for rel in relationships:
            await memory_store.add_relationship(rel)
        
        # Notify subscribers of graph update
        await graph_notifier.notify_update(
            "memory_created",
            {
                "memory_id": memory.id,
                "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "relationships_count": len(relationships),
                "source": input_data.source,
                "metadata": memory.metadata
            }
        )
        
        # If relationships were created, notify those too
        if relationships:
            await graph_notifier.notify_update(
                "relationships_added",
                {
                    "count": len(relationships),
                    "memory_id": memory.id,
                    "relationship_types": [rel.relation_type.value for rel in relationships]
                }
            )
        
        # Invalidate stats cache since data has changed
        await stats_cache.invalidate()
        
        return memory
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="create memory")

@app.post("/api/memories/pdf")
async def create_memory_from_pdf(file: UploadFile = File(...)):
    """Extract text from PDF and create memories"""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF content
        content = await file.read()
        
        # Extract text from PDF using pdfplumber (run in thread pool to avoid blocking)
        def extract_pdf_text(pdf_bytes):
            """Extract text from PDF bytes - synchronous function"""
            text_chunks = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)
                
                # Combine all pages into full text
                full_text = "\n\n".join(all_text)
                
                # Split into chunks if text is too long (max 2000 chars per chunk)
                # This ensures we don't hit embedding model limits and allows better search granularity
                max_chunk_size = 2000
                if len(full_text) <= max_chunk_size:
                    text_chunks = [full_text]
                else:
                    # Split by paragraphs/sentences, then by size
                    paragraphs = full_text.split("\n\n")
                    current_chunk = ""
                    for para in paragraphs:
                        # If adding this paragraph would exceed limit, save current chunk and start new one
                        if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                            text_chunks.append(current_chunk.strip())
                            current_chunk = para
                        else:
                            if current_chunk:
                                current_chunk += "\n\n" + para
                            else:
                                current_chunk = para
                    # Add last chunk
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
            return text_chunks
        
        # Run PDF extraction in thread pool to avoid blocking
        text_chunks = await asyncio.to_thread(extract_pdf_text, content)
        
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Create memories for each chunk
        created_memories = []
        skipped_duplicates = []
        pdf_source_id = str(uuid.uuid4())  # Unique ID for this PDF upload
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Check for exact hash duplicates (fast check)
            chunk_hash = MemoryProcessor.compute_content_hash(chunk_text)
            similar_hashes = await memory_store.find_similar_by_hash(chunk_hash)
            if similar_hashes:
                # Skip exact duplicate chunk
                skipped_duplicates.append(f"Chunk {chunk_idx + 1}: exact duplicate")
                continue
            
            # Check for semantic near-duplicates BEFORE creating memory
            duplicate_check = await processor.check_semantic_duplicates(
                chunk_text,
                similarity_threshold=0.95
            )
            if duplicate_check:
                # Skip semantic duplicate chunk
                existing_memory, similarity = duplicate_check
                skipped_duplicates.append(
                    f"Chunk {chunk_idx + 1}: semantic duplicate (similarity: {similarity:.2%})"
                )
                continue
            
            # No duplicates found - create memory
            memory = await processor.create_memory(
                chunk_text,
                {
                    "source_type": "pdf",
                    "filename": file.filename,
                    "pdf_source_id": pdf_source_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "page_info": f"Chunk {chunk_idx + 1} of {len(text_chunks)}"
                },
                source=file.filename
            )
            
            await memory_store.add_memory(memory)
            created_memories.append(memory)
            
            # Create relationships between consecutive chunks from the same PDF
            if len(created_memories) > 1:
                # Link this chunk to the previous created chunk from the same PDF
                prev_memory = created_memories[-2]  # Previous chunk in created_memories list
                relationship = MemoryRelationship(
                    from_id=prev_memory.id,
                    to_id=memory.id,
                    relation_type=RelationType.CHUNK_SEQUENCE,
                    confidence=0.8,
                    created_at=datetime.utcnow(),
                    reasoning=f"Sequential chunks from same PDF document (source: {pdf_source_id})"
                )
                await memory_store.add_relationship(relationship)
        
        # Infer relationships with existing memories (excluding the ones we just created)
        existing_memories = await memory_store.get_all_memories(include_inactive=False)
        existing_memories = [m for m in existing_memories 
                           if m.id not in [mem.id for mem in created_memories]]
        
        for memory in created_memories:
            relationships = await processor.infer_relationships(memory, existing_memories)
            for rel in relationships:
                await memory_store.add_relationship(rel)
            
            # Notify subscribers for each memory created from PDF
            await graph_notifier.notify_update(
                "memory_created",
                {
                    "memory_id": memory.id,
                    "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    "relationships_count": len(relationships),
                    "source": "pdf",
                    "filename": file.filename,
                    "metadata": memory.metadata
                }
            )
        
        # Notify of bulk PDF processing completion and invalidate cache
        if created_memories:
            await graph_notifier.notify_update(
                "graph_updated",
                {
                    "event": "pdf_processed",
                    "filename": file.filename,
                    "chunks_created": len(created_memories),
                    "chunks_skipped": len(skipped_duplicates),
                    "memory_ids": [m.id for m in created_memories]
                }
            )
            
            # Invalidate stats cache since data has changed
            await stats_cache.invalidate()
        
        # Return the first memory (or all if needed) - frontend can use this
        if not created_memories:
            raise HTTPException(
                status_code=400,
                detail="All chunks from PDF were duplicates. No new memories created."
            )
        
        return {
            "id": pdf_source_id,
            "filename": file.filename,
            "chunks_created": len(created_memories),
            "chunks_skipped": len(skipped_duplicates),
            "skipped_details": skipped_duplicates if skipped_duplicates else None,
            "memories": [{"id": m.id, "content": m.content[:200] + "..." if len(m.content) > 200 else m.content} 
                        for m in created_memories],
            "first_memory": created_memories[0] if created_memories else None
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="PDF upload")

@app.post("/api/search", response_model=List[Memory])
async def search_memories(query: SearchQuery):
    """Semantic search across memories using Neo4j vector search"""
    try:
        # Generate query embedding with "query" prompt for better retrieval performance
        query_embedding = await embedding_service.generate_embedding(query.query, prompt_name="query")
        
        # Use Neo4j vector search if available, otherwise fallback to in-memory
        results = await memory_store.vector_search(
            query_embedding,
            top_k=query.top_k,
            include_inactive=query.include_inactive
        )
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="search memories")

@app.get("/api/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str):
    """Retrieve a specific memory"""
    try:
        memory = await memory_store.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        return memory
    except HTTPException:
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="get memory")

@app.get("/api/memories/{memory_id}/relationships", response_model=List[MemoryRelationship])
async def get_memory_relationships(memory_id: str):
    """Get all relationships for a memory"""
    try:
        memory = await memory_store.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        
        relationships = await memory_store.get_relationships(memory_id)
        return relationships
    except HTTPException:
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="get memory relationships")

@app.get("/api/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    memory_ids: Optional[List[str]] = Query(None),
    depth: int = Query(2, ge=0, le=5),
    active_only: bool = Query(True)
):
    """Get knowledge graph centered on specific memories or entire graph"""
    try:
        if memory_ids:
            nodes, edges = await memory_store.get_subgraph(memory_ids, depth)
        else:
            # Return entire graph - use optimized single query (prevents N+1)
            nodes = await memory_store.get_all_memories(include_inactive=not active_only)
            # Fetch all relationships in a single efficient query
            edges = await memory_store.get_all_relationships(active_only=active_only)
        
        if active_only:
            active_ids = {n.id for n in nodes if n.is_active}
            nodes = [n for n in nodes if n.is_active]
            edges = [e for e in edges if e.from_id in active_ids and e.to_id in active_ids]
        
        graph_nodes = [
            GraphNode(
                id=n.id,
                content=n.content[:100] + "..." if len(n.content) > 100 else n.content,
                version=n.version,
                is_active=n.is_active,
                created_at=n.created_at,
                metadata=n.metadata
            ) for n in nodes
        ]
        
        graph_edges = [
            GraphEdge(
                from_id=e.from_id,
                to_id=e.to_id,
                relation_type=e.relation_type,
                confidence=e.confidence
            ) for e in edges
        ]
        
        return KnowledgeGraphResponse(
            nodes=graph_nodes,
            edges=graph_edges,
            total_nodes=len(graph_nodes),
            total_edges=len(graph_edges)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="get knowledge graph")

@app.get("/api/graph/stream")
async def stream_graph_updates():
    """
    Server-Sent Events (SSE) endpoint for real-time knowledge graph updates.
    Streams graph change events as they occur (memory creation, updates, relationships, etc.).
    
    Frontend can connect to this endpoint to receive live updates:
    ```javascript
    const eventSource = new EventSource('/api/graph/stream');
    eventSource.onmessage = (event) => {
        const update = JSON.parse(event.data);
        // Handle graph update
    };
    ```
    """
    async def event_generator():
        queue = await graph_notifier.subscribe()
        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'type': 'connected', 'message': 'Graph update stream connected'})}\n\n"
            
            # Keep connection alive with periodic heartbeats
            while True:
                try:
                    # Wait for event with timeout for heartbeat
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    # Format as SSE event
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield f": heartbeat\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            await graph_notifier.unsubscribe(queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

async def _compute_stats() -> Dict[str, Any]:
    """
    Compute platform statistics from database.
    This is separated so it can be called to refresh cache.
    """
    all_memories = await memory_store.get_all_memories(include_inactive=True)
    total_memories = len(all_memories)
    active_memories = sum(1 for m in all_memories if m.is_active)
    
    # Batch fetch all relationships in a single query (prevents N+1)
    if all_memories:
        memory_ids = [m.id for m in all_memories]
        relationships_by_id = await memory_store.get_relationships_batch(memory_ids)
        
        # Flatten relationships and deduplicate
        all_rels = []
        for rels in relationships_by_id.values():
            all_rels.extend(rels)
        
        # Deduplicate
        seen_rels = set()
        unique_rels = []
        for rel in all_rels:
            rel_key = (rel.from_id, rel.to_id, rel.relation_type)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
    else:
        unique_rels = []
    
    total_relationships = len(unique_rels)
    rel_type_counts = {}
    for rel in unique_rels:
        rel_type_counts[rel.relation_type.value] = rel_type_counts.get(rel.relation_type.value, 0) + 1
    
    stats = {
        "total_memories": total_memories,
        "active_memories": active_memories,
        "inactive_memories": total_memories - active_memories,
        "total_relationships": total_relationships,
        "relationship_types": rel_type_counts,
        "avg_relationships_per_memory": total_relationships / total_memories if total_memories > 0 else 0,
        "cached_at": datetime.utcnow().isoformat()
    }
    
    return stats

@app.get("/api/stats")
async def get_stats():
    """
    Get platform statistics with caching for performance.
    Returns cached stats if available, otherwise computes and caches them.
    Cache is invalidated when memories or relationships change.
    """
    try:
        # Try to get from cache first
        cached_stats = await stats_cache.get_stats()
        if cached_stats:
            # Remove cached_at from response if present (internal metadata)
            response_stats = {k: v for k, v in cached_stats.items() if k != "cached_at"}
            return response_stats
        
        # Cache miss - compute stats
        stats = await _compute_stats()
        
        # Cache the results (5 minute TTL)
        await stats_cache.set_stats(stats, ttl=300)
        
        # Remove cached_at from response
        response_stats = {k: v for k, v in stats.items() if k != "cached_at"}
        return response_stats
    
    except HTTPException:
        raise
    except Exception as e:
        raise handle_endpoint_error(e, context="get stats")

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "neo4j": False,
        "neo4j_vector": False,
        "embeddings": False
    }
    
    # Check Neo4j connection
    if memory_store.use_neo4j and memory_store.driver:
        try:
            async with memory_store.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            health_status["neo4j"] = True
            
            # Check if vector index exists (indicates vector search capability)
            try:
                async with memory_store.driver.session() as session:
                    result = await session.run("SHOW INDEXES YIELD name WHERE name = 'memory_embedding_idx' RETURN count(*) as count")
                    record = await result.single()
                    if record and record["count"] > 0:
                        health_status["neo4j_vector"] = True
            except Exception:
                pass  # Vector index check failed, but Neo4j is connected
        except Exception:
            pass  # Neo4j connection check failed
    
    # Check embeddings service
    if embedding_service.use_model and embedding_service.model:
        health_status["embeddings"] = True
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)