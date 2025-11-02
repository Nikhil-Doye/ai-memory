#!/bin/bash
# Migration Script - Upgrade to Neo4j + Vector Database
# This script updates your existing setup to use real databases

set -e

echo "��� AI Memory Platform - Database Migration"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if in correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "${RED}❌ Error: docker-compose.yml not found${NC}"
    echo "Please run this script from your ai-memory-platform directory"
    exit 1
fi

echo "${BLUE}This script will:${NC}"
echo "  1. Backup your current setup"
echo "  2. Update docker-compose.yml to use Neo4j for vectors"
echo "  3. Update backend/main.py with Neo4j + Vector integration"
echo "  4. Update requirements.txt"
echo "  5. Restart services"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Step 1: Backup
echo ""
echo "${BLUE}��� Step 1: Creating backup...${NC}"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || true
cp backend/main.py "$BACKUP_DIR/" 2>/dev/null || true
cp backend/requirements.txt "$BACKUP_DIR/" 2>/dev/null || true
cp .env "$BACKUP_DIR/" 2>/dev/null || true

echo "${GREEN}✅ Backup created in $BACKUP_DIR${NC}"

# Step 2: Check for DeepSeek API key
echo ""
echo "${BLUE}��� Step 2: OpenAI API Key${NC}"

if [ -f ".env" ]; then
    source .env
fi

if [ -z "$DEEPSEEK_API_KEY" ] || [ "$DEEPSEEK_API_KEY" == "sk-your-api-key-here" ]; then
    echo "${YELLOW}⚠️  No valid DeepSeek API key found${NC}"
    read -p "Enter your DeepSeek API key (or press Enter to use mock embeddings): " NEW_API_KEY
    
    if [ ! -z "$NEW_API_KEY" ]; then
        DEEPSEEK_API_KEY="$NEW_API_KEY"
        echo "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY" > .env.tmp
        cat .env | grep -v "DEEPSEEK_API_KEY" >> .env.tmp 2>/dev/null || true
        mv .env.tmp .env
        echo "${GREEN}✅ API key updated${NC}"
    else
        echo "${YELLOW}⚠️  Will use mock embeddings (limited functionality)${NC}"
    fi
else
    echo "${GREEN}✅ API key found${NC}"
fi

# Step 3: Update docker-compose.yml
echo ""
echo "${BLUE}��� Step 3: Updating docker-compose.yml...${NC}"

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: memory-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/memoryplatform2024
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=512M
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=512M
    volumes:
      - ./data/neo4j:/data
      - ./data/neo4j-logs:/logs
    networks:
      - memory-network
    healthcheck:
      test: ["CMD", "wget", "-O", "/dev/null", "-q", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: memory-redis
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - ./data/redis:/data
    networks:
      - memory-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

networks:
  memory-network:
    driver: bridge
EOF

echo "${GREEN}✅ docker-compose.yml updated (Neo4j for graph + vectors)${NC}"

# Step 4: Update requirements.txt
echo ""
echo "${BLUE}��� Step 4: Updating requirements.txt...${NC}"

cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic==2.5.3
neo4j==5.16.0
sentence-transformers>=2.7.0
transformers>=4.51.0
torch>=2.0.0
PyPDF2==3.0.1
pdfplumber==0.10.3
python-dotenv==1.0.0
aiofiles==23.2.1
redis==5.0.1
EOF

echo "${GREEN}✅ requirements.txt updated${NC}"

# Step 5: Update .env
echo ""
echo "${BLUE}⚙️  Step 5: Updating environment variables...${NC}"

cat > .env << EOF
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIMENSION=768
USE_HF_EMBEDDING_MODEL=true
USE_EMBEDDING_INSTRUCTION=true
EMBEDDING_INSTRUCTION=query
HF_TOKEN=your-huggingface-token-here
USE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=memoryplatform2024
LOG_LEVEL=info
EOF

echo "${GREEN}✅ Environment variables configured${NC}"

# Step 6: Stop current services
echo ""
echo "${BLUE}��� Step 6: Stopping current services...${NC}"
docker-compose down
echo "${GREEN}✅ Services stopped${NC}"

# Step 7: Update backend code
echo ""
echo "${BLUE}��� Step 7: Backend code update${NC}"
echo "${YELLOW}⚠️  You need to manually replace backend/main.py${NC}"
echo ""
echo "Please copy the content from the artifact:"
echo "${GREEN}'Backend with Neo4j + Vector Database'${NC}"
echo ""
echo "To: ${BLUE}backend/main.py${NC}"
echo ""
read -p "Press Enter when you've updated backend/main.py..."

# Step 8: Reinstall Python dependencies
echo ""
echo "${BLUE}��� Step 8: Installing Python dependencies...${NC}"

cd backend

if [ -d "venv" ]; then
    echo "Found existing venv, reinstalling..."
    rm -rf venv
fi

# Detect OS and use appropriate Python/venv command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "${RED}❌ Python not found. Please install Python first.${NC}"
    exit 1
fi

$PYTHON_CMD -m venv venv

# Detect OS and use appropriate activation script and pip command
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash, MINGW)
    source venv/Scripts/activate
    PYTHON_VENV="venv/Scripts/python.exe"
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac
    source venv/bin/activate
    PYTHON_VENV="venv/bin/python"
else
    echo "${YELLOW}⚠️  Could not find venv activation script${NC}"
    echo "Trying to install packages without activating venv..."
    $PYTHON_CMD -m pip install --upgrade pip -q 2>/dev/null || true
    $PYTHON_CMD -m pip install -r requirements.txt -q
    echo "${GREEN}✅ Python dependencies installed${NC}"
    cd ..
    exit 0
fi

echo "Installing packages..."
# Use python -m pip to avoid pip upgrade errors on Windows
if ! $PYTHON_VENV -m pip install --upgrade pip --quiet 2>/dev/null; then
    echo "${YELLOW}⚠️  Pip upgrade skipped (may already be latest or in use)${NC}"
fi

# Install requirements
if $PYTHON_VENV -m pip install -r requirements.txt --quiet 2>/dev/null; then
    echo "${GREEN}✅ Python dependencies installed${NC}"
else
    echo "${YELLOW}⚠️  Installing with verbose output...${NC}"
    $PYTHON_VENV -m pip install -r requirements.txt
    echo "${GREEN}✅ Python dependencies installed${NC}"
fi

cd ..

# Step 9: Start services
echo ""
echo "${BLUE}��� Step 9: Starting services...${NC}"

docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

# Check health
echo ""
echo "Checking service health..."

if docker-compose ps | grep -q "healthy"; then
    echo "${GREEN}✅ Services started successfully${NC}"
else
    echo "${YELLOW}⚠️  Some services may still be starting...${NC}"
fi

# Step 10: Summary
echo ""
echo "${GREEN}��� Migration Complete!${NC}"
echo ""
echo "${BLUE}What changed:${NC}"
echo "  ✅ Neo4j now handles both graph relationships and vector search"
echo "  ✅ Neo4j graph database configured"
echo "  ✅ Backend now uses real embeddings"
echo "  ✅ Automatic relationship inference enabled"
echo ""
echo "${BLUE}Services running:${NC}"
docker-compose ps
echo ""
echo "${BLUE}Next steps:${NC}"
echo "  1. Start backend:"
echo "     ${GREEN}cd backend && source venv/bin/activate && python main.py${NC}"
echo ""
echo "  2. Start frontend (in new terminal):"
echo "     ${GREEN}cd frontend && npm run dev${NC}"
echo ""
echo "  3. Test the system:"
echo "     ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo "  4. View Neo4j Browser:"
echo "     ${GREEN}http://localhost:7474${NC} (neo4j / memoryplatform2024)"
echo ""
echo "${BLUE}Documentation:${NC}"
echo "  • See 'Setup Guide - Neo4j + Vector Database' artifact"
echo "  • Backup location: ${GREEN}$BACKUP_DIR${NC}"
echo ""

if [ "$DEEPSEEK_API_KEY" == "sk-your-api-key-here" ]; then
    echo "${YELLOW}⚠️  Remember to add your DeepSeek API key to .env for real embeddings!${NC}"
fi

echo ""
echo "${GREEN}Happy building! ���${NC}"
