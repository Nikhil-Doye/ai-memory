#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script - Verify AI Memory Platform Setup
Tests Neo4j vector search and embeddings integration
"""

import asyncio
import httpx
import time
from typing import Dict, Any
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuration
API_BASE = "http://localhost:8000"
TIMEOUT = 30.0

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERROR] {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.END}")

async def test_health_check(client: httpx.AsyncClient) -> bool:
    """Test 1: Health check"""
    print("\n" + "="*50)
    print("Test 1: Health Check")
    print("="*50)
    
    try:
        response = await client.get(f"{API_BASE}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_info(f"Status: {data['status']}")
            
            if data.get('neo4j'):
                print_success("Neo4j connected")
            else:
                print_error("Neo4j not connected")
                return False
            
            if data.get('neo4j_vector'):
                print_success("Neo4j vector search enabled")
            else:
                print_warning("Neo4j vector search not enabled (using fallback)")
            
            if data.get('embeddings'):
                print_success("Embeddings service configured")
            else:
                print_warning("Embeddings service missing (using mock embeddings)")
            
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False

async def test_create_memory(client: httpx.AsyncClient, text: str, metadata: Dict = None) -> Dict[str, Any]:
    """Test 2: Create memory"""
    try:
        response = await client.post(
            f"{API_BASE}/api/memories",
            json={
                "text": text,
                "metadata": metadata or {}
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Created memory: {data['id']}")
            print_info(f"Content: {text[:50]}...")
            return data
        else:
            print_error(f"Failed to create memory: {response.status_code}")
            print_error(response.text)
            return None
    
    except Exception as e:
        print_error(f"Create memory error: {str(e)}")
        return None

async def test_search(client: httpx.AsyncClient, query: str) -> list:
    """Test 3: Semantic search"""
    print("\n" + "="*50)
    print(f"Test 3: Semantic Search - '{query}'")
    print("="*50)
    
    try:
        response = await client.post(
            f"{API_BASE}/api/search",
            json={"query": query, "top_k": 5},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            results = response.json()
            print_success(f"Found {len(results)} results")
            
            for i, result in enumerate(results[:3], 1):
                print_info(f"Result {i}: {result['content'][:60]}...")
            
            return results
        else:
            print_error(f"Search failed: {response.status_code}")
            return []
    
    except Exception as e:
        print_error(f"Search error: {str(e)}")
        return []

async def test_get_graph(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Test 4: Get knowledge graph"""
    print("\n" + "="*50)
    print("Test 4: Knowledge Graph")
    print("="*50)
    
    try:
        response = await client.get(
            f"{API_BASE}/api/graph?depth=2&active_only=true",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Graph has {data['total_nodes']} nodes and {data['total_edges']} edges")
            
            if data['total_edges'] > 0:
                print_info("Sample relationships:")
                for edge in data['edges'][:3]:
                    print_info(f"  {edge['relation_type']}: {edge['from_id'][:8]}... -> {edge['to_id'][:8]}...")
            
            return data
        else:
            print_error(f"Get graph failed: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Get graph error: {str(e)}")
        return None

async def test_get_stats(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Test 5: Get statistics"""
    print("\n" + "="*50)
    print("Test 5: Statistics")
    print("="*50)
    
    try:
        response = await client.get(f"{API_BASE}/api/stats", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Statistics retrieved:")
            print_info(f"  Total memories: {data['total_memories']}")
            print_info(f"  Active memories: {data['active_memories']}")
            print_info(f"  Inactive memories: {data['inactive_memories']}")
            print_info(f"  Total relationships: {data['total_relationships']}")
            print_info(f"  Avg relationships/memory: {data['avg_relationships_per_memory']:.2f}")
            return data
        else:
            print_error(f"Get stats failed: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Get stats error: {str(e)}")
        return None

async def run_all_tests():
    """Run complete test suite"""
    print("\n" + "��� AI Memory Platform Test Suite")
    print("="*50)
    print("Testing Neo4j (graph + vector search) integration")
    print("="*50)
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{API_BASE}/health", timeout=5)
    except Exception as e:
        print_error("Backend server is not running!")
        print_info("Start the backend with: cd backend && python main.py")
        sys.exit(1)
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health Check
        if await test_health_check(client):
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
            print_error("Health check failed - stopping tests")
            return test_results
        
        # Test 2: Create Test Memories
        print("\n" + "="*50)
        print("Test 2: Create Memories")
        print("="*50)
        
        test_memories = [
            ("User prefers morning workouts at 6 AM for better energy", {"category": "fitness"}),
            ("User changed workout time to 7 AM due to work schedule", {"category": "fitness"}),
            ("User follows high-protein diet with 150g daily protein target", {"category": "nutrition"}),
        ]
        
        created_memories = []
        for text, metadata in test_memories:
            memory = await test_create_memory(client, text, metadata)
            if memory:
                created_memories.append(memory)
                test_results["passed"] += 1
            else:
                test_results["failed"] += 1
        
        # Wait for indexing
        print_info("Waiting for embeddings to be indexed...")
        await asyncio.sleep(2)
        
        # Test 3: Semantic Search
        search_results = await test_search(client, "exercise routine")
        if search_results:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
        
        # Test 4: Knowledge Graph
        graph = await test_get_graph(client)
        if graph:
            test_results["passed"] += 1
            if graph['total_edges'] == 0:
                print_warning("No relationships found - this is normal for first run")
                test_results["warnings"] += 1
        else:
            test_results["failed"] += 1
        
        # Test 5: Statistics
        stats = await test_get_stats(client)
        if stats:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
    
    # Print summary
    print("\n" + "="*50)
    print("��� Test Summary")
    print("="*50)
    print(f"{Colors.GREEN}Passed: {test_results['passed']}{Colors.END}")
    print(f"{Colors.RED}Failed: {test_results['failed']}{Colors.END}")
    print(f"{Colors.YELLOW}Warnings: {test_results['warnings']}{Colors.END}")
    
    if test_results['failed'] == 0:
        print(f"\n{Colors.GREEN}��� All tests passed!{Colors.END}")
        print("\nYour AI Memory Platform is working correctly!")
        print("\nNext steps:")
        print("  • Open frontend: http://localhost:5173")
        print("  • View Neo4j Browser: http://localhost:7474")
        print("  • API docs: http://localhost:8000/docs")
    else:
        print(f"\n{Colors.RED}[WARN] Some tests failed{Colors.END}")
        print("\nTroubleshooting:")
        print("  1. Check all services are running: docker-compose ps")
        print("  2. View backend logs for errors")
        print("  3. Verify .env has correct credentials")
        print("  4. Ensure Neo4j is healthy and vector index is created")
    
    return test_results

if __name__ == "__main__":
    print("Starting tests...")
    print("Make sure your backend is running: python backend/main.py\n")
    
    try:
        results = asyncio.run(run_all_tests())
        sys.exit(0 if results["failed"] == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.END}")
        sys.exit(1)
