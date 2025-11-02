import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "react";
import {
  Search,
  Upload,
  Plus,
  Activity,
  Database,
  Network,
  TrendingUp,
  Clock,
  Zap,
} from "lucide-react";

// Mock API client (replace with actual axios/fetch calls)
const API_BASE = "http://localhost:8000/api";

const MemoryPlatform = () => {
  const [memories, setMemories] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [stats, setStats] = useState({});
  const [searchQuery, setSearchQuery] = useState("");
  const [newMemoryText, setNewMemoryText] = useState("");
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [activeTab, setActiveTab] = useState("graph");
  const [uploadingPDF, setUploadingPDF] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [initializing, setInitializing] = useState(true);
  const fileInputRef = useRef(null);

  const canvasRef = useRef(null);
  const [graphInstance, setGraphInstance] = useState(null);

  // Initialize: Try loading from backend first, fall back to sample data if backend unavailable
  useEffect(() => {
    const initializeData = async () => {
      setInitializing(true);
      try {
        // Try to load real data from backend
        await Promise.all([
          loadGraphData(),
          loadStats(),
        ]);
      } catch (error) {
        console.warn("Backend unavailable, using sample data:", error);
        // Fall back to sample data only if backend is unreachable
        loadSampleData();
        loadStatsFallback();
      } finally {
        setInitializing(false);
      }
    };
    initializeData();
  }, []);

  // Auto-clear success/error messages after 5 seconds
  useEffect(() => {
    if (uploadSuccess) {
      const timer = setTimeout(() => setUploadSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadSuccess]);

  useEffect(() => {
    if (uploadError) {
      const timer = setTimeout(() => setUploadError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadError]);

  const loadSampleData = () => {
    const sampleNodes = [
      {
        id: "1",
        content: "User prefers morning workouts at 6 AM",
        version: 1,
        is_active: true,
        created_at: new Date("2025-01-15"),
        metadata: { source: "user_input" },
      },
      {
        id: "2",
        content: "User increased workout time to 7 AM for better energy",
        version: 2,
        is_active: true,
        created_at: new Date("2025-02-10"),
        metadata: { source: "user_input" },
      },
      {
        id: "3",
        content: "User follows high-protein diet with 150g daily target",
        version: 1,
        is_active: true,
        created_at: new Date("2025-01-20"),
        metadata: { source: "user_input" },
      },
      {
        id: "4",
        content:
          "Derived insight: Morning workouts correlate with higher protein intake",
        version: 1,
        is_active: true,
        created_at: new Date("2025-02-15"),
        metadata: { source: "inferred" },
      },
      {
        id: "5",
        content:
          "User added resistance training 3x per week to workout routine",
        version: 1,
        is_active: true,
        created_at: new Date("2025-02-20"),
        metadata: { source: "user_input" },
      },
      {
        id: "6",
        content: "Original morning workout preference at 6 AM",
        version: 1,
        is_active: false,
        created_at: new Date("2025-01-15"),
        metadata: { source: "user_input" },
      },
    ];

    const sampleEdges = [
      { from_id: "6", to_id: "2", relation_type: "UPDATE", confidence: 0.98 },
      { from_id: "2", to_id: "4", relation_type: "DERIVE", confidence: 0.82 },
      { from_id: "3", to_id: "4", relation_type: "DERIVE", confidence: 0.85 },
      { from_id: "2", to_id: "5", relation_type: "EXTEND", confidence: 0.76 },
    ];

    setGraphData({ nodes: sampleNodes, edges: sampleEdges });
    setMemories(sampleNodes);
  };

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      } else {
        throw new Error(`Failed to load stats: ${response.statusText}`);
      }
    } catch (error) {
      console.error("Failed to load stats from backend:", error);
      throw error; // Re-throw to allow fallback in initialization
    }
  };

  // Fallback stats for when backend is unavailable
  const loadStatsFallback = () => {
    setStats({
      total_memories: 6,
      active_memories: 5,
      inactive_memories: 1,
      total_relationships: 4,
      relationship_types: { UPDATE: 1, EXTEND: 1, DERIVE: 2 },
      avg_relationships_per_memory: 0.67,
    });
  };

  const createMemory = async () => {
    if (!newMemoryText.trim()) return;

    setLoading(true);
    setUploadError(null);
    setUploadSuccess(null);
    
    try {
      const response = await fetch(`${API_BASE}/memories`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: newMemoryText,
          metadata: { source: "user_input" },
          source: "manual"
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to create memory' }));
        
        // Handle conflict (409) - duplicate memory
        if (response.status === 409) {
          setUploadError(
            errorData.detail || "This memory already exists (exact or semantic duplicate)."
          );
          setNewMemoryText("");
          setLoading(false);
          return;
        }
        
        // Handle other errors
        throw new Error(errorData.detail || `Failed to create memory: ${response.statusText}`);
      }

      // Get the created memory from API response
      const createdMemory = await response.json();
      
      // Show success message
      setUploadSuccess(`Memory created successfully!`);
      
      // Clear input
      setNewMemoryText("");
      
      // Reload graph and stats to reflect new memory and any inferred relationships
      // This ensures we get the latest state including relationships inferred by the backend
      await Promise.all([
        loadGraphData(),
        loadStats(),
      ]);
      
    } catch (error) {
      console.error("Failed to create memory:", error);
      
      // Handle network errors or other failures
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        setUploadError("Unable to connect to backend. Please check if the server is running.");
      } else {
        setUploadError(error.message || "Failed to create memory. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const performSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      setSearchError(null);
      return;
    }

    setSearching(true);
    setSearchError(null);
    setSearchResults([]);

    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          top_k: 10,
          include_inactive: false,
          relation_depth: 2
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ 
          detail: `Search failed: ${response.statusText}` 
        }));
        throw new Error(errorData.detail || `Search failed: ${response.statusText}`);
      }

      const results = await response.json();
      
      // Set results from backend semantic search
      setSearchResults(results || []);
      
      // Clear any previous errors on success
      setSearchError(null);
      
    } catch (error) {
      console.error("Semantic search failed:", error);
      
      // Show error to user - don't fall back to local search
      // This ensures users know they're getting semantic search results
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        setSearchError("Unable to connect to backend for semantic search. Please check if the server is running.");
      } else {
        setSearchError(error.message || "Search failed. Please try again.");
      }
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const handlePDFUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError("Please select a PDF file");
      setUploadSuccess(null);
      return;
    }

    setUploadingPDF(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/memories/pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Show success message
      setUploadSuccess(
        `Successfully uploaded "${file.name}"! Created ${result.chunks_created || 0} memory chunk(s).${result.chunks_skipped > 0 ? ` Skipped ${result.chunks_skipped} duplicate(s).` : ''}`
      );

      // Reload memories and graph data
      await loadGraphData();
      await loadStats();

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      setUploadError(error.message || "Failed to upload PDF. Please try again.");
      setUploadSuccess(null);
    } finally {
      setUploadingPDF(false);
    }
  };

  const loadGraphData = async () => {
    try {
      const response = await fetch(`${API_BASE}/graph?active_only=true`);
      if (response.ok) {
        const data = await response.json();
        setGraphData({
          nodes: data.nodes || [],
          edges: data.edges || [],
        });
        setMemories(data.nodes || []);
      } else {
        throw new Error(`Failed to load graph: ${response.statusText}`);
      }
    } catch (error) {
      console.error("Failed to load graph data from backend:", error);
      throw error; // Re-throw to allow fallback in initialization
    }
  };

  // Graph Visualization Component
  const GraphVisualization = () => {
    const svgRef = useRef(null);
    const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    // Track previous node IDs and positions using refs to prevent infinite loops
    const prevNodeIdsRef = useRef("");
    const isUpdatingRef = useRef(false);

    useEffect(() => {
      // Prevent concurrent updates
      if (isUpdatingRef.current) {
        return;
      }

      const nodeCount = graphData.nodes.length;

      // Create node IDs string for comparison
      const nodeIds = graphData.nodes
        .map((n) => n.id)
        .sort()
        .join(",");

      // Skip if nothing actually changed
      if (prevNodeIdsRef.current === nodeIds) {
        return;
      }

      // Mark as updating and update ref
      isUpdatingRef.current = true;
      prevNodeIdsRef.current = nodeIds;

      if (!nodeCount) {
        setGraphInstance(null);
        isUpdatingRef.current = false;
        return;
      }

      // Force-directed layout simulation
      const width = 800;
      const height = 600;
      const centerX = width / 2;
      const centerY = height / 2;

      // Simple circular layout with physics
      const positions = graphData.nodes.map((node, i) => {
        const angle = (i / nodeCount) * 2 * Math.PI;
        const radius = 200;
        return {
          ...node,
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
          vx: 0,
          vy: 0,
        };
      });

      // Use setTimeout to defer state update and break potential render cycle
      setTimeout(() => {
        setGraphInstance(positions);
        isUpdatingRef.current = false;
      }, 0);
      // Only trigger when node count changes - the ref check inside prevents unnecessary updates
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [graphData.nodes.length]);

    const handleMouseDown = (e) => {
      setIsDragging(true);
      setDragStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      setTransform({
        ...transform,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    // Memoize handleWheel to prevent unnecessary re-attachments
    const handleWheel = useCallback((e) => {
      e.preventDefault();
      const delta = e.deltaY * -0.001;
      setTransform((prev) => {
        const newScale = Math.min(Math.max(0.5, prev.scale + delta), 3);
        return { ...prev, scale: newScale };
      });
    }, []);

    // Attach wheel event listener manually with passive: false to allow preventDefault
    useEffect(() => {
      const svgElement = svgRef.current;
      if (!svgElement) return;

      svgElement.addEventListener("wheel", handleWheel, { passive: false });

      return () => {
        svgElement.removeEventListener("wheel", handleWheel);
      };
    }, [handleWheel]);

    if (!graphInstance)
      return (
        <div className="flex items-center justify-center h-full text-gray-400">
          Loading graph...
        </div>
      );

    return (
      <div className="relative w-full h-full bg-slate-900 rounded-lg overflow-hidden">
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          style={{ cursor: isDragging ? "grabbing" : "grab" }}
        >
          <defs>
            <marker
              id="arrowhead-update"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#ef4444" />
            </marker>
            <marker
              id="arrowhead-extend"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#3b82f6" />
            </marker>
            <marker
              id="arrowhead-derive"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#8b5cf6" />
            </marker>
          </defs>

          <g
            transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}
          >
            {/* Edges */}
            {graphData.edges.map((edge, i) => {
              const fromNode = graphInstance.find((n) => n.id === edge.from_id);
              const toNode = graphInstance.find((n) => n.id === edge.to_id);
              if (!fromNode || !toNode) return null;

              const markerMap = {
                UPDATE: "arrowhead-update",
                EXTEND: "arrowhead-extend",
                DERIVE: "arrowhead-derive",
              };

              const colorMap = {
                UPDATE: "#ef4444",
                EXTEND: "#3b82f6",
                DERIVE: "#8b5cf6",
              };

              return (
                <g key={`edge-${i}`}>
                  <line
                    x1={fromNode.x}
                    y1={fromNode.y}
                    x2={toNode.x}
                    y2={toNode.y}
                    stroke={colorMap[edge.relation_type]}
                    strokeWidth={2}
                    markerEnd={`url(#${markerMap[edge.relation_type]})`}
                    opacity={0.6}
                  />
                  <text
                    x={(fromNode.x + toNode.x) / 2}
                    y={(fromNode.y + toNode.y) / 2}
                    fill="#94a3b8"
                    fontSize="10"
                    textAnchor="middle"
                  >
                    {edge.relation_type}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {graphInstance.map((node) => {
              const isSelected = selectedNode?.id === node.id;
              const nodeColor = node.is_active ? "#10b981" : "#6b7280";

              return (
                <g
                  key={node.id}
                  onClick={() => setSelectedNode(node)}
                  style={{ cursor: "pointer" }}
                >
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={isSelected ? 25 : 20}
                    fill={nodeColor}
                    stroke={isSelected ? "#fbbf24" : "#1e293b"}
                    strokeWidth={isSelected ? 3 : 2}
                    opacity={node.is_active ? 1 : 0.5}
                  />
                  <text
                    x={node.x}
                    y={node.y + 35}
                    fill="#e2e8f0"
                    fontSize="11"
                    textAnchor="middle"
                    fontWeight={isSelected ? "bold" : "normal"}
                  >
                    v{node.version}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-slate-800 p-3 rounded-lg text-xs space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-gray-300">Active Memory</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-500"></div>
            <span className="text-gray-300">Inactive Memory</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-red-500"></div>
            <span className="text-gray-300">UPDATE</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span className="text-gray-300">EXTEND</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-purple-500"></div>
            <span className="text-gray-300">DERIVE</span>
          </div>
        </div>

        {/* Controls */}
        <div className="absolute top-4 right-4 bg-slate-800 p-2 rounded-lg text-xs space-y-1">
          <button
            onClick={() => setTransform({ x: 0, y: 0, scale: 1 })}
            className="w-full px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-gray-300"
          >
            Reset View
          </button>
          <div className="text-gray-400 text-center">
            Zoom: {transform.scale.toFixed(2)}x
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-gray-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network className="w-8 h-8 text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold">AI Memory Platform</h1>
                <p className="text-sm text-blue-200">
                  Semantic Knowledge Graph Management
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 bg-slate-900/50 px-3 py-2 rounded-lg">
                <Activity className="w-4 h-4 text-green-400" />
                <span className="text-sm font-mono">&lt;400ms</span>
              </div>
              <div className="flex items-center gap-2 bg-slate-900/50 px-3 py-2 rounded-lg">
                <Database className="w-4 h-4 text-blue-400" />
                <span className="text-sm">
                  {stats.active_memories || 0} Active
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Stats Dashboard */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Total Memories</p>
                <p className="text-2xl font-bold text-blue-400">
                  {stats.total_memories || 0}
                </p>
              </div>
              <Database className="w-8 h-8 text-blue-400 opacity-50" />
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Relationships</p>
                <p className="text-2xl font-bold text-purple-400">
                  {stats.total_relationships || 0}
                </p>
              </div>
              <Network className="w-8 h-8 text-purple-400 opacity-50" />
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Active Nodes</p>
                <p className="text-2xl font-bold text-green-400">
                  {stats.active_memories || 0}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400 opacity-50" />
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Avg Connections</p>
                <p className="text-2xl font-bold text-yellow-400">
                  {(stats.avg_relationships_per_memory || 0).toFixed(2)}
                </p>
              </div>
              <Zap className="w-8 h-8 text-yellow-400 opacity-50" />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-4">
            {/* Add Memory */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Plus className="w-5 h-5" />
                Add Memory
              </h3>
              <textarea
                value={newMemoryText}
                onChange={(e) => setNewMemoryText(e.target.value)}
                placeholder="Enter new memory text..."
                className="w-full h-24 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm mb-3 focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={createMemory}
                disabled={loading || !newMemoryText.trim()}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-gray-500 px-4 py-2 rounded font-medium transition-colors"
              >
                {loading ? "Processing..." : "Create Memory"}
              </button>

              {/* Error/Success messages for memory creation */}
              {uploadError && !uploadingPDF && (
                <div className="mt-3 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded px-2 py-1">
                  {uploadError}
                </div>
              )}
              {uploadSuccess && !uploadingPDF && (
                <div className="mt-3 text-sm text-green-400 bg-green-900/20 border border-green-800 rounded px-2 py-1">
                  {uploadSuccess}
                </div>
              )}

              <div className="mt-3 pt-3 border-t border-slate-800">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer hover:text-gray-300">
                  <Upload className="w-4 h-4" />
                  <span>{uploadingPDF ? "Uploading..." : "Upload PDF"}</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={handlePDFUpload}
                    disabled={uploadingPDF}
                  />
                </label>
                {/* Error/Success messages for PDF upload only */}
                {uploadError && uploadingPDF !== undefined && (
                  <div className="mt-2 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded px-2 py-1">
                    {uploadError}
                  </div>
                )}
                {uploadSuccess && uploadingPDF !== undefined && (
                  <div className="mt-2 text-sm text-green-400 bg-green-900/20 border border-green-800 rounded px-2 py-1">
                    {uploadSuccess}
                  </div>
                )}
              </div>
            </div>

            {/* Search */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Search className="w-5 h-5" />
                Semantic Search
              </h3>
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && !searching && performSearch()}
                  placeholder="Search memories semantically..."
                  disabled={searching}
                  className="flex-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <button
                  onClick={performSearch}
                  disabled={searching || !searchQuery.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-gray-500 disabled:cursor-not-allowed px-4 py-2 rounded transition-colors flex items-center justify-center"
                >
                  {searching ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <Search className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Loading state */}
              {searching && (
                <div className="flex items-center justify-center py-4 text-gray-400">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm">Searching with semantic similarity...</span>
                  </div>
                </div>
              )}

              {/* Error state */}
              {searchError && !searching && (
                <div className="mb-3 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded px-2 py-1">
                  {searchError}
                </div>
              )}

              {/* Results */}
              {!searching && searchResults.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-gray-400 mb-2">
                    Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} (semantic similarity)
                  </div>
                  {searchResults.map((result) => (
                    <div
                      key={result.id}
                      onClick={() => setSelectedNode(result)}
                      className="bg-slate-800 border border-slate-700 rounded p-2 cursor-pointer hover:border-blue-500 transition-colors"
                    >
                      <p className="text-xs text-gray-300">
                        {result.content.substring(0, 100)}{result.content.length > 100 ? '...' : ''}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        v{result.version} •{" "}
                        {result.is_active ? "Active" : "Inactive"}
                        {result.metadata?.source && ` • Source: ${result.metadata.source}`}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* No results state */}
              {!searching && !searchError && searchQuery.trim() && searchResults.length === 0 && (
                <div className="text-center py-4 text-gray-400 text-sm">
                  No memories found matching your query.
                </div>
              )}

              {/* Empty state */}
              {!searching && !searchError && !searchQuery.trim() && searchResults.length === 0 && (
                <div className="text-center py-4 text-gray-500 text-xs">
                  Enter a search query to find semantically similar memories.
                </div>
              )}
            </div>

            {/* Selected Node Details */}
            {selectedNode && (
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3">Memory Details</h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Content</p>
                    <p className="text-sm text-gray-200">
                      {selectedNode.content}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Version</p>
                      <p className="text-sm font-mono text-blue-400">
                        v{selectedNode.version}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Status</p>
                      <p
                        className={`text-sm font-medium ${
                          selectedNode.is_active
                            ? "text-green-400"
                            : "text-gray-400"
                        }`}
                      >
                        {selectedNode.is_active ? "Active" : "Inactive"}
                      </p>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Created</p>
                    <p className="text-sm text-gray-300 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(selectedNode.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Source</p>
                    <p className="text-sm text-gray-300">
                      {selectedNode.metadata?.source || "Unknown"}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Visualization */}
          <div className="col-span-2">
            <div
              className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden"
              style={{ height: "700px" }}
            >
              <div className="border-b border-slate-800 p-3 flex items-center justify-between">
                <h3 className="text-lg font-semibold">Knowledge Graph</h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setActiveTab("graph")}
                    className={`px-3 py-1 rounded text-sm ${
                      activeTab === "graph"
                        ? "bg-blue-600"
                        : "bg-slate-800 hover:bg-slate-700"
                    }`}
                  >
                    Graph View
                  </button>
                  <button
                    onClick={() => setActiveTab("timeline")}
                    className={`px-3 py-1 rounded text-sm ${
                      activeTab === "timeline"
                        ? "bg-blue-600"
                        : "bg-slate-800 hover:bg-slate-700"
                    }`}
                  >
                    Timeline
                  </button>
                </div>
              </div>

              <div className="h-full">
                {initializing ? (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                      <p>Loading graph data from backend...</p>
                    </div>
                  </div>
                ) : activeTab === "graph" ? (
                  <GraphVisualization />
                ) : (
                  <div className="p-6 overflow-y-auto h-full">
                    <h4 className="text-sm font-semibold text-gray-400 mb-4">
                      Memory Evolution Timeline
                    </h4>
                    <div className="space-y-4">
                      {memories
                        .sort(
                          (a, b) =>
                            new Date(a.created_at) - new Date(b.created_at)
                        )
                        .map((memory, idx) => (
                          <div key={memory.id} className="flex gap-4">
                            <div className="flex flex-col items-center">
                              <div
                                className={`w-3 h-3 rounded-full ${
                                  memory.is_active
                                    ? "bg-green-500"
                                    : "bg-gray-500"
                                }`}
                              ></div>
                              {idx < memories.length - 1 && (
                                <div className="w-0.5 h-full bg-slate-700 my-1"></div>
                              )}
                            </div>
                            <div className="flex-1 pb-4">
                              <p className="text-xs text-gray-400">
                                {new Date(memory.created_at).toLocaleString()}
                              </p>
                              <p className="text-sm text-gray-200 mt-1">
                                {memory.content}
                              </p>
                              <p className="text-xs text-gray-500 mt-1">
                                v{memory.version} •{" "}
                                {memory.is_active ? "Active" : "Superseded"}
                              </p>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  return <MemoryPlatform />;
}
