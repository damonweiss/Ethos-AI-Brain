# Unified MCP Architecture - Complete Design

## Overview

Comprehensive design for a unified MCP (Model Context Protocol) system that supports multiple execution modalities, multi-server federation, and universal tool wrapping capabilities.

## Core Architecture Principles

### Agent-Brain Separation
- **Agent** = Prompt Generator + Actor (interface to reality)
- **Brain** = Pure Processing Engine (processes thoughts, never acts)
- **MCP Tools** = Universal registry/plugin repo for both thinking patterns and action capabilities

### Dual Modality Support
- **Direct Execution** - Homebrew subprocess-based tool execution via `uv run`
- **JSON-RPC Protocol** - Standards-compliant MCP server communication
- **Unified Interface** - Same API regardless of backend modality

## Client Architecture

### Modular Client Structure
```
ethos_ai_brain/orchestration/mcp/client/
├── __init__.py                    # Public API exports
├── mcp_client_base.py            # Abstract base & config classes
├── mcp_client_direct.py          # Direct execution strategy
├── mcp_client_json_rpc.py        # JSON-RPC server strategy
├── mcp_client_router.py          # Router with hybrid strategy
├── mcp_client_factory.py         # Factory functions
└── mcp_server_registry.py        # Multi-server registry (future)
```

### Strategy Pattern Implementation
- **MCPStrategy** - Abstract base for execution strategies
- **DirectStrategy** - Wraps MCPToolManager for subprocess execution
- **JsonRpcStrategy** - Wraps EthosMCPClient for HTTP/JSON-RPC
- **HybridStrategy** - Routes tools to different backends based on rules

### Configuration-Driven Routing
```python
@dataclass
class RoutingConfig:
    routing_rules: Dict[str, str]      # tool_name -> preferred_mode
    namespace_rules: Dict[str, str]    # namespace -> preferred_mode
    fallback_rules: Dict[str, List[str]]  # tool -> fallback_chain

@dataclass
class MCPConfig:
    server_url: str = "http://localhost:8051"
    tools_directory: Optional[Path] = None
    default_mode: str = "auto"  # "auto", "direct", "json_rpc", "hybrid"
    routing_config: RoutingConfig
    timeout_settings: Dict[str, int]
```

### Factory Functions
```python
# Simple auto-detection
router = create_client()

# Development mode (direct execution)
router = create_dev_client()

# Production mode (server with fallback)
router = create_prod_client("http://prod-server:8051")

# Custom hybrid routing
router = create_hybrid_client(
    routing_rules={"special_tool": "direct"},
    namespace_rules={"brain.": "direct", "web.": "json_rpc"}
)
```

## Multi-Server Federation

### Server Registry Design
```python
@dataclass
class ServerConfig:
    name: str                    # "local", "anthropic", "partner"
    url: str                     # "http://localhost:8051"
    namespace_prefix: str        # "local.", "anthropic.", "partner."
    priority: int = 100          # Lower = higher priority
    health_check_interval: int = 30
    timeout: int = 30
    auth_token: Optional[str] = None

class MCPServerRegistry:
    def register_server(self, config: ServerConfig)
    def get_server_for_tool(self, tool_name: str) -> Optional[ServerConfig]
    def resolve_namespace_collision(self, tool_name: str) -> str
    def list_all_tools(self) -> Dict[str, List[Dict]]
    def health_check_all(self) -> Dict[str, Dict]
```

### Multi-Server Client Usage
```python
# Connect to multiple MCP servers
router = create_multi_server_client({
    "local": "http://localhost:8051",           # Your server
    "anthropic": "https://anthropic-mcp.com",   # External server 1
    "openai": "https://openai-mcp.com",         # External server 2
    "partner": "https://partner-tools.com"      # External server 3
})

# Smart routing by server namespace
result = router.execute_tool("local.web_scraper", params)     # Your server
result = router.execute_tool("anthropic.claude_tool", params) # Anthropic's server
result = router.execute_tool("openai.gpt_tool", params)       # OpenAI's server
```

### LLM Tool Visibility
- **Complete visibility** - LLM sees tools from all servers + direct execution
- **Unified interface** - Same API regardless of tool source
- **Automatic routing** - Router handles server selection transparently
- **Graceful degradation** - Failed servers don't break other tools
- **Namespace prefixing** - Prevents collisions, clear tool origins

## Server Architecture

### Shared Codebase Strategy
```python
# Same tool discovery and execution engine
tool_manager = MCPToolManager.get_instance()

# Different protocol wrappers around same engine
class DirectServer:
    def __init__(self):
        self.tool_manager = tool_manager  # Shared!

class JsonRpcServer:
    def __init__(self):
        self.tool_manager = tool_manager  # Same shared engine!
```

### Protocol Adapters Pattern
```
┌─────────────────────────────────────────────────────────────┐
│                MCPToolManager (Shared Core)                │
│  - Tool discovery                                          │
│  - Tool execution                                          │
│  - Namespace management                                    │
│  - Dependency handling                                     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Direct    │  │   Direct    │  │  JSON-RPC   │
    │ Execution   │  │  Protocol   │  │  Protocol   │
    │             │  │  Adapter    │  │  Adapter    │
    │ (No Server) │  │             │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Multi-Environment Deployment
- **Development**: Direct execution (fast iteration, debugging)
- **Internal Production**: Direct server (custom protocol, performance)
- **External/Public**: JSON-RPC server (standards compliance, interoperability)
- **Same tools** available via all deployment modes

## Elegant MCP Tool Management

### Unified MCPRegistry Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    MCPRegistry (Elegant)                   │
├─────────────────────────────────────────────────────────────┤
│  Single Discovery Interface - One way to find all tools    │
│  Unified Execution Model - Same interface for all types    │
│  Consistent Metadata - Same schema for all descriptions    │
│  Transparent Routing - Execution method chosen auto        │
│  Composable Wrappers - Easy to add new tool types         │
└─────────────────────────────────────────────────────────────┘
```

### Tool Management Structure
```
ethos_ai_brain/orchestration/mcp/tool_management/
├── __init__.py                   # Public API exports
├── mcp_registry.py              # Central registry (single interface)
├── tool_discovery.py            # Unified discovery system
├── tool_wrappers.py             # Composable wrapper system
├── tool_execution.py            # Transparent execution routing
└── tool_metadata.py             # Consistent metadata schema
```

### Elegant Single Interface
```python
# Single, elegant interface for everything
mcp = MCPRegistry()

# Discovers everything automatically
tools = mcp.discover_all()  # UV tools, Python functions, APIs, schemas

# Executes anything the same way  
result = mcp.execute("any_tool_name", params)  # Routes automatically

# Wraps anything easily
mcp.wrap_api("github.yaml")           # API wrapper
mcp.wrap_function(my_function)        # Function wrapper  
mcp.wrap_module("pandas")             # Module wrapper
mcp.wrap_uv_tool("./my_tool/")        # UV tool wrapper
```

### Composable Wrapper System
```python
# All wrappers implement same interface
class ToolWrapper(ABC):
    @abstractmethod
    def discover(self) -> List[ToolMetadata]
    @abstractmethod  
    def execute(self, tool_name: str, params: Dict) -> Dict

# Easy to add new wrapper types
class APIWrapper(ToolWrapper): ...      # YAML API configs
class FunctionWrapper(ToolWrapper): ... # Python functions
class ModuleWrapper(ToolWrapper): ...   # Python modules
class UVWrapper(ToolWrapper): ...       # UV-managed tools
```

### Unified Discovery System with Semantic Search
```python
class MCPRegistry:
    def __init__(self, semantic_search=True):
        self.wrappers = [
            UVWrapper(),           # UV-managed tools
            APIWrapper(),          # YAML API configs  
            FunctionWrapper(),     # Python functions
            ModuleWrapper(),       # Python modules
        ]
        self.semantic_search = PineconeIndex() if semantic_search else None
    
    def discover_all(self) -> List[ToolMetadata]:
        """Single method discovers all tool types."""
        tools = []
        for wrapper in self.wrappers:
            discovered = wrapper.discover()
            if self.semantic_search:
                # Auto-generate embeddings for semantic search
                for tool in discovered:
                    tool.embedding = self._generate_embedding(tool.description)
                    self.semantic_search.upsert(tool.name, tool.embedding, tool.metadata)
            tools.extend(discovered)
        return self._deduplicate_and_merge(tools)
    
    def execute(self, tool_name: str, params: Dict) -> Dict:
        """Direct execution by name - fast path."""
        wrapper = self._find_wrapper_for_tool(tool_name)
        return wrapper.execute(tool_name, params)
    
    def find_tools(self, description: str, top_k=10) -> List[str]:
        """Semantic tool discovery - essential for auto-MCP scale."""
        if not self.semantic_search:
            return self._fallback_text_search(description)
        
        query_embedding = self._generate_embedding(description)
        results = self.semantic_search.query(query_embedding, top_k=top_k)
        return [result.id for result in results.matches]
    
    def suggest_tools(self, context: str, task: str = None) -> List[str]:
        """Context-aware tool suggestions for Agent Zero."""
        query = f"Context: {context}. Task: {task}" if task else context
        return self.find_tools(query, top_k=5)
```

## Semantic Tool Discovery

### Auto-MCP Scale Problem
With auto-wrapping of libraries and APIs, tool count explodes:
- **pandas**: ~200+ functions
- **numpy**: ~500+ functions  
- **GitHub API**: ~100+ endpoints
- **OpenAI API**: ~20+ endpoints
- **Stripe API**: ~300+ endpoints
- **Total**: 1000+ tools easily

### Vector Database Options

#### **Pinecone (Recommended)**
- **Pros**: Managed service, excellent performance, existing library integration
- **Cons**: Cost scales with usage, vendor lock-in
- **Use Case**: Production-ready, handles large scale automatically

#### **AWS S3 + Vector Search**
- **Pros**: Potentially cheaper for large datasets, full control
- **Cons**: More infrastructure complexity, need to manage indexing
- **Use Case**: Cost-sensitive deployments, existing AWS infrastructure

#### **Local Vector Store (Development)**
- **Pros**: No external dependencies, free for development
- **Cons**: Limited scale, no persistence across restarts
- **Use Case**: Development and testing

### Semantic Search Architecture
```python
# Agent Zero natural language tool discovery
agent_context = "I need to analyze sales data from a CSV file"
suggested_tools = mcp.suggest_tools(agent_context)
# Returns: pandas.read_csv, pandas.groupby, matplotlib.plot

# Direct semantic search
tools = mcp.find_tools("HTTP request with authentication")
# Returns: requests.post, requests.get, httpx.post, urllib.request
```

## AI Engine Integration

### Global MCP Registry with Semantic Search
- **AI Engine** manages global MCPRegistry singleton with embeddings
- **Agents** use both direct execution and semantic discovery
- **Tool discovery** includes embedding generation at startup
- **Agent Zero** can explore massive tool ecosystems intelligently

### Agent Zero Integration
```python
class AIEngine:
    def __init__(self):
        self.zmq_engine = ZeroMQEngine()
        self.mcp_manager = MCPToolManager.get_instance()  # Singleton
        
class AIAgent:
    def __init__(self, mcp_manager):
        self.brain = AIBrain()
        self.mcp_client = create_client()  # Uses global manager
```

## Benefits Summary

### Development Benefits
- **Fast iteration** - Direct execution for development
- **Easy debugging** - Transparent tool execution
- **Modular architecture** - Clean separation of concerns
- **Testable components** - Strategy pattern enables isolated testing

### Production Benefits
- **Performance options** - Direct server for internal use
- **Standards compliance** - JSON-RPC for external integration
- **Multi-server federation** - Connect to external MCP ecosystems
- **Universal tool access** - Any Python library or API becomes MCP tool

### LLM Integration Benefits
- **Complete tool visibility** - LLM sees all tools from all sources
- **Unified interface** - Same API regardless of tool origin
- **Intelligent routing** - Automatic backend selection
- **Graceful degradation** - Robust fallback handling

## Implementation Priority

### Phase 1: Core Client Architecture
1. ✅ Modular client structure
2. ✅ Strategy pattern implementation
3. ✅ Factory functions
4. ✅ Configuration-driven routing

### Phase 2: Multi-Server Federation
1. Server registry implementation
2. Multi-server client strategy
3. Namespace collision resolution
4. Health monitoring and failover

### Phase 3: Universal Tool Wrapper
1. Python library wrapper
2. OpenAPI schema wrapper
3. Custom function wrapper
4. Enhanced discovery system

### Phase 4: Server Implementation
1. Direct protocol server
2. JSON-RPC protocol server
3. Shared codebase architecture
4. Multi-environment deployment

## Technical Decisions Made

### Architecture Choices
- **Strategy Pattern** over inheritance for execution modes
- **Configuration-driven** routing over hardcoded logic
- **Singleton MCPToolManager** for shared tool discovery
- **Protocol adapters** over monolithic server design

### Naming Conventions
- **MCPClientRouter** (not UnifiedMCPClient) - emphasizes routing capability
- **MCPStrategy** (not MCPExecutionStrategy) - cleaner, shorter
- **create_client()** (not create_auto_client()) - simple default
- **DirectStrategy/JsonRpcStrategy** - clear protocol distinction

### File Organization
- **Dedicated client/ folder** - Isolates complexity from legacy code
- **Descriptive file names** - mcp_client_base, mcp_client_direct, etc.
- **Clean imports** - Public API through __init__.py
- **Modular components** - Each file has single responsibility

## Future Enhancements

### Advanced Routing
- **Load balancing** across multiple servers
- **Circuit breaker** pattern for failed servers
- **Caching** of tool results
- **Rate limiting** per server

### Tool Enhancement
- **Tool versioning** and compatibility checking
- **Tool dependencies** and execution ordering
- **Tool composition** - chaining tools together
- **Tool analytics** - usage tracking and optimization

### Security & Monitoring
- **Authentication** for server connections
- **Authorization** for tool access
- **Audit logging** for tool execution
- **Performance monitoring** and alerting

This architecture provides a comprehensive foundation for a unified MCP system that can scale from development to enterprise production environments while maintaining clean separation of concerns and maximum flexibility.
