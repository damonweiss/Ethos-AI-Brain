"""
ZMQ Node Base Classes
Provides container abstraction for different processing units in ZMQ network
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Import your existing components
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

from ethos_zeromq import *


class ProcessorType(Enum):
    """Types of processors that can be contained in ZMQ nodes"""
    LLM = "llm"  # Large Language Model
    NEURAL_NETWORK = "neural_net"  # Custom neural networks
    MCP_TOOL = "mcp_tool"  # Model Context Protocol tools
    USER_INPUT = "user_input"  # Human input interfaces
    API_SERVICE = "api_service"  # External API calls
    DATABASE = "database"  # Database operations
    FILE_SYSTEM = "file_system"  # File operations
    CUSTOM_LOGIC = "custom_logic"  # Custom Python logic
    HYBRID = "hybrid"  # Multiple processors combined


@dataclass
class ProcessorCapability:
    """Defines what a processor can do"""
    name: str
    processor_type: ProcessorType
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    description: str
    cost_estimate: float = 0.0  # Processing cost/time estimate
    reliability: float = 1.0  # Success rate estimate


@dataclass
class ProcessingRequest:
    """Standard request format for processors"""
    request_id: str
    processor_type: ProcessorType
    operation: str
    input_data: Dict[str, Any]
    context: Dict[str, Any] = None
    timeout: float = 30.0
    priority: int = 1  # 1=high, 5=low


@dataclass
class ProcessingResponse:
    """Standard response format from processors"""
    request_id: str
    processor_type: ProcessorType
    status: str  # "success", "error", "timeout"
    output_data: Dict[str, Any]
    error_message: str = None
    processing_time: float = 0.0
    cost_consumed: float = 0.0


class ProcessorInterface(ABC):
    """
    Abstract interface that all processors must implement
    This allows LLMs, neural networks, MCP tools, etc. to be used interchangeably
    """

    def __init__(self, processor_id: str, processor_type: ProcessorType):
        self.processor_id = processor_id
        self.processor_type = processor_type
        self.capabilities: List[ProcessorCapability] = []
        self.is_initialized = False
        self.is_busy = False

    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the processor with configuration"""
        pass

    @abstractmethod
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a request and return response"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Cleanup and shutdown the processor"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[ProcessorCapability]:
        """Return list of processor capabilities"""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current processor status"""
        return {
            "processor_id": self.processor_id,
            "processor_type": self.processor_type.value,
            "is_initialized": self.is_initialized,
            "is_busy": self.is_busy,
            "capabilities_count": len(self.capabilities)
        }


class LLMProcessor(ProcessorInterface):
    """LLM processor implementation"""

    def __init__(self, processor_id: str, model_name: str = "gpt-4"):
        super().__init__(processor_id, ProcessorType.LLM)
        self.model_name = model_name

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize LLM processor"""
        try:
            self.capabilities = [
                ProcessorCapability(
                    name="text_generation",
                    processor_type=ProcessorType.LLM,
                    input_schema={"prompt": "str", "context": "dict"},
                    output_schema={"text": "str", "reasoning": "dict"},
                    description=f"Text generation using {self.model_name}",
                    cost_estimate=0.02,  # per request
                    reliability=0.95
                ),
                ProcessorCapability(
                    name="analysis",
                    processor_type=ProcessorType.LLM,
                    input_schema={"data": "any", "analysis_type": "str"},
                    output_schema={"analysis": "dict", "insights": "list"},
                    description="Intelligent data analysis",
                    cost_estimate=0.03,
                    reliability=0.90
                )
            ]
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            return False

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process LLM request"""
        if not self.is_initialized:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message="LLM not initialized"
            )

        self.is_busy = True
        start_time = asyncio.get_event_loop().time()

        try:
            if request.operation == "text_generation":
                prompt = request.input_data.get("prompt", "")
                context = request.input_data.get("context", {})

                result = await self.brain.think(prompt, context)

                return ProcessingResponse(
                    request_id=request.request_id,
                    processor_type=self.processor_type,
                    status="success",
                    output_data={
                        "text": result.get("response", ""),
                        "reasoning": result
                    },
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    cost_consumed=0.02
                )

            elif request.operation == "analysis":
                data = request.input_data.get("data")
                analysis_type = request.input_data.get("analysis_type", "general")

                analysis = await self.brain.think(
                    f"Perform {analysis_type} analysis on: {data}",
                    context=request.context or {}
                )

                return ProcessingResponse(
                    request_id=request.request_id,
                    processor_type=self.processor_type,
                    status="success",
                    output_data={
                        "analysis": analysis,
                        "insights": analysis.get("key_insights", [])
                    },
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    cost_consumed=0.03
                )

            else:
                return ProcessingResponse(
                    request_id=request.request_id,
                    processor_type=self.processor_type,
                    status="error",
                    output_data={},
                    error_message=f"Unknown operation: {request.operation}"
                )

        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )

        finally:
            self.is_busy = False

    async def shutdown(self) -> bool:
        """Shutdown LLM processor"""
        self.is_initialized = False
        self.brain = None
        return True

    def get_capabilities(self) -> List[ProcessorCapability]:
        return self.capabilities


class MCPToolProcessor(ProcessorInterface):
    """MCP Tool processor implementation"""

    def __init__(self, processor_id: str, tool_name: str):
        super().__init__(processor_id, ProcessorType.MCP_TOOL)
        self.tool_name = tool_name
        self.mcp_client = None

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize MCP tool connection"""
        try:
            # Initialize MCP client (placeholder - would use actual MCP client)
            self.mcp_client = f"mcp_client_{self.tool_name}"

            self.capabilities = [
                ProcessorCapability(
                    name=f"{self.tool_name}_operation",
                    processor_type=ProcessorType.MCP_TOOL,
                    input_schema={"operation": "str", "parameters": "dict"},
                    output_schema={"result": "any", "metadata": "dict"},
                    description=f"MCP tool operations for {self.tool_name}",
                    cost_estimate=0.01,
                    reliability=0.98
                )
            ]
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ MCP tool initialization failed: {e}")
            return False

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process MCP tool request"""
        if not self.is_initialized:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message="MCP tool not initialized"
            )

        self.is_busy = True
        start_time = asyncio.get_event_loop().time()

        try:
            # Simulate MCP tool operation
            operation = request.input_data.get("operation", "default")
            parameters = request.input_data.get("parameters", {})

            # Placeholder for actual MCP tool call
            result = {
                "tool": self.tool_name,
                "operation": operation,
                "parameters": parameters,
                "result": f"MCP {self.tool_name} executed {operation}",
                "success": True
            }

            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="success",
                output_data={
                    "result": result,
                    "metadata": {"tool": self.tool_name, "operation": operation}
                },
                processing_time=asyncio.get_event_loop().time() - start_time,
                cost_consumed=0.01
            )

        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )

        finally:
            self.is_busy = False

    async def shutdown(self) -> bool:
        """Shutdown MCP tool processor"""
        self.is_initialized = False
        self.mcp_client = None
        return True

    def get_capabilities(self) -> List[ProcessorCapability]:
        return self.capabilities


class UserInputProcessor(ProcessorInterface):
    """User input processor implementation"""

    def __init__(self, processor_id: str, input_method: str = "terminal"):
        super().__init__(processor_id, ProcessorType.USER_INPUT)
        self.input_method = input_method
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize user input interface"""
        try:
            self.capabilities = [
                ProcessorCapability(
                    name="user_query",
                    processor_type=ProcessorType.USER_INPUT,
                    input_schema={"question": "str", "options": "list"},
                    output_schema={"response": "str", "metadata": "dict"},
                    description=f"User input via {self.input_method}",
                    cost_estimate=0.0,  # No computational cost
                    reliability=1.0  # User always provides input (eventually)
                )
            ]
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ User input initialization failed: {e}")
            return False

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process user input request"""
        if not self.is_initialized:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message="User input processor not initialized"
            )

        self.is_busy = True
        start_time = asyncio.get_event_loop().time()

        try:
            question = request.input_data.get("question", "Please provide input:")
            options = request.input_data.get("options", [])

            # Display question to user
            print(f"\n❓ {question}")
            if options:
                for i, option in enumerate(options, 1):
                    print(f"   {i}. {option}")

            # Get user input (in real implementation, this would be async)
            if self.input_method == "terminal":
                user_response = input("Your response: ")
            else:
                # Placeholder for other input methods (GUI, web, etc.)
                user_response = "simulated_user_input"

            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="success",
                output_data={
                    "response": user_response,
                    "metadata": {
                        "input_method": self.input_method,
                        "question": question,
                        "options": options
                    }
                },
                processing_time=asyncio.get_event_loop().time() - start_time,
                cost_consumed=0.0
            )

        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )

        finally:
            self.is_busy = False

    async def shutdown(self) -> bool:
        """Shutdown user input processor"""
        self.is_initialized = False
        return True

    def get_capabilities(self) -> List[ProcessorCapability]:
        return self.capabilities


class CustomLogicProcessor(ProcessorInterface):
    """Custom logic processor for arbitrary Python functions"""

    def __init__(self, processor_id: str, logic_function: Callable):
        super().__init__(processor_id, ProcessorType.CUSTOM_LOGIC)
        self.logic_function = logic_function
        self.function_name = getattr(logic_function, '__name__', 'custom_logic')

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize custom logic processor"""
        try:
            self.capabilities = [
                ProcessorCapability(
                    name=self.function_name,
                    processor_type=ProcessorType.CUSTOM_LOGIC,
                    input_schema={"args": "list", "kwargs": "dict"},
                    output_schema={"result": "any"},
                    description=f"Custom logic: {self.function_name}",
                    cost_estimate=0.001,
                    reliability=0.99
                )
            ]
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ Custom logic initialization failed: {e}")
            return False

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process custom logic request"""
        if not self.is_initialized:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message="Custom logic processor not initialized"
            )

        self.is_busy = True
        start_time = asyncio.get_event_loop().time()

        try:
            args = request.input_data.get("args", [])
            kwargs = request.input_data.get("kwargs", {})

            # Execute custom function
            if asyncio.iscoroutinefunction(self.logic_function):
                result = await self.logic_function(*args, **kwargs)
            else:
                result = self.logic_function(*args, **kwargs)

            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="success",
                output_data={"result": result},
                processing_time=asyncio.get_event_loop().time() - start_time,
                cost_consumed=0.001
            )

        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=self.processor_type,
                status="error",
                output_data={},
                error_message=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )

        finally:
            self.is_busy = False

    async def shutdown(self) -> bool:
        """Shutdown custom logic processor"""
        self.is_initialized = False
        return True

    def get_capabilities(self) -> List[ProcessorCapability]:
        return self.capabilities


class ZMQNode:
    """
    Concrete ZMQ node that can contain different types of processors
    This is the container that holds LLMs, neural networks, MCP tools, etc.
    """

    def __init__(
            self,
            node_id: str,
            processors: List[ProcessorInterface] = None,
            base_port: int = 5555,
            shared_zmq_engine: Any = None
    ):
        self.node_id = node_id
        self.processors: Dict[str, ProcessorInterface] = {}
        self.base_port = base_port

        # ZMQ Infrastructure - use shared engine if provided
        if shared_zmq_engine:
            self.zmq_engine = shared_zmq_engine
            print(f"🔗 Node {node_id} using shared ZMQ engine")
        else:
            self.zmq_engine = ZeroMQEngine(name=f"Node_{node_id}")
            print(f"🆕 Node {node_id} created own ZMQ engine")

        self.zmq_servers: Dict[str, Any] = {}
        self.is_running = False

        # Add processors
        if processors:
            for processor in processors:
                self.add_processor(processor)

        # Message routing
        self.request_handlers: Dict[str, Callable] = {}
        self._setup_default_handlers()

    def add_processor(self, processor: ProcessorInterface):
        """Add a processor to this node"""
        self.processors[processor.processor_id] = processor
        print(f"🔧 Added {processor.processor_type.value} processor: {processor.processor_id}")

    def remove_processor(self, processor_id: str):
        """Remove a processor from this node"""
        if processor_id in self.processors:
            del self.processors[processor_id]
            print(f"🗑️ Removed processor: {processor_id}")

    def get_processor(self, processor_id: str) -> Optional[ProcessorInterface]:
        """Get a specific processor"""
        return self.processors.get(processor_id)

    def get_processors_by_type(self, processor_type: ProcessorType) -> List[ProcessorInterface]:
        """Get all processors of a specific type"""
        return [p for p in self.processors.values() if p.processor_type == processor_type]

    async def initialize_node(self):
        """Initialize all processors and ZMQ infrastructure"""
        print(f"🚀 Initializing ZMQ Node: {self.node_id}")

        # Initialize all processors
        for processor_id, processor in self.processors.items():
            success = await processor.initialize()
            if success:
                print(f"  ✅ {processor.processor_type.value}: {processor_id}")
            else:
                print(f"  ❌ {processor.processor_type.value}: {processor_id}")

        # Initialize ZMQ infrastructure
        await self._initialize_zmq()

        self.is_running = True
        print(f"✅ Node {self.node_id} initialized with {len(self.processors)} processors")

    async def shutdown_node(self):
        """Shutdown all processors and ZMQ infrastructure"""
        print(f"🛑 Shutting down ZMQ Node: {self.node_id}")

        self.is_running = False

        # Shutdown all processors
        for processor_id, processor in self.processors.items():
            await processor.shutdown()
            print(f"  🔄 Shutdown {processor.processor_type.value}: {processor_id}")

        # Shutdown ZMQ infrastructure
        self.zmq_engine.stop_all_servers()

        print(f"✅ Node {self.node_id} shutdown complete")

    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Route request to appropriate processor"""
        # Find suitable processor
        suitable_processors = [
            p for p in self.processors.values()
            if p.processor_type == request.processor_type and not p.is_busy
        ]

        if not suitable_processors:
            return ProcessingResponse(
                request_id=request.request_id,
                processor_type=request.processor_type,
                status="error",
                output_data={},
                error_message=f"No available {request.processor_type.value} processor"
            )

        # Use first available processor (could implement load balancing here)
        processor = suitable_processors[0]

        return await processor.process(request)

    def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive node status"""
        processor_status = {}
        for proc_id, processor in self.processors.items():
            processor_status[proc_id] = processor.get_status()

        return {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "processor_count": len(self.processors),
            "processors": processor_status,
            "capabilities": self.get_all_capabilities()
        }

    def get_all_capabilities(self) -> List[ProcessorCapability]:
        """Get all capabilities from all processors"""
        all_capabilities = []
        for processor in self.processors.values():
            all_capabilities.extend(processor.get_capabilities())
        return all_capabilities

    async def _initialize_zmq(self):
        """Initialize basic ZMQ infrastructure for node"""
        # Create a simple REQ-REP server for processing requests
        try:
            self.req_rep_server = self.zmq_engine.create_and_start_server(
                "reqrep", f"{self.node_id}_processor"
            )

            if self.req_rep_server:
                # Register basic handlers
                self.req_rep_server.register_handler("process_request", self._handle_zmq_process_request)
                self.req_rep_server.register_handler("status_query", self._handle_zmq_status_query)
                print(f"✅ ZMQ node {self.node_id} initialized")
            else:
                print(f"⚠️ ZMQ server creation failed for {self.node_id}")

        except Exception as e:
            print(f"❌ ZMQ initialization failed for {self.node_id}: {e}")

    async def _handle_zmq_process_request(self, message_data):
        """Handle ZMQ processing requests"""
        try:
            import json
            # Convert ZMQ message to ProcessingRequest
            request = ProcessingRequest(**json.loads(message_data))
            response = await self.process_request(request)
            return json.dumps(response.__dict__)
        except Exception as e:
            print(f"❌ ZMQ processing error in {self.node_id}: {e}")
            return json.dumps({
                "status": "error",
                "error_message": str(e)
            })

    async def _handle_zmq_status_query(self, message_data):
        """Handle ZMQ status queries"""
        import json
        return json.dumps(self.get_node_status())

    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.request_handlers.update({
            "process_request": self._handle_process_request,
            "capabilities_query": self._handle_capabilities_query
        })

    async def _handle_process_request(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle processing requests"""
        try:
            request = ProcessingRequest(**message_data)
            response = await self.process_request(request)
            return response.__dict__
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Request handling failed: {str(e)}"
            }

    async def _handle_status_query(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status queries"""
        return self.get_node_status()

    async def _handle_capabilities_query(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capabilities queries"""
        return {
            "node_id": self.node_id,
            "capabilities": [cap.__dict__ for cap in self.get_all_capabilities()]
        }


# Factory functions for creating common processor types
def create_llm_processor(processor_id: str, model_name: str = "gpt-4") -> LLMProcessor:
    """Create an LLM processor"""
    return LLMProcessor(processor_id, model_name)


def create_mcp_processor(processor_id: str, tool_name: str) -> MCPToolProcessor:
    """Create an MCP tool processor"""
    return MCPToolProcessor(processor_id, tool_name)


def create_user_input_processor(processor_id: str, input_method: str = "terminal") -> UserInputProcessor:
    """Create a user input processor"""
    return UserInputProcessor(processor_id, input_method)


def create_custom_processor(processor_id: str, logic_function: Callable) -> CustomLogicProcessor:
    """Create a custom logic processor"""
    return CustomLogicProcessor(processor_id, logic_function)
