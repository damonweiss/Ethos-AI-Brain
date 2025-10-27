"""
Test script to verify imports work correctly
"""

import sys
import os

# Add path to Ethos-ZeroMQ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'PycharmProjects', 'Ethos-ZeroMQ'))

print("Testing imports...")

try:
    from ethos_zeromq import ZeroMQEngine
    print("✅ ethos_zeromq imported successfully")
except ImportError as e:
    print(f"❌ ethos_zeromq import failed: {e}")

try:
    from ai_brain import AI_Brain
    print("✅ ai_brain imported successfully")
except ImportError as e:
    print(f"❌ ai_brain import failed: {e}")

try:
    from zmq_node_base import ProcessorType, LLMProcessor
    print("✅ zmq_node_base imported successfully")
except ImportError as e:
    print(f"❌ zmq_node_base import failed: {e}")

try:
    from agent_orchestration_base import AgentOrchestrationBase, OrchestrationPattern
    print("✅ agent_orchestration_base imported successfully")
except ImportError as e:
    print(f"❌ agent_orchestration_base import failed: {e}")

try:
    from orchestration_patterns import SupervisorOrchestrator
    print("✅ orchestration_patterns imported successfully")
except ImportError as e:
    print(f"❌ orchestration_patterns import failed: {e}")

print("\n🎯 All imports tested!")
