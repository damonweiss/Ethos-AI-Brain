#!/usr/bin/env python3
"""
Simple ZMQ Test - Just ReqRep
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the Ethos-ZeroMQ path
sys.path.append(r'C:\Users\DamonWeiss\PycharmProjects\Ethos-ZeroMQ')

def test_simple_reqrep():
    """Test basic ReqRep with your ZMQ library"""
    try:
        from ethos_zeromq import ZeroMQEngine
        
        print("=" * 60)
        print("🔍 DETAILED ZMQ VALIDATION TEST")
        print("=" * 60)
        
        logger.info("✅ ZMQ Import Success!")
        print(f"📦 Imported ZeroMQEngine from: {ZeroMQEngine.__module__}")
        
        # Create ZMQ engine
        print("\n🏗️  Creating ZMQ Engine...")
        zmq_engine = ZeroMQEngine()
        logger.info("✅ ZMQ Engine Created!")
        
        # Inspect the engine
        print(f"🔍 Engine type: {type(zmq_engine)}")
        print(f"🔍 Engine name: {getattr(zmq_engine, 'name', 'No name attribute')}")
        print(f"🔍 Engine attributes: {[attr for attr in dir(zmq_engine) if not attr.startswith('_')]}")
        
        # Check if ServerManager exists
        if hasattr(zmq_engine, 'sm'):
            print(f"🔍 ServerManager: {type(zmq_engine.sm)}")
            print(f"🔍 ServerManager attributes: {[attr for attr in dir(zmq_engine.sm) if not attr.startswith('_')]}")
        
        # Create simple ReqRep server
        print("\n🚀 Creating ReqRep Server...")
        server = zmq_engine.create_and_start_server('reqrep', 'test_server')
        logger.info("✅ ReqRep Server Created!")
        
        # Detailed server inspection
        if server:
            print(f"✅ Server object: {type(server)}")
            print(f"🔍 Server attributes: {[attr for attr in dir(server) if not attr.startswith('_')]}")
            
            # Check if server has expected methods
            expected_methods = ['start', 'stop', 'send', 'receive', 'close']
            for method in expected_methods:
                has_method = hasattr(server, method)
                print(f"🔍 Has {method}(): {'✅' if has_method else '❌'}")
            
            # Check server state
            if hasattr(server, 'is_running'):
                print(f"🔍 Server running: {server.is_running}")
            if hasattr(server, 'port'):
                print(f"🔍 Server port: {server.port}")
            if hasattr(server, 'address'):
                print(f"🔍 Server address: {server.address}")
                
            print("🎉 Simple ZMQ Test PASSED!")
            return True
        else:
            print("❌ Server creation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Simple ZMQ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_reqrep()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
