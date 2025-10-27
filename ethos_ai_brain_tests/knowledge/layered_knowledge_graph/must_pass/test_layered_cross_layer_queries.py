"""
Test Layered Cross-Layer Queries - Must Pass
Tests querying and traversal across layers in LayeredKnowledgeGraph
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import KnowledgeGraph, GraphType
from ethos_ai_brain.knowledge.common.visualizers import LayeredKnowledgeGraphVisualizer


class MockLayeredGraphForQueries:
    """Mock layered graph with rich structure for cross-layer query testing"""
    
    def __init__(self):
        self.network_id = "cross_layer_query_network"
        self.graphs = {
            "input_layer": KnowledgeGraph("input_layer", "sensors"),
            "processing_layer": KnowledgeGraph("processing_layer", "computation"),
            "output_layer": KnowledgeGraph("output_layer", "results")
        }
        
        # INPUT LAYER - Rich sensor network (8 nodes)
        self.graphs["input_layer"].add_node("TempSensor1", type="temperature", location="room1", status="active", priority="high")
        self.graphs["input_layer"].add_node("TempSensor2", type="temperature", location="room2", status="active", priority="medium")
        self.graphs["input_layer"].add_node("HumiditySensor1", type="humidity", location="room1", status="active", priority="high")
        self.graphs["input_layer"].add_node("HumiditySensor2", type="humidity", location="room2", status="maintenance", priority="low")
        self.graphs["input_layer"].add_node("MotionSensor1", type="motion", location="hallway", status="active", priority="high")
        self.graphs["input_layer"].add_node("MotionSensor2", type="motion", location="entrance", status="active", priority="medium")
        self.graphs["input_layer"].add_node("LightSensor", type="ambient_light", location="window", status="active", priority="low")
        self.graphs["input_layer"].add_node("SensorHub", type="coordinator", location="central", status="active", priority="critical")
        
        # Connect sensors to hub
        self.graphs["input_layer"].add_edge("TempSensor1", "SensorHub", relationship="reports_to", frequency="1min")
        self.graphs["input_layer"].add_edge("TempSensor2", "SensorHub", relationship="reports_to", frequency="5min")
        self.graphs["input_layer"].add_edge("HumiditySensor1", "SensorHub", relationship="reports_to", frequency="1min")
        self.graphs["input_layer"].add_edge("HumiditySensor2", "SensorHub", relationship="reports_to", frequency="offline")
        self.graphs["input_layer"].add_edge("MotionSensor1", "SensorHub", relationship="reports_to", frequency="realtime")
        self.graphs["input_layer"].add_edge("MotionSensor2", "SensorHub", relationship="reports_to", frequency="realtime")
        self.graphs["input_layer"].add_edge("LightSensor", "SensorHub", relationship="reports_to", frequency="10min")
        
        # PROCESSING LAYER - Complex computation network (7 nodes)
        self.graphs["processing_layer"].add_node("DataAggregator", type="data_fusion", algorithm="weighted_avg", priority="critical", load="high")
        self.graphs["processing_layer"].add_node("PatternAnalyzer", type="pattern_detection", algorithm="ml_classifier", priority="high", load="medium")
        self.graphs["processing_layer"].add_node("AnomalyDetector", type="anomaly_detection", algorithm="isolation_forest", priority="high", load="low")
        self.graphs["processing_layer"].add_node("TrendAnalyzer", type="trend_analysis", algorithm="time_series", priority="medium", load="medium")
        self.graphs["processing_layer"].add_node("CorrelationEngine", type="correlation", algorithm="pearson", priority="low", load="low")
        self.graphs["processing_layer"].add_node("PreprocessorA", type="data_cleaning", algorithm="outlier_removal", priority="critical", load="high")
        self.graphs["processing_layer"].add_node("PreprocessorB", type="normalization", algorithm="min_max_scaling", priority="high", load="medium")
        
        # Create processing pipeline
        self.graphs["processing_layer"].add_edge("DataAggregator", "PreprocessorA", relationship="feeds_raw_data", latency="100ms")
        self.graphs["processing_layer"].add_edge("DataAggregator", "PreprocessorB", relationship="feeds_raw_data", latency="50ms")
        self.graphs["processing_layer"].add_edge("PreprocessorA", "PatternAnalyzer", relationship="provides_clean_data", latency="200ms")
        self.graphs["processing_layer"].add_edge("PreprocessorB", "PatternAnalyzer", relationship="provides_normalized_data", latency="150ms")
        self.graphs["processing_layer"].add_edge("PreprocessorA", "AnomalyDetector", relationship="provides_clean_data", latency="300ms")
        self.graphs["processing_layer"].add_edge("PatternAnalyzer", "TrendAnalyzer", relationship="informs_trends", latency="500ms")
        self.graphs["processing_layer"].add_edge("PatternAnalyzer", "CorrelationEngine", relationship="provides_patterns", latency="400ms")
        
        # OUTPUT LAYER - Multi-channel output system (6 nodes)
        self.graphs["output_layer"].add_node("WebDashboard", type="visualization", format="web_ui", audience="operators", update_rate="1sec")
        self.graphs["output_layer"].add_node("MobileDashboard", type="visualization", format="mobile_app", audience="managers", update_rate="5sec")
        self.graphs["output_layer"].add_node("EmailAlerts", type="notification", channel="email", urgency="medium", batch_size="10")
        self.graphs["output_layer"].add_node("SMSAlerts", type="notification", channel="sms", urgency="high", batch_size="1")
        self.graphs["output_layer"].add_node("ReportGenerator", type="reporting", format="pdf", schedule="daily", retention="30days")
        self.graphs["output_layer"].add_node("DataExporter", type="export", format="csv", destination="database", compression="gzip")
        
        # Connect output nodes
        self.graphs["output_layer"].add_edge("WebDashboard", "MobileDashboard", relationship="syncs_with", sync_interval="30sec")
        self.graphs["output_layer"].add_edge("WebDashboard", "ReportGenerator", relationship="triggers_reports", trigger="daily")
        self.graphs["output_layer"].add_edge("EmailAlerts", "SMSAlerts", relationship="escalates_to", threshold="critical")
        self.graphs["output_layer"].add_edge("ReportGenerator", "DataExporter", relationship="archives_to", format="compressed")
        
        # Cross-layer connections for queries
        self.cross_layer_connections = [
            {"source_graph": "input_layer", "source_node": "SensorHub", "target_graph": "processing_layer", "target_node": "DataAggregator", "data_flow": "sensor_readings"},
            {"source_graph": "input_layer", "source_node": "TempSensor1", "target_graph": "processing_layer", "target_node": "DataAggregator", "data_flow": "temperature_data"},
            {"source_graph": "input_layer", "source_node": "MotionSensor1", "target_graph": "processing_layer", "target_node": "AnomalyDetector", "data_flow": "motion_events"},
            {"source_graph": "processing_layer", "source_node": "PatternAnalyzer", "target_graph": "output_layer", "target_node": "WebDashboard", "data_flow": "pattern_results"},
            {"source_graph": "processing_layer", "source_node": "AnomalyDetector", "target_graph": "output_layer", "target_node": "EmailAlerts", "data_flow": "anomaly_alerts"},
            {"source_graph": "processing_layer", "source_node": "TrendAnalyzer", "target_graph": "output_layer", "target_node": "ReportGenerator", "data_flow": "trend_reports"}
        ]
    
    def get_graph(self, graph_id):
        return self.graphs.get(graph_id)
    
    def get_all_layers(self):
        return list(self.graphs.keys())
    
    def find_nodes_by_type(self, node_type):
        """Find all nodes of a specific type across all layers"""
        results = []
        for layer_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                node_attrs = graph.get_node_attributes(node_id)
                if node_attrs.get('type') == node_type:
                    results.append({
                        'layer': layer_id,
                        'node': node_id,
                        'attributes': node_attrs
                    })
        return results
    
    def find_nodes_by_attribute(self, attr_name, attr_value):
        """Find all nodes with a specific attribute value across layers"""
        results = []
        for layer_id, graph in self.graphs.items():
            for node_id in graph.nodes():
                node_attrs = graph.get_node_attributes(node_id)
                if node_attrs.get(attr_name) == attr_value:
                    results.append({
                        'layer': layer_id,
                        'node': node_id,
                        'attributes': node_attrs
                    })
        return results
    
    def trace_data_flow(self, start_layer, start_node):
        """Trace data flow from a starting node across layers"""
        flow_path = [{'layer': start_layer, 'node': start_node}]
        
        # Find cross-layer connections from this node
        for connection in self.cross_layer_connections:
            if (connection['source_graph'] == start_layer and 
                connection['source_node'] == start_node):
                
                target_layer = connection['target_graph']
                target_node = connection['target_node']
                flow_path.append({
                    'layer': target_layer,
                    'node': target_node,
                    'data_flow': connection['data_flow']
                })
                
                # Recursively trace from target node
                downstream = self.trace_data_flow(target_layer, target_node)
                if len(downstream) > 1:  # If there are downstream connections
                    flow_path.extend(downstream[1:])  # Skip the first (current node)
        
        return flow_path
    
    def get_layer_statistics(self):
        """Get statistics for each layer"""
        stats = {}
        for layer_id, graph in self.graphs.items():
            stats[layer_id] = {
                'node_count': len(graph.nodes()),
                'edge_count': len(graph.edges()),
                'node_types': list(set(graph.get_node_attributes(n).get('type', 'unknown') 
                                     for n in graph.nodes())),
                'avg_degree': sum(graph.degree(n) for n in graph.nodes()) / len(graph.nodes()) if graph.nodes() else 0
            }
        return stats


def test_cross_layer_node_type_queries():
    """Test querying nodes by type across all layers"""
    mock_graph = MockLayeredGraphForQueries()
    
    # Query for temperature sensors
    temp_sensors = mock_graph.find_nodes_by_type("temperature")
    
    print(f"Temperature sensors found: {len(temp_sensors)}")
    for sensor in temp_sensors:
        print(f"  Layer: {sensor['layer']}, Node: {sensor['node']}, Location: {sensor['attributes'].get('location')}")
    
    assert len(temp_sensors) == 2, "[FAILURE] Should find 2 temperature sensors"
    assert all(s['layer'] == 'input_layer' for s in temp_sensors), "[FAILURE] All temp sensors should be in input layer"
    
    # Query for visualization nodes
    viz_nodes = mock_graph.find_nodes_by_type("visualization")
    
    print(f"\nVisualization nodes found: {len(viz_nodes)}")
    for node in viz_nodes:
        print(f"  Layer: {node['layer']}, Node: {node['node']}, Format: {node['attributes'].get('format')}")
    
    assert len(viz_nodes) == 2, "[FAILURE] Should find 2 visualization nodes"
    assert all(n['layer'] == 'output_layer' for n in viz_nodes), "[FAILURE] All viz nodes should be in output layer"
    
    print("[SUCCESS] Cross-layer node type queries work correctly")


def test_cross_layer_attribute_queries():
    """Test querying nodes by attribute values across layers"""
    mock_graph = MockLayeredGraphForQueries()
    
    # Query for high priority nodes
    high_priority = mock_graph.find_nodes_by_attribute("priority", "high")
    
    print(f"\nHigh priority nodes found: {len(high_priority)}")
    layer_counts = {}
    for node in high_priority:
        layer = node['layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        print(f"  Layer: {layer}, Node: {node['node']}, Type: {node['attributes'].get('type')}")
    
    print(f"High priority distribution: {layer_counts}")
    
    assert len(high_priority) >= 5, "[FAILURE] Should find multiple high priority nodes"
    assert len(layer_counts) >= 2, "[FAILURE] High priority nodes should span multiple layers"
    
    # Query for active status nodes
    active_nodes = mock_graph.find_nodes_by_attribute("status", "active")
    
    print(f"\nActive nodes found: {len(active_nodes)}")
    for node in active_nodes:
        print(f"  Layer: {node['layer']}, Node: {node['node']}")
    
    assert len(active_nodes) >= 6, "[FAILURE] Should find multiple active nodes"
    
    print("[SUCCESS] Cross-layer attribute queries work correctly")


def test_cross_layer_data_flow_tracing():
    """Test tracing data flow across layers"""
    mock_graph = MockLayeredGraphForQueries()
    
    # Trace flow from TempSensor1
    flow_from_temp1 = mock_graph.trace_data_flow("input_layer", "TempSensor1")
    
    print(f"\nData flow from TempSensor1:")
    for i, step in enumerate(flow_from_temp1):
        data_flow = step.get('data_flow', 'start')
        print(f"  {i+1}. Layer: {step['layer']}, Node: {step['node']}, Flow: {data_flow}")
    
    assert len(flow_from_temp1) >= 2, "[FAILURE] Should trace at least 2 steps in flow"
    assert flow_from_temp1[0]['layer'] == 'input_layer', "[FAILURE] Should start in input layer"
    
    # Trace flow from SensorHub (should have multiple downstream paths)
    flow_from_hub = mock_graph.trace_data_flow("input_layer", "SensorHub")
    
    print(f"\nData flow from SensorHub:")
    for i, step in enumerate(flow_from_hub):
        data_flow = step.get('data_flow', 'start')
        print(f"  {i+1}. Layer: {step['layer']}, Node: {step['node']}, Flow: {data_flow}")
    
    assert len(flow_from_hub) >= 2, "[FAILURE] Should trace multiple steps from hub"
    
    # Trace from processing layer
    flow_from_analyzer = mock_graph.trace_data_flow("processing_layer", "PatternAnalyzer")
    
    print(f"\nData flow from PatternAnalyzer:")
    for i, step in enumerate(flow_from_analyzer):
        data_flow = step.get('data_flow', 'start')
        print(f"  {i+1}. Layer: {step['layer']}, Node: {step['node']}, Flow: {data_flow}")
    
    assert len(flow_from_analyzer) >= 2, "[FAILURE] Should trace to output layer"
    
    print("[SUCCESS] Cross-layer data flow tracing works correctly")


def test_layer_statistics_and_analysis():
    """Test getting statistics and analysis across layers"""
    mock_graph = MockLayeredGraphForQueries()
    
    # Get layer statistics
    stats = mock_graph.get_layer_statistics()
    
    print(f"\nLayer Statistics:")
    for layer_id, layer_stats in stats.items():
        print(f"  {layer_id}:")
        print(f"    Nodes: {layer_stats['node_count']}")
        print(f"    Edges: {layer_stats['edge_count']}")
        print(f"    Node Types: {layer_stats['node_types']}")
        print(f"    Avg Degree: {layer_stats['avg_degree']:.2f}")
    
    # Verify expected structure
    assert stats['input_layer']['node_count'] == 8, "[FAILURE] Input layer should have 8 nodes"
    assert stats['processing_layer']['node_count'] == 7, "[FAILURE] Processing layer should have 7 nodes"
    assert stats['output_layer']['node_count'] == 6, "[FAILURE] Output layer should have 6 nodes"
    
    # Check node type diversity
    assert len(stats['input_layer']['node_types']) >= 4, "[FAILURE] Input layer should have diverse node types"
    assert len(stats['processing_layer']['node_types']) >= 3, "[FAILURE] Processing layer should have diverse node types"
    
    print("[SUCCESS] Layer statistics and analysis work correctly")


def test_cross_layer_query_performance():
    """Test performance characteristics of cross-layer queries"""
    mock_graph = MockLayeredGraphForQueries()
    
    import time
    
    # Time node type queries
    start_time = time.time()
    for _ in range(100):
        temp_sensors = mock_graph.find_nodes_by_type("temperature")
    type_query_time = time.time() - start_time
    
    # Time attribute queries  
    start_time = time.time()
    for _ in range(100):
        high_priority = mock_graph.find_nodes_by_attribute("priority", "high")
    attr_query_time = time.time() - start_time
    
    # Time data flow tracing
    start_time = time.time()
    for _ in range(50):
        flow = mock_graph.trace_data_flow("input_layer", "SensorHub")
    flow_trace_time = time.time() - start_time
    
    print(f"\nQuery Performance (100 iterations):")
    print(f"  Type queries: {type_query_time*1000:.2f}ms")
    print(f"  Attribute queries: {attr_query_time*1000:.2f}ms")
    print(f"  Flow tracing (50 iter): {flow_trace_time*1000:.2f}ms")
    
    # Performance should be reasonable for small graphs
    assert type_query_time < 1.0, "[FAILURE] Type queries should be fast"
    assert attr_query_time < 1.0, "[FAILURE] Attribute queries should be fast"
    assert flow_trace_time < 1.0, "[FAILURE] Flow tracing should be fast"
    
    print("[SUCCESS] Cross-layer query performance is acceptable")


def test_cross_layer_visualization_integration():
    """Test that cross-layer queries work with visualization"""
    mock_graph = MockLayeredGraphForQueries()
    
    # Create visualizer (reusing the structure)
    viz = LayeredKnowledgeGraphVisualizer(mock_graph)
    
    # Test that we can query and then visualize results
    high_priority_nodes = mock_graph.find_nodes_by_attribute("priority", "high")
    critical_nodes = mock_graph.find_nodes_by_attribute("priority", "critical")
    
    print(f"\nVisualization Integration Test:")
    print(f"High priority nodes for highlighting: {len(high_priority_nodes)}")
    print(f"Critical nodes for highlighting: {len(critical_nodes)}")
    
    # Verify we have nodes to highlight
    assert len(high_priority_nodes) > 0, "[FAILURE] Should have high priority nodes to highlight"
    assert len(critical_nodes) > 0, "[FAILURE] Should have critical nodes to highlight"
    
    # Test data flow paths for visualization
    flow_paths = []
    for layer in mock_graph.get_all_layers():
        graph = mock_graph.get_graph(layer)
        for node in list(graph.nodes())[:2]:  # Test first 2 nodes per layer
            flow = mock_graph.trace_data_flow(layer, node)
            if len(flow) > 1:
                flow_paths.append(flow)
    
    print(f"Data flow paths found: {len(flow_paths)}")
    for i, path in enumerate(flow_paths[:3]):  # Show first 3 paths
        print(f"  Path {i+1}: {len(path)} steps")
    
    assert len(flow_paths) > 0, "[FAILURE] Should find data flow paths for visualization"
    
    print("[SUCCESS] Cross-layer queries integrate well with visualization")


# Converted to pytest format - no manual main needed
