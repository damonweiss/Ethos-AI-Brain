#!/usr/bin/env python3
"""
Debug NetworkX to see if there's a method conflict
"""

import networkx as nx

def test_networkx():
    print("Testing basic NetworkX functionality...")
    
    # Create a simple graph
    g = nx.DiGraph()
    print(f"Created DiGraph: {type(g)}")
    
    # Test has_node on empty graph
    print(f"has_node('test') on empty graph: {g.has_node('test')}")
    
    # Add a node
    g.add_node('test', name='Test Node')
    print(f"Added node 'test'")
    
    # Test has_node on graph with node
    print(f"has_node('test') after adding: {g.has_node('test')}")
    
    # Test getting node attributes
    attrs = dict(g.nodes['test'])
    print(f"Node attributes: {attrs}")
    
    print("NetworkX basic functionality works!")

if __name__ == "__main__":
    test_networkx()
