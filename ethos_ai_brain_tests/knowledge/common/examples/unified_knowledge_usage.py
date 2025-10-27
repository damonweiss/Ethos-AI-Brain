#!/usr/bin/env python3
"""
Example: Using Unified Knowledge Interface with AI Brain
Shows how the AI Brain can work with any knowledge system transparently
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.knowledge import (
    KnowledgeGraph,
    GraphType,
    KnowledgeGraphAdapter,
    LayeredKnowledgeGraphAdapter,
    UnifiedKnowledgeManager
)


class SimpleAIBrain:
    """Simple AI Brain that uses unified knowledge interface"""
    
    def __init__(self):
        self.knowledge = UnifiedKnowledgeManager()
    
    def add_knowledge_source(self, source, is_primary=False):
        """Add a knowledge source to the brain"""
        self.knowledge.add_knowledge_source(source, is_primary)
        print(f"[SUCCESS] Added knowledge source with capabilities: {source.get_capabilities()}")
    
    def learn(self, content, metadata=None):
        """Learn new information"""
        knowledge_id = self.knowledge.add_knowledge(content, metadata)
        print(f"[SUCCESS] Learned: {content} (ID: {knowledge_id})")
        return knowledge_id
    
    def think_about(self, topic):
        """Think about a topic using all available knowledge"""
        print(f"\n[THINKING] About: {topic}")
        
        # Query all knowledge sources
        results = self.knowledge.query(topic)
        print(f"Found {len(results)} relevant pieces of knowledge:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.content}")
            print(f"     Type: {result.knowledge_type}, Confidence: {result.confidence}")
            print(f"     Path: {' -> '.join(result.context_path)}")
        
        # Get related concepts for the first result
        if results:
            related = self.knowledge.get_related(results[0].knowledge_id)
            if related:
                print(f"\nRelated concepts:")
                for rel in related:
                    print(f"  - {rel.content}")
        
        return {
            "topic": topic,
            "knowledge_found": len(results),
            "primary_results": [r.to_dict() for r in results],
            "capabilities": self.knowledge.get_all_capabilities()
        }


def main():
    """Demonstrate unified knowledge interface usage"""
    
    print("=== AI Brain with Unified Knowledge Interface ===\n")
    
    # Create AI Brain
    brain = SimpleAIBrain()
    
    # === Setup Basic Knowledge Graph ===
    print("1. Setting up basic knowledge graph...")
    basic_kg = KnowledgeGraph("programming_basics", GraphType.INTENT)
    basic_kg.add_node("python", type="language", difficulty="beginner")
    basic_kg.add_node("variables", type="concept", difficulty="beginner")
    basic_kg.add_node("functions", type="concept", difficulty="intermediate")
    basic_kg.add_edge("python", "variables", relationship="uses")
    basic_kg.add_edge("python", "functions", relationship="uses")
    
    basic_adapter = KnowledgeGraphAdapter(basic_kg)
    brain.add_knowledge_source(basic_adapter, is_primary=True)
    
    # === Setup Layered Knowledge Graph ===
    print("\n2. Setting up layered knowledge graph...")
    
    class MockLayeredGraph:
        def __init__(self):
            self.network_id = "ml_concepts"
            self.graphs = {
                "foundations": KnowledgeGraph("foundations", GraphType.INTENT),
                "algorithms": KnowledgeGraph("algorithms", GraphType.EXECUTION),
                "applications": KnowledgeGraph("applications", GraphType.LOCAL_RAG)
            }
            
            # Add ML concepts across layers
            self.graphs["foundations"].add_node("statistics", type="math", importance="critical")
            self.graphs["foundations"].add_node("linear_algebra", type="math", importance="critical")
            
            self.graphs["algorithms"].add_node("neural_networks", type="algorithm", complexity="high")
            self.graphs["algorithms"].add_node("decision_trees", type="algorithm", complexity="medium")
            
            self.graphs["applications"].add_node("image_recognition", type="application", domain="computer_vision")
            self.graphs["applications"].add_node("nlp", type="application", domain="language")
    
    layered_kg = MockLayeredGraph()
    layered_adapter = LayeredKnowledgeGraphAdapter(layered_kg)
    brain.add_knowledge_source(layered_adapter)
    
    # === AI Brain Learning ===
    print("\n3. AI Brain learning new concepts...")
    brain.learn("machine_learning", {"type": "field", "complexity": "high"})
    brain.learn("data_science", {"type": "field", "complexity": "high"})
    
    # === AI Brain Thinking ===
    print("\n4. AI Brain thinking about topics...")
    
    # Think about programming
    result1 = brain.think_about("python")
    
    # Think about statistics (should find in layered graph)
    result2 = brain.think_about("statistics")
    
    # Think about neural networks
    result3 = brain.think_about("neural")
    
    # === Show Combined Capabilities ===
    print(f"\n5. AI Brain's combined knowledge capabilities:")
    capabilities = brain.knowledge.get_all_capabilities()
    for capability, available in capabilities.items():
        status = "[YES]" if available else "[NO]"
        print(f"   {status} {capability}")
    
    print(f"\nTotal knowledge sources: {len(brain.knowledge.knowledge_sources)}")
    print(f"Primary source type: {type(brain.knowledge.primary_source).__name__}")
    
    print("\n=== Summary ===")
    print("The AI Brain can now work with:")
    print("- Basic Knowledge Graphs (nodes + edges)")
    print("- Layered Knowledge Graphs (multiple interconnected layers)")
    print("- Future: Knowledge Nebulae (graphs as nodes in meta-graph)")
    print("- All through the same unified interface!")
    print("\n[SUCCESS] Unified Knowledge Interface demonstration complete!")


if __name__ == "__main__":
    main()
