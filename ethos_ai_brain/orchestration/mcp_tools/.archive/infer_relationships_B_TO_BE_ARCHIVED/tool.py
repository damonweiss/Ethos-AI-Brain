#!/usr/bin/env python3
"""
Infer Relationships - Graph Building Tool

Intelligently discovers meaningful relationships between entities using heuristic analysis.
Supports hierarchical, temporal, dependency, and semantic relationship detection.
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

def infer_relationships(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer meaningful relationships between entities for graph construction.
    
    Args:
        params: {
            "entities": list - List of entities to analyze for relationships
            "relationship_types": list - Optional focus on specific relationship types
            "confidence_threshold": float - Minimum confidence for including relationships
        }
    
    Returns:
        {
            "relationships": list - Discovered relationships with confidence scores
            "relationship_count": int - Total relationships found
            "relationship_types_found": list - Types of relationships discovered
            "inference_metadata": dict - Analysis information
        }
    """
    
    entities = params.get("entities", [])
    focus_types = params.get("relationship_types", [])
    confidence_threshold = params.get("confidence_threshold", 0.6)
    
    if not entities:
        return {
            "error": "entities parameter is required",
            "relationships": []
        }
    
    relationships = []
    relationship_types_found = set()
    
    # Analyze all entity pairs for relationships
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:  # Don't relate entity to itself
                relationship = _analyze_entity_pair(entity1, entity2)
                if relationship and relationship["confidence"] >= confidence_threshold:
                    relationships.append(relationship)
                    relationship_types_found.add(relationship["type"])
    
    # Filter by focus types if specified
    if focus_types:
        relationships = [r for r in relationships if r["type"] in focus_types]
    
    # Sort by confidence
    relationships.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "relationships": relationships,
        "relationship_count": len(relationships),
        "relationship_types_found": list(relationship_types_found),
        "inference_metadata": {
            "inference_timestamp": datetime.now().isoformat(),
            "entities_analyzed": len(entities),
            "pairs_evaluated": len(entities) * (len(entities) - 1),
            "confidence_threshold": confidence_threshold,
            "inference_tool": "graph.analysis/infer_relationships"
        }
    }


def _analyze_entity_pair(entity1: Dict, entity2: Dict) -> Optional[Dict]:
    """Analyze a pair of entities for potential relationships."""
    
    relationships = []
    
    # Check for different types of relationships
    relationships.extend(_check_hierarchical_relationship(entity1, entity2))
    relationships.extend(_check_temporal_relationship(entity1, entity2))
    relationships.extend(_check_dependency_relationship(entity1, entity2))
    relationships.extend(_check_semantic_relationship(entity1, entity2))
    relationships.extend(_check_structural_relationship(entity1, entity2))
    
    # Return the highest confidence relationship
    if relationships:
        best_relationship = max(relationships, key=lambda x: x["confidence"])
        return best_relationship
    
    return None


def _check_hierarchical_relationship(entity1: Dict, entity2: Dict) -> List[Dict]:
    """Check for parent-child or containment relationships."""
    relationships = []
    
    path1 = entity1.get("path", "")
    path2 = entity2.get("path", "")
    
    # Check if one path is a parent of another
    if path1 and path2:
        if path2.startswith(path1 + ".") and path1 != path2:
            relationships.append({
                "from": entity1["id"],
                "to": entity2["id"],
                "type": "contains",
                "confidence": 0.9,
                "evidence": f"Path hierarchy: {path1} contains {path2}"
            })
        elif path1.startswith(path2 + ".") and path1 != path2:
            relationships.append({
                "from": entity2["id"],
                "to": entity1["id"],
                "type": "contains",
                "confidence": 0.9,
                "evidence": f"Path hierarchy: {path2} contains {path1}"
            })
    
    return relationships


def _check_temporal_relationship(entity1: Dict, entity2: Dict) -> List[Dict]:
    """Check for time-based relationships."""
    relationships = []
    
    # Look for temporal keywords
    temporal_indicators = {
        "before": ["before", "prior", "previous", "earlier"],
        "after": ["after", "following", "next", "later"],
        "triggers": ["triggers", "causes", "initiates", "starts"]
    }
    
    key1 = entity1.get("key", "").lower()
    key2 = entity2.get("key", "").lower()
    
    for rel_type, indicators in temporal_indicators.items():
        if any(indicator in key1 for indicator in indicators):
            relationships.append({
                "from": entity1["id"],
                "to": entity2["id"],
                "type": rel_type,
                "confidence": 0.7,
                "evidence": f"Temporal indicator in key: {key1}"
            })
    
    return relationships


def _check_dependency_relationship(entity1: Dict, entity2: Dict) -> List[Dict]:
    """Check for dependency relationships."""
    relationships = []
    
    # Look for dependency patterns
    key1 = entity1.get("key", "").lower()
    key2 = entity2.get("key", "").lower()
    
    dependency_patterns = {
        "depends_on": ["depends", "requires", "needs"],
        "enables": ["enables", "allows", "supports"],
        "influences": ["influences", "affects", "impacts"]
    }
    
    for rel_type, patterns in dependency_patterns.items():
        if any(pattern in key1 for pattern in patterns):
            relationships.append({
                "from": entity1["id"],
                "to": entity2["id"],
                "type": rel_type,
                "confidence": 0.8,
                "evidence": f"Dependency pattern in key: {key1}"
            })
    
    return relationships


def _check_semantic_relationship(entity1: Dict, entity2: Dict) -> List[Dict]:
    """Check for semantic relationships based on content similarity."""
    relationships = []
    
    semantic1 = entity1.get("semantic_type", "")
    semantic2 = entity2.get("semantic_type", "")
    
    # Related semantic types
    semantic_relationships = {
        ("intention", "approach"): ("informs", 0.8),
        ("assessment", "approach"): ("guides", 0.7),
        ("capability", "outcome"): ("produces", 0.8),
        ("analysis_aspect", "assessment"): ("contributes_to", 0.7)
    }
    
    for (sem1, sem2), (rel_type, confidence) in semantic_relationships.items():
        if (semantic1 == sem1 and semantic2 == sem2):
            relationships.append({
                "from": entity1["id"],
                "to": entity2["id"],
                "type": rel_type,
                "confidence": confidence,
                "evidence": f"Semantic relationship: {sem1} {rel_type} {sem2}"
            })
        elif (semantic1 == sem2 and semantic2 == sem1):
            relationships.append({
                "from": entity2["id"],
                "to": entity1["id"],
                "type": rel_type,
                "confidence": confidence,
                "evidence": f"Semantic relationship: {sem2} {rel_type} {sem1}"
            })
    
    return relationships


def _check_structural_relationship(entity1: Dict, entity2: Dict) -> List[Dict]:
    """Check for structural relationships based on data types and patterns."""
    relationships = []
    
    type1 = entity1.get("entity_type", "")
    type2 = entity2.get("entity_type", "")
    
    # Structural patterns
    if type1 == "collection" and type2 == "attribute":
        relationships.append({
            "from": entity1["id"],
            "to": entity2["id"],
            "type": "aggregates",
            "confidence": 0.6,
            "evidence": f"Collection aggregates attribute"
        })
    
    if type1 == "identifier" and type2 == "label":
        relationships.append({
            "from": entity1["id"],
            "to": entity2["id"],
            "type": "identifies",
            "confidence": 0.8,
            "evidence": f"Identifier identifies label"
        })
    
    return relationships


# MCP Tool Registry
TOOLS = [
    {
        "name": "infer_relationships",
        "namespace": "graph.analysis",
        "category": "relationship_inference",
        "description": "Intelligently discover meaningful relationships between entities using heuristic analysis",
        "function": infer_relationships,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of entities to analyze for relationships"
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional focus on specific relationship types"
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence for including relationships",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["entities"]
        }
    }
]
