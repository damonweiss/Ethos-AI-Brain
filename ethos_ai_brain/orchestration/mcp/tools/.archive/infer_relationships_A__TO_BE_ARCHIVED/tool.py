#!/usr/bin/env python3
"""
Infer Domain Relationships - Meta-Reasoning Linking Tool

Intelligently infers DOMAIN-SPECIFIC relationships between entities using business/intent heuristics.
This tool focuses on semantic, business, and domain-aware relationships.
For mathematical graph relationships, use graph.building.analysis/infer_graph_math_relationships.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

def infer_relationships(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently infer relationships between entities using meta-reasoning heuristics.
    
    Args:
        params: {
            "entities": list - List of entities to analyze for relationships
            "relationship_types": list - Optional focus on specific relationship types
            "confidence_threshold": float - Minimum confidence for including relationships
            "heuristic_strategies": list - Heuristic strategies to apply
        }
    
    Returns:
        {
            "relationships": list - Discovered relationships with confidence scores
            "relationship_count": int - Total relationships found
            "heuristic_analysis": dict - Analysis of heuristics applied
            "inference_metadata": dict - Process metadata
        }
    """
    
    entities = params.get("entities", [])
    focus_types = params.get("relationship_types", [])
    confidence_threshold = params.get("confidence_threshold", 0.6)
    heuristic_strategies = params.get("heuristic_strategies", ["semantic", "structural", "temporal"])
    
    if not entities:
        return {
            "error": "entities parameter is required",
            "relationships": []
        }
    
    try:
        relationships = []
        heuristic_analysis = {}
        
        # Apply semantic heuristics
        if "semantic" in heuristic_strategies:
            semantic_rels, semantic_stats = _apply_semantic_heuristics(entities, confidence_threshold)
            relationships.extend(semantic_rels)
            heuristic_analysis["semantic"] = semantic_stats
        
        # Apply structural heuristics
        if "structural" in heuristic_strategies:
            structural_rels, structural_stats = _apply_structural_heuristics(entities, confidence_threshold)
            relationships.extend(structural_rels)
            heuristic_analysis["structural"] = structural_stats
        
        # Apply hierarchical collection inference (NEW - addresses Problem 2)
        hierarchical_rels, hierarchical_stats = _apply_hierarchical_collection_inference(entities, confidence_threshold)
        relationships.extend(hierarchical_rels)
        heuristic_analysis["hierarchical_collections"] = hierarchical_stats
        
        # Apply temporal heuristics
        if "temporal" in heuristic_strategies:
            temporal_rels, temporal_stats = _apply_temporal_heuristics(entities, confidence_threshold)
            relationships.extend(temporal_rels)
            heuristic_analysis["temporal"] = temporal_stats
        
        # Filter by focus types if specified
        if focus_types:
            relationships = [r for r in relationships if r.get("type") in focus_types]
        
        # Sort by confidence
        relationships.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "relationships": relationships,
            "relationship_count": len(relationships),
            "heuristic_analysis": heuristic_analysis,
            "inference_metadata": {
                "inference_timestamp": datetime.now().isoformat(),
                "entities_analyzed": len(entities),
                "strategies_applied": heuristic_strategies,
                "confidence_threshold": confidence_threshold,
                "inference_tool": "meta_reasoning.linking/infer_relationships"
            }
        }
        
    except Exception as e:
        return {
            "error": f"Relationship inference failed: {str(e)}",
            "relationships": []
        }


def _apply_semantic_heuristics(entities: List[Dict], threshold: float) -> tuple[List[Dict], Dict]:
    """Apply semantic-based relationship inference."""
    relationships = []
    stats = {"patterns_matched": 0, "relationships_created": 0}
    
    # Enhanced semantic patterns for business/technical terminology
    semantic_patterns = {
        "business_synonyms": {
            "patterns": [
                (["goal", "objective", "target", "intent", "purpose"], "equivalent_concept"),
                (["problem", "issue", "challenge", "constraint", "limitation"], "equivalent_concept"),
                (["solution", "answer", "resolution", "strategy", "approach"], "equivalent_concept"),
                (["requirement", "need", "specification", "criteria"], "equivalent_concept"),
                (["stakeholder", "team", "group", "participant"], "equivalent_concept")
            ],
            "confidence": 0.85
        },
        "hierarchical": {
            "patterns": [
                (["system", "component", "module", "service"], "contains"),
                (["process", "step", "phase", "stage"], "contains"),
                (["category", "item", "element", "dimension"], "contains"),
                (["analysis", "result", "finding", "outcome"], "contains"),
                (["strategy", "tactic", "method", "approach"], "contains")
            ],
            "confidence": 0.8
        },
        "constraint_patterns": {
            "patterns": [
                (["constraint", "requirement", "limit", "restriction"], "constrains"),
                (["budget", "cost", "price", "expense"], "constrains"),
                (["time", "deadline", "schedule", "duration"], "constrains"),
                (["security", "compliance", "standard", "policy"], "constrains")
            ],
            "confidence": 0.9
        },
        "dependency_patterns": {
            "patterns": [
                (["requirement", "dependency", "prerequisite"], "requires"),
                (["stakeholder", "team", "resource"], "involves"),
                (["metric", "score", "measurement", "kpi"], "measures"),
                (["dimension", "aspect", "factor"], "influences")
            ],
            "confidence": 0.8
        }
    }
    
    # Extract content from entities
    entity_content = []
    for entity in entities:
        content = str(entity.get("content", "") or entity.get("label", "") or entity.get("value", "")).lower()
        if content:
            entity_content.append((entity.get("id"), content, entity))
    
    # Apply semantic patterns
    for pattern_type, pattern_data in semantic_patterns.items():
        for pattern_words, relationship_type in pattern_data["patterns"]:
            for i, (id1, content1, entity1) in enumerate(entity_content):
                for j, (id2, content2, entity2) in enumerate(entity_content):
                    if i != j:
                        matches1 = [word for word in pattern_words if word in content1]
                        matches2 = [word for word in pattern_words if word in content2]
                        
                        if matches1 and matches2:
                            confidence = pattern_data["confidence"]
                            if confidence >= threshold:
                                relationships.append({
                                    "source_id": id1,
                                    "target_id": id2,
                                    "type": relationship_type,
                                    "confidence": confidence,
                                    "evidence": f"Semantic pattern: {matches1} -> {matches2}",
                                    "heuristic": f"semantic_{pattern_type}"
                                })
                                stats["relationships_created"] += 1
                            stats["patterns_matched"] += 1
    
    return relationships, stats


def _apply_hierarchical_collection_inference(entities: List[Dict], threshold: float) -> tuple[List[Dict], Dict]:
    """Apply hierarchical collection inference - if LLM says 5 constraints, create 5 constraint relationships."""
    relationships = []
    stats = {"collections_analyzed": 0, "relationships_created": 0, "parent_child_pairs": 0}
    
    # Group entities by path hierarchy
    path_groups = defaultdict(list)
    for entity in entities:
        path = entity.get("path", "")
        if path:
            path_groups[path].append(entity)
    
    # Create parent-child relationships based on path hierarchy
    for entity in entities:
        entity_path = entity.get("path", "")
        entity_type = entity.get("entity_type", "")
        
        # Find potential children (entities with this entity's path as prefix)
        for other_entity in entities:
            other_path = other_entity.get("path", "")
            other_type = other_entity.get("entity_type", "")
            
            if (entity.get("id") != other_entity.get("id") and 
                other_path.startswith(entity_path) and 
                len(other_path) > len(entity_path)):
                
                # Determine relationship type based on entity types and content
                relationship_type = "contains"
                confidence = 0.8
                
                # Enhanced relationship typing based on content
                entity_key = entity.get("key", "").lower()
                other_key = other_entity.get("key", "").lower()
                
                if "constraint" in entity_key or "constraint" in other_key:
                    relationship_type = "constrains"
                    confidence = 0.9
                elif "requirement" in entity_key or "requirement" in other_key:
                    relationship_type = "requires"
                    confidence = 0.9
                elif "stakeholder" in entity_key or "team" in other_key:
                    relationship_type = "involves"
                    confidence = 0.85
                elif entity_type == "collection" and other_type in ["attribute", "metric", "label"]:
                    relationship_type = "contains"
                    confidence = 0.9
                elif "dimension" in entity_key:
                    relationship_type = "influences"
                    confidence = 0.8
                
                if confidence >= threshold:
                    relationships.append({
                        "source_id": entity.get("id"),
                        "target_id": other_entity.get("id"),
                        "type": relationship_type,
                        "confidence": confidence,
                        "evidence": f"Hierarchical path: {entity_path} -> {other_path}",
                        "heuristic": "hierarchical_collection"
                    })
                    stats["relationships_created"] += 1
                    stats["parent_child_pairs"] += 1
    
    # Create collection-item relationships for arrays/lists
    for entity in entities:
        if entity.get("entity_type") == "collection":
            entity_value = entity.get("value", [])
            entity_key = entity.get("key", "").lower()
            
            if isinstance(entity_value, list) and len(entity_value) > 0:
                stats["collections_analyzed"] += 1
                
                # Find entities that represent items in this collection
                for other_entity in entities:
                    other_value = str(other_entity.get("value", "")).lower()
                    other_key = other_entity.get("key", "").lower()
                    
                    # Check if this entity's value appears in the collection
                    if (entity.get("id") != other_entity.get("id") and
                        any(str(item).lower() in other_value or other_value in str(item).lower() 
                            for item in entity_value)):
                        
                        # Determine specific relationship type
                        relationship_type = "contains"
                        confidence = 0.85
                        
                        if "constraint" in entity_key:
                            relationship_type = "constrains"
                            confidence = 0.9
                        elif "requirement" in entity_key:
                            relationship_type = "requires"
                            confidence = 0.9
                        elif "dimension" in entity_key:
                            relationship_type = "influences"
                            confidence = 0.8
                        elif "stakeholder" in entity_key:
                            relationship_type = "involves"
                            confidence = 0.85
                        
                        if confidence >= threshold:
                            relationships.append({
                                "source_id": entity.get("id"),
                                "target_id": other_entity.get("id"),
                                "type": relationship_type,
                                "confidence": confidence,
                                "evidence": f"Collection membership: {entity_key} contains {other_key}",
                                "heuristic": "collection_membership"
                            })
                            stats["relationships_created"] += 1
    
    return relationships, stats


def _apply_structural_heuristics(entities: List[Dict], threshold: float) -> tuple[List[Dict], Dict]:
    """Apply structure-based relationship inference."""
    relationships = []
    stats = {"type_rules_applied": 0, "relationships_created": 0}
    
    # Structural rules based on entity types
    structural_rules = {
        ("intent", "constraint"): ("constrained_by", 0.9),
        ("intent", "requirement"): ("requires", 0.85),
        ("requirement", "solution"): ("addressed_by", 0.8),
        ("problem", "solution"): ("solved_by", 0.85),
        ("goal", "strategy"): ("achieved_through", 0.8)
    }
    
    # Apply structural rules
    for entity1 in entities:
        for entity2 in entities:
            if entity1.get("id") != entity2.get("id"):
                type1 = entity1.get("type", "unknown")
                type2 = entity2.get("type", "unknown")
                
                if (type1, type2) in structural_rules:
                    relationship_type, confidence = structural_rules[(type1, type2)]
                    stats["type_rules_applied"] += 1
                    
                    if confidence >= threshold:
                        relationships.append({
                            "source_id": entity1.get("id"),
                            "target_id": entity2.get("id"),
                            "type": relationship_type,
                            "confidence": confidence,
                            "evidence": f"Structural rule: {type1} {relationship_type} {type2}",
                            "heuristic": "structural_type_based"
                        })
                        stats["relationships_created"] += 1
    
    return relationships, stats


def _apply_temporal_heuristics(entities: List[Dict], threshold: float) -> tuple[List[Dict], Dict]:
    """Apply temporal-based relationship inference."""
    relationships = []
    stats = {"temporal_patterns": 0, "relationships_created": 0}
    
    # Temporal indicators
    temporal_patterns = {
        "sequence": {
            "before_words": ["before", "prior", "previous", "first"],
            "after_words": ["after", "following", "next", "then"],
            "relationship": "precedes",
            "confidence": 0.8
        },
        "causation": {
            "cause_words": ["causes", "triggers", "leads to"],
            "effect_words": ["results", "produces", "creates"],
            "relationship": "causes",
            "confidence": 0.85
        }
    }
    
    # Extract text content
    entity_text = []
    for entity in entities:
        text = str(entity.get("content", "") or entity.get("label", "")).lower()
        if text:
            entity_text.append((entity.get("id"), text, entity))
    
    # Apply temporal patterns
    for pattern_name, pattern_data in temporal_patterns.items():
        for i, (id1, text1, entity1) in enumerate(entity_text):
            for j, (id2, text2, entity2) in enumerate(entity_text):
                if i != j:
                    temporal_score = 0
                    evidence = []
                    
                    if pattern_name == "sequence":
                        before_matches = [w for w in pattern_data["before_words"] if w in text1]
                        after_matches = [w for w in pattern_data["after_words"] if w in text2]
                        
                        if before_matches and after_matches:
                            temporal_score = pattern_data["confidence"]
                            evidence.append(f"Sequence: {before_matches} -> {after_matches}")
                    
                    elif pattern_name == "causation":
                        cause_matches = [w for w in pattern_data["cause_words"] if w in text1]
                        effect_matches = [w for w in pattern_data["effect_words"] if w in text2]
                        
                        if cause_matches and effect_matches:
                            temporal_score = pattern_data["confidence"]
                            evidence.append(f"Causation: {cause_matches} -> {effect_matches}")
                    
                    if temporal_score >= threshold:
                        relationships.append({
                            "source_id": id1,
                            "target_id": id2,
                            "type": pattern_data["relationship"],
                            "confidence": temporal_score,
                            "evidence": "; ".join(evidence),
                            "heuristic": f"temporal_{pattern_name}"
                        })
                        stats["relationships_created"] += 1
                    
                    if evidence:
                        stats["temporal_patterns"] += 1
    
    return relationships, stats


# MCP Tool Registry
TOOLS = [
    {
        "name": "infer_relationships",
        "namespace": "meta_reasoning.linking",
        "category": "relationship_inference",
        "description": "Intelligently infer relationships between entities using meta-reasoning heuristics",
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
                },
                "heuristic_strategies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Heuristic strategies to apply",
                    "default": ["semantic", "structural", "temporal"]
                }
            },
            "required": ["entities"]
        }
    }
]
