"""
Knowledge Graph System - Core knowledge management for Ethos AI Brain

This module provides the foundational knowledge graph capabilities including:
- Core knowledge graph operations (nodes, edges, queries)
- Knowledge graph manager for multi-graph orchestration
- Global registry for cross-graph operations
- Graph type definitions and utilities
"""

from .common.knowledge_graph import KnowledgeGraph, GraphType, KnowledgeGraphRegistry
from .common.layered_knowledge_graph import LayeredKnowledgeGraph
from .common.knowledge_system import (
    KnowledgeSystem, 
    KnowledgeResult, 
    UnifiedKnowledgeManager
)

__all__ = [
    'KnowledgeGraph', 
    'GraphType', 
    'KnowledgeGraphRegistry', 
    'LayeredKnowledgeGraph',
    'KnowledgeSystem',
    'KnowledgeResult',
    'UnifiedKnowledgeManager'
]

__version__ = "0.1.0"
