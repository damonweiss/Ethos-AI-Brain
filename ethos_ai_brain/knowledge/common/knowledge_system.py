#!/usr/bin/env python3
"""
Knowledge System - Base Interface
Provides a single interface for AI Brain to interact with any knowledge system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class KnowledgeResult:
    """Standardized result format for all knowledge systems"""
    content: str = ""
    knowledge_id: str = ""
    knowledge_type: str = ""  # "graph_node", "layered_graph", etc.
    confidence: float = 0.0
    metadata: Dict = None
    relationships: List[Dict] = None
    context_path: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.relationships is None:
            self.relationships = []
        if self.context_path is None:
            self.context_path = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for AI brain consumption"""
        return {
            "content": self.content,
            "id": self.knowledge_id,
            "type": self.knowledge_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "relationships": self.relationships,
            "context_path": self.context_path
        }


class KnowledgeSystem(ABC):
    """
    Universal interface for all knowledge structures.
    The AI Brain only needs to know this interface.
    """
    
    def __init__(self):
        """Initialize common functionality"""
        from datetime import datetime
        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0",
            "description": "",
            "tags": [],
            "source": "",
            "confidence": 1.0
        }
        self._global_position: Optional[Tuple[float, float, float]] = None
    
    # ========================================
    # Abstract Properties (Must Implement)
    # ========================================
    
    @property
    @abstractmethod
    def knowledge_id(self) -> str:
        """Unique identifier for this knowledge structure"""
        pass
    
    @property
    @abstractmethod
    def knowledge_type(self) -> str:
        """Type of knowledge structure (graph, layered_graph, nebula)"""
        pass
    
    # ========================================
    # Abstract Methods (Must Implement)
    # ========================================
    
    @abstractmethod
    def query(self, query: str, context: Dict = None, **filters) -> List[KnowledgeResult]:
        """Universal query method - works for any knowledge structure"""
        pass
    
    @abstractmethod
    def add_knowledge(self, content: Any, metadata: Dict = None) -> str:
        """Add new knowledge - returns knowledge_id"""
        pass
    
    @abstractmethod
    def get_related(self, knowledge_id: str, relationship_type: str = "semantic") -> List[KnowledgeResult]:
        """Find related knowledge"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """What can this knowledge system do?"""
        pass
    
    @abstractmethod
    def get_components(self) -> List[str]:
        """Get components (nodes for graphs, layers for layered graphs, graphs for nebulae)"""
        pass
    
    @abstractmethod
    def get_relationships(self) -> List[Tuple[str, str, Dict]]:
        """Get relationships (edges, connections, links) with metadata"""
        pass
    
    # ========================================
    # Common Functionality (Implemented)
    # ========================================
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata"""
        return self._metadata.copy()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self._metadata[key] = value
        self._update_modified_time()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update multiple metadata values"""
        self._metadata.update(metadata)
        self._update_modified_time()
    
    def _update_modified_time(self) -> None:
        """Update the last modified timestamp"""
        from datetime import datetime
        self._metadata["last_modified"] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary - base implementation"""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type,
            "metadata": self.get_metadata(),
            "global_position": self._global_position,
            "components": self.get_components(),
            "relationships": self.get_relationships()
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize from dictionary - base implementation"""
        if "metadata" in data:
            self._metadata.update(data["metadata"])
        if "global_position" in data:
            self._global_position = data["global_position"]
    
    # ========================================
    # 3D Positioning (Common to All)
    # ========================================
    
    def set_global_position(self, x: float, y: float, z: float) -> None:
        """Set global position in 3D space"""
        self._global_position = (x, y, z)
        self._update_modified_time()
    
    def get_global_position(self) -> Optional[Tuple[float, float, float]]:
        """Get global position in 3D space"""
        return self._global_position
    
    def get_3d_bounds(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """Get 3D bounding box - base implementation returns global position as point"""
        if self._global_position:
            x, y, z = self._global_position
            return ((x, x), (y, y), (z, z))
        return None
    
    # ========================================
    # Physics & Visualization (Common Interface)
    # ========================================
    
    def supports_physics(self) -> bool:
        """Check if this knowledge structure supports physics simulation"""
        return self.get_capabilities().get("physics_simulation", False)
    
    def supports_3d_positioning(self) -> bool:
        """Check if this knowledge structure supports 3D positioning"""
        return self.get_capabilities().get("3d_positioning", False)
    
    def get_visualization_color(self) -> Optional[str]:
        """Get visualization color - can be overridden"""
        return self._metadata.get("color")
    
    def set_visualization_color(self, color: str) -> None:
        """Set visualization color"""
        self.set_metadata("color", color)


# Adapter classes removed - KnowledgeGraph and LayeredKnowledgeGraph now implement KnowledgeSystem directly


class UnifiedKnowledgeManager:
    """
    Manages multiple knowledge sources with unified interface
    """
    
    def __init__(self):
        self.knowledge_sources: List[KnowledgeSystem] = []
        self.primary_source: Optional[KnowledgeSystem] = None
    
    def add_knowledge_source(self, source: KnowledgeSystem, is_primary: bool = False):
        """Add a knowledge source"""
        self.knowledge_sources.append(source)
        if is_primary or not self.primary_source:
            self.primary_source = source
    
    def query(self, query: str, context: Dict = None, **filters) -> List[KnowledgeResult]:
        """Query all knowledge sources"""
        all_results = []
        
        for source in self.knowledge_sources:
            try:
                results = source.query(query, context, **filters)
                all_results.extend(results)
            except Exception as e:
                print(f"[WARNING] Knowledge source failed: {e}")
        
        return all_results
    
    def add_knowledge(self, content: Any, metadata: Dict = None) -> str:
        """Add knowledge to primary source"""
        if self.primary_source:
            return self.primary_source.add_knowledge(content, metadata)
        return ""
    
    def get_related(self, knowledge_id: str, relationship_type: str = "semantic") -> List[KnowledgeResult]:
        """Get related knowledge from appropriate source"""
        # Find source that created this knowledge_id
        for source in self.knowledge_sources:
            try:
                related = source.get_related(knowledge_id, relationship_type)
                if related:
                    return related
            except Exception as e:
                print(f"[WARNING] Failed to get related from source: {e}")
        
        return []
    
    def get_all_capabilities(self) -> Dict[str, bool]:
        """Get combined capabilities of all sources"""
        combined = {}
        for source in self.knowledge_sources:
            capabilities = source.get_capabilities()
            for capability, available in capabilities.items():
                combined[capability] = combined.get(capability, False) or available
        
        return combined
