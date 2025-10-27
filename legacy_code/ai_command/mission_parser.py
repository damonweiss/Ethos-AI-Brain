"""
Mission Parser for AI Command System
Handles parsing and validation of mission objectives
"""

import uuid
import logging
import networkx as nx
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class MissionPriority(Enum):
    LOW = "low"
    NORMAL = "normal" 
    HIGH = "high"
    CRITICAL = "critical"

class MissionStatus(Enum):
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Mission:
    """Mission data structure"""
    id: str
    objective: str
    priority: MissionPriority
    context: Dict[str, Any] = field(default_factory=dict)
    status: MissionStatus = MissionStatus.RECEIVED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_agents: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, new_status: MissionStatus):
        """Update mission status with timestamp"""
        self.status = new_status
        self.updated_at = datetime.now()
        logger.info(f"Mission {self.id} status updated to {new_status.value}")
    
    def assign_agent(self, agent_id: str):
        """Assign an agent to this mission"""
        if agent_id not in self.assigned_agents:
            self.assigned_agents.append(agent_id)
            logger.info(f"Agent {agent_id} assigned to mission {self.id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mission to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'objective': self.objective,
            'priority': self.priority.value,
            'context': self.context,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'assigned_agents': self.assigned_agents,
            'results': self.results
        }

class MissionParser:
    """Parse and validate mission objectives"""
    
    def __init__(self):
        self.mission_keywords = {
            'analyze': MissionPriority.NORMAL,
            'urgent': MissionPriority.HIGH,
            'critical': MissionPriority.CRITICAL,
            'emergency': MissionPriority.CRITICAL,
            'routine': MissionPriority.LOW,
            'optimize': MissionPriority.NORMAL,
            'investigate': MissionPriority.NORMAL,
            'monitor': MissionPriority.LOW
        }
        
        # NetworkX Intelligence Graphs
        self.mission_relationship_graph = nx.DiGraph()        # Mission dependencies and relationships
        self.priority_dependency_graph = nx.DiGraph()         # Priority cascading relationships
        self.resource_conflict_graph = nx.Graph()             # Resource competition analysis
        self.mission_similarity_graph = nx.Graph()            # Mission similarity network
        
        # Initialize mission intelligence graphs
        self._initialize_mission_graphs()
    
    def parse_mission(self, objective: str, context: Optional[Dict[str, Any]] = None) -> Mission:
        """Parse mission objective and create Mission object"""
        try:
            # Generate unique mission ID
            mission_id = f"mission_{uuid.uuid4().hex[:8]}"
            
            # Determine priority from objective text
            priority = self._determine_priority(objective)
            
            # Clean and validate objective
            cleaned_objective = self._clean_objective(objective)
            
            # Create mission object
            mission = Mission(
                id=mission_id,
                objective=cleaned_objective,
                priority=priority,
                context=context or {}
            )
            
            logger.info(f"Parsed mission {mission_id}: {cleaned_objective} (Priority: {priority.value})")
            return mission
            
        except Exception as e:
            logger.error(f"Failed to parse mission: {e}")
            raise ValueError(f"Invalid mission objective: {e}")
    
    def _determine_priority(self, objective: str) -> MissionPriority:
        """Determine mission priority based on keywords"""
        objective_lower = objective.lower()
        
        # Check for priority keywords
        for keyword, priority in self.mission_keywords.items():
            if keyword in objective_lower:
                return priority
        
        # Default priority
        return MissionPriority.NORMAL
    
    def _clean_objective(self, objective: str) -> str:
        """Clean and validate mission objective"""
        if not objective or not objective.strip():
            raise ValueError("Mission objective cannot be empty")
        
        # Clean whitespace and ensure reasonable length
        cleaned = objective.strip()
        
        if len(cleaned) < 10:
            raise ValueError("Mission objective too short (minimum 10 characters)")
        
        if len(cleaned) > 1000:
            raise ValueError("Mission objective too long (maximum 1000 characters)")
        
        return cleaned
    
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate mission context data"""
        if not isinstance(context, dict):
            return False
        
        # Check for required context fields if any
        # Add validation logic as needed
        
        return True
    
    def _initialize_mission_graphs(self):
        """Initialize NetworkX graphs for mission intelligence"""
        try:
            # Priority dependency relationships (how priorities affect each other)
            priority_relationships = [
                (MissionPriority.CRITICAL, MissionPriority.HIGH, {'impact': 'blocks', 'weight': 1.0}),
                (MissionPriority.HIGH, MissionPriority.NORMAL, {'impact': 'delays', 'weight': 0.7}),
                (MissionPriority.NORMAL, MissionPriority.LOW, {'impact': 'postpones', 'weight': 0.3}),
                (MissionPriority.CRITICAL, MissionPriority.NORMAL, {'impact': 'overrides', 'weight': 0.9}),
                (MissionPriority.CRITICAL, MissionPriority.LOW, {'impact': 'cancels', 'weight': 1.0})
            ]
            
            for source, target, attributes in priority_relationships:
                self.priority_dependency_graph.add_edge(source.value, target.value, **attributes)
            
            # Resource conflict patterns (missions that compete for same resources)
            resource_conflicts = [
                ('analysis', 'investigation', {'conflict_type': 'data_access', 'severity': 0.6}),
                ('optimization', 'monitoring', {'conflict_type': 'system_resources', 'severity': 0.4}),
                ('critical_response', 'routine_maintenance', {'conflict_type': 'personnel', 'severity': 0.9}),
                ('documentation', 'implementation', {'conflict_type': 'expert_time', 'severity': 0.3})
            ]
            
            for mission_type1, mission_type2, attributes in resource_conflicts:
                self.resource_conflict_graph.add_edge(mission_type1, mission_type2, **attributes)
            
            logger.info("Mission intelligence graphs initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize mission graphs: {e}")
    
    def add_mission_to_graphs(self, mission: Mission):
        """Add mission to NetworkX graphs for relationship analysis"""
        try:
            mission_type = self._classify_mission_type(mission.objective)
            
            # Add to mission relationship graph
            self.mission_relationship_graph.add_node(
                mission.id,
                mission_type=mission_type,
                priority=mission.priority.value,
                created_at=mission.created_at.isoformat(),
                objective_keywords=self._extract_keywords(mission.objective)
            )
            
            # Find similar missions for similarity graph
            similar_missions = self._find_similar_missions(mission)
            for similar_mission_id, similarity_score in similar_missions:
                self.mission_similarity_graph.add_edge(
                    mission.id,
                    similar_mission_id,
                    similarity=similarity_score,
                    relationship_type='similar_objective'
                )
            
            logger.info(f"Mission {mission.id} added to intelligence graphs")
            
        except Exception as e:
            logger.error(f"Failed to add mission to graphs: {e}")
    
    def _classify_mission_type(self, objective: str) -> str:
        """Classify mission into type categories"""
        objective_lower = objective.lower()
        
        if any(word in objective_lower for word in ['analyze', 'analysis', 'study', 'examine']):
            return 'analysis'
        elif any(word in objective_lower for word in ['investigate', 'research', 'explore']):
            return 'investigation'
        elif any(word in objective_lower for word in ['optimize', 'improve', 'enhance']):
            return 'optimization'
        elif any(word in objective_lower for word in ['monitor', 'track', 'observe']):
            return 'monitoring'
        elif any(word in objective_lower for word in ['create', 'build', 'develop', 'implement']):
            return 'implementation'
        elif any(word in objective_lower for word in ['document', 'record', 'report']):
            return 'documentation'
        elif any(word in objective_lower for word in ['critical', 'emergency', 'urgent']):
            return 'critical_response'
        else:
            return 'general'
    
    def _extract_keywords(self, objective: str) -> List[str]:
        """Extract key terms from mission objective"""
        # Simple keyword extraction (could be enhanced with NLP)
        words = objective.lower().split()
        keywords = []
        
        # Filter out common words and keep meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if len(clean_word) > 3 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _find_similar_missions(self, mission: Mission) -> List[tuple]:
        """Find missions with similar objectives using graph analysis"""
        similar_missions = []
        mission_keywords = set(self._extract_keywords(mission.objective))
        
        for node_id in self.mission_relationship_graph.nodes():
            if node_id != mission.id:
                node_data = self.mission_relationship_graph.nodes[node_id]
                other_keywords = set(node_data.get('objective_keywords', []))
                
                # Calculate similarity based on keyword overlap
                if mission_keywords and other_keywords:
                    intersection = mission_keywords & other_keywords
                    union = mission_keywords | other_keywords
                    similarity = len(intersection) / len(union) if union else 0
                    
                    if similarity > 0.3:  # Threshold for similarity
                        similar_missions.append((node_id, similarity))
        
        return similar_missions
    
    # NetworkX Intelligence Methods
    
    def analyze_mission_conflicts(self, mission: Mission) -> Dict[str, Any]:
        """Use NetworkX to analyze potential mission conflicts"""
        conflicts = {
            'priority_conflicts': [],
            'resource_conflicts': [],
            'dependency_conflicts': []
        }
        
        try:
            mission_type = self._classify_mission_type(mission.objective)
            
            # Check for resource conflicts
            if mission_type in self.resource_conflict_graph:
                conflicting_types = list(self.resource_conflict_graph.neighbors(mission_type))
                
                for conflict_type in conflicting_types:
                    edge_data = self.resource_conflict_graph.get_edge_data(mission_type, conflict_type)
                    conflicts['resource_conflicts'].append({
                        'conflict_with': conflict_type,
                        'conflict_type': edge_data.get('conflict_type', 'unknown'),
                        'severity': edge_data.get('severity', 0.5)
                    })
            
            # Check for priority conflicts with existing missions
            for node_id in self.mission_relationship_graph.nodes():
                if node_id != mission.id:
                    node_data = self.mission_relationship_graph.nodes[node_id]
                    other_priority = node_data.get('priority', 'normal')
                    
                    if self.priority_dependency_graph.has_edge(mission.priority.value, other_priority):
                        edge_data = self.priority_dependency_graph.get_edge_data(mission.priority.value, other_priority)
                        conflicts['priority_conflicts'].append({
                            'mission_id': node_id,
                            'impact': edge_data.get('impact', 'unknown'),
                            'weight': edge_data.get('weight', 0.5)
                        })
            
            logger.info(f"Conflict analysis completed for mission {mission.id}")
            return conflicts
            
        except Exception as e:
            logger.error(f"Error analyzing mission conflicts: {e}")
            return conflicts
    
    def get_mission_recommendations(self, mission: Mission) -> Dict[str, Any]:
        """Use NetworkX to provide mission optimization recommendations"""
        recommendations = {
            'similar_missions': [],
            'priority_adjustments': [],
            'scheduling_suggestions': []
        }
        
        try:
            # Find similar missions for learning
            if mission.id in self.mission_similarity_graph:
                similar_nodes = list(self.mission_similarity_graph.neighbors(mission.id))
                
                for similar_id in similar_nodes:
                    edge_data = self.mission_similarity_graph.get_edge_data(mission.id, similar_id)
                    similarity = edge_data.get('similarity', 0)
                    
                    recommendations['similar_missions'].append({
                        'mission_id': similar_id,
                        'similarity_score': similarity,
                        'recommendation': 'Review similar mission outcomes for insights'
                    })
            
            # Priority adjustment recommendations
            mission_type = self._classify_mission_type(mission.objective)
            if mission_type == 'critical_response' and mission.priority != MissionPriority.CRITICAL:
                recommendations['priority_adjustments'].append({
                    'current_priority': mission.priority.value,
                    'suggested_priority': MissionPriority.CRITICAL.value,
                    'reason': 'Mission type indicates critical response needed'
                })
            
            # Scheduling suggestions based on conflicts
            conflicts = self.analyze_mission_conflicts(mission)
            if conflicts['resource_conflicts']:
                recommendations['scheduling_suggestions'].append({
                    'suggestion': 'Schedule during low-conflict time window',
                    'reason': f"Detected {len(conflicts['resource_conflicts'])} resource conflicts"
                })
            
            logger.info(f"Generated {len(recommendations['similar_missions'])} recommendations for mission {mission.id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating mission recommendations: {e}")
            return recommendations
    
    def analyze_mission_network_metrics(self) -> Dict[str, Any]:
        """Use NetworkX to analyze mission network properties"""
        metrics = {
            'relationship_analysis': {},
            'similarity_analysis': {},
            'conflict_analysis': {}
        }
        
        try:
            # Mission relationship graph metrics
            if len(self.mission_relationship_graph) > 0:
                metrics['relationship_analysis'] = {
                    'total_missions': len(self.mission_relationship_graph.nodes()),
                    'mission_types': len(set(data.get('mission_type', 'unknown') for _, data in self.mission_relationship_graph.nodes(data=True))),
                    'priority_distribution': self._get_priority_distribution(),
                    'average_keywords_per_mission': self._get_average_keywords()
                }
            
            # Mission similarity graph metrics
            if len(self.mission_similarity_graph) > 0:
                metrics['similarity_analysis'] = {
                    'total_similarity_connections': len(self.mission_similarity_graph.edges()),
                    'most_similar_mission': max(self.mission_similarity_graph.nodes(), key=lambda x: self.mission_similarity_graph.degree(x), default=None),
                    'average_similarity': sum(data.get('similarity', 0) for _, _, data in self.mission_similarity_graph.edges(data=True)) / max(len(self.mission_similarity_graph.edges()), 1),
                    'similarity_clusters': len(list(nx.connected_components(self.mission_similarity_graph)))
                }
            
            # Resource conflict analysis
            if len(self.resource_conflict_graph) > 0:
                metrics['conflict_analysis'] = {
                    'potential_conflict_types': len(self.resource_conflict_graph.nodes()),
                    'total_conflict_relationships': len(self.resource_conflict_graph.edges()),
                    'highest_conflict_severity': max((data.get('severity', 0) for _, _, data in self.resource_conflict_graph.edges(data=True)), default=0),
                    'most_conflicted_type': max(self.resource_conflict_graph.nodes(), key=lambda x: self.resource_conflict_graph.degree(x), default=None)
                }
            
            logger.info("Mission network metrics analysis completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing mission network metrics: {e}")
            return metrics
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of mission priorities"""
        distribution = {}
        for _, data in self.mission_relationship_graph.nodes(data=True):
            priority = data.get('priority', 'unknown')
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution
    
    def _get_average_keywords(self) -> float:
        """Get average number of keywords per mission"""
        total_keywords = sum(
            len(data.get('objective_keywords', []))
            for _, data in self.mission_relationship_graph.nodes(data=True)
        )
        total_missions = len(self.mission_relationship_graph.nodes())
        return total_keywords / max(total_missions, 1)
