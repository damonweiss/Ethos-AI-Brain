#!/usr/bin/env python3
"""
Execution Knowledge Graph - Specialized knowledge graph for execution planning and workflow analysis
Provides execution-specific query methods for critical path, workflow blockers, and timeline analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from knowledge_graph import KnowledgeGraph, GraphType

logger = logging.getLogger(__name__)

class ExecutionKnowledgeGraph(KnowledgeGraph):
    """
    Specialized knowledge graph for execution planning and workflow analysis
    Provides execution-specific query methods that don't make sense for RAG or intent graphs
    """
    
    def __init__(self, graph_id: str, execution_name: str = None):
        """
        Initialize execution knowledge graph
        
        Args:
            graph_id: Unique identifier for this graph
            execution_name: Human-readable execution name
        """
        super().__init__(GraphType.EXECUTION, graph_id)
        self.execution_name = execution_name or graph_id
        self.execution_start_date = None
        self.execution_deadline = None
        
        logger.info(f"ExecutionKnowledgeGraph created: {graph_id} ({self.execution_name})")
    
    # ========================================
    # Execution-Specific Query Methods
    # ========================================
    
    def find_critical_path(self) -> List[str]:
        """
        Find the critical path through the execution workflow
        Returns list of node IDs representing the longest path through the execution
        """
        if not self.nodes():
            return []
        
        # Find nodes with no dependencies (start nodes)
        start_nodes = []
        for node_id in self.nodes():
            predecessors = list(self.predecessors(node_id))
            if not predecessors:
                start_nodes.append(node_id)
        
        if not start_nodes:
            # If no start nodes found, pick the first node
            start_nodes = [list(self.nodes())[0]]
        
        # Find longest path from each start node
        longest_path = []
        max_duration = 0
        
        for start_node in start_nodes:
            path, duration = self._find_longest_path_from_node(start_node)
            if duration > max_duration:
                max_duration = duration
                longest_path = path
        
        logger.info(f"Critical path found: {len(longest_path)} nodes, {max_duration} duration")
        return longest_path
    
    def detect_workflow_blockers(self) -> List[Dict[str, Any]]:
        """
        Detect tasks that are blocking execution progress
        Returns list of blocker information with details
        """
        blockers = []
        
        for node_id in self.nodes():
            node_attrs = self.get_node_attributes(node_id)
            
            # Check if this node is blocking others
            successors = list(self.successors(node_id))
            if successors:
                # Check node status
                status = node_attrs.get('status', 'unknown')
                priority = node_attrs.get('priority', 'normal')
                
                # Node is a blocker if it's not completed and has successors
                if status in ['pending', 'in_progress', 'blocked']:
                    blocker_info = {
                        'node_id': node_id,
                        'status': status,
                        'priority': priority,
                        'blocked_tasks': successors,
                        'blocking_count': len(successors),
                        'description': node_attrs.get('description', ''),
                        'estimated_completion': node_attrs.get('estimated_completion'),
                        'blocker_severity': self._calculate_blocker_severity(node_id, successors)
                    }
                    blockers.append(blocker_info)
        
        # Sort by severity (most severe first)
        blockers.sort(key=lambda x: x['blocker_severity'], reverse=True)
        
        logger.info(f"Found {len(blockers)} workflow blockers")
        return blockers
    
    def analyze_timeline_dependencies(self) -> Dict[str, Any]:
        """
        Analyze execution timeline and task dependencies
        Returns comprehensive timeline analysis
        """
        analysis = {
            'total_tasks': len(self.nodes()),
            'total_dependencies': len(self.edges()),
            'parallel_chains': [],
            'sequential_chains': [],
            'dependency_depth': 0,
            'estimated_duration': 0,
            'critical_milestones': [],
            'timeline_risks': []
        }
        
        if not self.nodes():
            return analysis
        
        # Find parallel vs sequential task chains
        analysis['parallel_chains'] = self._find_parallel_chains()
        analysis['sequential_chains'] = self._find_sequential_chains()
        
        # Calculate dependency depth (longest chain)
        analysis['dependency_depth'] = self._calculate_dependency_depth()
        
        # Estimate total execution duration
        analysis['estimated_duration'] = self._estimate_execution_duration()
        
        # Identify critical milestones
        analysis['critical_milestones'] = self._identify_critical_milestones()
        
        # Assess timeline risks
        analysis['timeline_risks'] = self._assess_timeline_risks()
        
        logger.info(f"Timeline analysis: {analysis['total_tasks']} tasks, "
                   f"{analysis['dependency_depth']} depth, "
                   f"{analysis['estimated_duration']} estimated duration")
        
        return analysis
    
    def calculate_execution_progress(self) -> Dict[str, Any]:
        """
        Calculate overall execution progress and completion metrics
        """
        progress = {
            'total_tasks': len(self.nodes()),
            'completed_tasks': 0,
            'in_progress_tasks': 0,
            'pending_tasks': 0,
            'blocked_tasks': 0,
            'completion_percentage': 0.0,
            'critical_path_progress': 0.0,
            'estimated_completion_date': None,
            'progress_velocity': 0.0
        }
        
        if not self.nodes():
            return progress
        
        # Count tasks by status
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            status = attrs.get('status', 'pending')
            
            if status == 'completed':
                progress['completed_tasks'] += 1
            elif status == 'in_progress':
                progress['in_progress_tasks'] += 1
            elif status == 'blocked':
                progress['blocked_tasks'] += 1
            else:
                progress['pending_tasks'] += 1
        
        # Calculate completion percentage
        if progress['total_tasks'] > 0:
            progress['completion_percentage'] = (
                progress['completed_tasks'] / progress['total_tasks']
            ) * 100
        
        # Calculate critical path progress
        critical_path = self.find_critical_path()
        if critical_path:
            completed_critical = sum(
                1 for node_id in critical_path
                if self.get_node_attributes(node_id).get('status') == 'completed'
            )
            progress['critical_path_progress'] = (completed_critical / len(critical_path)) * 100
        
        # Estimate completion date based on current velocity
        progress['estimated_completion_date'] = self._estimate_completion_date()
        progress['progress_velocity'] = self._calculate_progress_velocity()
        
        logger.info(f"Execution progress: {progress['completion_percentage']:.1f}% complete, "
                   f"{progress['completed_tasks']}/{progress['total_tasks']} tasks")
        
        return progress
    
    def identify_risk_factors(self) -> List[Dict[str, Any]]:
        """
        Identify potential risk factors that could impact execution success
        """
        risks = []
        
        # Check for single points of failure
        for node_id in self.nodes():
            successors = list(self.successors(node_id))
            if len(successors) > 3:  # Node with many dependencies
                risks.append({
                    'type': 'single_point_of_failure',
                    'node_id': node_id,
                    'severity': 'high',
                    'description': f"Task {node_id} blocks {len(successors)} other tasks",
                    'impact': len(successors),
                    'mitigation': 'Consider breaking down task or adding parallel paths'
                })
        
        # Check for long sequential chains
        sequential_chains = self._find_sequential_chains()
        for chain in sequential_chains:
            if len(chain) > 5:  # Long sequential chain
                risks.append({
                    'type': 'long_sequential_chain',
                    'chain': chain,
                    'severity': 'medium',
                    'description': f"Sequential chain of {len(chain)} tasks with no parallelization",
                    'impact': len(chain),
                    'mitigation': 'Look for opportunities to parallelize some tasks'
                })
        
        # Check for overdue tasks
        current_time = datetime.now()
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            deadline = attrs.get('deadline')
            status = attrs.get('status', 'pending')
            
            if deadline and status != 'completed':
                try:
                    deadline_date = datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline
                    if current_time > deadline_date:
                        risks.append({
                            'type': 'overdue_task',
                            'node_id': node_id,
                            'severity': 'high',
                            'description': f"Task {node_id} is overdue",
                            'days_overdue': (current_time - deadline_date).days,
                            'mitigation': 'Immediate attention required or timeline adjustment'
                        })
                except (ValueError, TypeError):
                    pass  # Skip invalid date formats
        
        # Sort risks by severity
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        risks.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
        
        logger.info(f"Identified {len(risks)} risk factors")
        return risks
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _get_node_weight(self, node_attrs: Dict[str, Any]) -> float:
        """
        Override base class to use mission-specific duration weighting
        """
        return node_attrs.get('duration', 1.0)
    
    def _find_longest_path_from_node(self, start_node: str) -> Tuple[List[str], float]:
        """Find the longest path from a given start node using mission-specific duration weighting"""
        return self.find_longest_path_from_node(start_node)
    
    def _calculate_blocker_severity(self, node_id: str, blocked_tasks: List[str]) -> float:
        """Calculate how severe a blocker this node is"""
        base_severity = len(blocked_tasks)  # More blocked tasks = higher severity
        
        # Check if any blocked tasks are on critical path
        critical_path = self.find_critical_path()
        critical_blocked = sum(1 for task in blocked_tasks if task in critical_path)
        
        # Check priority of the blocker task
        attrs = self.get_node_attributes(node_id)
        priority_multiplier = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0
        }.get(attrs.get('priority', 'medium'), 1.0)
        
        severity = base_severity * priority_multiplier + (critical_blocked * 2)
        return severity
    
    def _find_parallel_chains(self) -> List[List[str]]:
        """Find chains of tasks that can be executed in parallel"""
        return self.find_parallel_chains()
    
    def _find_sequential_chains(self) -> List[List[str]]:
        """Find chains of tasks that must be executed sequentially"""
        return self.find_sequential_chains()
    
    
    def _calculate_dependency_depth(self) -> int:
        """Calculate the maximum dependency depth in the execution"""
        return self.calculate_dependency_depth()
    
    def _get_node_depth(self, node_id: str, visited: Set[str] = None) -> int:
        """Get the dependency depth of a specific node"""
        return self.get_node_depth(node_id, visited)
    
    def _estimate_execution_duration(self) -> float:
        """Estimate total execution duration based on critical path"""
        critical_path = self.find_critical_path()
        if not critical_path:
            return 0.0
        
        total_duration = 0.0
        for node_id in critical_path:
            attrs = self.get_node_attributes(node_id)
            duration = attrs.get('duration', 1.0)
            total_duration += duration
        
        return total_duration
    
    def _identify_critical_milestones(self) -> List[Dict[str, Any]]:
        """Identify critical milestones in the execution"""
        milestones = []
        
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if attrs.get('is_milestone', False) or attrs.get('type') == 'milestone':
                milestone = {
                    'node_id': node_id,
                    'name': attrs.get('name', node_id),
                    'deadline': attrs.get('deadline'),
                    'status': attrs.get('status', 'pending'),
                    'dependencies': list(self.predecessors(node_id)),
                    'blocks': list(self.successors(node_id))
                }
                milestones.append(milestone)
        
        return milestones
    
    def _assess_timeline_risks(self) -> List[Dict[str, Any]]:
        """Assess risks to the execution timeline"""
        risks = []
        
        # Check for tasks without duration estimates
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            if 'duration' not in attrs:
                risks.append({
                    'type': 'missing_duration_estimate',
                    'node_id': node_id,
                    'severity': 'medium',
                    'description': f"Task {node_id} has no duration estimate"
                })
        
        # Check for unrealistic durations
        for node_id in self.nodes():
            attrs = self.get_node_attributes(node_id)
            duration = attrs.get('duration', 1.0)
            if duration > 30:  # More than 30 time units seems unrealistic
                risks.append({
                    'type': 'unrealistic_duration',
                    'node_id': node_id,
                    'severity': 'medium',
                    'description': f"Task {node_id} has unusually long duration: {duration}"
                })
        
        return risks
    
    def _estimate_completion_date(self) -> Optional[datetime]:
        """Estimate execution completion date based on current progress"""
        if not self.execution_start_date:
            return None
        
        progress = self.calculate_execution_progress()
        if progress['completion_percentage'] == 0:
            return None
        
        # Simple linear projection based on current progress
        elapsed_time = datetime.now() - self.execution_start_date
        total_estimated_time = elapsed_time / (progress['completion_percentage'] / 100)
        estimated_completion = self.execution_start_date + total_estimated_time
        
        return estimated_completion
    
    def _calculate_progress_velocity(self) -> float:
        """Calculate the rate of progress (tasks completed per day)"""
        if not self.execution_start_date:
            return 0.0
        
        elapsed_days = (datetime.now() - self.execution_start_date).days
        if elapsed_days == 0:
            return 0.0
        
        progress = self.calculate_execution_progress()
        velocity = progress['completed_tasks'] / elapsed_days
        
        return velocity
    
    # ========================================
    # Execution Management Methods
    # ========================================
    
    def set_execution_timeline(self, start_date: datetime, deadline: datetime = None):
        """Set execution start date and optional deadline"""
        self.execution_start_date = start_date
        self.execution_deadline = deadline
        
        logger.info(f"Execution timeline set: {start_date} to {deadline}")
    
    def add_task(self, task_id: str, **attributes) -> None:
        """Add a task to the execution with execution-specific attributes"""
        # Set default execution-specific attributes
        execution_attrs = {
            'type': 'task',
            'status': 'pending',
            'priority': 'medium',
            'duration': 1.0,
            'created_date': datetime.now().isoformat(),
            **attributes
        }
        
        self.add_node(task_id, **execution_attrs)
        logger.info(f"Added execution task: {task_id}")
    
    def add_dependency(self, from_task: str, to_task: str, dependency_type: str = 'depends_on') -> None:
        """Add a dependency relationship between tasks"""
        self.add_edge(from_task, to_task, relationship=dependency_type)
        logger.info(f"Added dependency: {from_task} -> {to_task} ({dependency_type})")
    
    def update_task_status(self, task_id: str, status: str, completion_date: datetime = None) -> None:
        """Update the status of a task"""
        updates = {'status': status}
        if status == 'completed' and completion_date:
            updates['completion_date'] = completion_date.isoformat()
        elif status == 'completed':
            updates['completion_date'] = datetime.now().isoformat()
        
        self.update_node_attributes(task_id, updates)
        logger.info(f"Updated task {task_id} status: {status}")
