"""
Test Prompt Intent Schema Functionality - Must Pass
Light tests focusing on intent analysis schemas
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.schemas.prompt_intent import (
    IntentAnalysis, IntentEntity, IntentRelationship, 
    IntentCategory, IntentConfidence
)


def test_intent_entity_creation():
    """Test creating IntentEntity with various types"""
    entity = IntentEntity(
        entity_type="action",
        value="user_request",
        importance=0.85,
        confidence=0.90
    )
    
    assert entity.value == "user_request"
    assert entity.entity_type == "action"
    assert entity.importance == 0.85
    assert entity.confidence == 0.90


def test_intent_relationship_creation():
    """Test creating IntentRelationship between entities"""
    relationship = IntentRelationship(
        source_entity="user",
        target_entity="analysis_task",
        relationship_type="requests",
        strength=0.90
    )
    
    assert relationship.source_entity == "user"
    assert relationship.target_entity == "analysis_task"
    assert relationship.relationship_type == "requests"
    assert relationship.strength == 0.90


def test_intent_confidence_enum():
    """Test IntentConfidence enum values"""
    # Test all confidence levels
    confidences = [
        IntentConfidence.HIGH,
        IntentConfidence.MEDIUM, 
        IntentConfidence.LOW
    ]
    
    expected_values = ["high", "medium", "low"]
    
    for i, confidence in enumerate(confidences):
        assert confidence.value == expected_values[i]


def test_intent_category_enum():
    """Test IntentCategory enum values"""
    print("Running Intent Category Enum Tests...")
    
    # Test some key categories
    categories = [
        IntentCategory.INFORMATION_SEEKING,
        IntentCategory.TASK_EXECUTION,
        IntentCategory.PROBLEM_SOLVING,
        IntentCategory.ANALYSIS_REQUEST
    ]
    
    expected_values = ["information_seeking", "task_execution", "problem_solving", "analysis_request"]
    
    for i, category in enumerate(categories):
        print(f"Category {i+1}: {category.value}")
        assert category.value == expected_values[i], f"[FAILURE] Category {i+1} should be {expected_values[i]}"
    
    print("[SUCCESS] IntentCategory enum works correctly")


def test_intent_analysis_complete():
    """Test complete IntentAnalysis with all components"""
    print("Running Complete Intent Analysis Tests...")
    
    # Create entities
    entities = [
        IntentEntity(
            entity_type="actor",
            value="user",
            importance=0.95,
            confidence=0.95
        ),
        IntentEntity(
            entity_type="task",
            value="data_analysis",
            importance=0.88,
            confidence=0.88
        )
    ]
    
    # Create relationships
    relationships = [
        IntentRelationship(
            source_entity="user",
            target_entity="data_analysis",
            relationship_type="requests",
            strength=0.92
        )
    ]
    
    # Create complete analysis
    analysis = IntentAnalysis(
        primary_intent="data_analysis_request",
        intent_category=IntentCategory.ANALYSIS_REQUEST,
        intent_confidence=IntentConfidence.HIGH,
        goals=entities,
        relationships=relationships,
        complexity_score=3.0
    )
    
    print(f"Primary intent: {analysis.primary_intent}")
    print(f"Intent confidence: {analysis.intent_confidence.value}")
    print(f"Intent category: {analysis.intent_category.value}")
    print(f"Number of goals: {len(analysis.goals)}")
    print(f"Number of relationships: {len(analysis.relationships)}")
    
    assert analysis.primary_intent == "data_analysis_request", "[FAILURE] Primary intent should match"
    assert analysis.intent_confidence == IntentConfidence.HIGH, "[FAILURE] Intent confidence should match"
    assert analysis.intent_category == IntentCategory.ANALYSIS_REQUEST, "[FAILURE] Intent category should match"
    assert len(analysis.goals) == 2, "[FAILURE] Should have 2 goal entities"
    assert len(analysis.relationships) == 1, "[FAILURE] Should have 1 relationship"
    
    print("[SUCCESS] Complete IntentAnalysis works correctly")


def test_confidence_validation():
    """Test confidence score validation (0.0 to 1.0)"""
    print("Running Confidence Validation Tests...")
    
    # Test valid confidence scores
    valid_entity = IntentEntity(
        entity_type="test",
        value="test",
        importance=0.5,
        confidence=0.5
    )
    
    print(f"Valid confidence: {valid_entity.confidence}")
    assert 0.0 <= valid_entity.confidence <= 1.0, "[FAILURE] Confidence should be in valid range"
    
    # Test boundary values
    min_entity = IntentEntity(entity_type="test", value="test", importance=0.0, confidence=0.0)
    max_entity = IntentEntity(entity_type="test", value="test", importance=1.0, confidence=1.0)
    
    print(f"Min confidence: {min_entity.confidence}")
    print(f"Max confidence: {max_entity.confidence}")
    
    assert min_entity.confidence == 0.0, "[FAILURE] Min confidence should work"
    assert max_entity.confidence == 1.0, "[FAILURE] Max confidence should work"
    
    print("[SUCCESS] Confidence validation works correctly")


def test_schema_serialization():
    """Test intent schema serialization"""
    print("Running Intent Schema Serialization Tests...")
    
    entity = IntentEntity(
        entity_type="concept",
        value="test_entity",
        importance=0.75,
        confidence=0.75
    )
    
    # Test dict serialization
    entity_dict = entity.model_dump()
    print(f"Serialized keys: {list(entity_dict.keys())}")
    print(f"Value in dict: {entity_dict['value']}")
    
    assert "value" in entity_dict, "[FAILURE] Should contain value"
    assert "entity_type" in entity_dict, "[FAILURE] Should contain entity_type"
    assert "confidence" in entity_dict, "[FAILURE] Should contain confidence"
    assert entity_dict["value"] == "test_entity", "[FAILURE] Value should be preserved"
    
    # Test JSON serialization
    import json
    entity_json = entity.model_dump_json()
    parsed_json = json.loads(entity_json)
    
    print(f"JSON value: {parsed_json['value']}")
    assert parsed_json["value"] == "test_entity", "[FAILURE] JSON should preserve value"
    assert parsed_json["confidence"] == 0.75, "[FAILURE] JSON should preserve confidence"
    
    print("[SUCCESS] Intent schema serialization works correctly")


if __name__ == "__main__":
    print("Running Prompt Intent Schema Tests...")
    test_intent_entity_creation()
    test_intent_relationship_creation()
    test_intent_confidence_enum()
    test_intent_category_enum()
    test_intent_analysis_complete()
    test_confidence_validation()
    test_schema_serialization()
    print("[SUCCESS] All prompt intent schema tests passed!")
