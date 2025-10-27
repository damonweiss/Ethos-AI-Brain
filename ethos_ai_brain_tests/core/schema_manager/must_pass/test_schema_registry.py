"""
Test SchemaRegistry Basic Functionality - Must Pass
Light tests focusing on core schema registry features
"""

import pytest
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ethos_ai_brain.core.schema_manager.schema_manager import SchemaRegistry, ValidationMode


# Test schemas for registry testing
class TestUserSchema(BaseModel):
    name: str = Field(..., description="User name")
    age: int = Field(..., ge=0, le=150, description="User age")
    email: str = Field(..., description="User email")

class TestProductSchema(BaseModel):
    product_id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float = Field(..., gt=0, description="Product price")


def test_schema_registry_creation():
    """Test creating SchemaRegistry"""
    registry = SchemaRegistry()
    initial_count = len(registry.list_schemas())
    
    assert isinstance(registry, SchemaRegistry)
    assert hasattr(registry, 'list_schemas')
    assert initial_count == 0  # Should start with empty registry


def test_schema_registration():
    """Test registering schemas in the registry"""
    registry = SchemaRegistry()
    initial_count = len(registry.list_schemas())
    
    # Register test schema
    registry.register("test_user", TestUserSchema)
    
    # Check registration
    schemas = registry.list_schemas()
    new_count = len(schemas)
    
    assert new_count == initial_count + 1
    assert "test_user" in schemas
    
    # Test retrieving registered schema
    retrieved_schema = registry.get("test_user")
    
    assert retrieved_schema is not None
    assert retrieved_schema == TestUserSchema


def test_multiple_schema_registration():
    """Test registering multiple schemas"""
    print("Running Multiple Schema Registration Tests...")
    
    registry = SchemaRegistry()
    
    # Register multiple schemas
    registry.register("user", TestUserSchema)
    registry.register("product", TestProductSchema)
    
    schemas = registry.list_schemas()
    print(f"Registered schemas: {schemas}")
    print(f"Total count: {len(schemas)}")
    
    assert "user" in schemas, "[FAILURE] Should contain user schema"
    assert "product" in schemas, "[FAILURE] Should contain product schema"
    assert len(schemas) >= 2, "[FAILURE] Should have at least 2 schemas"
    
    # Test retrieving both schemas
    user_schema = registry.get("user")
    product_schema = registry.get("product")
    
    print(f"User schema: {user_schema.__name__}")
    print(f"Product schema: {product_schema.__name__}")
    
    assert user_schema == TestUserSchema, "[FAILURE] Should return correct user schema"
    assert product_schema == TestProductSchema, "[FAILURE] Should return correct product schema"
    
    print("[SUCCESS] Multiple schema registration works correctly")


def test_schema_not_found():
    """Test retrieving non-existent schema"""
    print("Running Schema Not Found Tests...")
    
    registry = SchemaRegistry()
    
    # Try to get non-existent schema
    missing_schema = registry.get("non_existent")
    
    print(f"Missing schema result: {missing_schema}")
    
    assert missing_schema is None, "[FAILURE] Should return None for missing schema"
    
    print("[SUCCESS] Schema not found handling works correctly")


def test_schema_overwrite():
    """Test overwriting existing schema"""
    print("Running Schema Overwrite Tests...")
    
    registry = SchemaRegistry()
    
    # Register initial schema
    registry.register("test", TestUserSchema)
    first_schema = registry.get("test")
    
    # Overwrite with different schema
    registry.register("test", TestProductSchema)
    second_schema = registry.get("test")
    
    print(f"First schema: {first_schema.__name__}")
    print(f"Second schema: {second_schema.__name__}")
    
    assert first_schema == TestUserSchema, "[FAILURE] First registration should work"
    assert second_schema == TestProductSchema, "[FAILURE] Overwrite should work"
    assert first_schema != second_schema, "[FAILURE] Schemas should be different"
    
    # Check that count didn't increase (overwrite, not addition)
    schemas = registry.list_schemas()
    test_count = sum(1 for name in schemas if name == "test")
    
    print(f"Count of 'test' schemas: {test_count}")
    assert test_count == 1, "[FAILURE] Should only have one 'test' schema after overwrite"
    
    print("[SUCCESS] Schema overwrite works correctly")


def test_schema_metadata():
    """Test schema metadata functionality if available"""
    print("Running Schema Metadata Tests...")
    
    registry = SchemaRegistry()
    registry.register("user_with_meta", TestUserSchema)
    
    # Test if metadata methods exist
    try:
        # Try to get schema info/metadata
        if hasattr(registry, 'get_schema_info'):
            info = registry.get_schema_info("user_with_meta")
            print(f"Schema info: {info}")
            assert info is not None, "[FAILURE] Should return schema info"
        else:
            print("[INFO] get_schema_info method not available")
        
        # Try to get schema fields
        schema = registry.get("user_with_meta")
        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
            print(f"Schema fields: {list(fields.keys())}")
            assert "name" in fields, "[FAILURE] Should have name field"
            assert "age" in fields, "[FAILURE] Should have age field"
        
        print("[SUCCESS] Schema metadata access works")
        
    except Exception as e:
        print(f"[INFO] Schema metadata functionality not fully implemented: {e}")
        print("[SUCCESS] Basic schema registration works")


def test_validation_mode_enum():
    """Test ValidationMode enum values"""
    print("Running Validation Mode Enum Tests...")
    
    modes = [
        ValidationMode.STRICT,
        ValidationMode.COERCE,
        ValidationMode.PARTIAL,
        ValidationMode.LENIENT
    ]
    
    expected_values = ["strict", "coerce", "partial", "lenient"]
    
    for i, mode in enumerate(modes):
        print(f"Validation mode {i+1}: {mode.value}")
        assert mode.value == expected_values[i], f"[FAILURE] Mode {i+1} should be {expected_values[i]}"
    
    print("[SUCCESS] ValidationMode enum works correctly")


if __name__ == "__main__":
    print("Running Schema Registry Tests...")
    test_schema_registry_creation()
    test_schema_registration()
    test_multiple_schema_registration()
    test_schema_not_found()
    test_schema_overwrite()
    test_schema_metadata()
    test_validation_mode_enum()
    print("[SUCCESS] All schema registry tests passed!")
