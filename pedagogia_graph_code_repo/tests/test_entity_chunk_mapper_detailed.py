"""
Tests for EntityChunkMapper functionality
"""

import pytest
from RepoKnowledgeGraphLib.EntityChunkMapper import EntityChunkMapper


def test_entity_chunk_mapper_cpp_basic():
    """Test EntityChunkMapper can find C++ entities in chunks"""
    
    chunk = """#include "math/calculator.h"

using namespace math;

int main() {
    Calculator calc;
    return calc.add(1, 2);
}"""
    
    mapper = EntityChunkMapper()
    chunks = [chunk]
    
    # Test finding unqualified name
    result = mapper.find_entity_in_chunks('Calculator', chunks, entity_type=None, file_name='main.cpp')
    assert 0 in result, "Should find Calculator in chunk 0"
    
    # Test finding qualified name
    result = mapper.find_entity_in_chunks('math::Calculator', chunks, entity_type=None, file_name='main.cpp')
    assert 0 in result, "Should find math::Calculator in chunk 0"
    
    # Test finding method call
    result = mapper.find_entity_in_chunks('add', chunks, entity_type=None, file_name='main.cpp')
    assert 0 in result, "Should find add method call in chunk 0"


def test_entity_chunk_mapper_map_entities():
    """Test EntityChunkMapper.map_entities_to_chunks"""
    
    chunk = """#include "math/calculator.h"

using namespace math;

int main() {
    Calculator calc;
    return calc.add(1, 2);
}"""
    
    mapper = EntityChunkMapper()
    chunks = [chunk]
    
    declared_entities = [
        {"name": "main", "type": "function"},
        {"name": "calc", "type": "variable"}
    ]
    
    called_entities = ["math/calculator.h", "math::Calculator", "Calculator", "add"]
    
    chunk_declared, chunk_called = mapper.map_entities_to_chunks(
        declared_entities, called_entities, chunks, file_name='main.cpp'
    )
    
    # Verify declared entities are mapped
    assert 0 in chunk_declared
    assert len(chunk_declared[0]) == 2, "Should have 2 declared entities in chunk 0"
    
    declared_names = [e['name'] for e in chunk_declared[0]]
    assert 'main' in declared_names
    assert 'calc' in declared_names
    
    # Verify called entities are mapped
    assert 0 in chunk_called
    assert len(chunk_called[0]) > 0, "Should have called entities in chunk 0"
    
    # All called entities should be found in the chunk
    assert 'math/calculator.h' in chunk_called[0]
    assert 'Calculator' in chunk_called[0] or 'math::Calculator' in chunk_called[0]
    assert 'add' in chunk_called[0]


def test_entity_chunk_mapper_python():
    """Test EntityChunkMapper works with Python code"""
    
    chunk = """
class MyClass:
    def my_method(self):
        return 42

def use_class():
    obj = MyClass()
    return obj.my_method()
"""
    
    mapper = EntityChunkMapper()
    chunks = [chunk]
    
    # Test finding class
    result = mapper.find_entity_in_chunks('MyClass', chunks, entity_type='class', file_name='test.py')
    assert 0 in result, "Should find MyClass in chunk 0"
    
    # Test finding method
    result = mapper.find_entity_in_chunks('my_method', chunks, entity_type='method', file_name='test.py')
    assert 0 in result, "Should find my_method in chunk 0"
    
    # Test finding function
    result = mapper.find_entity_in_chunks('use_class', chunks, entity_type='function', file_name='test.py')
    assert 0 in result, "Should find use_class in chunk 0"


def test_entity_chunk_mapper_excludes_comments():
    """Test that EntityChunkMapper excludes entities found only in comments"""
    
    chunk = """
// This is a comment about Calculator
/* Another comment about Calculator */

int main() {
    // Calculator is not used here, just mentioned in comment
    return 0;
}
"""
    
    mapper = EntityChunkMapper()
    chunks = [chunk]
    
    # Calculator appears in comments but not in actual code
    result = mapper.find_entity_in_chunks('Calculator', chunks, entity_type=None, file_name='test.cpp')
    
    # Should not find it since it's only in comments
    assert 0 not in result, "Should not find Calculator in chunk 0 (only in comments)"


def test_entity_chunk_mapper_multiple_chunks():
    """Test EntityChunkMapper with multiple chunks"""
    
    chunk1 = """
class MyClass:
    def method1(self):
        pass
"""
    
    chunk2 = """
def use_class():
    obj = MyClass()
    obj.method1()
"""
    
    mapper = EntityChunkMapper()
    chunks = [chunk1, chunk2]
    
    declared_entities = [
        {"name": "MyClass", "type": "class"},
        {"name": "use_class", "type": "function"}
    ]
    
    called_entities = ["MyClass", "method1"]
    
    chunk_declared, chunk_called = mapper.map_entities_to_chunks(
        declared_entities, called_entities, chunks, file_name='test.py'
    )
    
    # MyClass should be declared in chunk 0
    assert 0 in chunk_declared
    declared_names_0 = [e['name'] for e in chunk_declared[0]]
    assert 'MyClass' in declared_names_0
    
    # use_class should be declared in chunk 1
    assert 1 in chunk_declared
    declared_names_1 = [e['name'] for e in chunk_declared[1]]
    assert 'use_class' in declared_names_1
    
    # MyClass should be called in chunk 1 (not chunk 0 where it's defined)
    assert 1 in chunk_called
    assert 'MyClass' in chunk_called[1]
    
    # method1 should be called in chunk 1
    assert 'method1' in chunk_called[1]
