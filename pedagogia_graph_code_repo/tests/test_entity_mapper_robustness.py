#!/usr/bin/env python3
"""
Test script to demonstrate that EntityChunkMapper now correctly ignores
entity matches in comments and docstrings.
"""

from RepoKnowledgeGraphLib.EntityChunkMapper import EntityChunkMapper


def test_entity_in_comments():
    """Test that entities mentioned in comments are not matched"""
    mapper = EntityChunkMapper()
    
    # Chunk with function name only in comment
    chunks = [
        """
# This function calls calculate_total() to get the sum
def some_function():
    pass
""",
        """
def calculate_total(items):
    '''Calculate the total of all items'''
    return sum(items)
""",
        """
# calculate_total is a useful function
result = calculate_total([1, 2, 3])
"""
    ]
    
    # Find where calculate_total is actually used (not just mentioned in comments)
    matches = mapper.find_entity_in_chunks("calculate_total", chunks, entity_type="function")
    
    print("Test 1: Entity in comments")
    print(f"Chunks with 'calculate_total': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comment)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_entity_in_docstring():
    """Test that entities mentioned in docstrings are not matched"""
    mapper = EntityChunkMapper()
    
    chunks = [
        '''
def process_data(data):
    """
    Process data using the transform_value function.
    The transform_value function should be called for each item.
    """
    return [x * 2 for x in data]
''',
        '''
def transform_value(x):
    """Transform a single value"""
    return x * 3
''',
        '''
def main():
    """Main function"""
    result = transform_value(10)
    return result
'''
    ]
    
    matches = mapper.find_entity_in_chunks("transform_value", chunks, entity_type="function")
    
    print("Test 2: Entity in docstring")
    print(f"Chunks with 'transform_value': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in docstring)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_partial_name_matching():
    """Test that partial name matches are not matched"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
def calculate():
    return 42
""",
        """
def calculate_total():
    # Should not match 'calculate' alone
    return calculate()
""",
        """
# This has calculate_total_sum which should not match 'calculate_total'
calculate_total_sum = 100
"""
    ]
    
    matches = mapper.find_entity_in_chunks("calculate", chunks, entity_type="function")
    
    print("Test 3: Partial name matching")
    print(f"Chunks with 'calculate' (exact): {matches}")
    print(f"Expected: {0, 1} (chunk 2 should be excluded - different name)")
    print(f"Chunk 0 included (definition): {0 in matches}")
    print(f"Chunk 1 included (usage): {1 in matches}")
    print(f"Chunk 2 excluded: {2 not in matches}")
    print()


def test_class_in_comment():
    """Test that class names in comments are not matched"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
# The DataProcessor class is defined elsewhere
def helper_function():
    pass
""",
        """
class DataProcessor:
    '''Process data'''
    def process(self):
        pass
""",
        """
# Create a DataProcessor instance
processor = DataProcessor()
"""
    ]
    
    matches = mapper.find_entity_in_chunks("DataProcessor", chunks, entity_type="class")
    
    print("Test 4: Class name in comment")
    print(f"Chunks with 'DataProcessor': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comment)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


# Test functions will be auto-discovered and run by pytest

