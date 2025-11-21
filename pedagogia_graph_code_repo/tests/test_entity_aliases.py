#!/usr/bin/env python3
"""
Test script to verify entity alias functionality.
Tests that entities have proper aliases based on file paths.
"""

from RepoKnowledgeGraphLib.EntityExtractor import HybridEntityExtractor
from RepoKnowledgeGraphLib.utils.path_utils import generate_entity_aliases, file_path_to_module_path


def test_path_to_module_conversion():
    """Test file path to module path conversion"""
    print("=" * 80)
    print("Testing file path to module path conversion")
    print("=" * 80)
    
    test_cases = [
        ("path/to/repo/python_script.py", "path.to.repo.python_script"),
        ("src/utils/helper.py", "src.utils.helper"),
        ("module.py", "module"),
        ("deep/nested/path/to/file.py", "deep.nested.path.to.file"),
    ]
    
    for file_path, expected in test_cases:
        result = file_path_to_module_path(file_path)
        status = "✓" if result == expected else "✗"
        print(f"{status} {file_path} -> {result} (expected: {expected})")
    print()


def test_entity_alias_generation():
    """Test entity alias generation"""
    print("=" * 80)
    print("Testing entity alias generation")
    print("=" * 80)
    
    test_cases = [
        ("Class_1", "path/to/repo/python_script.py", 
         ["Class_1", "path.to.repo.python_script.Class_1"]),
        ("my_function", "src/utils/helper.py",
         ["my_function", "src.utils.helper.my_function"]),
        ("MyClass.my_method", "app/models/user.py",
         ["MyClass.my_method", "app.models.user.MyClass.my_method"]),
    ]
    
    for entity_name, file_path, expected_aliases in test_cases:
        aliases = generate_entity_aliases(entity_name, file_path)
        match = set(aliases) == set(expected_aliases)
        status = "✓" if match else "✗"
        print(f"{status} Entity: {entity_name} in {file_path}")
        print(f"   Generated aliases: {aliases}")
        print(f"   Expected aliases:  {expected_aliases}")
        print()


def test_python_entity_extraction_with_aliases():
    """Test that Python entity extraction includes aliases"""
    print("=" * 80)
    print("Testing Python entity extraction with aliases")
    print("=" * 80)
    
    python_code = """
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def my_method(self):
        return self.value

def my_function(x, y):
    return x + y

CONSTANT = 42
"""
    
    file_name = "src/module/test.py"
    extractor = HybridEntityExtractor()
    
    declared_entities, called_entities = extractor.extract_entities(python_code, file_name)
    
    print(f"File: {file_name}")
    print(f"Found {len(declared_entities)} declared entities:")
    print()
    
    for entity in declared_entities:
        name = entity.get('name')
        entity_type = entity.get('type')
        aliases = entity.get('aliases', [])
        print(f"  Entity: {name} (type: {entity_type})")
        print(f"    Aliases: {aliases}")
    
    print()
    print("Verification:")
    
    # Check that MyClass has proper aliases
    my_class = next((e for e in declared_entities if e.get('name') == 'MyClass'), None)
    if my_class:
        expected_aliases = ['MyClass', 'src.module.test.MyClass']
        actual_aliases = my_class.get('aliases', [])
        match = set(actual_aliases) == set(expected_aliases)
        status = "✓" if match else "✗"
        print(f"{status} MyClass aliases: {actual_aliases}")
    
    # Check that my_function has proper aliases
    my_func = next((e for e in declared_entities if e.get('name') == 'my_function'), None)
    if my_func:
        expected_aliases = ['my_function', 'src.module.test.my_function']
        actual_aliases = my_func.get('aliases', [])
        match = set(actual_aliases) == set(expected_aliases)
        status = "✓" if match else "✗"
        print(f"{status} my_function aliases: {actual_aliases}")
    
    # Check nested method has proper aliases
    my_method = next((e for e in declared_entities if e.get('name') == 'MyClass.my_method'), None)
    if my_method:
        expected_aliases = ['MyClass.my_method', 'src.module.test.MyClass.my_method']
        actual_aliases = my_method.get('aliases', [])
        match = set(actual_aliases) == set(expected_aliases)
        status = "✓" if match else "✗"
        print(f"{status} MyClass.my_method aliases: {actual_aliases}")
    
    print()


def test_javascript_entity_extraction_with_aliases():
    """Test that JavaScript entity extraction includes aliases"""
    print("=" * 80)
    print("Testing JavaScript entity extraction with aliases")
    print("=" * 80)
    
    js_code = """
class UserController {
    constructor(name) {
        this.name = name;
    }
    
    getUser() {
        return this.name;
    }
}

function createUser(name) {
    return new UserController(name);
}

const API_URL = 'http://api.example.com';
"""
    
    file_name = "app/controllers/user.js"
    extractor = HybridEntityExtractor()
    
    declared_entities, called_entities = extractor.extract_entities(js_code, file_name)
    
    print(f"File: {file_name}")
    print(f"Found {len(declared_entities)} declared entities:")
    print()
    
    for entity in declared_entities:
        name = entity.get('name')
        entity_type = entity.get('type')
        aliases = entity.get('aliases', [])
        print(f"  Entity: {name} (type: {entity_type})")
        print(f"    Aliases: {aliases}")
    
    print()


def test_java_entity_extraction_with_aliases():
    """Test that Java entity extraction includes aliases"""
    print("=" * 80)
    print("Testing Java entity extraction with aliases")
    print("=" * 80)
    
    java_code = """
package com.example.app;

import java.util.List;

public class UserService {
    private String name;
    
    public UserService(String name) {
        this.name = name;
    }
    
    public String getName() {
        return this.name;
    }
    
    public void updateName(String newName) {
        this.name = newName;
    }
}
"""
    
    file_name = "src/main/java/com/example/app/UserService.java"
    extractor = HybridEntityExtractor()
    
    declared_entities, called_entities = extractor.extract_entities(java_code, file_name)
    
    print(f"File: {file_name}")
    print(f"Found {len(declared_entities)} declared entities:")
    print()
    
    for entity in declared_entities:
        name = entity.get('name')
        entity_type = entity.get('type')
        aliases = entity.get('aliases', [])
        print(f"  Entity: {name} (type: {entity_type})")
        print(f"    Aliases: {aliases}")
    
    print()


# Test functions will be auto-discovered and run by pytest
    print("TESTING COMPLETE")
    print("=" * 80)

