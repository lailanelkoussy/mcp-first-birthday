#!/usr/bin/env python3
"""
Test C++ entity extraction with #include directives and alias-based matching.
Demonstrates how entity calls can be matched to their definitions using aliases.
"""

from RepoKnowledgeGraphLib.EntityExtractor import HybridEntityExtractor
from RepoKnowledgeGraphLib.utils.path_utils import (
    generate_entity_aliases,
    normalize_include_path,
    build_entity_alias_map,
    resolve_entity_call
)


def test_cpp_include_normalization():
    """Test that C++ include paths are normalized correctly"""
    print("=" * 80)
    print("Testing C++ include path normalization")
    print("=" * 80)
    
    test_cases = [
        ('<vector>', 'vector'),
        ('<iostream>', 'iostream'),
        ('"myheader.h"', 'myheader'),
        ('"utils/helper.h"', 'utils.helper'),
        ('<boost/algorithm/string.hpp>', 'boost.algorithm.string'),
    ]
    
    for include_path, expected in test_cases:
        result = normalize_include_path(include_path)
        status = "✓" if result == expected else "✗"
        print(f"{status} {include_path} -> {result} (expected: {expected})")
    print()


def test_cpp_entity_extraction_with_includes():
    """Test C++ entity extraction detects includes and entities"""
    print("=" * 80)
    print("Testing C++ entity extraction with #include directives")
    print("=" * 80)
    
    # Header file: utils/math.h
    math_header = """
#ifndef MATH_H
#define MATH_H

namespace utils {
    class Calculator {
    public:
        Calculator();
        int add(int a, int b);
        int multiply(int a, int b);
    };
}

#endif
"""
    
    # Implementation file that includes the header
    main_cpp = """
#include "utils/math.h"
#include <iostream>
#include <vector>

using namespace utils;

int main() {
    Calculator calc;
    int result = calc.add(5, 3);
    std::cout << "Result: " << result << std::endl;
    
    std::vector<int> numbers;
    numbers.push_back(result);
    
    return 0;
}
"""
    
    extractor = HybridEntityExtractor()
    
    # Extract from header
    print("Extracting from utils/math.h:")
    header_declared, header_called = extractor.extract_entities(math_header, "utils/math.h")
    print(f"  Declared entities: {len(header_declared)}")
    for entity in header_declared:
        name = entity.get('name')
        aliases = entity.get('aliases', [])
        print(f"    - {name}")
        print(f"      Aliases: {aliases}")
    print(f"  Called entities: {len(header_called)}")
    for entity in header_called:
        print(f"    - {entity}")
    print()
    
    # Extract from main file
    print("Extracting from src/main.cpp:")
    main_declared, main_called = extractor.extract_entities(main_cpp, "src/main.cpp")
    print(f"  Declared entities: {len(main_declared)}")
    for entity in main_declared:
        name = entity.get('name')
        aliases = entity.get('aliases', [])
        print(f"    - {name}")
        print(f"      Aliases: {aliases}")
    print(f"  Called entities: {len(main_called)}")
    for entity in main_called:
        print(f"    - {entity}")
    print()
    
    # Check that includes were detected
    includes = [e for e in main_called if e in ['utils/math.h', 'iostream', 'vector']]
    print(f"✓ Detected {len(includes)} includes: {includes}")
    print()


def test_alias_based_entity_matching():
    """Test that entity calls can be matched to definitions using aliases"""
    print("=" * 80)
    print("Testing alias-based entity call matching")
    print("=" * 80)
    
    # Simulate entities from different files
    entities = {
        # From utils/math.h
        'utils::Calculator': {
            'declaring_chunk_ids': ['utils/math.h_0'],
            'calling_chunk_ids': [],
            'aliases': ['utils::Calculator', 'utils.math.utils::Calculator'],
            'type': ['class']
        },
        'utils::Calculator::add': {
            'declaring_chunk_ids': ['utils/math.h_0'],
            'calling_chunk_ids': [],
            'aliases': ['utils::Calculator::add', 'utils.math.utils::Calculator::add'],
            'type': ['method']
        },
        # Called entities from main.cpp (without aliases initially)
        'Calculator': {
            'declaring_chunk_ids': [],
            'calling_chunk_ids': ['src/main.cpp_0'],
            'aliases': [],
            'type': []
        },
        'add': {
            'declaring_chunk_ids': [],
            'calling_chunk_ids': ['src/main.cpp_0'],
            'aliases': [],
            'type': []
        },
        'iostream': {
            'declaring_chunk_ids': [],
            'calling_chunk_ids': ['src/main.cpp_0'],
            'aliases': [],
            'type': []
        },
    }
    
    # Build alias map
    alias_map = build_entity_alias_map(entities)
    print(f"Built alias map with {len(alias_map)} entries")
    print()
    
    # Test resolution
    test_cases = [
        ('Calculator', 'utils::Calculator', 'Should match Calculator to utils::Calculator'),
        ('utils::Calculator', 'utils::Calculator', 'Direct match'),
        ('add', None, 'Cannot resolve just "add" without context'),
    ]
    
    for called_name, expected, description in test_cases:
        resolved = resolve_entity_call(called_name, alias_map)
        match = resolved == expected
        status = "✓" if match else "✗"
        print(f"{status} {description}")
        print(f"   Called: '{called_name}' -> Resolved: '{resolved}' (expected: '{expected}')")
    print()


def test_full_workflow():
    """Test the complete workflow of extracting, aliasing, and matching entities"""
    print("=" * 80)
    print("Testing complete workflow: extract -> alias -> match")
    print("=" * 80)
    
    # Simulate a small codebase
    files = {
        'math/calculator.h': """
namespace math {
    class Calculator {
    public:
        int add(int a, int b);
        int subtract(int a, int b);
    };
}
""",
        'src/app.cpp': """
#include "math/calculator.h"

int main() {
    math::Calculator calc;
    int result = calc.add(10, 20);
    return 0;
}
"""
    }
    
    extractor = HybridEntityExtractor()
    all_entities = {}
    
    # Step 1: Extract entities from all files
    print("Step 1: Extracting entities from all files")
    for file_path, code in files.items():
        declared, called = extractor.extract_entities(code, file_path)
        
        # Register declared entities
        for entity in declared:
            name = entity.get('name')
            if name:
                if name not in all_entities:
                    all_entities[name] = {
                        'declaring_chunk_ids': [],
                        'calling_chunk_ids': [],
                        'aliases': [],
                        'type': []
                    }
                all_entities[name]['declaring_chunk_ids'].append(f"{file_path}_0")
                all_entities[name]['aliases'].extend(entity.get('aliases', []))
                entity_type = entity.get('type')
                if entity_type and entity_type not in all_entities[name]['type']:
                    all_entities[name]['type'].append(entity_type)
        
        # Register called entities
        for called_name in called:
            if called_name not in all_entities:
                all_entities[called_name] = {
                    'declaring_chunk_ids': [],
                    'calling_chunk_ids': [],
                    'aliases': [],
                    'type': []
                }
            all_entities[called_name]['calling_chunk_ids'].append(f"{file_path}_0")
    
    print(f"  Found {len(all_entities)} total entities")
    print()
    
    # Step 2: Build alias map
    print("Step 2: Building alias map")
    alias_map = build_entity_alias_map(all_entities)
    print(f"  Built alias map with {len(alias_map)} entries")
    print()
    
    # Step 3: Display entity summary
    print("Step 3: Entity summary")
    declared_count = sum(1 for e in all_entities.values() if e['declaring_chunk_ids'])
    called_count = sum(1 for e in all_entities.values() if e['calling_chunk_ids'] and not e['declaring_chunk_ids'])
    print(f"  Declared entities: {declared_count}")
    print(f"  Called-only entities: {called_count}")
    print()
    
    # Step 4: Show declared entities with aliases
    print("Step 4: Declared entities and their aliases")
    for name, info in all_entities.items():
        if info['declaring_chunk_ids']:
            aliases = info.get('aliases', [])
            if len(aliases) > 1:  # More than just the simple name
                print(f"  {name}")
                print(f"    Aliases: {aliases}")
    print()
    
    # Step 5: Resolve called entities
    print("Step 5: Resolving called entities to declarations")
    for name, info in all_entities.items():
        if info['calling_chunk_ids'] and not info['declaring_chunk_ids']:
            resolved = resolve_entity_call(name, alias_map)
            if resolved:
                print(f"  ✓ '{name}' -> '{resolved}'")
            else:
                print(f"  ✗ '{name}' -> (unresolved)")
    print()


# Test functions will be auto-discovered and run by pytest

