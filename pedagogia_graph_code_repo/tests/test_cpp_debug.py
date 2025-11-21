#!/usr/bin/env python3
"""
Simple debug test for C++ entity extraction.
"""

from RepoKnowledgeGraphLib.EntityExtractor import CppEntityExtractor


def test_cpp_entity_extraction_with_namespace():
    """Test C++ entity extraction with namespace and class."""
    # Simple C++ code with namespace and class
    cpp_code = """
namespace utils {
    class Calculator {
    public:
        Calculator();
        int add(int a, int b);
        int multiply(int a, int b);
    };
}
"""

    print("Testing C++ entity extraction with simple namespace + class")
    print("=" * 80)
    print("Code:")
    print(cpp_code)
    print("=" * 80)

    extractor = CppEntityExtractor()
    declared, called = extractor.extract_entities(cpp_code)

    print(f"\nDeclared entities ({len(declared)}):")
    for entity in declared:
        print(f"  - {entity}")

    print(f"\nCalled entities ({len(called)}):")
    for entity in called:
        print(f"  - {entity}")

    print("\n" + "=" * 80)
    print("Expected to see:")
    print("  - utils::Calculator (class)")
    print("  - utils::Calculator::Calculator (method/constructor)")
    print("  - utils::Calculator::add (method)")
    print("  - utils::Calculator::multiply (method)")
    print("=" * 80)

    # Assertions for pytest
    assert declared is not None, "Declared entities should not be None"
    assert len(declared) > 0, "Should have at least one declared entity"

    # Check that we have the Calculator class
    entity_names = [e.get('name', '') for e in declared]
    assert any('Calculator' in name for name in entity_names), "Should find Calculator entity"

