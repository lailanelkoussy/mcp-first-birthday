from RepoKnowledgeGraphLib.EntityExtractor import CppEntityExtractor


def test_cpp_entity_extraction():
    """Test C++ entity extraction with namespace and class."""
    cpp_code = """
#include <iostream>
namespace NS {
    class Foo {
    public:
        void bar(int x) { std::cout << x; }
    };

    void baz() {
        Foo f;
        f.bar(42);
    }
}
"""

    extractor = CppEntityExtractor()
    declared, called = extractor.extract_entities(cpp_code)

    print("Declared:")
    for d in declared:
        print(d)

    print("\nCalled:")
    print(called)

    # Assertions for pytest
    assert declared is not None, "Declared entities should not be None"
    assert called is not None, "Called entities should not be None"
    assert len(declared) > 0, "Should have at least one declared entity"
