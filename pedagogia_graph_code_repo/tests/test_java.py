from RepoKnowledgeGraphLib.EntityExtractor import JavaEntityExtractor


def test_java_entity_extraction():
    """Test Java entity extraction with class and inheritance."""
    java_code = """
package com.example;

import java.util.List;

public class Foo extends BaseClass implements Runnable {
    private int count;

    public Foo(int c) { this.count = c; }

    public void run() {
        Helper.doSomething();
        System.out.println(count);
    }
}
"""

    extractor = JavaEntityExtractor()
    declared, called = extractor.extract_entities(java_code)

    print("Declared:")
    for d in declared:
        print(d)

    print("\nCalled:")
    print(called)

    # Assertions for pytest
    assert declared is not None, "Declared entities should not be None"
    assert called is not None, "Called entities should not be None"
    assert len(declared) > 0, "Should have at least one declared entity"
