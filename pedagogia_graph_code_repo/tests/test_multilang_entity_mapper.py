#!/usr/bin/env python3
"""
Test script to demonstrate EntityChunkMapper support for C, C++, and Java
"""

from RepoKnowledgeGraphLib.EntityChunkMapper import EntityChunkMapper


def test_c_comments():
    """Test that C comments are correctly excluded"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
// calculate_sum is defined elsewhere
int helper() {
    return 0;
}
""",
        """
/* This function calculates the sum
   using calculate_sum helper
*/
int calculate_sum(int a, int b) {
    return a + b;
}
""",
        """
int main() {
    int result = calculate_sum(5, 10);
    return 0;
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("calculate_sum", chunks, entity_type="function", file_name="test.c")
    
    print("Test 1: C comments")
    print(f"Chunks with 'calculate_sum': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comment)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_cpp_class():
    """Test C++ class detection"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
// The DataProcessor class handles data
void helper() {}
""",
        """
class DataProcessor {
public:
    void process();
};
""",
        """
void main() {
    DataProcessor dp;
    dp.process();
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("DataProcessor", chunks, entity_type="class", file_name="test.cpp")
    
    print("Test 2: C++ class")
    print(f"Chunks with 'DataProcessor': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comment)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_cpp_method():
    """Test C++ method detection with scope resolution"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
class MyClass {
    void doSomething();
};
""",
        """
// Implementation of doSomething
void MyClass::doSomething() {
    // Do work
}
""",
        """
int main() {
    MyClass obj;
    obj.doSomething();
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("MyClass::doSomething", chunks, entity_type="method", file_name="test.cpp")
    
    print("Test 3: C++ method with ::")
    print(f"Chunks with 'MyClass::doSomething': {matches}")
    print(f"Expected: {1, 2} (chunk 0 is declaration only)")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_java_multiline_comments():
    """Test that Java multi-line comments are correctly excluded"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
/**
 * The processData method is very important
 * It uses processData internally
 */
public void helper() {
    return;
}
""",
        """
public void processData(String data) {
    System.out.println(data);
}
""",
        """
public void main() {
    processData("Hello");
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("processData", chunks, entity_type="method", file_name="Test.java")
    
    print("Test 4: Java multi-line comments")
    print(f"Chunks with 'processData': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in javadoc)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_java_class():
    """Test Java class detection"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
// Employee class is defined below
public void helper() {}
""",
        """
public class Employee {
    private String name;
    
    public Employee(String name) {
        this.name = name;
    }
}
""",
        """
public class Main {
    public static void main(String[] args) {
        Employee emp = new Employee("John");
    }
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("Employee", chunks, entity_type="class", file_name="Employee.java")
    
    print("Test 5: Java class")
    print(f"Chunks with 'Employee': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comment)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_c_struct():
    """Test C struct detection"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
/* Point struct definition */
struct Point {
    int x;
    int y;
};
""",
        """
void movePoint(struct Point* p, int dx, int dy) {
    p->x += dx;
    p->y += dy;
}
""",
        """
int main() {
    struct Point p = {0, 0};
    movePoint(&p, 10, 20);
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("Point", chunks, entity_type="class", file_name="point.c")
    
    print("Test 6: C struct")
    print(f"Chunks with 'Point': {matches}")
    print(f"Expected: {0, 2} (struct definition and usage)")
    print(f"Chunk 0 included (definition): {0 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_mixed_comments():
    """Test mixed comment styles in C++"""
    mapper = EntityChunkMapper()
    
    chunks = [
        """
// Single-line comment about compute
/* Multi-line comment
   also mentioning compute
*/
int helper() { return 0; }
""",
        """
int compute(int x) {
    return x * 2;
}
""",
        """
int main() {
    return compute(5);
}
"""
    ]
    
    matches = mapper.find_entity_in_chunks("compute", chunks, entity_type="function", file_name="test.cpp")
    
    print("Test 7: Mixed comment styles")
    print(f"Chunks with 'compute': {matches}")
    print(f"Expected: {1, 2} (chunk 0 should be excluded - only in comments)")
    print(f"Chunk 0 excluded: {0 not in matches}")
    print(f"Chunk 1 included (definition): {1 in matches}")
    print(f"Chunk 2 included (usage): {2 in matches}")
    print()


def test_language_detection():
    """Test language detection from file extensions"""
    mapper = EntityChunkMapper()
    
    test_cases = [
        ("test.py", "PYTHON"),
        ("test.c", "C"),
        ("test.h", "C"),
        ("test.cpp", "CPP"),
        ("test.hpp", "CPP"),
        ("test.java", "JAVA"),
        ("test.cc", "CPP"),
        ("test.cxx", "CPP"),
        (None, "PYTHON"),  # Default
    ]
    
    print("Test 8: Language detection")
    all_correct = True
    for filename, expected_lang in test_cases:
        detected = mapper._detect_language(filename)
        is_correct = detected.name == expected_lang
        all_correct = all_correct and is_correct
        status = "✓" if is_correct else "✗"
        print(f"  {status} {filename or 'None':<15} -> {detected.name:<10} (expected: {expected_lang})")
    
    print(f"All correct: {all_correct}")
    print()


# Test functions will be auto-discovered and run by pytest
    print("=" * 70)

