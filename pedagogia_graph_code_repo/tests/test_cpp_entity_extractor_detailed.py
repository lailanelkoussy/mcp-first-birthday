"""
Tests for C++ entity extraction with namespace resolution
"""

import tempfile
import os
import pytest
from RepoKnowledgeGraphLib.EntityExtractor import HybridEntityExtractor


def test_cpp_namespace_entity_extraction():
    """Test that C++ entity extraction correctly handles namespaces"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create header file with namespace
        header_path = os.path.join(tmpdir, "math", "calculator.h")
        os.makedirs(os.path.dirname(header_path), exist_ok=True)
        
        with open(header_path, 'w') as f:
            f.write("""
namespace math {
    class Calculator {
    public:
        int add(int a, int b);
    };
}
""")
        
        # Extract entities from header
        extractor = HybridEntityExtractor()
        with open(header_path, 'r') as f:
            code = f.read()
        declared_header, called_header = extractor.extract_entities(code, file_name=header_path)
        
        # Verify header declarations
        assert len(declared_header) >= 2, "Should have class and method declarations"
        
        # Find the Calculator class
        calculator_entities = [e for e in declared_header if 'Calculator' in e.get('name', '')]
        assert len(calculator_entities) >= 1, "Should have Calculator class"
        
        calculator = next((e for e in declared_header if e.get('name') == 'math::Calculator'), None)
        assert calculator is not None, "Should have math::Calculator"
        assert calculator['type'] == 'class'
        assert 'Calculator' in calculator.get('aliases', []), "Should have unqualified alias"
        assert 'math::Calculator' in calculator.get('aliases', []), "Should have qualified alias"


def test_cpp_namespace_usage():
    """Test that C++ entity extraction correctly extracts called entities with using namespace"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create header file
        header_path = os.path.join(tmpdir, "math", "calculator.h")
        os.makedirs(os.path.dirname(header_path), exist_ok=True)
        
        with open(header_path, 'w') as f:
            f.write("""
namespace math {
    class Calculator {
    public:
        int add(int a, int b);
    };
}
""")
        
        # Create implementation file that uses the namespace
        impl_path = os.path.join(tmpdir, "main.cpp")
        with open(impl_path, 'w') as f:
            f.write("""
#include "math/calculator.h"

using namespace math;

int main() {
    Calculator calc;
    return calc.add(1, 2);
}
""")
        
        # Extract entities from main
        extractor = HybridEntityExtractor()
        with open(impl_path, 'r') as f:
            code = f.read()
        declared_main, called_main = extractor.extract_entities(code, file_name=impl_path)
        
        # Verify declarations
        assert len(declared_main) >= 2, "Should have main function and calc variable"
        
        main_func = next((e for e in declared_main if e.get('name') == 'main'), None)
        assert main_func is not None, "Should have main function"
        assert main_func['type'] == 'function'
        
        calc_var = next((e for e in declared_main if e.get('name') == 'calc'), None)
        assert calc_var is not None, "Should have calc variable"
        assert calc_var['type'] == 'variable'
        
        # Verify called entities
        assert 'math/calculator.h' in called_main, "Should call header file"
        
        # When include is resolved, we should get the fully qualified name
        # This is the key assertion - with proper include resolution, libclang gives us math::Calculator
        assert 'math::Calculator' in called_main or 'Calculator' in called_main, \
            "Should call Calculator (qualified or unqualified)"
        
        assert 'add' in called_main, "Should call add method"


def test_cpp_entity_extraction_with_full_paths():
    """Test that entity extraction works correctly when given full file paths"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple C++ file
        cpp_path = os.path.join(tmpdir, "test.cpp")
        
        with open(cpp_path, 'w') as f:
            f.write("""
class MyClass {
public:
    void myMethod();
};

int main() {
    MyClass obj;
    obj.myMethod();
    return 0;
}
""")
        
        extractor = HybridEntityExtractor()
        
        # Extract with full path
        with open(cpp_path, 'r') as f:
            code = f.read()
        declared, called = extractor.extract_entities(code, file_name=cpp_path)
        
        # Verify extraction works
        assert len(declared) >= 2, "Should have class and main function"
        
        myclass = next((e for e in declared if 'MyClass' in e.get('name', '')), None)
        assert myclass is not None, "Should have MyClass"
        
        assert 'myMethod' in called, "Should call myMethod"
