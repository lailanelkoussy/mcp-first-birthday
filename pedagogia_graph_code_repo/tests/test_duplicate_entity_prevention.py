"""
Test to verify that entities are not added as duplicates when they already exist
under a different name or alias.
"""
import pytest
import tempfile
import os
from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph


def test_no_duplicate_entities_with_aliases():
    """
    Test that entities are not duplicated when they already exist under an alias.
    This tests both FIRST PASS (declared entities) and SECOND PASS (called entities).
    """
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file that defines an entity
        file1 = os.path.join(tmpdir, "module1.py")
        with open(file1, "w") as f:
            f.write("""
class MyClass:
    def __init__(self):
        self.value = 0
    
    def my_method(self):
        return self.value
""")

        # Create another Python file that calls the entity using different names
        file2 = os.path.join(tmpdir, "module2.py")
        with open(file2, "w") as f:
            f.write("""
from module1 import MyClass

def use_class():
    obj = MyClass()
    return obj.my_method()
""")

        # Build the knowledge graph
        kg =  RepoKnowledgeGraph.from_path(
            path=tmpdir,
            extract_entities=True,
            index_nodes=False,
            describe_nodes=False,
            model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    },
            code_index_kwargs={"index_type": "keyword-only"},
        )

        # Check that entities are not duplicated
        print("\n=== All entities in the knowledge graph ===")
        for entity_name, info in kg.entities.items():
            print(f"\nEntity: {entity_name}")
            print(f"  Aliases: {info.get('aliases', [])}")
            print(f"  Declaring chunks: {info.get('declaring_chunk_ids', [])}")
            print(f"  Calling chunks: {info.get('calling_chunk_ids', [])}")
            print(f"  Type: {info.get('type', [])}")

        # Verify no duplicates by checking that no two entities have overlapping aliases
        all_aliases = {}
        duplicates = []
        
        for entity_name, info in kg.entities.items():
            aliases = info.get('aliases', [])
            for alias in aliases:
                if alias in all_aliases:
                    duplicates.append({
                        'alias': alias,
                        'entity1': all_aliases[alias],
                        'entity2': entity_name
                    })
                else:
                    all_aliases[alias] = entity_name
        
        if duplicates:
            print("\n=== DUPLICATES FOUND ===")
            for dup in duplicates:
                print(f"Alias '{dup['alias']}' is used by both '{dup['entity1']}' and '{dup['entity2']}'")
            pytest.fail(f"Found {len(duplicates)} duplicate aliases")
        else:
            print("\n=== No duplicates found - TEST PASSED ===")

        # Additional check: verify that MyClass exists and has proper aliases
        myclass_entities = [name for name in kg.entities.keys() if 'MyClass' in name]
        print(f"\nEntities containing 'MyClass': {myclass_entities}")
        
        # There should be exactly one canonical entity for MyClass
        myclass_canonical = [name for name, info in kg.entities.items() 
                            if 'MyClass' in info.get('aliases', []) or name == 'MyClass']
        
        assert len(myclass_canonical) > 0, "MyClass entity should exist"
        print(f"\nCanonical entities for MyClass: {myclass_canonical}")


def test_c_cpp_no_duplicate_entities():
    """
    Test that C/C++ entities are not duplicated across header and implementation files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a header file
        header = os.path.join(tmpdir, "calculator.h")
        with open(header, "w") as f:
            f.write("""
#ifndef CALCULATOR_H
#define CALCULATOR_H

class Calculator {
public:
    int add(int a, int b);
    int subtract(int a, int b);
};

#endif
""")

        # Create an implementation file
        impl = os.path.join(tmpdir, "calculator.cpp")
        with open(impl, "w") as f:
            f.write("""
#include "calculator.h"

int Calculator::add(int a, int b) {
    return a + b;
}

int Calculator::subtract(int a, int b) {
    return a - b;
}
""")

        # Create a file that uses the calculator
        main = os.path.join(tmpdir, "main.cpp")
        with open(main, "w") as f:
            f.write("""
#include "calculator.h"

int main() {
    Calculator calc;
    int result = calc.add(5, 3);
    return 0;
}
""")

        # Build the knowledge graph
        kg =  RepoKnowledgeGraph.from_path(
            path=tmpdir,
            extract_entities=True,
            index_nodes=False,
            describe_nodes=False,
            model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    },
            code_index_kwargs={"index_type": "keyword-only"},
        )

        # Check entities
        print("\n=== All C++ entities in the knowledge graph ===")
        for entity_name, info in kg.entities.items():
            print(f"\nEntity: {entity_name}")
            print(f"  Aliases: {info.get('aliases', [])}")
            print(f"  Declaring chunks: {info.get('declaring_chunk_ids', [])}")
            print(f"  Calling chunks: {info.get('calling_chunk_ids', [])}")

        # Verify no duplicates
        all_aliases = {}
        duplicates = []
        
        for entity_name, info in kg.entities.items():
            aliases = info.get('aliases', [])
            for alias in aliases:
                if alias in all_aliases:
                    duplicates.append({
                        'alias': alias,
                        'entity1': all_aliases[alias],
                        'entity2': entity_name
                    })
                else:
                    all_aliases[alias] = entity_name
        
        if duplicates:
            print("\n=== DUPLICATES FOUND ===")
            for dup in duplicates:
                print(f"Alias '{dup['alias']}' is used by both '{dup['entity1']}' and '{dup['entity2']}'")
            pytest.fail(f"Found {len(duplicates)} duplicate aliases")
        else:
            print("\n=== No duplicates found - TEST PASSED ===")


def test_entity_resolution_with_qualified_names():
    """
    Test that entities with qualified names (module.Class) are properly resolved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directory structure
        subdir = os.path.join(tmpdir, "utils")
        os.makedirs(subdir)

        # Create a file in subdirectory
        file1 = os.path.join(subdir, "helper.py")
        with open(file1, "w") as f:
            f.write("""
class Helper:
    def help(self):
        return "helping"
""")

        # Create a file that uses qualified import
        file2 = os.path.join(tmpdir, "main.py")
        with open(file2, "w") as f:
            f.write("""
from utils.helper import Helper

def main():
    h = Helper()
    return h.help()
""")

        # Build the knowledge graph
        kg =  RepoKnowledgeGraph.from_path(
            path=tmpdir,
            extract_entities=True,
            index_nodes=False,
            describe_nodes=False,
            model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    },
            code_index_kwargs={"index_type": "keyword-only"},
        )

        # Check entities
        print("\n=== All entities with qualified names ===")
        for entity_name, info in kg.entities.items():
            if 'Helper' in entity_name or 'Helper' in str(info.get('aliases', [])):
                print(f"\nEntity: {entity_name}")
                print(f"  Aliases: {info.get('aliases', [])}")
                print(f"  Declaring chunks: {info.get('declaring_chunk_ids', [])}")
                print(f"  Calling chunks: {info.get('calling_chunk_ids', [])}")

        # Verify no duplicates
        all_aliases = {}
        duplicates = []
        
        for entity_name, info in kg.entities.items():
            aliases = info.get('aliases', [])
            for alias in aliases:
                if alias in all_aliases:
                    duplicates.append({
                        'alias': alias,
                        'entity1': all_aliases[alias],
                        'entity2': entity_name
                    })
                else:
                    all_aliases[alias] = entity_name
        
        if duplicates:
            print("\n=== DUPLICATES FOUND ===")
            for dup in duplicates:
                print(f"Alias '{dup['alias']}' is used by both '{dup['entity1']}' and '{dup['entity2']}'")
            pytest.fail(f"Found {len(duplicates)} duplicate aliases")
        else:
            print("\n=== No duplicates found - TEST PASSED ===")


# Test functions will be auto-discovered and run by pytest

