#!/usr/bin/env python3
"""
Test script for re-added methods in RepoKnowledgeGraph.
Tests: print_tree, to_dict, from_dict, save_graph_to_file, load_graph_from_file,
       get_neighbors, get_previous_chunk, get_next_chunk, get_all_chunks,
       get_all_files, get_chunks_of_file, find_path, get_subgraph
"""

import os
import json
import tempfile
from io import StringIO
import sys

from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph
from RepoKnowledgeGraphLib.Node import Node, FileNode, ChunkNode, DirectoryNode


def test_graph_serialization():
    """Test to_dict and from_dict methods"""
    print("=" * 80)
    print("Testing to_dict and from_dict methods")
    print("=" * 80)
    
    # Create a simple test graph
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
""")
        
        # Build graph from path
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Test to_dict
        graph_dict = kg.to_dict()
        assert 'nodes' in graph_dict, "to_dict should return dict with 'nodes' key"
        assert 'edges' in graph_dict, "to_dict should return dict with 'edges' key"
        assert len(graph_dict['nodes']) > 0, "Graph should have nodes"
        print(f"✓ to_dict created dictionary with {len(graph_dict['nodes'])} nodes and {len(graph_dict['edges'])} edges")
        
        # Test from_dict
        kg_restored = RepoKnowledgeGraph.from_dict(
            graph_dict,
            index_nodes=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        assert kg_restored.graph.number_of_nodes() == kg.graph.number_of_nodes(), "Restored graph should have same number of nodes"
        assert kg_restored.graph.number_of_edges() == kg.graph.number_of_edges(), "Restored graph should have same number of edges"
        print(f"✓ from_dict restored graph with {kg_restored.graph.number_of_nodes()} nodes and {kg_restored.graph.number_of_edges()} edges")
        print()


def test_save_and_load_graph():
    """Test save_graph_to_file and load_graph_from_file methods"""
    print("=" * 80)
    print("Testing save_graph_to_file and load_graph_from_file methods")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_py = os.path.join(tmpdir, "example.py")
        with open(test_py, 'w') as f:
            f.write("""
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
""")
        
        # Build graph
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Save to file
        graph_file = os.path.join(tmpdir, "test_graph.json")
        kg.save_graph_to_file(graph_file)
        assert os.path.exists(graph_file), "Graph file should be created"
        print(f"✓ Graph saved to {graph_file}")
        
        # Load from file
        kg_loaded = RepoKnowledgeGraph.load_graph_from_file(
            graph_file,
            index_nodes=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        assert kg_loaded.graph.number_of_nodes() == kg.graph.number_of_nodes(), "Loaded graph should have same number of nodes"
        assert kg_loaded.graph.number_of_edges() == kg.graph.number_of_edges(), "Loaded graph should have same number of edges"
        print(f"✓ Graph loaded with {kg_loaded.graph.number_of_nodes()} nodes and {kg_loaded.graph.number_of_edges()} edges")
        
        # Verify JSON structure
        with open(graph_file, 'r') as f:
            data = json.load(f)
            assert 'nodes' in data, "Saved JSON should have 'nodes' key"
            assert 'edges' in data, "Saved JSON should have 'edges' key"
            print(f"✓ Saved JSON has correct structure")
        print()


def test_print_tree():
    """Test print_tree method"""
    print("=" * 80)
    print("Testing print_tree method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directory structure
        subdir = os.path.join(tmpdir, "src")
        os.makedirs(subdir)
        
        test_py1 = os.path.join(tmpdir, "main.py")
        with open(test_py1, 'w') as f:
            f.write("print('Main')")
        
        test_py2 = os.path.join(subdir, "utils.py")
        with open(test_py2, 'w') as f:
            f.write("def util(): pass")
        
        # Build graph
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            kg.print_tree(max_depth=3)
            output = captured_output.getvalue()
            
            # Check that output contains expected elements
            assert 'root' in output, "Tree should contain root node"
            assert any(char in output for char in ['📁', '📂', '📄', '📝']), "Tree should contain node symbols"
            print(f"✓ print_tree generated output with {len(output)} characters", file=old_stdout)
            print(f"Sample output:\n{output[:200]}...", file=old_stdout)
        finally:
            sys.stdout = old_stdout
        print()


def test_get_neighbors():
    """Test get_neighbors method"""
    print("=" * 80)
    print("Testing get_neighbors method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
def function1():
    return 1

def function2():
    return 2
""")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Get neighbors of root
        neighbors = kg.get_neighbors('root')
        assert len(neighbors) > 0, "Root should have neighbors"
        print(f"✓ Root node has {len(neighbors)} neighbors")
        
        # Get neighbors of a file node
        file_nodes = [n for n in kg.graph.nodes() if isinstance(kg[n], FileNode)]
        if file_nodes:
            file_neighbors = kg.get_neighbors(file_nodes[0])
            assert len(file_neighbors) >= 0, "File node should have neighbors or be isolated"
            print(f"✓ File node has {len(file_neighbors)} neighbors")
        print()


def test_get_all_chunks():
    """Test get_all_chunks method"""
    print("=" * 80)
    print("Testing get_all_chunks method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
# This is a test file
def function1():
    return 1

def function2():
    return 2

def function3():
    return 3
""")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        chunks = kg.get_all_chunks()
        assert len(chunks) > 0, "Graph should have chunks"
        assert all(isinstance(chunk, ChunkNode) for chunk in chunks), "All returned items should be ChunkNodes"
        print(f"✓ Found {len(chunks)} chunk nodes")
        
        # Verify chunks have required attributes
        if chunks:
            first_chunk = chunks[0]
            assert hasattr(first_chunk, 'content'), "Chunk should have content"
            assert hasattr(first_chunk, 'order_in_file'), "Chunk should have order_in_file"
            print(f"✓ Chunks have required attributes")
        print()


def test_get_all_files():
    """Test get_all_files method"""
    print("=" * 80)
    print("Testing get_all_files method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple files
        for i in range(3):
            test_py = os.path.join(tmpdir, f"test{i}.py")
            with open(test_py, 'w') as f:
                f.write(f"# File {i}\ndef func{i}(): pass")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        files = kg.get_all_files()
        assert len(files) == 3, f"Should have 3 files, got {len(files)}"
        assert all(isinstance(f, FileNode) for f in files), "All returned items should be FileNodes"
        print(f"✓ Found {len(files)} file nodes")
        
        # Verify files have required attributes
        if files:
            first_file = files[0]
            assert hasattr(first_file, 'path'), "File should have path"
            assert hasattr(first_file, 'content'), "File should have content"
            print(f"✓ Files have required attributes")
        print()


def test_get_chunks_of_file():
    """Test get_chunks_of_file method"""
    print("=" * 80)
    print("Testing get_chunks_of_file method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
def function1():
    return 1

def function2():
    return 2

def function3():
    return 3

def function4():
    return 4
""")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Get file nodes
        file_nodes = kg.get_all_files()
        assert len(file_nodes) > 0, "Should have at least one file"
        
        # Get chunks for first file
        file_node_id = file_nodes[0].id
        chunks = kg.get_chunks_of_file(file_node_id)
        assert len(chunks) > 0, "File should have chunks"
        assert all(isinstance(chunk, ChunkNode) for chunk in chunks), "All returned items should be ChunkNodes"
        print(f"✓ File '{file_node_id}' has {len(chunks)} chunks")
        
        # Verify chunks belong to the file
        for chunk in chunks:
            assert chunk.path == file_node_id, f"Chunk path should match file id"
        print(f"✓ All chunks belong to the correct file")
        print()


def test_get_previous_and_next_chunk():
    """Test get_previous_chunk and get_next_chunk methods"""
    print("=" * 80)
    print("Testing get_previous_chunk and get_next_chunk methods")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
def function1():
    return 1

def function2():
    return 2

def function3():
    return 3
""")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        chunks = kg.get_all_chunks()
        # Sort chunks by order_in_file to ensure correct ordering
        chunks_sorted = sorted(chunks, key=lambda c: c.order_in_file)
        
        if len(chunks_sorted) >= 2:
            # Test get_next_chunk
            first_chunk = chunks_sorted[0]
            next_chunk = kg.get_next_chunk(first_chunk.id)
            if next_chunk:
                assert next_chunk.order_in_file == first_chunk.order_in_file + 1, "Next chunk should have order_in_file + 1"
                print(f"✓ get_next_chunk works correctly")
            
            # Test get_previous_chunk
            second_chunk = chunks_sorted[1]
            prev_chunk = kg.get_previous_chunk(second_chunk.id)
            assert prev_chunk is not None, "Second chunk should have a previous chunk"
            assert prev_chunk.order_in_file == second_chunk.order_in_file - 1, "Previous chunk should have order_in_file - 1"
            print(f"✓ get_previous_chunk works correctly")
            
            # Test boundary conditions
            first_prev = kg.get_previous_chunk(chunks_sorted[0].id)
            assert first_prev is None, "First chunk should not have a previous chunk"
            print(f"✓ Boundary condition: first chunk has no previous")
            
            last_next = kg.get_next_chunk(chunks_sorted[-1].id)
            assert last_next is None, "Last chunk should not have a next chunk"
            print(f"✓ Boundary condition: last chunk has no next")
        else:
            print("⚠ Not enough chunks to test prev/next functionality")
        print()


def test_find_path():
    """Test find_path method"""
    print("=" * 80)
    print("Testing find_path method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = os.path.join(tmpdir, "src")
        os.makedirs(subdir)
        
        test_py = os.path.join(subdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("def test(): pass")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Get some nodes
        nodes = list(kg.graph.nodes())
        if len(nodes) >= 2:
            # Test path finding
            result = kg.find_path(nodes[0], nodes[-1], max_depth=10)
            assert 'source_id' in result, "Result should have source_id"
            assert 'target_id' in result, "Result should have target_id"
            assert 'path' in result, "Result should have path"
            assert 'length' in result, "Result should have length"
            assert 'text' in result, "Result should have text"
            
            if 'error' not in result and result['length'] >= 0:
                print(f"✓ Found path of length {result['length']}")
                assert len(result['path']) == result['length'] + 1, "Path length should match node count"
                print(f"✓ Path has correct number of nodes")
            else:
                print(f"✓ find_path handled no-path case correctly")
        
        # Test error cases
        result_error = kg.find_path('nonexistent_node', nodes[0])
        assert 'error' in result_error, "Should return error for nonexistent node"
        print(f"✓ Handles nonexistent node correctly")
        print()


def test_get_subgraph():
    """Test get_subgraph method"""
    print("=" * 80)
    print("Testing get_subgraph method")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure with multiple files
        subdir = os.path.join(tmpdir, "src")
        os.makedirs(subdir)
        
        test_py1 = os.path.join(tmpdir, "main.py")
        with open(test_py1, 'w') as f:
            f.write("def main(): pass")
        
        test_py2 = os.path.join(subdir, "utils.py")
        with open(test_py2, 'w') as f:
            f.write("def util(): pass")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=False,
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Test get_subgraph around root
        result = kg.get_subgraph('root', depth=2)
        assert 'center_node_id' in result, "Result should have center_node_id"
        assert 'depth' in result, "Result should have depth"
        assert 'nodes' in result, "Result should have nodes"
        assert 'edges' in result, "Result should have edges"
        assert 'node_count' in result, "Result should have node_count"
        assert 'edge_count' in result, "Result should have edge_count"
        assert 'text' in result, "Result should have text"
        
        assert result['node_count'] > 0, "Subgraph should have nodes"
        assert result['depth'] == 2, "Depth should match requested depth"
        print(f"✓ Subgraph around 'root' has {result['node_count']} nodes and {result['edge_count']} edges")
        
        # Test with edge type filtering
        result_filtered = kg.get_subgraph('root', depth=2, edge_types=['contains'])
        assert result_filtered['node_count'] > 0, "Filtered subgraph should have nodes"
        print(f"✓ Filtered subgraph (contains only) has {result_filtered['node_count']} nodes")
        
        # Test error case
        result_error = kg.get_subgraph('nonexistent_node', depth=1)
        assert 'error' in result_error, "Should return error for nonexistent node"
        print(f"✓ Handles nonexistent node correctly")
        print()


def test_entity_alias_methods():
    """Test get_entity_by_alias and resolve_entity_references methods"""
    print("=" * 80)
    print("Testing get_entity_by_alias and resolve_entity_references methods")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_py = os.path.join(tmpdir, "test.py")
        with open(test_py, 'w') as f:
            f.write("""
class MyClass:
    def my_method(self):
        pass

def my_function():
    obj = MyClass()
    obj.my_method()
""")
        
        kg = RepoKnowledgeGraph.from_path(
            tmpdir,
            index_nodes=False,
            describe_nodes=False,
            extract_entities=True,  # Enable entity extraction
            model_service_kwargs={"embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}
        )
        
        # Test get_entity_by_alias
        if kg.entities:
            entity_names = list(kg.entities.keys())
            first_entity = entity_names[0]
            result = kg.get_entity_by_alias(first_entity)
            # Result should be the entity name itself or a canonical name
            print(f"✓ get_entity_by_alias returned result for '{first_entity}': {result}")
        else:
            print("⚠ No entities found to test get_entity_by_alias")
        
        # Test resolve_entity_references
        resolutions = kg.resolve_entity_references()
        assert isinstance(resolutions, dict), "resolve_entity_references should return a dict"
        print(f"✓ resolve_entity_references returned dict with {len(resolutions)} resolutions")
        print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS FOR RE-ADDED METHODS")
    print("=" * 80 + "\n")
    
    test_functions = [
        test_graph_serialization,
        test_save_and_load_graph,
        test_print_tree,
        test_get_neighbors,
        test_get_all_chunks,
        test_get_all_files,
        test_get_chunks_of_file,
        test_get_previous_and_next_chunk,
        test_find_path,
        test_get_subgraph,
        test_entity_alias_methods,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
