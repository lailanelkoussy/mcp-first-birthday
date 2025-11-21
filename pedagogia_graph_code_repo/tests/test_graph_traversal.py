"""
Test script for the new graph traversal methods in RepoKnowledgeGraph.
"""
import sys
from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph

def test_find_path():
    """Test the find_path method."""
    print("=" * 80)
    print("Testing find_path method")
    print("=" * 80)
    
    # Load a test graph
    try:
        kg = RepoKnowledgeGraph.load_graph_from_file(
            'data/solving-challenges-c/knowledge_graph.json',
            index_nodes=False
        )
        
        # Get a couple of nodes to test with
        nodes = list(kg.graph.nodes())[:10]
        print(f"\nAvailable nodes (showing first 10):")
        for i, node_id in enumerate(nodes):
            node = kg[node_id]
            print(f"  {i}. {node_id}: {getattr(node, 'name', 'Unknown')} ({getattr(node, 'node_type', 'Unknown')})")
        
        if len(nodes) >= 2:
            # Test path finding between first two nodes
            source_id = nodes[0]
            target_id = nodes[5] if len(nodes) > 5 else nodes[-1]
            
            print(f"\n\nFinding path from '{source_id}' to '{target_id}':")
            print("-" * 80)
            result = kg.find_path(source_id, target_id, max_depth=10)
            print(result['text'])
            print(f"\nResult keys: {list(result.keys())}")
        else:
            print("\nNot enough nodes to test path finding")
            
    except FileNotFoundError:
        print("Test graph file not found. Skipping test.")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def test_get_subgraph():
    """Test the get_subgraph method."""
    print("\n\n" + "=" * 80)
    print("Testing get_subgraph method")
    print("=" * 80)
    
    try:
        kg = RepoKnowledgeGraph.load_graph_from_file(
            'data/solving-challenges-c/knowledge_graph.json',
            index_nodes=False
        )
        
        # Get a node to test with
        nodes = list(kg.graph.nodes())[:5]
        if len(nodes) > 0:
            node_id = nodes[0]
            print(f"\n\nGetting subgraph around '{node_id}' with depth=2:")
            print("-" * 80)
            result = kg.get_subgraph(node_id, depth=2)
            print(result['text'])
            print(f"\nResult keys: {list(result.keys())}")
            
            # Test with edge type filtering
            print(f"\n\nGetting subgraph around '{node_id}' with depth=2, filtering by 'contains' edges:")
            print("-" * 80)
            result = kg.get_subgraph(node_id, depth=2, edge_types=['contains'])
            print(result['text'])
        else:
            print("\nNo nodes found to test")
            
    except FileNotFoundError:
        print("Test graph file not found. Skipping test.")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


# Test functions will be auto-discovered and run by pytest

