"""
Test script for the RepoKnowledgeGraph library using VLLM (OpenAI-compatible API).

This script demonstrates how to create, save, load, and query a repository knowledge graph
using the RepoKnowledgeGraph class with OpenAI embeddings via local VLLM servers.
"""

from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph
import pytest


def test_repo_knowledge_graph_with_vllm():
    """
    Test repository knowledge graph with VLLM (OpenAI-compatible API).

    This function performs the following steps:
    1. Creates a knowledge graph from the repository path.
    2. Retrieves and prints information about chunks and entities.
    3. Saves the graph to a file.
    4. Loads the graph from the file and performs queries.

    Note: This test requires VLLM servers running on localhost:8080 and localhost:8081
    """
    pytest.skip("This test requires VLLM servers running locally")

    import os
    path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Configure the model service parameters for OpenAI embeddings via VLLM
    model_service_kwargs = {
        'embedder_type': 'openai',
        'embed_model_name': "Salesforce/SFR-Embedding-Code-400M_R",
        'openai_embed_base_url': "http://localhost:8081/v1",
        'openai_embed_token': "no-need",

        'model_name': "meta-llama/Llama-3.2-3B-Instruct",
        'openai_base_url': "http://localhost:8080/v1",
        'openai_token': "no-need",
    }

    # Create the knowledge graph from the repository path
    # index_nodes=False: Do not index nodes initially
    # describe_nodes=False: Do not describe nodes
    # extract_entities=True: Extract entities from the code
    knowledge_graph = RepoKnowledgeGraph.from_path(path_name, index_nodes=False, describe_nodes=False,
                                                   extract_entities=True, model_service_kwargs=model_service_kwargs)

    # Get all chunks from the knowledge graph and examine the first chunk
    all_chunks = knowledge_graph.get_all_chunks()
    chunk = all_chunks[0]
    print(f'Number of called entities: {len(chunk.called_entities)}')
    print(f'Number of defined entities: {len(chunk.declared_entities)}')
    print(f'Number of neighboring nodes: {len(knowledge_graph.get_neighbors(chunk.id))}')

    # Save the knowledge graph to a JSON file
    knowledge_graph.save_graph_to_file(filepath=path_name + '/knowledge_graph.json')

    # Load the knowledge graph from the saved file
    # index_nodes=True: Index nodes for querying
    # use_embed=False: Do not use embeddings initially
    knowledge_graph_loaded = RepoKnowledgeGraph.load_graph_from_file(filepath=path_name + '/knowledge_graph.json',
                                                                     index_nodes=True, use_embed=False,
                                                                     model_service_kwargs=model_service_kwargs)
    all_chunks = knowledge_graph_loaded.get_all_chunks()
    chunk = all_chunks[0]
    print(f'Number of chunks: {len(all_chunks)}')
    print(f'Number of called entities: {len(chunk.called_entities)}')
    print(f'Number of defined entities: {len(chunk.declared_entities)}')
    print(f'Number of neighboring nodes: {len(knowledge_graph_loaded.get_neighbors(chunk.id))}')

    # Save the loaded graph again (demonstrating save/load cycle)
    knowledge_graph_loaded.save_graph_to_file(filepath=path_name + '/knowledge_graph_1.json')

    # Load the graph again with embeddings enabled
    # use_embed=True: Enable embeddings for querying
    knowledge_graph_loaded_2 = RepoKnowledgeGraph.load_graph_from_file(filepath=path_name + '/knowledge_graph_1.json',
                                                                       index_nodes=True, use_embed=True,
                                                                       model_service_kwargs=model_service_kwargs)
    all_chunks = knowledge_graph_loaded_2.get_all_chunks()
    chunk = all_chunks[0]
    print(f'Number of chunks: {len(all_chunks)}')
    print(f'Number of called entities: {len(chunk.called_entities)}')
    print(f'Number of defined entities: {len(chunk.declared_entities)}')
    print(f'Number of neighboring nodes: {len(knowledge_graph_loaded_2.get_neighbors(chunk.id))}')

    # Query the code index for 'function' with 3 results
    results = knowledge_graph_loaded_2.code_index.query('function', n_results=3)
    print(f'Query result available keys: {results.keys()}')
    print(f'Query results:')

    # Print the query results
    for res in results['metadatas'][0]:
        print(f'===================={res["id"]}===================')
        print(res['content'])

        print('--------------------------------------------------')
        print(f'Declared entities: {res["declared_entities"]}')
        print(f'Called entities: {res["called_entities"]}')
        print('==================================================')

    # Assertions for pytest
    assert knowledge_graph is not None, "Knowledge graph should be created"
    assert len(all_chunks) > 0, "Should have at least one chunk"
    assert knowledge_graph_loaded is not None, "Loaded knowledge graph should not be None"
    assert knowledge_graph_loaded_2 is not None, "Second loaded knowledge graph should not be None"
    assert results is not None, "Query results should not be None"
    assert 'metadatas' in results, "Query results should contain metadatas"



