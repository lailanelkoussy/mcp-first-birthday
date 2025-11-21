"""
Test script for the RepoKnowledgeGraph library using Transformers.

This script demonstrates how to create, save, load, and query a repository knowledge graph
using the RepoKnowledgeGraph class with sentence-transformers embeddings.
"""

from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph


def test_repo_knowledge_graph_with_transformers():
    """
    Test repository knowledge graph with sentence-transformers embeddings.

    This function performs the following steps:
    1. Creates a knowledge graph from a repository.
    2. Retrieves and prints information about chunks and entities.
    3. Saves the graph to a file.
    4. Loads the graph from the file and performs queries.
    """
    import os
    path_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Configure the model service parameters for sentence-transformers embedding
    model_service_kwargs = {'embedder_type': 'sentence-transformers',
        'embed_model_name': "Salesforce/SFR-Embedding-Code-400M_R" }
    code_index_kwargs = { "index_type": "keyword-only",}
    
    try: 

        knowledge_graph = RepoKnowledgeGraph.from_repo(repo_url='https://github.com/lailanelkoussy/streamlit-fastapi-github-authentification', index_nodes=False, describe_nodes=False, extract_entities=True, model_service_kwargs=model_service_kwargs, code_index_kwargs={'index_type': 'keyword-only',})
        #knowledge_graph = RepoKnowledgeGraph.from_repo(repo_url='https://github.com/eliben/pycparser',
                                                    #index_nodes=False, describe_nodes=False, extract_entities=True,
                                                    #model_service_kwargs=model_service_kwargs, github_token=os.getenv('GITHUB_TOKEN', None))

        # Get all chunks from the knowledge graph and examine the first chunk
        all_chunks = knowledge_graph.get_all_chunks()
        chunk = all_chunks[0]
        print(f'Number of called entities: {len(chunk.called_entities)}')
        print(f'Number of defined entities: {len(chunk.declared_entities)}')
        print(f'Number of neighboring nodes: {len(knowledge_graph.get_neighbors(chunk.id))}')

        # Save the knowledge graph to a JSON file
        knowledge_graph.save_graph_to_file(filepath=path_name+ '/knowledge_graph.json')

        # Load the knowledge graph from the saved file
        # index_nodes=True: Index nodes for querying
        # use_embed=False: Do not use embeddings initially
        knowledge_graph_loaded = RepoKnowledgeGraph.load_graph_from_file(filepath=path_name + '/knowledge_graph.json',
                                                                    index_nodes=True,
                                                                    use_embed=False,
                                                                    model_service_kwargs=model_service_kwargs, code_index_kwargs={'index_type': 'hybrid',})
        all_chunks = knowledge_graph_loaded.get_all_chunks()
        chunk = all_chunks[0]
        print(f'Number of chunks: {len(all_chunks)}')
        print(f'Number of called entities: {len(chunk.called_entities)}')
        print(f'Number of defined entities: {len(chunk.declared_entities)}')
        print(f'Number of neighboring nodes: {len(knowledge_graph_loaded.get_neighbors(chunk.id))}')

        # Save the loaded graph again (demonstrating save/load cycle)
        knowledge_graph_loaded.save_graph_to_file(filepath=path_name+ '/knowledge_graph_1.json')

        # Load the graph again with embeddings enabled
        # use_embed=True: Enable embeddings for querying
        knowledge_graph_loaded_2 = RepoKnowledgeGraph.load_graph_from_file(filepath=path_name+ '/knowledge_graph_1.json',
                                                                        index_nodes=True,
                                                                        use_embed=True,
                                                                        model_service_kwargs=model_service_kwargs, code_index_kwargs={'index_type': 'hybrid',})
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
        #print((results['metadatas'][0])[0].keys())

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
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        assert False, f"Test failed due to an exception: {e}"


