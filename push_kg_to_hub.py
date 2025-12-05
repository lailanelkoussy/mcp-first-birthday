from pedagogia_graph_code_repo.RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph

knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(
            "/app/pedagogia_graph_code_repo/data/hf-repos/transformers/multihop_knowledge_graph_with_embeddings.json",
            #repo_id="lailaelkoussy/transformers-knowledge-graph",
            index_nodes=True,
            model_service_kwargs={
                "embedder_type": "sentence-transformers",
                "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
            },
        code_index_kwargs = {
        "index_type": "keyword-only",
        "backend": "lancedb",
        "use_embed": False,
    },
        )
knowledge_graph.to_hf_dataset("lailaelkoussy/transformers-library-knowledge-graph", save_embeddings=False, private=True,)

print(knowledge_graph.code_index.query("class", n_results=1))