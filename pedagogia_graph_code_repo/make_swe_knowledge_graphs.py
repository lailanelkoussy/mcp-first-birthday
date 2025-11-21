from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph
import os

from tqdm import tqdm

from pathlib import Path
import os.path
import tempfile
import subprocess


links = [
    "https://github.com/astropy/astropy.git",
    "https://github.com/django/django.git",
    "https://github.com/matplotlib/matplotlib.git",
    "https://github.com/mwaskom/seaborn.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/requests.git",
    "https://github.com/pydata/xarray.git",
    "https://github.com/pylint-dev/pylint.git",
    "https://github.com/pytest-dev/pytest.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/sphinx-doc/sphinx.git",
    "https://github.com/sympy/sympy.git"
]



def create_knowledge_graph_from_repo(repo_url: str, data_dir: str = '~/pedagogia-code-questions/data'):
    data_dir = os.path.expanduser(data_dir)
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    group_dir = os.path.join(data_dir, repo_name)
    print(f'group dir: {group_dir}')
    knowledge_graph_path = os.path.join(group_dir, 'multihop_knowledge_graph_with_embeddings.json')
    #if not os.path.exists(knowledge_graph_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f'Cloning repo to temp dir: {tmpdirname}')
        subprocess.run(['git', 'clone', repo_url, tmpdirname], check=True)
        Path(group_dir).mkdir(parents=True, exist_ok=True)
        knowledge_graph = RepoKnowledgeGraph.from_path(tmpdirname, index_nodes=True, describe_nodes=False, extract_entities=True, model_service_kwargs={"embedder_type": "sentence-transformers","embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",}, code_index_kwargs= {"index_type": "hybrid",'embedding_batch_size' : 20})
        knowledge_graph.save_graph_to_file(filepath=knowledge_graph_path)

DATA_DIR = '/app/data/swe_knowledge_graphs'
KG_FILENAME = 'multihop_knowledge_graph_with_embeddings.json'

def main():
    kg_paths = []
    for link in tqdm(links, desc="Processing GitHub repos"):
        repo_name = link.rstrip('/').split('/')[-1].replace('.git', '')
        group_dir = os.path.join(DATA_DIR, repo_name)
        kg_path = os.path.join(group_dir, KG_FILENAME)
        # Call the function to create the knowledge graph
        create_knowledge_graph_from_repo(link, data_dir=DATA_DIR)
        if os.path.exists(kg_path):
            kg_paths.append(kg_path)
        else:
            print(f"Knowledge graph not found for {repo_name}")



if __name__ == "__main__":
    main()
