from RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph
import os
import sys

from tqdm import tqdm

from pathlib import Path
import os.path
import tempfile
import subprocess

# Ensure stdout is unbuffered for docker logs
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

links = [
    "https://github.com/huggingface/transformers",       # Transformers :contentReference[oaicite:0]{index=0}  
    "https://github.com/huggingface/datasets",           # Datasets :contentReference[oaicite:1]{index=1}  
    "https://github.com/huggingface/tokenizers",         # Tokenizers  
    "https://github.com/huggingface/diffusers",          # Diffusion models :contentReference[oaicite:2]{index=2}  
    "https://github.com/huggingface/peft",                # PEFT (Parameter-Efficient Fine-Tuning) :contentReference[oaicite:3]{index=3}  
    "https://github.com/huggingface/accelerate",          # Accelerate :contentReference[oaicite:4]{index=4}  
    "https://github.com/huggingface/optimum",             # Optimum (hardware optimization) :contentReference[oaicite:5]{index=5}  
    "https://github.com/huggingface/huggingface_hub",     # Hugging Face Hub client :contentReference[oaicite:6]{index=6}  
    "https://github.com/huggingface/trl",                 # Transformer RL training (RLHF) :contentReference[oaicite:7]{index=7}  
    "https://github.com/huggingface/evaluate",            # Evaluation library  
    "https://github.com/huggingface/safetensors",         # Safe tensor format  
    "https://github.com/huggingface/lighteval",           # Lightweight evaluation toolkit :contentReference[oaicite:8]{index=8}  
    "https://github.com/huggingface/course",              # Hugging Face Course :contentReference[oaicite:9]{index=9}  
    "https://github.com/huggingface/blog",                # Hugging Face Blog :contentReference[oaicite:10]{index=10}  
    "https://github.com/huggingface/awesome-huggingface", # Community projects list :contentReference[oaicite:11]{index=11}  
    "https://github.com/huggingface/xet-core",            # Xet client for large files / storage :contentReference[oaicite:12]{index=12}  
    "https://github.com/huggingface/sentence-transformers", # Sentence Transformers :contentReference[oaicite:13]{index=13}  
    "https://github.com/huggingface/optimum-neuron",      # Optimum for AWS Neuron (Trainium/Inferentia) :contentReference[oaicite:14]{index=14}  
]


# Get batch size from environment or use optimized default
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 64))

def create_knowledge_graph_from_repo(repo_url: str, data_dir: str = '~/pedagogia-code-questions/data'):
    data_dir = os.path.expanduser(data_dir)
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    group_dir = os.path.join(data_dir, repo_name)
    print(f'group dir: {group_dir}', flush=True)
    knowledge_graph_path = os.path.join(group_dir, 'multihop_knowledge_graph_with_embeddings.json')
    #if not os.path.exists(knowledge_graph_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f'Cloning repo to temp dir: {tmpdirname}', flush=True)
        subprocess.run(['git', 'clone', repo_url, tmpdirname], check=True)
        Path(group_dir).mkdir(parents=True, exist_ok=True)
        print(f'Starting knowledge graph creation with batch_size={EMBEDDING_BATCH_SIZE}', flush=True)
        knowledge_graph = RepoKnowledgeGraph.from_path(
            tmpdirname, 
            index_nodes=True, 
            describe_nodes=False, 
            extract_entities=True, 
            model_service_kwargs={
                "embedder_type": "sentence-transformers",
                "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
                "model_kwargs": {
                    "ENCODE_BATCH_SIZE": EMBEDDING_BATCH_SIZE,  # Pass batch size to model service
                }
            }, 
            code_index_kwargs={
                "index_type": "hybrid",
                "embedding_batch_size": EMBEDDING_BATCH_SIZE,  # Larger batch size for faster processing
            }
        )
        knowledge_graph.save_graph_to_file(filepath=knowledge_graph_path)
        print(f'Knowledge graph saved to: {knowledge_graph_path}', flush=True)

DATA_DIR = '/app/data/hf-repos'
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
