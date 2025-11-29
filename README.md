---
title: Transformers Library Knowledge Graph
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Knowledge Graph MCP Explorer

This is a Gradio-based interactive tool for exploring code repository knowledge graphs. It provides a web interface to search, navigate, and analyze code relationships using the Model Context Protocol (MCP).

## Features

- **Search Nodes**: Search for code entities, functions, classes, and more using semantic search
- **Graph Navigation**: Explore relationships between code elements
- **Entity Tracking**: View declared and called entities within code chunks
- **Path Finding**: Find paths between different nodes in the knowledge graph
- **Subgraph Extraction**: Extract and visualize subgraphs around specific nodes
- **File Structure**: View the hierarchical structure of the repository

## Usage

The application loads a pre-built knowledge graph from the HuggingFace Transformers repository. You can:

1. **Search**: Use the search tab to find relevant code snippets and entities
2. **Explore**: Navigate through the graph using node IDs
3. **Analyze**: Get statistics about the code structure and relationships

## Technical Details

- Built with Gradio for the web interface
- Uses LanceDB for efficient code indexing and search
- Supports hybrid search (keyword + semantic embeddings)
- Pre-computed embeddings using Salesforce/SFR-Embedding-Code-400M_R model

## Data Sources

The application supports loading knowledge graphs from:

### 1. HuggingFace Hub Dataset (Recommended)

Load directly from a HuggingFace dataset:

```bash
python gradio_mcp.py --host 0.0.0.0 --port 7860 --hf-dataset "username/dataset-name"
```

### 2. Local JSON File

Use a local JSON file (e.g., `multihop_knowledge_graph_with_embeddings.json`):

```bash
python gradio_mcp.py --host 0.0.0.0 --port 7860 --graph-file data/multihop_knowledge_graph_with_embeddings.json
```

### Creating and Publishing a Dataset

You can save an existing knowledge graph to HuggingFace Hub:

```python
from RepoKnowledgeGraphLib import RepoKnowledgeGraph

# Load from local file
kg = RepoKnowledgeGraph.load("path/to/graph.json")

# Push to HuggingFace Hub (without embeddings to reduce size)
kg.to_hf_dataset("username/my-knowledge-graph", save_embeddings=False, private=False)

# Or with embeddings (larger dataset)
kg.to_hf_dataset("username/my-knowledge-graph-with-embeddings", save_embeddings=True)
```

## Docker Configuration

The default Dockerfile uses a local JSON file. To use HuggingFace datasets instead, modify the CMD line in `Dockerfile`:

```dockerfile
# Using HuggingFace dataset (recommended for smaller Docker image)
CMD ["python", "-u", "gradio_mcp.py", "--host", "0.0.0.0", "--port", "7860", "--hf-dataset", "username/dataset-name"]

# Using local file (requires large data file in image)
CMD ["python", "-u", "gradio_mcp.py", "--host", "0.0.0.0", "--port", "7860", "--graph-file", "/app/data/multihop_knowledge_graph_with_embeddings.json"]
```

## Local Development

To run locally:

```bash
docker build -t gradio-mcp-space .
docker run -p 7860:7860 gradio-mcp-space
```

Or without Docker:

```bash
pip install -r requirements.txt
python gradio_mcp.py --host 0.0.0.0 --port 7860 --hf-dataset "username/dataset-name"
```

## Deployment to HuggingFace Spaces

### Option 1: Using HuggingFace Dataset (Recommended)

1. First, push your knowledge graph to a HuggingFace dataset
2. Update the Dockerfile CMD to use `--hf-dataset`
3. Push to the Space repository (no large files needed)

### Option 2: Using Local JSON File

1. Create a new Space on HuggingFace with Docker SDK
2. Enable Git LFS in your Space repository
3. Push this directory to the Space repository:
   ```bash
   git lfs install
   git lfs track "data/*.json"
   git add .
   git commit -m "Initial commit"
   git push
   ```





## 👥 Team

**Team Name:** CEPIA Ionis Team

**Team Members:**
- **Laila ELKOUSSY** - [@lailaelkoussy](https://huggingface.co/lailaelkoussy) - Research Engineer, Data Scientist
- **Julien PEREZ** - [@jnm38](https://huggingface.co/jnm38) - 