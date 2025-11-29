---
title: Code Repository Knowledge Graph MCP Server
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎓 Code Knowledge Graph MCP Server

> **Helping LLM-based agents navigate and understand large codebases**

## 📚 What is this project?

This project provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that transforms code repositories into navigable **knowledge graphs**. It enables Large Language Model (LLM) based agents to efficiently explore, understand, and reason about complex codebases — a critical capability for modern software engineering education and practice.

## 🔬 Use Case: EPITA Coding Courses

This project was developed with **educational applications** in mind, specifically to support **EPITA coding courses**:

### 🔍 Enhanced Code Discovery for Agents

LLM-based coding agents can use this tool to **better discover and navigate large repositories**. Instead of blindly searching through files, agents can:

- Query the knowledge graph to understand the overall architecture
- Follow relationships between modules, classes, and functions
- Identify entry points and critical code paths
- Understand how different parts of the codebase interact

### 📈 Detecting Areas for Code Improvement

For EPITA courses, this tool helps agents **identify areas where student code can be improved**:

- **Dead Code Detection**: Find unused functions, classes, or variables
- **Circular Dependencies**: Detect problematic import cycles between modules
- **Code Coupling Analysis**: Identify tightly coupled components that should be refactored
- **Missing Documentation**: Find undocumented public APIs and complex functions
- **Complexity Hotspots**: Locate chunks with many outgoing calls (high coupling)
- **Orphan Code**: Detect code that is declared but never called

### 🎓 EPITA Course Integration

- **Project Reviews**: Quickly understand student project architectures before grading
- **Automated Feedback**: Integrate with LLM tutors to provide targeted improvement suggestions
- **Code Quality Assessment**: Consistent evaluation criteria across student submissions
- **Learning Tool**: Help students navigate and understand unfamiliar codebases (e.g., open-source projects)
- **Research**: Study code organization patterns across student projects

The MCP interface makes it easy to integrate with any LLM-based tutoring or code review system used in EPITA courses.

---

### 🎯 The Problem We Solve

At **EPITA** (École pour l'informatique et les techniques avancées), students work on increasingly complex software projects throughout their curriculum. Understanding large codebases — whether their own, their teammates', or open-source libraries — is a fundamental skill for any computer science engineer.

However, LLM-based coding assistants face significant challenges when working with large repositories:

- **Context window limitations**: LLMs cannot process entire codebases at once
- **Lack of structural awareness**: Without understanding how code is organized, LLMs struggle to locate relevant files
- **Missing relationships**: Function calls, class inheritance, and module dependencies are not immediately visible
- **Inefficient search**: Simple keyword search fails to capture semantic meaning

### 💡 Our Solution: Knowledge Graphs + MCP

This project addresses these challenges by:

1. **Parsing repositories** into a structured knowledge graph (files → chunks → entities)
2. **Extracting relationships** between code elements (calls, contains, declares, imports)
3. **Indexing content** with hybrid search (semantic embeddings + keyword matching)
4. **Exposing tools via MCP** that allow LLM agents to navigate the codebase intelligently

```
┌─────────────────────────────────────────────────────────────────┐
│                     CODE REPOSITORY                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  File A  │  │  File B  │  │  File C  │  │  File D  │   ...   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│               KNOWLEDGE GRAPH CONSTRUCTION                       │
│  • AST Parsing (Python, C/C++, Java, JavaScript, Rust, HTML)    │
│  • Entity Extraction (classes, functions, variables, methods)   │
│  • Relationship Detection (calls, inheritance, imports)         │
│  • Code Chunking & Embedding (semantic vectors)                 │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP SERVER (FastMCP)                          │
│  ┌─────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │search_nodes │ │go_to_def   │ │find_usages   │ │get_neighbors│ │
│  └─────────────┘ └────────────┘ └──────────────┘ └────────────┘ │
│  ┌─────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │get_file_    │ │get_related │ │find_path     │ │print_tree  │ │
│  │structure    │ │_chunks     │ │              │ │            │ │
│  └─────────────┘ └────────────┘ └──────────────┘ └────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM-BASED AGENT                               │
│  • Can search for relevant code using natural language          │
│  • Navigate from function calls to their definitions            │
│  • Understand the structure of files and directories            │
│  • Trace dependencies and relationships across the codebase     │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ MCP Tools Available

The MCP server exposes the following tools for LLM agents:

| Tool | Description |
|------|-------------|
| `search_nodes` | Semantic + keyword search for code chunks |
| `get_node_info` | Detailed information about any node (file, chunk, entity) |
| `get_node_edges` | Incoming and outgoing relationships of a node |
| `go_to_definition` | Find where a function/class/variable is declared |
| `find_usages` | Find all places where an entity is called/used |
| `get_neighbors` | Get all directly connected nodes |
| `get_file_structure` | Overview of a file's chunks and entities |
| `get_related_chunks` | Find chunks related by a specific relationship type |
| `list_all_entities` | List all tracked entities in the codebase |
| `get_graph_stats` | Statistics about the knowledge graph |
| `find_path` | Find shortest path between two nodes |
| `get_subgraph` | Extract a subgraph around a node |
| `print_tree` | Display repository structure as a tree |
| `diff_chunks` | Compare content between two code chunks |
| `search_by_type_and_name` | Search entities by type (class, function, etc.) and name |
| `get_chunk_context` | Get a chunk with its surrounding context |

## 🌐 Supported Languages

The knowledge graph builder uses **AST-based entity extraction** for accurate parsing:

| Language | Parser | Entity Types |
|----------|--------|--------------|
| Python | `ast` module | classes, functions, methods, variables, imports |
| C | `libclang` | functions, structs, typedefs, variables |
| C++ | `libclang` | classes, namespaces, methods, templates |
| Java | `javalang` | classes, interfaces, methods, fields |
| JavaScript/TypeScript | `esprima` | classes, functions, variables, imports |
| Rust | `tree-sitter` | structs, enums, traits, functions, modules |
| HTML | `BeautifulSoup` | DOM elements, inline JS extraction |

The system also detects **API endpoints** for web frameworks (FastAPI, Flask, Spring Boot, Actix-web, etc.).

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- CUDA-capable GPU (optional, for faster embeddings)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/lailanelkoussy/mcp-first-birthday.git
cd mcp-first-birthday

# Start the MCP server with a sample knowledge graph
docker-compose up
```

### Building a Knowledge Graph from Your Repository

```python
from pedagogia_graph_code_repo.RepoKnowledgeGraphLib import RepoKnowledgeGraph

# From a local path
kg = RepoKnowledgeGraph.from_path(
    "/path/to/your/repo",
    skip_dirs=["node_modules", ".git", "__pycache__"],
    extract_entities=True,
    index_nodes=True
)

# Save for later use
kg.save_graph_to_file("my_knowledge_graph.json")
```

### Running the MCP Server

```bash
# Using the Gradio interface (recommended for exploration)
python gradio_mcp.py --graph-file my_knowledge_graph.json --host 0.0.0.0 --port 7860

# Or directly as an MCP server
python pedagogia_graph_code_repo/run_mcp_server.py --graph-file my_knowledge_graph.json
```

## 📊 Interactive Explorer (Gradio UI)

The project includes a Gradio-based web interface for exploring knowledge graphs interactively:

- **Search**: Use natural language or keywords to find relevant code
- **Navigate**: Click through nodes to explore relationships  
- **Analyze**: Get statistics about code structure and dependencies
- **Visualize**: View the repository tree and entity relationships

## 📁 Data Sources

The application supports loading knowledge graphs from multiple sources:

### 1. HuggingFace Hub Dataset (Recommended for Sharing)

Load directly from a HuggingFace dataset:

```bash
python gradio_mcp.py --host 0.0.0.0 --port 7860 --hf-dataset "username/dataset-name"
```

### 2. Local JSON File

Use a local JSON file (e.g., `multihop_knowledge_graph_with_embeddings.json`):

```bash
python gradio_mcp.py --host 0.0.0.0 --port 7860 --graph-file data/multihop_knowledge_graph_with_embeddings.json
```

### 3. Direct from Git Repository

Clone and analyze a repository on-the-fly:

```bash
python gradio_mcp.py --host 0.0.0.0 --port 7860 --repo-url "https://github.com/user/repo.git"
```

### Publishing to HuggingFace Hub

You can save an existing knowledge graph to HuggingFace Hub for sharing:

```python
from RepoKnowledgeGraphLib import RepoKnowledgeGraph

# Load from local file
kg = RepoKnowledgeGraph.load("path/to/graph.json")

# Push to HuggingFace Hub (without embeddings to reduce size)
kg.to_hf_dataset("username/my-knowledge-graph", save_embeddings=False, private=False)

# Or with embeddings (larger dataset)
kg.to_hf_dataset("username/my-knowledge-graph-with-embeddings", save_embeddings=True)
```

## 🐳 Docker Configuration

The default Dockerfile uses a local JSON file. To use HuggingFace datasets instead, modify the CMD line in `Dockerfile`:

```dockerfile
# Using HuggingFace dataset (recommended for smaller Docker image)
CMD ["python", "-u", "gradio_mcp.py", "--host", "0.0.0.0", "--port", "7860", "--hf-dataset", "username/dataset-name"]

# Using local file (requires large data file in image)
CMD ["python", "-u", "gradio_mcp.py", "--host", "0.0.0.0", "--port", "7860", "--graph-file", "/app/data/multihop_knowledge_graph_with_embeddings.json"]
```

## 💻 Local Development

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

## ☁️ Deployment to HuggingFace Spaces

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

## 🏗️ Architecture Overview

```
mcp-first-birthday/
├── gradio_mcp.py              # Main Gradio web interface
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── pedagogia_graph_code_repo/  # Core library
│   ├── RepoKnowledgeGraphLib/  # Knowledge graph implementation
│   │   ├── RepoKnowledgeGraph.py    # Main graph class
│   │   ├── KnowledgeGraphMCPServer.py # MCP server implementation
│   │   ├── EntityExtractor.py       # AST-based entity extraction
│   │   ├── CodeParser.py            # Code chunking
│   │   ├── CodeIndex.py             # Hybrid search (LanceDB/Weaviate)
│   │   ├── ModelService.py          # Embedding generation
│   │   └── Node.py                  # Graph node types
│   ├── run_mcp_server.py            # Standalone MCP server
│   └── tests/                       # Test suite
└── docker-compose*.yml         # Docker configurations
```





## 👥 Team

**Team Name:** CEPIA Ionis Team

**Team Members:**
- **Laila ELKOUSSY** - [@lailaelkoussy](https://huggingface.co/lailaelkoussy) - Research Engineer, Data Scientist
- **Julien PEREZ** - [@jnm38](https://huggingface.co/jnm38) - Research Director

---

## 📄 License

This project is developed as part of research at EPITA / Ionis Group.

## 🔗 Related Resources

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) - The protocol standard
- [FastMCP](https://github.com/jlowin/fastmcp) - Python MCP framework used
- [LanceDB](https://lancedb.github.io/lancedb/) - Vector database for code indexing
- [Salesforce SFR-Embedding-Code](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R) - Code embedding model 