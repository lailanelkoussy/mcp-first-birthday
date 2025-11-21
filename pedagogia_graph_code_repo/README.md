# Pedagogia Graph Code Repository

## Project Structure

```
├── docker-compose.yml                # Compose file for vllm and chroma services
├── docker-compose-transformers.yml   # Compose file for transformers-based testing
├── Dockerfile                        # Builds the main Python app image
├── requirements.txt                  # Python dependencies
├── test_transformers.py              # Example script using transformers
├── test_vllm.py                      # Example script using vllm
├── test_mcp.py                       # Test script for MCP server
├── test_mcp_smolagent.py             # Smolagents-based test for MCP server
├── knowledge_graph.json              # Output knowledge graph
├── knowledge_graph_1.json            # Output knowledge graph (alternate)
├── RepoKnowledgeGraphLib/            # Main library code
│   ├── __init__.py
│   ├── CodeIndex.py
│   ├── CodeParser.py
│   ├── Entity.py
│   ├── EntityChunkMapper.py
│   ├── EntityExtractor.py
│   ├── mcp_server.py                 # MCP server for accessing the knowledge graph
```

## Docker Compose Settings Overview

This project uses several Docker Compose files for different workflows. Here is a summary of each:

- **docker-compose.yml**: 
  - **Purpose**: Launches the core model services (vllm for LLM inference, Chroma for vector DB) and provides a base for running scripts in containers.
  - **Services**:
    - `chroma`: Chroma vector database (port 8000)
    - `vllm-embed`: vLLM server for code embedding model (port 8081)
    - `vllm-model`: vLLM server for main LLM model (port 8080)
    - `app-runner`: (builds from Dockerfile) for running custom scripts or apps
  - **When to use**: For general model serving, running scripts that need access to LLMs or Chroma DB, or as a base for other Compose setups.

- **docker-compose-transformers.yml**:
  - **Purpose**: For running Python scripts (e.g., `test_transformers.py`, `test_vllm.py`) that use HuggingFace Transformers and Chroma DB, in a containerized environment.
  - **Services**:
    - `app-runner`: Runs your Python script (default: `test_transformers.py`)
    - `chroma`: Chroma vector DB (port 8000)
  - **When to use**: For local or containerized testing of code that uses Transformers and Chroma, or for running batch jobs with mounted code/data.

- **docker-compose-mcp-transformers.yml**:
  - **Purpose**: For end-to-end testing of the MCP server and client, with Chroma DB as backend.
  - **Services**:
    - `mcp-server`: Runs the MCP server (default: `python run_mcp_server.py`)
    - `mcp-client`: Runs the MCP client test (default: `python test_mcp_client.py`)
    - `chroma`: Chroma vector DB (port 8000)
  - **When to use**: For integration testing of the MCP server and client, or to simulate LLM/agent interaction with the knowledge graph.

**Tip:** Use the `-f` option to specify which Compose file to use, e.g.,

```sh
docker compose -f docker-compose-transformers.yml up app-runner
```

See below for detailed usage examples for each workflow.

## MCP Server

The project includes an MCP (Model Context Protocol) server that provides programmatic access to the code knowledge graph.

### Running the MCP Server

#### Using Docker Compose

```bash
docker-compose up mcp-server
```

The MCP server will be available at `http://localhost:3000/sse`

#### Manual Execution

```bash
python RepoKnowledgeGraphLib/mcp_server.py --graph-file RepoKnowledgeGraphLib/knowledge_graph.json --port 3000
```

### Available MCP Tools

The MCP server exposes the following tools:

- `get_node_info(node_id)`: Get detailed information about a specific node
- `get_neighbors(node_id)`: Get the neighbors of a specific node
- `search_entities(entity_name)`: Search for entities by name
- `query_code(query, n_results=5)`: Query the code index for relevant chunks
- `get_graph_stats()`: Get basic statistics about the knowledge graph

### Testing the MCP Server

#### Using Docker Compose

```bash
# Start both MCP server and test
docker-compose up mcp-server mcp-test
```

#### Smolagents Test

The project also includes a test using the Smolagents library to demonstrate agent-based interaction with the MCP server. The test uses a CodeAgent with HuggingFace models for intelligent mission execution.

> **Note:** There is no Docker Compose setup for Smolagents. Please use the manual instructions below.

##### Manual Testing

```bash
# Install smolagents first
pip install smolagents

# Run the smolagent test
python test_mcp_smolagent.py

# For interactive mode
python test_mcp_smolagent.py --interactive

# Execute a specific mission
python test_mcp_smolagent.py --mission "Get graph statistics"
python test_mcp_smolagent.py --mission "Search for classes"
python test_mcp_smolagent.py --mission "List file nodes"
```

##### Mission Examples

The `--mission` option supports various natural language queries:

- `"Get graph statistics"` - Shows overall knowledge graph stats
- `"Search for classes"` - Finds nodes containing "class"
- `"List file nodes"` - Lists all file-type nodes
- `"Get node information"` - Shows info for the first available node
- `"Show node connections"` - Displays edges for the first available node

The smolagent test demonstrates how to wrap MCP server tools as agent tools and use them for knowledge graph queries.

#### Manual Testing

First, start the MCP server:

```bash
python RepoKnowledgeGraphLib/mcp_server.py --graph-file RepoKnowledgeGraphLib/knowledge_graph.json --port 3000
```

Then, in another terminal, run the test script:

```bash
python test_mcp.py --url http://localhost:3000
```

The test script will exercise all available MCP tools and report the results.
│   ├── ModelService.py
│   ├── Node.py
│   ├── QuestionMaker.py
│   ├── RepoKnowledgeGraph.py
│   └── utils/
│       ├── __init__.py
│       ├── chunk_utils.py
│       ├── data_utils.py
│       ├── logger_utils.py
│       ├── parsing_utils.py
│       └── path_utils.py
└── .env                              # (Optional) Environment variable overrides for docker-compose-app.yml
```

## Library Overview

- **RepoKnowledgeGraphLib/**: Contains the core logic for parsing code, extracting entities, building and saving knowledge graphs, and interacting with models.
- **test_transformers.py**: Example script using HuggingFace Transformers for model inference.
- **test_vllm.py**: Example script using vllm for model inference.
- **docker-compose.yml**: Launches vllm model servers and Chroma vector DB for serving models as APIs.
- **docker-compose-app.yml**: Runs Python scripts in a containerized environment, mounting your code/data as needed.

## Building and Running Examples

### 1. Build the Docker Image

```sh
docker compose -f docker-compose-transformers.yml build
```

### 2. Run Example Scripts

#### Run `test_transformers.py` on a specific path

Edit `docker-compose-app.yml` to mount your data/code path:

```yaml
    volumes:
      - .:/app
      - /your/host/path:/data/repo
    command: python test_transformers.py --path-name /data/repo
```

Then run:

```sh
docker compose -f docker-compose-transformers.yml up app-runner
```

#### Run `test_vllm.py` (or any other script)

Override the command as needed:

```sh
docker compose -f docker-compose-transformers.yml run app-runner python test_vllm.py --path-name /data/repo
```

Or edit the `command:` field in `docker-compose-app.yml`.

#### Use an Environment Variable for the Data Path (Optional)

You can use a `.env` file to set `DATA_PATH`:

```
DATA_PATH=/your/host/path
```

And in `docker-compose-app.yml`:

```yaml
    environment:
      - DATA_PATH=/your/host/path
    volumes:
      - .:/app
      - ${DATA_PATH}:/data/repo
    command: python test_transformers.py --path-name /data/repo
```

### 3. Launch Model Services (vllm) and Chroma Vector DB

To launch the vllm model servers and Chroma DB (for OpenAI-compatible API endpoints):

```sh
docker compose up
```

This will start:
- `vllm-embed`: vllm server for the embedding model (port 8081)
- `vllm-model`: vllm server for the main model (port 8080)
- `chroma`: Chroma vector DB (port 8000)

### 4. Example: Full Workflow

1. Start model services:
   ```sh
   docker compose up -d
   ```
2. Run your script (e.g., test_vllm.py) in the app container:
   ```sh
   docker compose -f docker-compose-transformers.yml run app-runner python test_vllm.py --path-name /data/repo
   ```

## Notes
- Always ensure the host path you mount exists and is accessible.
- You can add more scripts and run them in the same way by changing the `command`.
- For advanced configuration, edit the compose files or Dockerfile as needed.

## Troubleshooting
- If you get permission errors, check your host path permissions.
- If you change requirements.txt, rebuild the image: `docker compose -f docker-compose-app.yml build`
- For logs, check the `logs/` directory or container logs via `docker logs <container_name>`.

---

For further details, see comments in the compose files and Dockerfile.
