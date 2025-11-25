# Smolagent Chat Interface Setup

This setup provides a complete Knowledge Graph exploration system with two interfaces:
1. **Gradio MCP Interface** - Direct access to MCP server tools
2. **Smolagent Chat Interface** - AI-powered conversational agent

## Architecture

```
┌─────────────────────┐      ┌─────────────────────┐
│  Smolagent Chat     │      │   Gradio MCP        │
│  (Port 7861)        │      │   (Port 7860)       │
│  AI Chat Interface  │      │   Direct Tools      │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           │                            │
           └────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   MCP Server    │
              │   (Port 4000)   │
              │  /mcp endpoint  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Weaviate     │
              │   (Port 8080)   │
              │  Vector Store   │
              └─────────────────┘
```

## Prerequisites

1. **Docker and Docker Compose** installed
2. **OpenAI API Key** (or compatible endpoint like Azure OpenAI, vLLM, etc.)
3. **Environment Configuration** (`.env` file)

## Quick Start

### 1. Create `.env` file

Create a `.env` file in the project root with your configuration:

```bash
# OpenAI Configuration (required for smolagent-chat)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_API_VERSION=2024-02-15-preview

# Repository Configuration
REPO_URL=https://github.com/huggingface/transformers
# Or use local path:
# REPO_PATH=/path/to/your/repo

# Agent Configuration
MAX_STEPS=5
```

### 2. Launch All Services

```bash
docker-compose -f docker-compose-chat.yml up --build
```

This will start:
- **Weaviate** (vector database) on port 8080
- **MCP Server** (knowledge graph server) on port 4000
- **Gradio MCP** (direct tools interface) on port 7860
- **Smolagent Chat** (AI chat interface) on port 7861

### 3. Access the Interfaces

Once all services are healthy (check with `docker-compose -f docker-compose-chat.yml ps`):

- **AI Chat Interface**: http://localhost:7861
  - Conversational AI agent that can explore your codebase
  - Ask natural language questions
  - Get intelligent analysis and explanations

- **Direct MCP Interface**: http://localhost:7860
  - Direct access to all MCP server tools
  - More control over specific queries
  - Technical interface for developers

- **MCP Server API**: http://localhost:4000/mcp
  - REST API endpoint
  - Used by the chat agent

## Usage Examples

### Smolagent Chat Interface (Port 7861)

The AI-powered chat interface allows you to have natural conversations about your codebase:

**Example Questions:**
- "What does the RepoKnowledgeGraph class do?"
- "Show me all the functions that handle file parsing"
- "Find all usages of the search_nodes method"
- "Explain the architecture of this repository"
- "What are the main components and how do they interact?"
- "Find all classes that inherit from Node"
- "Which files have the most dependencies?"

**Features:**
- Natural language processing
- Context-aware responses
- Multi-step reasoning
- Automatic tool selection
- Follow-up question support

### Gradio MCP Interface (Port 7860)

Direct access to knowledge graph tools with structured inputs:

**Available Tools:**
- **Graph Overview**: Statistics and structure
- **Search**: Find nodes and entities
- **Node Info**: Detailed node information
- **Structure**: File and directory structure
- **Entities**: Entity definitions and usages
- **Relationships**: Node connections and paths
- **Analysis**: Code analysis and comparisons

## Configuration Options

### Environment Variables

#### Required for Chat Agent:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: API endpoint URL (default: OpenAI)
- `OPENAI_MODEL_NAME`: Model to use (default: gpt-4o-mini)

#### Optional:
- `MAX_STEPS`: Max reasoning steps for agent (default: 5)
- `REPO_URL`: GitHub repository URL to analyze
- `REPO_PATH`: Local path to repository
- `MCP_SERVER_URL`: MCP server endpoint (default: http://mcp-server:4000/mcp)

### Using Different LLM Providers

#### Azure OpenAI
```bash
OPENAI_API_KEY=your_azure_key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_MODEL_NAME=gpt-4
OPENAI_API_VERSION=2024-02-15-preview
```

#### Local vLLM Server
```bash
OPENAI_API_KEY=none
OPENAI_BASE_URL=http://vllm:8000/v1
OPENAI_MODEL_NAME=HuggingFaceTB/SmolLM3-3B
```

## Service Management

### Start Services
```bash
docker-compose -f docker-compose-chat.yml up -d
```

### Stop Services
```bash
docker-compose -f docker-compose-chat.yml down
```

### View Logs
```bash
# All services
docker-compose -f docker-compose-chat.yml logs -f

# Specific service
docker-compose -f docker-compose-chat.yml logs -f smolagent-chat
docker-compose -f docker-compose-chat.yml logs -f mcp-server
```

### Restart a Service
```bash
docker-compose -f docker-compose-chat.yml restart smolagent-chat
```

### Check Service Health
```bash
docker-compose -f docker-compose-chat.yml ps
```

## Troubleshooting

### Service Won't Start

1. **Check logs**:
   ```bash
   docker-compose -f docker-compose-chat.yml logs [service-name]
   ```

2. **Verify environment variables**:
   ```bash
   cat .env
   ```

3. **Check port conflicts**:
   ```bash
   lsof -i :7860
   lsof -i :7861
   lsof -i :4000
   lsof -i :8080
   ```

### MCP Server Connection Issues

The chat agent needs the MCP server to be healthy. Check:

```bash
# Check MCP server health
curl http://localhost:4000/health

# Verify MCP endpoint
curl http://localhost:4000/mcp
```

### Chat Agent Not Responding

1. **Verify OpenAI API key is valid**
2. **Check if MCP server is accessible**
3. **Review agent logs**:
   ```bash
   docker-compose -f docker-compose-chat.yml logs smolagent-chat
   ```

### Slow Initial Startup

The first run takes time because:
- MCP server needs to clone and analyze the repository
- This can take 2-5 minutes for large repos
- Wait until the MCP server health check passes

## Development

### Run Chat Agent Locally (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key
export MCP_SERVER_URL=http://localhost:4000/mcp

# Run the chat agent
python smolagent_chat.py --host 0.0.0.0 --port 7861
```

### Run MCP Server Locally

```bash
cd pedagogia_graph_code_repo
export REPO_URL=https://github.com/your/repo
python run_mcp_server.py
```

## Advanced Usage

### Custom Repository Analysis

Edit `docker-compose-chat.yml` to analyze your own repository:

```yaml
mcp-server:
  environment:
    - REPO_URL=https://github.com/your/repository
    # Or use local path:
    # - REPO_PATH=/data/your-repo
```

### Adjust Agent Behavior

Modify agent parameters in `docker-compose-chat.yml`:

```yaml
smolagent-chat:
  environment:
    - MAX_STEPS=10  # More reasoning steps
    - OPENAI_MODEL_NAME=gpt-4  # More powerful model
```

### Save Knowledge Graph

To save the generated knowledge graph for faster subsequent loads:

```yaml
volumes:
  - ./data:/data  # Persist graph data

command: >
  python gradio_mcp.py
  --graph-file /data/knowledge_graph.pkl
```

## Architecture Details

### Smolagent Chat Agent

The chat agent (`smolagent_chat.py`) provides:
- **MCPClient**: Connects to the MCP server over HTTP
- **CodeAgent**: Uses OpenAI models for reasoning
- **Gradio Interface**: User-friendly chat interface
- **Tool Integration**: Automatic selection and use of MCP tools

### MCP Server

The MCP server (`pedagogia_graph_code_repo/run_mcp_server.py`) provides:
- **Knowledge Graph**: Parsed code structure
- **MCP Tools**: RESTful tool endpoints
- **Entity Extraction**: Code entities and relationships
- **Vector Search**: Semantic code search

## License

See the main project LICENSE file.

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify your configuration
3. Review the troubleshooting section
4. Open an issue on the repository
