# 🚀 EPITA CodeVoyager

> **A conversational AI agent that helps you navigate and understand large codebases through natural language**

## 📚 What is EPITA CodeVoyager?

**EPITA CodeVoyager** is an interactive **chat agent** powered by [Smolagents](https://github.com/huggingface/smolagents) that connects to the **EPITA Codebase Knowledge Graph MCP Server**. It enables users to ask natural language questions about a codebase and receive accurate, grounded answers based on the actual code — not hallucinations.

### 🤗 Showcase: Hugging Face Transformers Library

We demonstrate EPITA CodeVoyager on the [**Hugging Face Transformers**](https://github.com/huggingface/transformers) library — one of the most popular open-source ML libraries with:
- **4,000+ files** across multiple modules
- **400,000+ lines of code**
- **Hundreds of model implementations** (BERT, GPT, LLaMA, etc.)
- **Complex inheritance hierarchies** and cross-file dependencies

This showcase demonstrates how the agent can help users understand and navigate even the most complex codebases through simple conversational queries.

### 🎯 Why This Matters for Education

Understanding large codebases is a **fundamental skill** for software engineers. As explained in the [main README](./README.md), at **EPITA** (École pour l'informatique et les techniques avancées), students work on increasingly complex projects and need to understand codebases — whether their own, their teammates', or open-source libraries.

LLM-based coding assistants face significant challenges with large repositories: context window limitations, lack of structural awareness, missing relationships, and inefficient search. **EPITA CodeVoyager** solves these problems by using MCP tools to **search**, **navigate**, and **understand** code repositories intelligently, making it an ideal assistant for developers, students, and educators exploring complex codebases.

## 🔬 Use Case: EPITA Coding Courses

Just like the [EPITA Codebase Knowledge Graph MCP Server](./README.md), **EPITA CodeVoyager** was developed with **educational applications** in mind, specifically to support **EPITA coding courses**.

### 🎯 The Educational Challenge

At **EPITA**, students work on increasingly complex software projects throughout their curriculum. Understanding large codebases — whether their own, their teammates', or open-source libraries like Transformers — is a fundamental skill for any computer science engineer.

However, navigating a library with **thousands of files** is overwhelming. Students often:
- Struggle to find where specific functionality is implemented
- Don't understand how different components connect
- Spend hours reading code without grasping the big picture
- Miss important design patterns and architectural decisions

### 💡 How EPITA CodeVoyager Helps

**EPITA CodeVoyager** addresses these challenges by enabling students to **ask questions in natural language**:

### 🔍 Intelligent Code Q&A

Instead of manually searching through thousands of files, users can simply ask:
- *"What classes inherit from `PreTrainedModel`?"*
- *"How is tokenization implemented in the library?"*
- *"What files are involved in the BERT implementation?"*

The agent uses MCP tools to explore the codebase, gather relevant information, and provide accurate, well-structured answers.

### 📈 Learning Through Exploration

For EPITA courses, this agent helps students:

- **Understand Architecture**: Ask about how components are organized and connected
- **Trace Code Flow**: Follow function calls and understand execution paths
- **Learn Design Patterns**: Discover implementation patterns used in real-world libraries
- **Prepare for Code Reviews**: Understand unfamiliar code before reviewing or contributing

### 🎓 EPITA Course Integration

- **Interactive Learning**: Students can explore open-source libraries conversationally
- **Office Hours Support**: Integrate with tutoring systems to answer code-related questions
- **Project Onboarding**: Help students understand project codebases quickly
- **Self-Paced Study**: Enable students to learn complex libraries at their own pace

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Gradio UI)                         │
│                                                                   │
│   "How does BertModel's forward method work?"                    │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EPITA CODEVOYAGER                             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    ToolCallingAgent                          │ │
│  │  • Receives natural language question                       │ │
│  │  • Decides which MCP tools to call                          │ │
│  │  • Chains multiple tool calls if needed                     │ │
│  │  • Synthesizes final answer from tool results               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Supported LLM Backends:                                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐             │
│  │  OpenAI  │  │ Azure OpenAI │  │ HF Inference   │             │
│  │ gpt-4o   │  │    gpt-4     │  │ Qwen2.5-Coder  │             │
│  └──────────┘  └──────────────┘  └────────────────┘             │
└───────────────────────────────┬───────────────────────────────────┘
                                │ MCP Protocol (HTTP)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         EPITA CODEBASE KNOWLEDGE GRAPH MCP SERVER                │
│                                                                   │
│  Tools Used by Agent:                                            │
│  ┌─────────────┐ ┌────────────┐ ┌──────────────┐                │
│  │search_nodes │ │go_to_def   │ │find_usages   │                │
│  └─────────────┘ └────────────┘ └──────────────┘                │
│  ┌─────────────┐ ┌────────────┐ ┌──────────────┐                │
│  │get_node_info│ │get_file_   │ │get_neighbors │                │
│  │             │ │structure   │ │              │                │
│  └─────────────┘ └────────────┘ └──────────────┘                │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                               │
│           (Hugging Face Transformers Library)                    │
│                                                                   │
│  • 4,000+ files indexed                                          │
│  • 400k+ lines of code                                           │
│  • Functions, classes, relationships extracted                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Features

### Multi-Provider LLM Support

The agent supports multiple LLM backends:

| Provider | Models | Configuration |
|----------|--------|---------------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo | API key + base URL |
| **Azure OpenAI** | gpt-4, gpt-4o (deployed) | API key + endpoint + version |
| **HuggingFace Inference** | Qwen2.5-Coder-32B, Llama-3.1, etc. | HF token + optional provider |

### Streaming Responses

The agent streams responses in real-time, showing:
- 🧠 **Model Thinking**: Internal reasoning displayed in collapsible sections
- 🔧 **Tool Calls**: Which MCP tools are being invoked
- 💬 **Final Answer**: Synthesized response based on code exploration

### Configurable Reasoning Steps

Control how deeply the agent explores:
- **Max Steps**: Limit the number of tool calls per query (default: 5)
- Lower values = faster responses, higher values = more thorough exploration

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A running Code Knowledge Graph MCP Server (see main README)
- API key for one of the supported LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/lailanelkoussy/mcp-first-birthday.git
cd mcp-first-birthday

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

#### 1. Start the MCP Server First

```bash
# Using the Gradio MCP server
python gradio_mcp.py --graph-file my_knowledge_graph.json --host 0.0.0.0 --port 7860
```

#### 2. Launch the Chat Agent

```bash
# Connect to local MCP server
python smolagent_chat.py --mcp-server-url http://localhost:7860/gradio_api/mcp/

# Or connect to a remote server (e.g., HuggingFace Space)
python smolagent_chat.py --mcp-server-url https://your-space.hf.space/gradio_api/mcp/

# With custom host/port
python smolagent_chat.py --host 0.0.0.0 --port 7861

# Create a public shareable link
python smolagent_chat.py --share
```

#### 3. Configure the LLM in the Web UI

Once launched, open the Gradio interface and configure your LLM provider:

**For OpenAI:**
- Model Type: `openai`
- Model Name: `gpt-4o-mini` (or `gpt-4o`, `gpt-4-turbo`)
- API Key: Your OpenAI API key
- Base URL: `https://api.openai.com/v1`

**For Azure OpenAI:**
- Model Type: `azure`
- Model Name: Your deployment name
- Azure API Key: Your Azure API key
- Azure Endpoint: `https://your-resource.openai.azure.com`
- API Version: `2024-02-15-preview`

**For HuggingFace Inference:**
- Model Type: `hf_inference`
- Model Name: `Qwen/Qwen2.5-Coder-32B-Instruct`
- HuggingFace Token: Your HF API token
- Provider (optional): `together`, `fireworks-ai`, `cerebras`

---

## ⚙️ Configuration

### Environment Variables

You can configure the agent using environment variables:

```bash
# MCP Server
export MCP_SERVER_URL="http://localhost:7860/gradio_api/mcp/"

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL_NAME="gpt-4o-mini"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export OPENAI_API_VERSION="2024-02-15-preview"

# HuggingFace Inference
export HF_TOKEN="hf_..."
export HF_MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export HF_INFERENCE_PROVIDER="together"  # optional

# Agent Settings
export MAX_STEPS=5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mcp-server-url` | URL of the MCP server | `http://localhost:4000/mcp` |
| `--host` | Host to bind the Gradio server | `0.0.0.0` |
| `--port` | Port for the Gradio server | `7861` |
| `--share` | Create a public Gradio link | `False` |

---

## 💡 Example Interactions

### Understanding a Class

**User:** *"How does the AutoModel class work?"*

**Agent:**
1. Calls `search_nodes("AutoModel")`
2. Calls `get_node_info("src/transformers/models/auto/auto_factory.py_3")`
3. Calls `get_file_structure("src/transformers/models/auto/auto_factory.py")`
4. Synthesizes response explaining the auto-class factory pattern

### Tracing Dependencies

**User:** *"What classes inherit from PreTrainedModel?"*

**Agent:**
1. Calls `go_to_definition("PreTrainedModel")`
2. Calls `find_usages("PreTrainedModel")`
3. Returns list of model classes with inheritance relationships

### Exploring Implementation

**User:** *"How does tokenization work in the library?"*

**Agent:**
1. Calls `search_nodes("tokenization")`
2. Calls `get_neighbors("src/transformers/tokenization_utils_base.py")`
3. Calls `get_file_structure("src/transformers/tokenization_utils.py")`
4. Explains the tokenizer hierarchy and key methods

---

## 🔧 Agent Internals

### KnowledgeGraphChatAgent Class

The main agent class handles:

```python
class KnowledgeGraphChatAgent:
    def __init__(self, mcp_server_url: str):
        # Connect to MCP server and load tools
        self._initialize_mcp_tools()
    
    def _initialize_model(self, model_type, api_key, ...):
        # Configure OpenAI, Azure, or HF Inference backend
    
    def _initialize_agent(self, max_steps):
        # Create ToolCallingAgent with MCP tools
    
    def chat(self, message, history):
        # Stream responses using stream_to_gradio
```

### Custom Instructions

EPITA CodeVoyager is configured with domain-specific instructions for the Transformers library:

```python
CUSTOM_INSTRUCTIONS = """You are an expert assistant for understanding the Hugging Face Transformers library.

Your role is to help users understand the Transformers codebase by exploring the repository using the available tools. You can:
- Search for functions, classes, and methods in the codebase
- Navigate the file structure and understand code organization
- Find relationships between different components
- Trace how code flows through the library
- Explain implementation details and design patterns

When answering questions:
1. Use the available tools to explore the repository and gather accurate information
2. Provide clear, well-structured explanations based on the actual code
3. Reference specific files, functions, or classes when relevant
4. If you're unsure about something, search the codebase to verify before answering

Always base your answers on the actual code in the repository, not assumptions."""
```

---

## 👥 Team

**Team Name:** CEPIA Ionis Team

**Team Members:**
- **Laila ELKOUSSY** - [@lailaelkoussy](https://huggingface.co/lailaelkoussy) - Research Engineer, Data Scientist
- **Julien PEREZ** - [@jnm38](https://huggingface.co/jnm38) - Research Director

---

## 📄 License

This project is developed as part of research at EPITA / Ionis Group.

## 🔗 Related Resources

- [Main README](./README.md) - Code Knowledge Graph MCP Server documentation
- [Smolagents](https://github.com/huggingface/smolagents) - The agent framework used
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) - The protocol standard
- [Gradio](https://gradio.app/) - Web interface framework
