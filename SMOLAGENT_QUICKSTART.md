# Smolagent Chat Agent - Quick Reference

## 🎯 What's New

This setup adds an AI-powered chat interface to your Knowledge Graph MCP server:

### New Files Created:
1. **`smolagent_chat.py`** - Smolagents agent with Gradio chat interface
2. **`docker-compose-chat.yml`** - Complete Docker Compose setup
3. **`SMOLAGENT_CHAT_SETUP.md`** - Comprehensive documentation
4. **`.env.example`** - Environment configuration template
5. **`start-chat.sh`** - Quick start script

## 🚀 Quick Start (30 seconds)

```bash
# 1. Copy .env.example to .env and add your OpenAI API key
cp .env.example .env
nano .env  # Set OPENAI_API_KEY

# 2. Run the quick start script
./start-chat.sh

# 3. Open your browser
# AI Chat:  http://localhost:7861
# MCP Tools: http://localhost:7860
```

## 🏗️ What Gets Launched

| Service | Port | Description |
|---------|------|-------------|
| Smolagent Chat | 7861 | AI-powered conversational interface |
| Gradio MCP | 7860 | Direct MCP tools interface |
| MCP Server | 4000 | Knowledge graph API |
| Weaviate | 8080 | Vector database |

## 💬 Example Chat Queries

Try asking the AI agent:
- "What does the RepoKnowledgeGraph class do?"
- "Find all functions that parse code"
- "Show me the file structure of this repository"
- "Which classes inherit from Node?"
- "Explain the architecture of this codebase"

## 🛠️ Manual Commands

```bash
# Start all services
docker-compose -f docker-compose-chat.yml up -d

# Stop all services
docker-compose -f docker-compose-chat.yml down

# View logs
docker-compose -f docker-compose-chat.yml logs -f

# Restart chat agent only
docker-compose -f docker-compose-chat.yml restart smolagent-chat
```

## 📚 Documentation

- **Full Setup Guide**: `SMOLAGENT_CHAT_SETUP.md`
- **Configuration**: `.env.example`
- **Architecture**: See diagram in SMOLAGENT_CHAT_SETUP.md

## 🔧 Configuration

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_MODEL_NAME=gpt-4o-mini  # or gpt-4
MAX_STEPS=5  # Agent reasoning steps
REPO_URL=https://github.com/your/repo  # Repository to analyze
```

## 🐛 Troubleshooting

### Chat agent won't start
```bash
# Check if MCP server is healthy
docker-compose -f docker-compose-chat.yml ps

# View chat agent logs
docker-compose -f docker-compose-chat.yml logs smolagent-chat
```

### Connection refused
Wait for MCP server to be healthy (takes 2-5 minutes on first run)

### API errors
Verify your `OPENAI_API_KEY` in `.env`

## 🎨 Features

### AI Chat Agent (`smolagent_chat.py`)
- ✅ Natural language queries
- ✅ Multi-step reasoning
- ✅ Automatic tool selection
- ✅ Context-aware responses
- ✅ Follow-up questions
- ✅ Clean Gradio interface

### Integration
- ✅ Connects to MCP server via HTTP
- ✅ Uses all available MCP tools
- ✅ Supports OpenAI, Azure OpenAI, vLLM
- ✅ Configurable agent behavior
- ✅ Docker Compose orchestration

## 📊 Differences from MCP Interface

| Feature | MCP Interface (7860) | Chat Agent (7861) |
|---------|---------------------|-------------------|
| Input | Structured forms | Natural language |
| Tools | Manual selection | Automatic selection |
| Reasoning | Single-step | Multi-step |
| Context | None | Conversation history |
| Best for | Precise queries | Exploration |

## 🔗 Learn More

- **Smolagents**: https://github.com/huggingface/smolagents
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Gradio**: https://gradio.app/

---

**Need help?** Check `SMOLAGENT_CHAT_SETUP.md` for detailed documentation.
