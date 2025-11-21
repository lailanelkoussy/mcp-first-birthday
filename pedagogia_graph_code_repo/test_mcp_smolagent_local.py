#!/usr/bin/env python3
"""
Test script for the Knowledge Graph MCP Server using Smolagents with remote MCP connection.
This script creates an agent that connects to a remote MCP server over HTTP
and uses the tools exposed by that server.
"""

import os
import argparse
from smolagents import MCPClient, CodeAgent, OpenAIServerModel, AzureOpenAIModel

# Optional: Color output helpers (simplified)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.YELLOW}ℹ️  {message}{Colors.ENDC}")

def create_knowledge_graph_agent():
    """Create an agent configuration that can interact with the knowledge graph MCP server over HTTP."""
    # Get configuration from environment variables
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    openai_base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    openai_model = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
    openai_api_version = os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://mcp-server:4000/mcp")
    is_azure = 'azure' in openai_base_url.lower()
    if not openai_api_key:
        print_error("OPENAI_API_KEY environment variable is not set!")
        print_info("Please set: export OPENAI_API_KEY=your_api_key")
        return None
    print_info(f"Using OpenAI configuration:")
    print(f"  Model: {openai_model}")
    print(f"  Base URL: {openai_base_url}")
    print(f"  Azure: {is_azure}")
    if is_azure:
        print(f"  API Version: {openai_api_version}")
    print(f"  MCP Server: {mcp_server_url}")
    try:
        if is_azure:
            model = AzureOpenAIModel(
                api_key=openai_api_key,
                api_base=openai_base_url,
                api_version=openai_api_version,
                model_id=openai_model
            )
        else:
            model = OpenAIServerModel(
                model_id=openai_model,
                api_key=openai_api_key,
                api_base=openai_base_url
            )
        print_success("Model configuration created successfully!")
        return model, mcp_server_url
    except Exception as e:
        print_error(f"Could not create agent configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

def execute_mission(mission: str):
    """Execute a specific mission using the knowledge graph agent."""
    print_info(f"Executing mission: {mission}")
    agent_config = create_knowledge_graph_agent()
    if agent_config is None:
        print_error("Cannot execute mission without a proper agent configuration.")
        print_info("Please ensure smolagents and openai are installed and OPENAI_API_KEY is set.")
        return
    model, mcp_server_url = agent_config
    try:
        print_info(f"Connecting to MCP server at {mcp_server_url}...")
        with MCPClient({"url": mcp_server_url, "transport": "streamable-http"}) as tools:
            agent = CodeAgent(
                tools=tools,
                model=model,
                name="KnowledgeGraphAgent",
                description="An agent that can query and analyze a source code knowledge graph using MCP server tools.",
                max_steps=int(os.getenv("MAX_STEPS", 5)),
                add_base_tools=True
            )
            result = agent.run(mission)
            print_success("Mission Result:")
            print(result)
    except Exception as e:
        print_error(f"Error executing mission: {e}")
        import traceback
        traceback.print_exc()

def interactive_agent_session():
    """Run an interactive session with the agent."""
    print_info("Interactive Knowledge Graph Agent Session. Type 'quit' to exit.")
    agent_config = create_knowledge_graph_agent()
    if agent_config is None:
        print_error("Cannot run interactive session without a proper agent configuration.")
        return
    model, mcp_server_url = agent_config
    try:
        print_info(f"Connecting to MCP server at {mcp_server_url}...")
        with MCPClient({"url": mcp_server_url, "transport": "streamable-http"}) as tools:
            agent = CodeAgent(
                tools=tools,
                model=model,
                name="KnowledgeGraphAgent",
                description="An agent that can query and analyze a source code knowledge graph using MCP server tools.",
                max_steps=3,
                add_base_tools=True
            )
            print_success("Connected to MCP server and agent initialized!")
            while True:
                try:
                    user_input = input(f"\n{Colors.CYAN}Enter your query: {Colors.ENDC}").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print_info("Goodbye!")
                        break
                    if not user_input:
                        continue
                    print_info("Processing query...")
                    result = agent.run(user_input)
                    print(f"\n{Colors.GREEN}Agent: {Colors.ENDC}{result}")
                except KeyboardInterrupt:
                    print_info("\nGoodbye!")
                    break
                except Exception as e:
                    print_error(f"Error: {e}")
    except Exception as e:
        print_error(f"Error connecting to MCP server: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test Knowledge Graph MCP Server with Smolagents (Remote Connection)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--mission", type=str, help="Execute a specific mission/task with the agent")
    args = parser.parse_args()
    print_info("Knowledge Graph Smolagent - Remote MCP Connection")
    if args.mission:
        execute_mission(args.mission)
    elif args.interactive:
        interactive_agent_session()
    else:
        print_info("Please specify a mode:")
        print("  --mission 'your mission here'  : Execute a specific task")
        print("  --interactive                   : Start interactive session")

if __name__ == "__main__":
    main()