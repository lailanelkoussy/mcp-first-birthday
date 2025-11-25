#!/usr/bin/env python3
"""
Smolagents Agent with Gradio Chat Interface that connects to the MCP server.
This script creates an interactive chat interface where users can query the knowledge graph
through a conversational AI agent.
"""

import os
import sys
import argparse
from typing import List, Tuple
import gradio as gr
from smolagents import MCPClient, CodeAgent, OpenAIServerModel, AzureOpenAIModel


class Colors:
    """Color codes for terminal output."""
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


class KnowledgeGraphChatAgent:
    """A chat agent that connects to the Knowledge Graph MCP server."""
    
    def __init__(self, mcp_server_url: str = None):
        """Initialize the chat agent with MCP server connection."""
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL", "http://localhost:4000/mcp")
        self.model = None
        self.agent = None
        self.mcp_client = None
        self.conversation_history = []
        
        # Initialize the model and agent
        self._initialize_model()
        self._initialize_agent()
    
    def _initialize_model(self):
        """Initialize the OpenAI or Azure OpenAI model."""
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai_base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        openai_model = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
        openai_api_version = os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
        is_azure = 'azure' in openai_base_url.lower()
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
        
        print_info(f"Using OpenAI configuration:")
        print(f"  Model: {openai_model}")
        print(f"  Base URL: {openai_base_url}")
        print(f"  Azure: {is_azure}")
        if is_azure:
            print(f"  API Version: {openai_api_version}")
        
        try:
            if is_azure:
                self.model = AzureOpenAIModel(
                    api_key=openai_api_key,
                    api_base=openai_base_url,
                    api_version=openai_api_version,
                    model_id=openai_model
                )
            else:
                self.model = OpenAIServerModel(
                    model_id=openai_model,
                    api_key=openai_api_key,
                    api_base=openai_base_url
                )
            print_success("Model initialized successfully!")
        except Exception as e:
            print_error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_agent(self):
        """Initialize the MCP client and agent."""
        try:
            print_info(f"Connecting to MCP server at {self.mcp_server_url}...")
            self.mcp_client = MCPClient({"url": self.mcp_server_url, "transport": "streamable-http"})
            tools = self.mcp_client.__enter__()
            
            self.agent = CodeAgent(
                tools=tools,
                model=self.model,
                name="KnowledgeGraphAgent",
                description="An agent that can query and analyze a source code knowledge graph using MCP server tools.",
                max_steps=int(os.getenv("MAX_STEPS", 5)),
                add_base_tools=True
            )
            print_success("Agent initialized successfully!")
        except Exception as e:
            print_error(f"Failed to initialize agent: {e}")
            if self.mcp_client:
                self.mcp_client.__exit__(None, None, None)
            raise
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Process a chat message and return the response.
        
        Args:
            message: The user's message
            history: The conversation history as list of (user_msg, bot_msg) tuples
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not message.strip():
            return "", history
        
        try:
            print_info(f"Processing query: {message}")
            result = self.agent.run(message)
            response = str(result)
            print_success("Query processed successfully!")
            
            # Update history
            history.append((message, response))
            
            return "", history
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print_error(error_msg)
            history.append((message, error_msg))
            return "", history
    
    def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            try:
                self.mcp_client.__exit__(None, None, None)
            except Exception as e:
                print_error(f"Error during cleanup: {e}")


def create_gradio_interface(agent: KnowledgeGraphChatAgent):
    """Create the Gradio chat interface."""
    
    with gr.Blocks(title="Knowledge Graph Chat Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Knowledge Graph Chat Agent
        
        Chat with an AI agent that can explore and analyze your codebase knowledge graph.
        Ask questions about your code structure, functions, classes, dependencies, and more!
        
        ### Example Questions:
        - "What does the class RepoKnowledgeGraph do?"
        - "Show me all the functions in the codebase"
        - "Find usages of the search_nodes function"
        - "What are the main modules in this repository?"
        - "Explain the architecture of this codebase"
        """)
        
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
            show_copy_button=True,
            type="messages"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Ask me anything about your codebase...",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        gr.Markdown("""
        ### Tips:
        - Be specific in your questions for better results
        - You can ask follow-up questions based on previous responses
        - The agent has access to all MCP server tools for code analysis
        """)
        
        # Handle message submission
        def submit_message(message, history):
            return agent.chat(message, history)
        
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            fn=submit_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=chatbot
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Smolagents Chat Agent with Gradio Interface")
    parser.add_argument("--mcp-server-url", type=str, help="URL of the MCP server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    try:
        # Initialize the agent
        print_info("Initializing Knowledge Graph Chat Agent...")
        agent = KnowledgeGraphChatAgent(mcp_server_url=args.mcp_server_url)
        print_success("Agent ready!")
        
        # Create and launch the Gradio interface
        demo = create_gradio_interface(agent)
        print_info(f"Launching Gradio interface on {args.host}:{args.port}")
        
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
        
    except KeyboardInterrupt:
        print_info("\nShutting down gracefully...")
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'agent' in locals():
            agent.cleanup()


if __name__ == "__main__":
    main()
