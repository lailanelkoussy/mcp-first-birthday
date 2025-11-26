#!/usr/bin/env python3
"""
Smolagents Agent with Gradio Chat Interface that connects to the MCP server.
This script creates an interactive chat interface where users can query the knowledge graph
through a conversational AI agent.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
import gradio as gr
from gradio import ChatMessage
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
        self.tools = None
        self.conversation_history = []
        
        # Initialize MCP tools first (required for agent)
        self._initialize_mcp_tools()
    
    def _initialize_mcp_tools(self):
        """Initialize MCP client and load tools (must be done before agent creation)."""
        try:
            print_info(f"Connecting to MCP server at {self.mcp_server_url}...")
            self.mcp_client = MCPClient({"url": self.mcp_server_url, "transport": "streamable-http"})
            self.tools = self.mcp_client.__enter__()
            print_success(f"MCP tools loaded successfully! ({len(self.tools)} tools available)")
        except Exception as e:
            print_error(f"Failed to connect to MCP server: {e}")
            raise
    
    def _initialize_model(self, model_type: str = "openai", api_key: str = None, 
                         base_url: str = None, model_name: str = None, 
                         api_version: str = None):
        """Initialize the OpenAI or Azure OpenAI model with provided configuration."""
        # Use provided values or fall back to environment variables
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        base_url = base_url or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        model_name = model_name or os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
        api_version = api_version or os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
        is_azure = model_type == "azure" or 'azure' in base_url.lower()
        
        if not api_key:
            raise ValueError("API key is required!")
        
        print_info(f"Using OpenAI configuration:")
        print(f"  Model Type: {model_type}")
        print(f"  Model: {model_name}")
        print(f"  Base URL: {base_url}")
        print(f"  Azure: {is_azure}")
        if is_azure:
            print(f"  API Version: {api_version}")
        
        try:
            if is_azure:
                self.model = AzureOpenAIModel(
                    api_key=api_key,
                    api_base=base_url,
                    api_version=api_version,
                    model_id=model_name
                )
            else:
                self.model = OpenAIServerModel(
                    model_id=model_name,
                    api_key=api_key,
                    api_base=base_url
                )
            print_success("Model initialized successfully!")
        except Exception as e:
            print_error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_agent(self, max_steps: int = None):
        """Initialize the agent using the configured model and pre-loaded MCP tools."""
        if not self.model:
            raise ValueError("Model must be initialized before creating agent!")
        if not self.tools:
            raise ValueError("MCP tools must be loaded before creating agent!")
        
        try:
            max_steps = max_steps or int(os.getenv("MAX_STEPS", 5))
            
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.model,
                name="KnowledgeGraphAgent",
                description="An agent that can query and analyze a source code knowledge graph using MCP server tools.",
                max_steps=max_steps,
                add_base_tools=True
            )
            print_success("Agent initialized successfully!")
        except Exception as e:
            print_error(f"Failed to initialize agent: {e}")
            raise
    
    def is_ready(self):
        """Check if the agent is fully initialized and ready to chat."""
        return self.agent is not None and self.model is not None
    
    def chat(self, message: str, history: List[Dict[str, Any]]):
        """
        Process a chat message and return the response using messages format.
        
        Args:
            message: The user's message
            history: The conversation history as list of message dictionaries
            
        Yields:
            Updated history with new messages including thinking and tool usage
        """
        if not message.strip():
            yield history
            return
        
        # Add user message
        history.append(ChatMessage(role="user", content=message))
        yield history
        
        try:
            print_info(f"Processing query: {message}")
            
            # Show thinking indicator
            history.append(ChatMessage(
                role="assistant", 
                content="Analyzing your question and planning the approach...",
                metadata={"title": "🧠 Thinking"}
            ))
            yield history
            
            # Run the agent and capture the result
            result = self.agent.run(message)
            
            # Remove thinking indicator
            history = history[:-1]
            
            # Check if there are any tool calls in the agent's execution
            if hasattr(self.agent, 'logs') and self.agent.logs:
                for log_entry in self.agent.logs:
                    if 'tool' in str(log_entry).lower():
                        history.append(ChatMessage(
                            role="assistant",
                            content=str(log_entry),
                            metadata={"title": "🛠️ Using Tools"}
                        ))
                        yield history
            
            # Add final response
            response = str(result)
            print_success("Query processed successfully!")
            history.append(ChatMessage(role="assistant", content=response))
            yield history
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print_error(error_msg)
            # Remove thinking indicator if present
            if history and hasattr(history[-1], 'metadata') and history[-1].metadata:
                history = history[:-1]
            history.append(ChatMessage(role="assistant", content=error_msg))
            yield history
    
    def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            try:
                self.mcp_client.__exit__(None, None, None)
            except Exception as e:
                print_error(f"Error during cleanup: {e}")


def create_gradio_interface(agent: KnowledgeGraphChatAgent):
    """Create the Gradio chat interface with model configuration."""
    
    with gr.Blocks(title="Knowledge Graph Chat Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Knowledge Graph Chat Agent
        
        Chat with an AI agent that can explore and analyze your codebase knowledge graph.
        Ask questions about your code structure, functions, classes, dependencies, and more!
        """)
        
        # Configuration Section
        with gr.Accordion("⚙️ Model Configuration", open=not agent.is_ready()):
            gr.Markdown("Configure your AI model before starting the chat. The knowledge graph tools are already connected!")
            
            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["openai", "azure"],
                    value="openai",
                    label="Model Type",
                    info="Choose between OpenAI or Azure OpenAI"
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    value=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                    info="e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo"
                )
            
            with gr.Row():
                api_key = gr.Textbox(
                    label="API Key",
                    value=os.environ.get('OPENAI_API_KEY', ''),
                    type="password",
                    info="Your OpenAI or Azure OpenAI API key"
                )
                base_url = gr.Textbox(
                    label="Base URL",
                    value=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                    info="API endpoint URL"
                )
            
            with gr.Row():
                api_version = gr.Textbox(
                    label="API Version (Azure only)",
                    value=os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview'),
                    info="Required for Azure OpenAI"
                )
                max_steps = gr.Number(
                    label="Max Steps",
                    value=int(os.getenv("MAX_STEPS", 5)),
                    minimum=1,
                    maximum=20,
                    info="Maximum reasoning steps for the agent"
                )
            
            init_status = gr.Markdown("**Status:** " + ("✅ Agent Ready!" if agent.is_ready() else "⚠️ Please initialize the agent"))
            init_btn = gr.Button("Initialize Agent", variant="primary", visible=not agent.is_ready())
        
        # Chat Section
        gr.Markdown("""
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
            type="messages",
            avatar_images=(None, "🤖")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Initialize the agent first..." if not agent.is_ready() else "Ask me anything about your codebase...",
                scale=4,
                lines=2,
                interactive=agent.is_ready()
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1, interactive=agent.is_ready())
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        gr.Markdown("""
        ### Tips:
        - Configure your preferred AI model before starting
        - Be specific in your questions for better results
        - You can ask follow-up questions based on previous responses
        - The agent has access to all MCP server tools for code analysis
        """)
        
        # Handle agent initialization
        def initialize_agent(mtype, mname, akey, burl, aversion, msteps):
            try:
                agent._initialize_model(
                    model_type=mtype,
                    api_key=akey,
                    base_url=burl,
                    model_name=mname,
                    api_version=aversion
                )
                agent._initialize_agent(max_steps=int(msteps))
                return (
                    gr.update(value="**Status:** ✅ Agent Ready!"),
                    gr.update(visible=False),
                    gr.update(placeholder="Ask me anything about your codebase...", interactive=True),
                    gr.update(interactive=True)
                )
            except Exception as e:
                error_msg = f"**Status:** ❌ Initialization failed: {str(e)}"
                return (
                    gr.update(value=error_msg),
                    gr.update(visible=True),
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )
        
        init_btn.click(
            fn=initialize_agent,
            inputs=[model_type, model_name, api_key, base_url, api_version, max_steps],
            outputs=[init_status, init_btn, msg, submit_btn]
        )
        
        # Handle message submission with streaming
        def submit_message(message, history):
            if not agent.is_ready():
                history.append(ChatMessage(
                    role="assistant", 
                    content="❌ Please initialize the agent first using the configuration section above."
                ))
                yield "", history
                return
            for updated_history in agent.chat(message, history):
                yield "", updated_history
        
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
