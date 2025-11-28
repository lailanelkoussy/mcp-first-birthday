#!/usr/bin/env python3
"""
Smolagents Agent with Gradio Chat Interface that connects to the MCP server.
This script creates an interactive chat interface where users can query the knowledge graph
through a conversational AI agent.
"""

import os
import sys
import argparse
import re
from typing import List, Dict, Any
import gradio as gr
from gradio import ChatMessage
from smolagents import MCPClient, ToolCallingAgent, OpenAIServerModel, AzureOpenAIModel, InferenceClientModel, stream_to_gradio


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
        """Initialize the OpenAI, Azure OpenAI, or HF Inference model with provided configuration."""
        
        print_info(f"Initializing model:")
        print(f"  Model Type: {model_type}")
        print(f"  Model: {model_name}")
        
        try:
            if model_type == "azure":
                api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
                base_url = base_url or os.environ.get('AZURE_OPENAI_ENDPOINT')
                api_version = api_version or os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
                
                if not api_key:
                    raise ValueError("Azure API key is required!")
                if not base_url:
                    raise ValueError("Azure endpoint is required!")
                
                print(f"  Endpoint: {base_url}")
                print(f"  API Version: {api_version}")
                
                self.model = AzureOpenAIModel(
                    model_id=model_name,
                    azure_endpoint=base_url,
                    api_key=api_key,
                    api_version=api_version
                )
            elif model_type == "hf_inference":
                api_key = api_key or os.environ.get('HF_TOKEN')
                model_name = model_name or os.environ.get('HF_MODEL_NAME', 'Qwen/Qwen2.5-Coder-32B-Instruct')
                provider = base_url or os.environ.get('HF_INFERENCE_PROVIDER', '')
                
                if not api_key:
                    raise ValueError("HuggingFace token is required!")
                
                print(f"  Model: {model_name}")
                if provider:
                    print(f"  Provider: {provider}")
                
                # Build kwargs for InferenceClientModel
                model_kwargs = {
                    "model_id": model_name,
                    "token": api_key
                }
                if provider:
                    model_kwargs["provider"] = provider
                
                self.model = InferenceClientModel(**model_kwargs)
            else:  # openai
                api_key = api_key or os.environ.get('OPENAI_API_KEY')
                base_url = base_url or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                model_name = model_name or os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
                
                if not api_key:
                    raise ValueError("OpenAI API key is required!")
                
                print(f"  Base URL: {base_url}")
                
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
            
            self.agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                name="KnowledgeGraphAgent",
                max_steps=max_steps,
                add_base_tools=False,
                instructions=CUSTOM_INSTRUCTIONS
            )
            print_success("Agent initialized successfully!")
        except Exception as e:
            print_error(f"Failed to initialize agent: {e}")
            raise
    
    def is_ready(self):
        """Check if the agent is fully initialized and ready to chat."""
        return self.agent is not None and self.model is not None
    
    def _parse_thinking_tags(self, text: str):
        """
        Extract content from <think> tags and return both thinking content and clean text.
        
        Args:
            text: Text that may contain <think>...</think> tags
            
        Returns:
            tuple: (thinking_content, clean_text)
        """
        # Find all <think>...</think> blocks
        think_pattern = r'<think>(.*?)</think>'
        thoughts = re.findall(think_pattern, text, re.DOTALL)
        
        # Remove <think> tags from the text
        clean_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        
        return thoughts, clean_text
    
    def chat(self, message: str, history: List[Dict[str, Any]]):
        """
        Process a chat message and stream the response using messages format.
        
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
            
            # Stream agent output using stream_to_gradio
            for chat_message in stream_to_gradio(self.agent, message):
                # Parse for <think> tags
                content = chat_message.content if isinstance(chat_message.content, str) else str(chat_message.content)
                thoughts, clean_content = self._parse_thinking_tags(content)
                
                # Display thinking content if present
                for thought in thoughts:
                    history.append(ChatMessage(
                        role="assistant",
                        content=thought.strip(),
                        metadata={"title": "🧠 Model Thinking"}
                    ))
                    yield history
                
                # Add the message with cleaned content
                if clean_content:
                    if hasattr(chat_message, 'metadata') and chat_message.metadata:
                        # Preserve original metadata from stream_to_gradio
                        history.append(ChatMessage(
                            role=chat_message.role,
                            content=clean_content,
                            metadata=chat_message.metadata
                        ))
                    else:
                        # Regular message without metadata
                        history.append(ChatMessage(
                            role=chat_message.role,
                            content=clean_content
                        ))
                    yield history
            
            print_success("Query processed successfully!")
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print_error(error_msg)
            # Remove pending messages if present
            if history and len(history) > 0:
                last_msg = history[-1]
                if hasattr(last_msg, 'metadata') and last_msg.metadata and last_msg.metadata.get('status') == 'pending':
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
    
    with gr.Blocks(title="🤗 Transformers Q&A Agent", theme=gr.themes.Soft()) as demo:
        
        # ==================== INITIALIZATION SECTION ====================
        with gr.Column(visible=not agent.is_ready()) as init_section:
            gr.Markdown("""
            # 🤗 Transformers Library Q&A Agent
            
            Welcome! This AI agent helps you understand the **Hugging Face Transformers** library.
            Ask questions about the codebase, find functions, explore classes, and understand how components work together.
            
            Configure your AI model below to get started. The MCP server tools are already connected!
            """)
            
            with gr.Group():
                gr.Markdown("### ⚙️ Model Configuration")
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["openai", "azure", "hf_inference"],
                        value="openai",
                        label="Model Type",
                        info="Choose between OpenAI, Azure OpenAI, or HuggingFace Inference"
                    )
                
                # Model name field (shown for all types)
                with gr.Row() as model_name_row:
                    model_name = gr.Textbox(
                        label="Model Name",
                        value=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                        info="e.g., gpt-4o-mini, gpt-4, Qwen/Qwen2.5-Coder-32B-Instruct"
                    )
                
                # OpenAI specific fields
                with gr.Row(visible=True) as openai_fields:
                    api_key = gr.Textbox(
                        label="API Key",
                        value=os.environ.get('OPENAI_API_KEY', ''),
                        type="password",
                        info="Your OpenAI API key"
                    )
                    base_url = gr.Textbox(
                        label="Base URL",
                        value=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                        info="API endpoint URL"
                    )
                
                # Azure specific fields
                with gr.Row(visible=False) as azure_fields:
                    azure_api_key = gr.Textbox(
                        label="Azure API Key",
                        value=os.environ.get('AZURE_OPENAI_API_KEY', ''),
                        type="password",
                        info="Your Azure OpenAI API key"
                    )
                    azure_endpoint = gr.Textbox(
                        label="Azure Endpoint",
                        value=os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
                        info="Azure OpenAI endpoint URL"
                    )
                
                with gr.Row(visible=False) as azure_version_row:
                    api_version = gr.Textbox(
                        label="API Version",
                        value=os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview'),
                        info="Azure OpenAI API version"
                    )
                
                # HuggingFace Inference specific fields
                with gr.Row(visible=False) as hf_fields:
                    hf_token = gr.Textbox(
                        label="HuggingFace Token",
                        value=os.environ.get('HF_TOKEN', ''),
                        type="password",
                        info="Your HuggingFace API token"
                    )
                    hf_provider = gr.Textbox(
                        label="Inference Provider (Optional)",
                        value=os.environ.get('HF_INFERENCE_PROVIDER', ''),
                        info="Provider name (e.g., 'together', 'fireworks-ai', 'cerebras'). Leave empty for auto."
                    )
                
                with gr.Row():
                    max_steps = gr.Number(
                        label="Max Steps",
                        value=int(os.getenv("MAX_STEPS", 5)),
                        minimum=1,
                        maximum=20,
                        info="Maximum reasoning steps for the agent"
                    )
                
                init_status = gr.Markdown("**Status:** ⚠️ Please configure and initialize the agent")
                init_btn = gr.Button("🚀 Initialize Agent", variant="primary", size="lg")
        
        # Toggle visibility based on model type
        def toggle_model_fields(mtype):
            if mtype == "azure":
                return (
                    gr.update(visible=False),  # openai_fields
                    gr.update(visible=True),   # azure_fields
                    gr.update(visible=True),   # azure_version_row
                    gr.update(visible=False)   # hf_fields
                )
            elif mtype == "hf_inference":
                return (
                    gr.update(visible=False),  # openai_fields
                    gr.update(visible=False),  # azure_fields
                    gr.update(visible=False),  # azure_version_row
                    gr.update(visible=True)    # hf_fields
                )
            else:  # openai
                return (
                    gr.update(visible=True),   # openai_fields
                    gr.update(visible=False),  # azure_fields
                    gr.update(visible=False),  # azure_version_row
                    gr.update(visible=False)   # hf_fields
                )
        
        model_type.change(
            fn=toggle_model_fields,
            inputs=[model_type],
            outputs=[openai_fields, azure_fields, azure_version_row, hf_fields]
        )
        
        # ==================== CHAT SECTION ====================
        with gr.Column(visible=agent.is_ready()) as chat_section:
            gr.Markdown("""
            # 🤗 Transformers Library Q&A Agent
            
            Ask me anything about the **Hugging Face Transformers** library! I can help you:
            - 🔍 Find and explain functions, classes, and methods
            - 🗺️ Navigate the codebase structure and understand file organization
            - 🔗 Trace relationships and dependencies between components
            - 📖 Explain implementation details and design patterns
            """)
            
            chatbot = gr.Chatbot(
                label="Transformers Q&A",
                height=500,
                show_copy_button=True,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about the Transformers library... (e.g., 'How does BertModel work?')",
                    scale=4,
                    lines=1
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            gr.Markdown("""
            ### 💡 Example Questions:
            - "How does the `AutoModel` class work?"
            - "What is the structure of a model's `forward` method?"
            - "Find all classes that inherit from `PreTrainedModel`"
            - "How does tokenization work in the library?"
            - "What files are involved in the BERT implementation?"
            """)
        
        # Handle agent initialization
        def initialize_agent(mtype, mname, akey, burl, azure_akey, azure_ep, aversion, hf_tok, hf_prov, msteps):
            try:
                if mtype == "azure":
                    agent._initialize_model(
                        model_type=mtype,
                        api_key=azure_akey,
                        base_url=azure_ep,
                        model_name=mname,
                        api_version=aversion
                    )
                elif mtype == "hf_inference":
                    agent._initialize_model(
                        model_type=mtype,
                        api_key=hf_tok,
                        model_name=mname,
                        base_url=hf_prov if hf_prov else None
                    )
                else:  # openai
                    agent._initialize_model(
                        model_type=mtype,
                        api_key=akey,
                        base_url=burl,
                        model_name=mname
                    )
                agent._initialize_agent(max_steps=int(msteps))
                return (
                    gr.update(value="**Status:** ✅ Agent Ready!"),
                    gr.update(visible=False),  # Hide init section
                    gr.update(visible=True)    # Show chat section
                )
            except Exception as e:
                error_msg = f"**Status:** ❌ Initialization failed: {str(e)}"
                return (
                    gr.update(value=error_msg),
                    gr.update(visible=True),   # Keep init section visible
                    gr.update(visible=False)   # Keep chat section hidden
                )
        
        init_btn.click(
            fn=initialize_agent,
            inputs=[model_type, model_name, api_key, base_url, azure_api_key, azure_endpoint, api_version, hf_token, hf_provider, max_steps],
            outputs=[init_status, init_section, chat_section]
        )
        
        # Handle message submission with streaming
        def submit_message(message, history):
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
