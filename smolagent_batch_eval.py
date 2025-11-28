#!/usr/bin/env python3
"""
Smolagents Batch Evaluation Script

This script runs a batch of pre-generated questions about the Transformers library
through the smolagent, logging all steps and responses to a JSON file.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from smolagents import MCPClient, ToolCallingAgent, OpenAIServerModel, AzureOpenAIModel, InferenceClientModel
from smolagents.agents import RunResult, ActionStep
from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# Initialize Langfuse client for tracing
langfuse = get_client()

# Verify Langfuse connection
if langfuse.auth_check():
    print("✓ Langfuse client is authenticated and ready!")
else:
    print("⚠️ Langfuse authentication failed. Tracing may not work correctly.")

# Instrument smolagents for automatic tracing
SmolagentsInstrumentor().instrument()


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


# Pre-generated questions about the Transformers library
TRANSFORMERS_QUESTIONS = [
    # Architecture & Model Structure
    {
        "id": "q001",
        "question": "How does the `AutoModel` class determine which specific model architecture to instantiate when loading a pretrained model?",
        "category": "architecture",
        "difficulty": "medium"
    },
    {
        "id": "q002",
        "question": "What is the purpose of the `PreTrainedModel` base class and what key methods does it provide for all transformer models?",
        "category": "architecture",
        "difficulty": "medium"
    },
    {
        "id": "q003",
        "question": "How does the attention mechanism in BERT differ from the one in GPT-2, and where are these differences implemented in the code?",
        "category": "architecture",
        "difficulty": "hard"
    },
    {
        "id": "q004",
        "question": "What is the role of the `ModelOutput` class and how does it standardize model outputs across different architectures?",
        "category": "architecture",
        "difficulty": "medium"
    },
    {
        "id": "q005",
        "question": "How does the library handle model configuration through the `PretrainedConfig` class?",
        "category": "architecture",
        "difficulty": "medium"
    },
    
    # Tokenization
    {
        "id": "q006",
        "question": "How does the `AutoTokenizer` class select the appropriate tokenizer for a given model?",
        "category": "tokenization",
        "difficulty": "medium"
    },
    {
        "id": "q007",
        "question": "What is the difference between `PreTrainedTokenizer` and `PreTrainedTokenizerFast`, and when should each be used?",
        "category": "tokenization",
        "difficulty": "medium"
    },
    {
        "id": "q008",
        "question": "How does the BPE (Byte-Pair Encoding) tokenization algorithm work in the Transformers library?",
        "category": "tokenization",
        "difficulty": "hard"
    },
    {
        "id": "q009",
        "question": "What is the purpose of special tokens like [CLS], [SEP], [PAD], and how are they handled during tokenization?",
        "category": "tokenization",
        "difficulty": "easy"
    },
    {
        "id": "q010",
        "question": "How does the tokenizer handle truncation and padding when preparing inputs for batch processing?",
        "category": "tokenization",
        "difficulty": "medium"
    },
    
    # Training & Fine-tuning
    {
        "id": "q011",
        "question": "What is the purpose of the `Trainer` class and what are its main components?",
        "category": "training",
        "difficulty": "medium"
    },
    {
        "id": "q012",
        "question": "How does the library implement gradient checkpointing to reduce memory usage during training?",
        "category": "training",
        "difficulty": "hard"
    },
    {
        "id": "q013",
        "question": "What is the role of `TrainingArguments` and what are the most important parameters for fine-tuning?",
        "category": "training",
        "difficulty": "medium"
    },
    {
        "id": "q014",
        "question": "How does the `DataCollator` work and what different collators are available for various tasks?",
        "category": "training",
        "difficulty": "medium"
    },
    {
        "id": "q015",
        "question": "How does the library handle mixed precision training (FP16/BF16)?",
        "category": "training",
        "difficulty": "hard"
    },
    
    # Generation & Inference
    {
        "id": "q016",
        "question": "How does the `generate()` method implement different decoding strategies like greedy, beam search, and sampling?",
        "category": "generation",
        "difficulty": "hard"
    },
    {
        "id": "q017",
        "question": "What is the purpose of the `GenerationConfig` class and how does it control text generation parameters?",
        "category": "generation",
        "difficulty": "medium"
    },
    {
        "id": "q018",
        "question": "How does the library implement KV-cache for efficient autoregressive generation?",
        "category": "generation",
        "difficulty": "hard"
    },
    {
        "id": "q019",
        "question": "What are `LogitsProcessor` and `StoppingCriteria`, and how are they used to customize text generation?",
        "category": "generation",
        "difficulty": "medium"
    },
    {
        "id": "q020",
        "question": "How does the library handle streaming generation for real-time output?",
        "category": "generation",
        "difficulty": "medium"
    },
    
    # Model Loading & Saving
    {
        "id": "q021",
        "question": "How does `from_pretrained()` handle downloading and caching models from the Hugging Face Hub?",
        "category": "loading",
        "difficulty": "medium"
    },
    {
        "id": "q022",
        "question": "What is the safetensors format and how does the library support it for model serialization?",
        "category": "loading",
        "difficulty": "medium"
    },
    {
        "id": "q023",
        "question": "How does model sharding work for loading large models that don't fit in memory?",
        "category": "loading",
        "difficulty": "hard"
    },
    {
        "id": "q024",
        "question": "What is the purpose of `device_map='auto'` and how does it distribute model layers across devices?",
        "category": "loading",
        "difficulty": "medium"
    },
    {
        "id": "q025",
        "question": "How does the library handle loading models with different weight precisions (float32, float16, bfloat16, int8)?",
        "category": "loading",
        "difficulty": "medium"
    },
    
    # Specific Models
    {
        "id": "q026",
        "question": "How does the BERT model implement masked language modeling in the `BertForMaskedLM` class?",
        "category": "models",
        "difficulty": "medium"
    },
    {
        "id": "q027",
        "question": "What is the difference between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures in the library?",
        "category": "models",
        "difficulty": "medium"
    },
    {
        "id": "q028",
        "question": "How does the Llama model implement rotary positional embeddings (RoPE)?",
        "category": "models",
        "difficulty": "hard"
    },
    {
        "id": "q029",
        "question": "What is the implementation of flash attention in the library and which models support it?",
        "category": "models",
        "difficulty": "hard"
    },
    {
        "id": "q030",
        "question": "How does the ViT (Vision Transformer) model process images and what is the role of the patch embedding layer?",
        "category": "models",
        "difficulty": "medium"
    },
    
    # Pipelines & Tasks
    {
        "id": "q031",
        "question": "How does the `pipeline()` function abstract different NLP tasks and select appropriate models?",
        "category": "pipelines",
        "difficulty": "easy"
    },
    {
        "id": "q032",
        "question": "What tasks are supported by the Transformers pipelines and how can custom pipelines be created?",
        "category": "pipelines",
        "difficulty": "medium"
    },
    {
        "id": "q033",
        "question": "How does the question-answering pipeline extract answers from context passages?",
        "category": "pipelines",
        "difficulty": "medium"
    },
    {
        "id": "q034",
        "question": "How does the text-classification pipeline handle multi-label vs multi-class classification?",
        "category": "pipelines",
        "difficulty": "medium"
    },
    {
        "id": "q035",
        "question": "What is the implementation of the conversational pipeline and how does it manage conversation history?",
        "category": "pipelines",
        "difficulty": "medium"
    },
    
    # Advanced Features
    {
        "id": "q036",
        "question": "How does the library implement PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA?",
        "category": "advanced",
        "difficulty": "hard"
    },
    {
        "id": "q037",
        "question": "What is the implementation of quantization (4-bit, 8-bit) using bitsandbytes integration?",
        "category": "advanced",
        "difficulty": "hard"
    },
    {
        "id": "q038",
        "question": "How does the library support distributed training across multiple GPUs using accelerate?",
        "category": "advanced",
        "difficulty": "hard"
    },
    {
        "id": "q039",
        "question": "What is the implementation of attention mask handling for different model types?",
        "category": "advanced",
        "difficulty": "medium"
    },
    {
        "id": "q040",
        "question": "How does the library implement model parallelism for very large models?",
        "category": "advanced",
        "difficulty": "hard"
    },
    
    # Code Organization
    {
        "id": "q041",
        "question": "How is the transformers library organized in terms of file structure and module hierarchy?",
        "category": "codebase",
        "difficulty": "easy"
    },
    {
        "id": "q042",
        "question": "What is the role of the `modeling_utils.py` file and what utilities does it provide?",
        "category": "codebase",
        "difficulty": "medium"
    },
    {
        "id": "q043",
        "question": "How does the library implement backwards compatibility when updating model implementations?",
        "category": "codebase",
        "difficulty": "medium"
    },
    {
        "id": "q044",
        "question": "What is the purpose of the `_init_weights` method in model classes?",
        "category": "codebase",
        "difficulty": "medium"
    },
    {
        "id": "q045",
        "question": "How does the library handle the registration of new models in the Auto classes?",
        "category": "codebase",
        "difficulty": "medium"
    },
    
    # Integration & Ecosystem
    {
        "id": "q046",
        "question": "How does the library integrate with the Hugging Face Hub for model sharing and versioning?",
        "category": "ecosystem",
        "difficulty": "easy"
    },
    {
        "id": "q047",
        "question": "What is the integration between transformers and the datasets library for data loading?",
        "category": "ecosystem",
        "difficulty": "medium"
    },
    {
        "id": "q048",
        "question": "How does the library support ONNX export for production deployment?",
        "category": "ecosystem",
        "difficulty": "medium"
    },
    {
        "id": "q049",
        "question": "What is the integration with TensorBoard for training visualization?",
        "category": "ecosystem",
        "difficulty": "easy"
    },
    {
        "id": "q050",
        "question": "How does the library handle interoperability between PyTorch, TensorFlow, and JAX backends?",
        "category": "ecosystem",
        "difficulty": "hard"
    },
]


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


@dataclass
class ToolCall:
    """Represents a single tool call made by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    timestamp: str = ""
    duration_seconds: float = 0.0
    tool_call_id: str = ""


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning process."""
    step_number: int
    thought: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    observation: Optional[str] = None
    timestamp: str = ""
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    is_final_answer: bool = False


@dataclass
class QuestionResult:
    """Represents the complete result for a single question."""
    question_id: str
    question: str
    category: str
    difficulty: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_tool_calls: int = 0
    total_steps: int = 0
    total_duration_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    success: bool = True
    error_message: Optional[str] = None
    timestamp_start: str = ""
    timestamp_end: str = ""


@dataclass
class EvaluationRun:
    """Represents a complete evaluation run."""
    run_id: str
    model_type: str
    model_name: str
    mcp_server_url: str
    max_steps: int
    timestamp_start: str
    timestamp_end: str = ""
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    total_duration_seconds: float = 0.0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    results: List[QuestionResult] = field(default_factory=list)


class BatchEvaluationAgent:
    """Agent that runs batch evaluation of questions against the knowledge graph."""
    
    def __init__(self, mcp_server_url: str = None):
        """Initialize the batch evaluation agent."""
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL", "http://localhost:4000/mcp")
        self.model = None
        self.agent = None
        self.mcp_client = None
        self.tools = None
        
        # Initialize MCP tools
        self._initialize_mcp_tools()
    
    def _initialize_mcp_tools(self):
        """Initialize MCP client and load tools."""
        try:
            print_info(f"Connecting to MCP server at {self.mcp_server_url}...")
            self.mcp_client = MCPClient({"url": self.mcp_server_url, "transport": "streamable-http"})
            self.tools = self.mcp_client.__enter__()
            print_success(f"MCP tools loaded successfully! ({len(self.tools)} tools available)")
        except Exception as e:
            print_error(f"Failed to connect to MCP server: {e}")
            raise
    
    def initialize_model(self, model_type: str = "openai", api_key: str = None, 
                        base_url: str = None, model_name: str = None, 
                        api_version: str = None):
        """Initialize the model with provided configuration."""
        
        print_info(f"Initializing model: {model_type} - {model_name}")
        
        try:
            if model_type == "azure":
                api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
                base_url = base_url or os.environ.get('AZURE_OPENAI_ENDPOINT')
                api_version = api_version or os.environ.get('OPENAI_API_VERSION', '2024-02-15-preview')
                
                if not api_key or not base_url:
                    raise ValueError("Azure API key and endpoint are required!")
                
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
                
                model_kwargs = {"model_id": model_name, "token": api_key, "bill_to": "epita", "timeout": 240}
                if provider:
                    model_kwargs["provider"] = provider
                
                self.model = InferenceClientModel(**model_kwargs)
            else:  # openai
                api_key = api_key or os.environ.get('OPENAI_API_KEY')
                base_url = base_url or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                model_name = model_name or os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
                
                if not api_key:
                    raise ValueError("OpenAI API key is required!")
                
                self.model = OpenAIServerModel(
                    model_id=model_name,
                    api_key=api_key,
                    api_base=base_url
                )
            print_success("Model initialized successfully!")
            return model_name
        except Exception as e:
            print_error(f"Failed to initialize model: {e}")
            raise
    
    def initialize_agent(self, max_steps: int = 5):
        """Initialize the agent with configured model and tools."""
        if not self.model:
            raise ValueError("Model must be initialized first!")
        if not self.tools:
            raise ValueError("MCP tools must be loaded first!")
        
        try:
            self.agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                name="TransformersEvalAgent",
                max_steps=max_steps,
                add_base_tools=False,
                instructions=CUSTOM_INSTRUCTIONS
            )
            print_success(f"Agent initialized with max_steps={max_steps}")
        except Exception as e:
            print_error(f"Failed to initialize agent: {e}")
            raise
    
    def run_question(self, question_data: Dict[str, Any]) -> QuestionResult:
        """Run a single question through the agent and capture all steps."""
        question_id = question_data["id"]
        question = question_data["question"]
        category = question_data["category"]
        difficulty = question_data["difficulty"]
        
        print_info(f"Processing question {question_id}: {question[:60]}...")
        
        result = QuestionResult(
            question_id=question_id,
            question=question,
            category=category,
            difficulty=difficulty,
            timestamp_start=datetime.now().isoformat()
        )
        
        start_time = time.time()
        
        try:
            # Run the agent with return_full_result=True to capture all steps
            run_result: RunResult = self.agent.run(question, return_full_result=True)
            
            # Extract steps from RunResult
            if run_result.steps:
                for i, step in enumerate(run_result.steps):
                    if isinstance(step, ActionStep):
                        agent_step = AgentStep(
                            step_number=step.step_number,
                            timestamp=datetime.now().isoformat(),
                            is_final_answer=step.is_final_answer
                        )
                        
                        # Extract timing information
                        if step.timing:
                            agent_step.duration_seconds = step.timing.duration or 0.0
                        
                        # Extract token usage
                        if step.token_usage:
                            agent_step.input_tokens = step.token_usage.input_tokens
                            agent_step.output_tokens = step.token_usage.output_tokens
                            result.total_input_tokens += step.token_usage.input_tokens
                            result.total_output_tokens += step.token_usage.output_tokens
                        
                        # Extract thought/reasoning from model_output
                        if step.model_output:
                            agent_step.thought = str(step.model_output)
                        
                        # Extract tool calls
                        if step.tool_calls:
                            for tc in step.tool_calls:
                                tool_call = ToolCall(
                                    tool_name=tc.name,
                                    arguments=tc.arguments if isinstance(tc.arguments, dict) else {"raw": str(tc.arguments)},
                                    timestamp=datetime.now().isoformat(),
                                    tool_call_id=tc.id if hasattr(tc, 'id') else ""
                                )
                                agent_step.tool_calls.append(tool_call)
                                result.total_tool_calls += 1
                                
                                # Log tool call to console
                                print(f"  📞 Tool call: {tc.name}")
                                if tc.arguments:
                                    args_str = str(tc.arguments)[:100] + "..." if len(str(tc.arguments)) > 100 else str(tc.arguments)
                                    print(f"     Args: {args_str}")
                        
                        # Extract observation/result
                        if step.observations:
                            agent_step.observation = str(step.observations)
                            obs_preview = step.observations[:200] + "..." if len(step.observations) > 200 else step.observations
                            print(f"  👁️  Observation: {obs_preview}")
                        
                        # Check for errors in this step
                        if step.error:
                            agent_step.observation = f"ERROR: {step.error}"
                            print(f"  ⚠️  Step error: {step.error}")
                        
                        result.steps.append(agent_step)
                        
                        # Log step completion with timing info
                        timing_info = f" ({agent_step.duration_seconds:.2f}s)" if agent_step.duration_seconds > 0 else ""
                        token_info = f" [tokens: {agent_step.input_tokens}→{agent_step.output_tokens}]" if agent_step.input_tokens > 0 else ""
                        print(f"  ✅ Step {step.step_number} completed{timing_info}{token_info}")
            
            # Get final answer from RunResult output
            result.final_answer = str(run_result.output) if run_result.output else "No answer generated"
            result.total_steps = len(result.steps)
            result.success = True
            
            # Log summary
            print_success(f"Question {question_id} completed successfully")
            print(f"  📊 Total steps: {result.total_steps}")
            print(f"  🔧 Total tool calls: {result.total_tool_calls}")
            if result.total_input_tokens > 0:
                print(f"  🎫 Total tokens: {result.total_input_tokens} input, {result.total_output_tokens} output")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print_error(f"Question {question_id} failed: {e}")
            
            # Try to extract any steps that were completed before the error
            try:
                if self.agent and hasattr(self.agent, 'memory') and self.agent.memory:
                    memory_steps = self.agent.memory.get_full_steps()
                    for step_dict in memory_steps:
                        # Check if this is an ActionStep-like dict
                        if 'step_number' in step_dict:
                            agent_step = AgentStep(
                                step_number=step_dict.get('step_number', 0),
                                timestamp=datetime.now().isoformat(),
                                is_final_answer=step_dict.get('is_final_answer', False)
                            )
                            
                            # Extract timing information
                            if 'timing' in step_dict and step_dict['timing']:
                                agent_step.duration_seconds = step_dict['timing'].get('duration', 0.0) or 0.0
                            
                            # Extract token usage
                            if 'token_usage' in step_dict and step_dict['token_usage']:
                                agent_step.input_tokens = step_dict['token_usage'].get('input_tokens', 0) or 0
                                agent_step.output_tokens = step_dict['token_usage'].get('output_tokens', 0) or 0
                                result.total_input_tokens += agent_step.input_tokens
                                result.total_output_tokens += agent_step.output_tokens
                            
                            # Extract thought/reasoning from model_output
                            if 'model_output' in step_dict and step_dict['model_output']:
                                agent_step.thought = str(step_dict['model_output'])
                            
                            # Extract tool calls
                            if 'tool_calls' in step_dict and step_dict['tool_calls']:
                                for tc in step_dict['tool_calls']:
                                    tool_call = ToolCall(
                                        tool_name=tc.get('name', ''),
                                        arguments=tc.get('arguments', {}) if isinstance(tc.get('arguments'), dict) else {"raw": str(tc.get('arguments', ''))},
                                        timestamp=datetime.now().isoformat(),
                                        tool_call_id=tc.get('id', '')
                                    )
                                    agent_step.tool_calls.append(tool_call)
                                    result.total_tool_calls += 1
                            
                            # Extract observation/result
                            if 'observations' in step_dict and step_dict['observations']:
                                agent_step.observation = str(step_dict['observations'])
                            
                            # Check for errors in this step
                            if 'error' in step_dict and step_dict['error']:
                                agent_step.observation = f"ERROR: {step_dict['error']}"
                            
                            result.steps.append(agent_step)
                    
                    if result.steps:
                        print_info(f"Captured {len(result.steps)} steps before error occurred")
            except Exception as memory_err:
                print_error(f"Could not extract steps from memory: {memory_err}")
        
        result.total_steps = len(result.steps)
        result.total_duration_seconds = time.time() - start_time
        result.timestamp_end = datetime.now().isoformat()
        
        return result
    
    def run_evaluation(self, questions: List[Dict[str, Any]], output_file: str,
                      model_type: str, model_name: str, max_steps: int) -> EvaluationRun:
        """Run evaluation on all questions and save results to JSON."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        evaluation = EvaluationRun(
            run_id=run_id,
            model_type=model_type,
            model_name=model_name,
            mcp_server_url=self.mcp_server_url,
            max_steps=max_steps,
            timestamp_start=datetime.now().isoformat(),
            total_questions=len(questions)
        )
        
        print_info(f"Starting evaluation run {run_id} with {len(questions)} questions")
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*60}")
            
            result = self.run_question(question_data)
            evaluation.results.append(result)
            
            if result.success:
                evaluation.successful_questions += 1
            else:
                evaluation.failed_questions += 1
            
            # Aggregate totals
            evaluation.total_tool_calls += result.total_tool_calls
            evaluation.total_input_tokens += result.total_input_tokens
            evaluation.total_output_tokens += result.total_output_tokens
            
            # Save intermediate results after each question
            self._save_results(evaluation, output_file)
        
        evaluation.timestamp_end = datetime.now().isoformat()
        evaluation.total_duration_seconds = sum(r.total_duration_seconds for r in evaluation.results)
        
        # Final save
        self._save_results(evaluation, output_file)
        
        print(f"\n{'='*60}")
        print_success(f"Evaluation complete!")
        print(f"  Total questions: {evaluation.total_questions}")
        print(f"  Successful: {evaluation.successful_questions}")
        print(f"  Failed: {evaluation.failed_questions}")
        print(f"  Total tool calls: {evaluation.total_tool_calls}")
        print(f"  Total duration: {evaluation.total_duration_seconds:.2f}s")
        if evaluation.total_input_tokens > 0:
            print(f"  Total tokens: {evaluation.total_input_tokens} input, {evaluation.total_output_tokens} output")
        print(f"  Results saved to: {output_file}")
        print(f"{'='*60}")
        
        return evaluation
    
    def _save_results(self, evaluation: EvaluationRun, output_file: str):
        """Save evaluation results to JSON file."""
        # Convert dataclasses to dicts
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            return obj
        
        data = convert_to_dict(evaluation)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def cleanup(self):
        """Clean up resources."""
        if self.mcp_client:
            try:
                self.mcp_client.__exit__(None, None, None)
            except Exception as e:
                print_error(f"Error during cleanup: {e}")


def filter_questions(questions: List[Dict], categories: List[str] = None, 
                    difficulties: List[str] = None, limit: int = None) -> List[Dict]:
    """Filter questions by category and difficulty."""
    filtered = questions
    
    if categories:
        filtered = [q for q in filtered if q["category"] in categories]
    
    if difficulties:
        filtered = [q for q in filtered if q["difficulty"] in difficulties]
    
    if limit:
        filtered = filtered[:limit]
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Smolagents Batch Evaluation")
    parser.add_argument("--mcp-server-url", type=str, help="URL of the MCP server")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--model-type", type=str, default="openai",
                       choices=["openai", "azure", "hf_inference"],
                       help="Model type to use")
    parser.add_argument("--model-name", type=str, help="Model name/ID")
    parser.add_argument("--api-key", type=str, help="API key for the model provider")
    parser.add_argument("--base-url", type=str, help="Base URL for API endpoint")
    parser.add_argument("--api-version", type=str, help="API version (for Azure)")
    parser.add_argument("--max-steps", type=int, default=5, help="Max agent steps per question")
    parser.add_argument("--categories", type=str, nargs="+",
                       help="Filter questions by category")
    parser.add_argument("--difficulties", type=str, nargs="+",
                       choices=["easy", "medium", "hard"],
                       help="Filter questions by difficulty")
    parser.add_argument("--limit", type=int, help="Limit number of questions to evaluate")
    parser.add_argument("--questions-file", type=str,
                       help="Path to custom questions JSON file")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        print_info("Initializing Batch Evaluation Agent...")
        agent = BatchEvaluationAgent(mcp_server_url=args.mcp_server_url)
        
        # Initialize model
        model_name = agent.initialize_model(
            model_type=args.model_type,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            api_version=args.api_version
        )
        
        # Initialize agent
        agent.initialize_agent(max_steps=args.max_steps)
        
        # Load questions
        if args.questions_file:
            with open(args.questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print_info(f"Loaded {len(questions)} questions from {args.questions_file}")
        else:
            questions = TRANSFORMERS_QUESTIONS
            print_info(f"Using {len(questions)} built-in questions")
        
        # Filter questions
        questions = filter_questions(
            questions,
            categories=args.categories,
            difficulties=args.difficulties,
            limit=args.limit
        )
        print_info(f"Evaluating {len(questions)} questions after filtering")
        
        # Run evaluation
        agent.run_evaluation(
            questions=questions,
            output_file=args.output,
            model_type=args.model_type,
            model_name=model_name,
            max_steps=args.max_steps
        )
        
    except KeyboardInterrupt:
        print_info("\nEvaluation interrupted by user")
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
