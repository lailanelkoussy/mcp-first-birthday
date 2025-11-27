from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import httpx 
from sentence_transformers import SentenceTransformer

# Optional torch import for CUDA detection
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

from .utils.logger_utils import setup_logger

LOGGER_NAME = "MODEL_SERVICE_LOGGER"
# GENERATION ENV VARIABLES (defaults)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", 'http://0.0.0.0:8000/v1')
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN", 'no-need')
MODEL_NAME = os.getenv('MODEL_NAME', "meta-llama/Llama-3.2-3B-Instruct")
# EMBED ENV VARIABLES (defaults)
OPENAI_EMBED_BASE_URL = os.getenv("OPENAI_EMBED_BASE_URL", 'http://0.0.0.0:8001/v1')
OPENAI_EMBED_TOKEN = os.getenv("OPENAI_EMBED_TOKEN", 'no-need')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', "Alibaba-NLP/gte-Qwen2-1.5B-instruct")

# Additional ENV defaults requested
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
TOP_P = float(os.getenv("TOP_P", 0.95))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", 0))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 0))
EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL", "")
EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "no_need")
EMBEDDING_NUMBER_DIMENSIONS = int(os.getenv("EMBEDDING_NUMBER_DIMENSIONS", 1024))

STOP_AFTER_ATTEMPT = int(os.getenv("STOP_AFTER_ATTEMPT", 5))
WAIT_BETWEEN_RETRIES = int(os.getenv("WAIT_BETWEEN_RETRIES", 2))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 240))

# Note: module-level clients remain for backward compatibility but instances will create their own if timeout is overridden.
long_timeout_client = httpx.Client(timeout=REQUEST_TIMEOUT)
long_timeout_async_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


class ModelServiceInterface(ABC):
    """
    Abstract base class defining the interface for model services.
    All model services should implement these methods.
    """

    # accept model_kwargs so variables can be overridden at runtime
    def __init__(self, model_name: str = None, model_kwargs: dict = None):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)

        model_kwargs = model_kwargs or {}

        # allow overriding via model_kwargs; fall back to module-level defaults
        self.openai_base_url = model_kwargs.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
        self.openai_token = model_kwargs.get("OPENAI_TOKEN", OPENAI_TOKEN)
        # model_name param takes precedence, then model_kwargs then default env
        self.model_name = model_name or model_kwargs.get("MODEL_NAME", MODEL_NAME)

        # embed defaults (may be overridden by subclasses or model_kwargs)
        self.openai_embed_base_url = model_kwargs.get("OPENAI_EMBED_BASE_URL", OPENAI_EMBED_BASE_URL)
        self.openai_embed_token = model_kwargs.get("OPENAI_EMBED_TOKEN", OPENAI_EMBED_TOKEN)
        self.embed_model_name = model_kwargs.get("EMBED_MODEL_NAME", EMBED_MODEL_NAME)

        # other configurable parameters
        self.max_tokens = int(model_kwargs.get("MAX_TOKENS", MAX_TOKENS))
        self.temperature = float(model_kwargs.get("TEMPERATURE", TEMPERATURE))
        self.top_p = float(model_kwargs.get("TOP_P", TOP_P))
        self.frequency_penalty = float(model_kwargs.get("FREQUENCY_PENALTY", FREQUENCY_PENALTY))
        self.presence_penalty = float(model_kwargs.get("PRESENCE_PENALTY", PRESENCE_PENALTY))
        self.embedding_model_url = model_kwargs.get("EMBEDDING_MODEL_URL", EMBEDDING_MODEL_URL)
        self.embedding_model_api_key = model_kwargs.get("EMBEDDING_MODEL_API_KEY", EMBEDDING_MODEL_API_KEY)
        self.embedding_number_dimensions = int(model_kwargs.get("EMBEDDING_NUMBER_DIMENSIONS", EMBEDDING_NUMBER_DIMENSIONS))

        self.stop_after_attempt = int(model_kwargs.get("STOP_AFTER_ATTEMPT", STOP_AFTER_ATTEMPT))
        self.wait_between_retries = int(model_kwargs.get("WAIT_BETWEEN_RETRIES", WAIT_BETWEEN_RETRIES))
        request_timeout = int(model_kwargs.get("REQUEST_TIMEOUT", REQUEST_TIMEOUT))

        # create per-instance httpx clients in case REQUEST_TIMEOUT was overridden
        self.long_timeout_client = httpx.Client(timeout=request_timeout)
        self.long_timeout_async_client = httpx.AsyncClient(timeout=request_timeout)

        # Initialize query client (shared by all implementations)
        self.client = OpenAI(
            base_url=self.openai_base_url,
            api_key=self.openai_token,
            http_client=self.long_timeout_client,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.openai_base_url,
            api_key=self.openai_token,
            http_client=self.long_timeout_async_client,
        )

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def query(self, prompt: str, model_name: str) -> str:
        """Query the model with a prompt."""
        if model_name is None:
            model_name = self.model_name
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def query_with_instructions(self, prompt: str, instructions: str, model_name: str) -> str:
        """Query the model with additional system instructions."""
        if model_name is None:
            model_name = self.model_name
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    async def query_async(self, prompt: str, model_name: str ) -> str:
        """Async version of query."""
        if model_name is None:
            model_name = self.model_name
        completion = await self.async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    async def query_with_instructions_async(self, prompt: str, instructions: str, model_name: str) -> str:
        """Async version of query with instructions."""
        if model_name is None:
            model_name = self.model_name
        completion = await self.async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    @abstractmethod
    def embed(self, text_to_embed: str) -> list:
        """Embed text using the configured embedding model."""
        pass

    @abstractmethod
    async def embed_async(self, text_to_embed: str) -> list:
        """Async version of embed."""
        pass

    @abstractmethod
    def embed_chunk_code(self, code_to_embed: str) -> list:
        """Embed code chunk for storage/indexing."""
        pass

    @abstractmethod
    def embed_query(self, query_to_embed: str) -> list:
        """Embed query for retrieval."""
        pass

    @abstractmethod
    def embed_batch(self, texts_to_embed: list[str]) -> list[list]:
        """Embed multiple texts in a batch for better performance."""
        pass

    @abstractmethod
    def embed_chunk_code_batch(self, codes_to_embed: list[str]) -> list[list]:
        """Embed multiple code chunks in a batch for storage/indexing."""
        pass


class OpenAIModelService(ModelServiceInterface):
    """
    Model service that uses OpenAI client for both queries and embeddings.
    """

    def __init__(self, model_name: str = None, embed_model_name: str = None, model_kwargs: dict = None):
        # forward model_kwargs to base so it can set instance-wide config
        super().__init__(model_name=model_name, model_kwargs=model_kwargs)

        # allow override of embed model name via param or model_kwargs
        model_kwargs = model_kwargs or {}
        self.embed_model_name = embed_model_name or model_kwargs.get("EMBED_MODEL_NAME", self.embed_model_name)

        # embed client should use the instance-level embed base/token
        self.embed_client = OpenAI(
            base_url=model_kwargs.get("OPENAI_EMBED_BASE_URL", self.openai_embed_base_url),
            api_key=model_kwargs.get("OPENAI_EMBED_TOKEN", self.openai_embed_token),
            http_client=self.long_timeout_client,
        )
        self.async_embed_client = AsyncOpenAI(
            base_url=model_kwargs.get("OPENAI_EMBED_BASE_URL", self.openai_embed_base_url),
            api_key=model_kwargs.get("OPENAI_EMBED_TOKEN", self.openai_embed_token),
            http_client=self.long_timeout_async_client,
        )

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def embed(self, text_to_embed: str) -> list:
        """Embed text using OpenAI embeddings API."""
        response = self.embed_client.embeddings.create(
            input=text_to_embed,
            model=self.embed_model_name,
        )
        return response.data[0].embedding

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    async def embed_async(self, text_to_embed: str) -> list:
        """Async version of embed using OpenAI embeddings API."""
        response = await self.async_embed_client.embeddings.create(
            input=text_to_embed,
            model=self.embed_model_name,
        )
        return response.data[0].embedding

    def embed_chunk_code(self, code_to_embed: str) -> list:
        """Embed code chunk using OpenAI embeddings API (same as embed)."""
        return self.embed(code_to_embed)

    def embed_query(self, query_to_embed: str) -> list:
        """Embed query using OpenAI embeddings API (same as embed)."""
        return self.embed(query_to_embed)

    @retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_fixed(WAIT_BETWEEN_RETRIES))
    def embed_batch(self, texts_to_embed: list[str]) -> list[list]:
        """Embed multiple texts in a batch using OpenAI embeddings API."""
        if not texts_to_embed:
            return []
        response = self.embed_client.embeddings.create(
            input=texts_to_embed,
            model=self.embed_model_name,
        )
        return [item.embedding for item in response.data]

    def embed_chunk_code_batch(self, codes_to_embed: list[str]) -> list[list]:
        """Embed multiple code chunks in a batch using OpenAI embeddings API."""
        return self.embed_batch(codes_to_embed)


class SentenceTransformersModelService(ModelServiceInterface):
    """
    Model service that uses OpenAI client for queries and SentenceTransformers for embeddings.
    Optimized for high-throughput batch embedding with GPU support.
    """

    def __init__(self, model_name: str = None, embed_model_name: str = None, model_kwargs: dict = None):
        super().__init__(model_name=model_name, model_kwargs=model_kwargs)
        model_kwargs = model_kwargs or {}
        # embed_model_name may be overridden by model_kwargs
        self.embed_model_name = embed_model_name or model_kwargs.get("EMBED_MODEL_NAME", self.embed_model_name)

        # Debug GPU detection
        self.logger.info(f'PyTorch available: {_TORCH_AVAILABLE}')
        if _TORCH_AVAILABLE:
            self.logger.info(f'CUDA available: {torch.cuda.is_available()}')
            self.logger.info(f'CUDA device count: {torch.cuda.device_count()}')
            if torch.cuda.is_available():
                self.logger.info(f'CUDA device name: {torch.cuda.get_device_name(0)}')

        # Select device: prefer CUDA if available
        self.device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.logger.info(f'Initializing SentenceTransformer on device: {self.device}')

        # Set batch size based on device and available memory
        # Larger batch sizes significantly improve GPU throughput
        self.encode_batch_size = int(model_kwargs.get("ENCODE_BATCH_SIZE", 64 if self.device == "cuda" else 32))
        
        # Show CUDA memory info if available
        if self.device == "cuda" and _TORCH_AVAILABLE:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f'GPU memory available: {gpu_memory:.2f} GB')
                # Adjust batch size based on available GPU memory
                if gpu_memory > 16:
                    self.encode_batch_size = max(self.encode_batch_size, 128)
                elif gpu_memory > 8:
                    self.encode_batch_size = max(self.encode_batch_size, 64)
            except Exception as e:
                self.logger.warning(f'Could not get GPU memory info: {e}')

        self.logger.info(f'Using encode batch size: {self.encode_batch_size}')

        # Initialize embedding model on the chosen device with performance optimizations
        self.embedding_model = SentenceTransformer(
            self.embed_model_name,
            trust_remote_code=True,
            device=self.device
        )
        
        # Enable half precision for faster inference on CUDA
        if self.device == "cuda" and _TORCH_AVAILABLE:
            try:
                # Check if model supports half precision
                self.embedding_model.half()
                self.logger.info('Enabled half precision (FP16) for faster GPU inference')
            except Exception as e:
                self.logger.warning(f'Could not enable half precision: {e}')

    def embed(self, text_to_embed: str) -> list:
        """Embed text using SentenceTransformers."""
        embeddings = self.embedding_model.encode(
            [text_to_embed],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])

    async def embed_async(self, text_to_embed: str) -> list:
        """
        Async version of embed using SentenceTransformers.
        Note: SentenceTransformers doesn't have native async support,
        so this runs synchronously but maintains the async interface.
        """
        return self.embed(text_to_embed)

    def embed_chunk_code(self, code_to_embed: str) -> list:
        """Embed code chunk using SentenceTransformers (no special prompt)."""
        self.logger.debug(f'Embedding code using {self.embed_model_name}')
        embeddings = self.embedding_model.encode(
            [code_to_embed],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])

    def embed_query(self, query_to_embed: str) -> list:
        """Embed query using SentenceTransformers with retrieval prompt."""
        self.logger.debug(f'Embedding query using {self.embed_model_name}')
        embeddings = self.embedding_model.encode(
            [query_to_embed],
            prompt='Given this prompt, retrieve relevant content\n Query:',
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])

    def embed_batch(self, texts_to_embed: list[str]) -> list[list]:
        """Embed multiple texts in a batch using SentenceTransformers with optimized settings."""
        if not texts_to_embed:
            return []
        self.logger.info(f'Batch embedding {len(texts_to_embed)} texts using {self.embed_model_name}')
        embeddings = self.embedding_model.encode(
            texts_to_embed,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts_to_embed) > 100,  # Only show progress for large batches
            normalize_embeddings=True  # Normalize for better similarity computation
        )
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    def embed_chunk_code_batch(self, codes_to_embed: list[str]) -> list[list]:
        """Embed multiple code chunks in a batch using SentenceTransformers with optimized settings."""
        if not codes_to_embed:
            return []
        self.logger.info(f'Batch embedding {len(codes_to_embed)} code chunks using {self.embed_model_name}')
        embeddings = self.embedding_model.encode(
            codes_to_embed,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(codes_to_embed) > 100,  # Only show progress for large batches
            normalize_embeddings=True  # Normalize for better similarity computation
        )
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]


def create_model_service(**kwargs) -> ModelServiceInterface:
    """
    Factory function to create the appropriate ModelService based on embedder_type.

    Args:
        **kwargs: Additional arguments including 'embedder_type' ('openai' or 'sentence-transformers')
                and optional 'model_kwargs' dict which can override any env var defaults.
    Returns:
        ModelServiceInterface: An instance of the appropriate ModelService
    """
    model_kwargs = kwargs.pop('model_kwargs', None)
    embedder_type = kwargs.pop('embedder_type', 'openai')

    if embedder_type == 'openai':
        return OpenAIModelService(model_kwargs=model_kwargs, **kwargs)
    elif embedder_type == 'sentence-transformers':
        return SentenceTransformersModelService(model_kwargs=model_kwargs, **kwargs)
    else:
        logging.getLogger(LOGGER_NAME).warning(
            f'Unknown embedder type: {embedder_type}, defaulting to OpenAI'
        )
        return OpenAIModelService(model_kwargs=model_kwargs, **kwargs)
