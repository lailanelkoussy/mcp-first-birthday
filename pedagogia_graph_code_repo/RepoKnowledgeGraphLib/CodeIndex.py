import logging
from tqdm import tqdm
import uuid
from typing import Literal
from abc import ABC, abstractmethod
import lancedb
import os
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

try:
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from .utils.logger_utils import setup_logger

LOGGER_NAME = 'CODE_INDEX_LOGGER'
STOP_AFTER_ATTEMPT = int(os.getenv("STOP_AFTER_ATTEMPT", 5))
WAIT_BETWEEN_RETRIES = int(os.getenv("WAIT_BETWEEN_RETRIES", 2))
MODEL_ID = os.getenv("MODEL_ID")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.2))
TOP_P = float(os.getenv('TOP_P', 0.95))
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
STOP = None
EMBEDDING_MODEL_URL = os.getenv('EMBEDDING_MODEL_URL')
EMBEDDING_MODEL_API_KEY = os.getenv('EMBEDDING_MODEL_API_KEY', "no_need")
EMBEDDING_NUMBER_DIMENSIONS = int(os.getenv('EMBEDDING_NUMBER_DIMENSIONS', 1024))

WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', "localhost")
WEAVIATE_PORT = int(os.getenv('WEAVIATE_PORT', 8080))
WEAVIATE_GRPC_PORT = int(os.getenv('WEAVIATE_GRPC_PORT', 50051))
ALPHA_SEARCH_VALUE = float(os.getenv('ALPHA_SEARCH_VALUE', 0.8))
LANCEDB_PATH = os.getenv('LANCEDB_PATH', './local_code_index_db')


class BaseCodeIndex(ABC):
    """Abstract base class for code indexing implementations"""

    def __init__(self, nodes: list, model_service, index_type: Literal['embedding-only', 'keyword-only', 'hybrid'] = 'hybrid',
                 embedding_batch_size: int = 20, use_embed: bool = True):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.model_service = model_service
        self.index_type = index_type
        self.embedding_batch_size = embedding_batch_size
        self.use_embed = use_embed

    @abstractmethod
    def query(self, query: str, n_results: int) -> dict:
        """Query the index and return results"""
        pass

    @abstractmethod
    def __del__(self):
        """Clean up resources"""
        pass


class WeaviateCodeIndex(BaseCodeIndex):
    """Weaviate-based code index implementation"""

    def __init__(self, nodes: list, model_service, index_type: Literal['embedding-only', 'keyword-only', 'hybrid'] = 'hybrid',
                 embedding_batch_size: int = 20, use_embed: bool = True,
                 host: str = None, port: int = None, grpc_port: int = None):
        super().__init__(nodes, model_service, index_type, embedding_batch_size, use_embed)

        # Use provided parameters or fall back to environment variables
        weaviate_host = host or WEAVIATE_HOST
        weaviate_port = port or WEAVIATE_PORT
        weaviate_grpc_port = grpc_port or WEAVIATE_GRPC_PORT

        # Connect to Weaviate
        self.weaviate_client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_port,
            grpc_port=weaviate_grpc_port
        )

        # Create a unique collection name
        self.collection_name = f"CodeChunks_{str(uuid.uuid4()).replace('-', '_')}"

        # Create collection with schema using the v4 API
        # Use vector_config with Configure.Vectors.self_provided() - the modern approach
        self.collection = self.weaviate_client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="node_id", data_type=DataType.TEXT),
                Property(name="name", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="node_type", data_type=DataType.TEXT),
                Property(name="order_in_file", data_type=DataType.INT),
                Property(name="declared_entities", data_type=DataType.TEXT),
                Property(name="called_entities", data_type=DataType.TEXT),
            ],
            # We provide our own vectors using the modern vector_config API
            vector_config=Configure.Vectors.self_provided(),
        )

        chunk_nodes = [node for node in nodes if node.node_type == 'chunk']
        

        # Pre-generate embeddings in batches for better performance
        if self.index_type != 'keyword-only':
            # Identify nodes that need embeddings
            nodes_needing_embeddings = [
                node for node in chunk_nodes 
                if node.embedding is None or not use_embed
            ]
            
            if nodes_needing_embeddings:
                self.logger.info(f'Batch embedding {len(nodes_needing_embeddings)} nodes')
                
                # Process in batches
                for i in tqdm(range(0, len(nodes_needing_embeddings), self.embedding_batch_size), 
                             desc="Batch embedding nodes"):
                    batch_nodes = nodes_needing_embeddings[i:i + self.embedding_batch_size]
                    texts_to_embed = [node.get_field_to_embed() for node in batch_nodes]
                    
                    # Batch embed all texts
                    embeddings = self.model_service.embed_chunk_code_batch(texts_to_embed)
                    
                    # Assign embeddings back to nodes
                    for node, embedding in zip(batch_nodes, embeddings):
                        node.embedding = embedding

        # Batch insert data into Weaviate
        with self.collection.batch.dynamic() as batch:
            for node in tqdm(chunk_nodes, desc="Indexing nodes"):
                self.logger.debug(f'Indexing node : {node.id}')

                # Use pre-computed embedding
                embedding = None
                if self.index_type != 'keyword-only':
                    embedding = node.embedding

                # Prepare properties
                properties = {
                    "node_id": node.id,
                    "name": node.name,
                    "content": node.content,
                    "description": node.description or "",
                    "path": node.path,
                    "language": node.language,
                    "node_type": node.node_type,
                    "order_in_file": node.order_in_file,
                    "declared_entities": str(node.declared_entities),
                    "called_entities": str(node.called_entities),
                }

                # Add object with or without vector based on index_type
                if self.index_type == 'keyword-only':
                    # No vector needed for keyword-only search
                    batch.add_object(properties=properties)
                else:
                    # Add with vector for embedding-only and hybrid modes
                    batch.add_object(
                        properties=properties,
                        vector=embedding
                    )


    def query(self, query: str, n_results:int) -> dict:
        """
        Perform search based on index_type:
        - 'embedding-only': pure vector search
        - 'keyword-only': pure keyword search (BM25)
        - 'hybrid': hybrid search combining both (alpha controls weighting)

        Weaviate's hybrid search uses:
        - alpha=0: pure keyword search (BM25)
        - alpha=1: pure vector search
        - alpha=0.5-0.8: balanced hybrid search (recommended)
        """
        try:
            # Execute search based on index_type
            if self.index_type == 'keyword-only':
                # Pure BM25 keyword search
                response = self.collection.query.bm25(
                    query=query,
                    limit=n_results,
                    return_metadata=MetadataQuery(score=True)
                )
            elif self.index_type == 'embedding-only':
                # Pure vector search
                embedding = self.model_service.embed_query(query)
                response = self.collection.query.near_vector(
                    near_vector=embedding,
                    limit=n_results,
                    return_metadata=MetadataQuery(distance=True)
                )
            else:  # 'hybrid'
                # Hybrid search combining keyword and vector
                embedding = self.model_service.embed_query(query)
                response = self.collection.query.hybrid(
                    query=query,
                    vector=embedding,
                    limit=n_results,
                    alpha=ALPHA_SEARCH_VALUE,
                    return_metadata=MetadataQuery(distance=True, score=True)
                )

            # Convert to ChromaDB-like format for compatibility
            results = {
                'ids': [[]],
                'distances': [[]],
                'metadatas': [[]],
                'documents': [[]]
            }

            for obj in response.objects:
                results['ids'][0].append(obj.properties['node_id'])
                results['distances'][0].append(obj.metadata.distance if obj.metadata.distance else 0.0)
                results['metadatas'][0].append({
                    'id': obj.properties['node_id'],
                    'name': obj.properties['name'],
                    'content': obj.properties['content'],
                    'description': obj.properties['description'],
                    'path': obj.properties['path'],
                    'language': obj.properties['language'],
                    'node_type': obj.properties['node_type'],
                    'order_in_file': str(obj.properties['order_in_file']),
                    'declared_entities': obj.properties['declared_entities'],
                    'called_entities': obj.properties['called_entities'],
                })
                results['documents'][0].append(obj.properties['content'])

            return results

        except Exception as e:
            self.logger.error(f'Failed to query: {e}', exc_info=True)
            raise e

    def __del__(self):
        """Clean up Weaviate connection"""
        if hasattr(self, 'weaviate_client'):
            try:
                self.weaviate_client.close()
            except:
                pass


class LanceDBCodeIndex(BaseCodeIndex):
    """LanceDB-based code index implementation"""

    def __init__(self, nodes: list, model_service, index_type: Literal['embedding-only', 'keyword-only', 'hybrid'] = 'hybrid',
                 embedding_batch_size: int = 20, use_embed: bool = True, db_path: str = None):
        super().__init__(nodes, model_service, index_type, embedding_batch_size, use_embed)

        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB is not available. Please install it with: pip install lancedb")

        # Embedded DB
        self.db_path = db_path or LANCEDB_PATH
        self.db = lancedb.connect(self.db_path)
        self.table_name = f"code_chunks_{uuid.uuid4().hex}"
        self.table = None

        chunk_nodes = [node for node in nodes if node.node_type == "chunk"]

        # -----------------------------------------------------------
        # Create embeddings IF using vector search
        # -----------------------------------------------------------
        if self.index_type != "keyword-only":
            nodes_needing_embeddings = [
                node for node in chunk_nodes
                if node.embedding is None or not use_embed
            ]

            if nodes_needing_embeddings:
                self.logger.info(f"Embedding {len(nodes_needing_embeddings)} chunks...")
                for i in tqdm(range(0, len(nodes_needing_embeddings), self.embedding_batch_size),
                             desc="Batch embedding nodes"):
                    batch = nodes_needing_embeddings[i:i + self.embedding_batch_size]
                    texts = [n.get_field_to_embed() for n in batch]
                    batch_embeds = self.model_service.embed_chunk_code_batch(texts)

                    for n, emb in zip(batch, batch_embeds):
                        n.embedding = np.array(emb, dtype=np.float32)

        # -----------------------------------------------------------
        # Prepare rows (only include vector column when allowed)
        # -----------------------------------------------------------
        rows = []
        for node in chunk_nodes:
            row = {
                "node_id": node.id,
                "name": node.name,
                "content": node.content,
                "description": node.description or "",
                "path": node.path,
                "language": node.language,
                "node_type": node.node_type,
                "order_in_file": node.order_in_file,
                "declared_entities": str(node.declared_entities),
                "called_entities": str(node.called_entities),
            }

            # Add embeddings only for hybrid/embedding-only
            if self.index_type != "keyword-only":
                row["vector"] = node.embedding

            rows.append(row)

        # Create table
        self.table = self.db.create_table(self.table_name, data=rows)
        self.logger.info(f"Created LanceDB table: {self.table_name}")

    def query(self, query: str, n_results: int) -> dict:
        """
        Perform search based on index_type:
        - 'embedding-only': pure vector search
        - 'keyword-only': keyword search using SQL-like filtering
        - 'hybrid': vector search with keyword filtering
        """
        try:
            # ---------------------- KEYWORD ONLY ----------------------
            if self.index_type == "keyword-only":
                # Use SQL-like query for keyword search when no vector column
                q = self.table.search().where(
                    f"content LIKE '%{query}%' OR name LIKE '%{query}%' OR description LIKE '%{query}%'",
                    prefilter=True
                )

            # ---------------------- VECTOR ONLY -----------------------
            elif self.index_type == "embedding-only":
                emb = np.array(self.model_service.embed_query(query), dtype=np.float32)
                q = self.table.search(
                    emb,
                    vector_column_name="vector"
                )

            # ---------------------- HYBRID ----------------------------
            else:
                emb = np.array(self.model_service.embed_query(query), dtype=np.float32)
                q = (
                    self.table.search(emb, vector_column_name="vector")
                    .where(
                        f"content LIKE '%{query}%' OR name LIKE '%{query}%' OR description LIKE '%{query}%'",
                        prefilter=False
                    )
                )

            df = q.limit(n_results).to_pandas()

            # Build result format (ChromaDB-like format for compatibility)
            results = {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
            }

            for _, row in df.iterrows():
                results["ids"][0].append(row["node_id"])
                results["documents"][0].append(row["content"])
                results["distances"][0].append(float(row.get("_distance", 0)))

                results["metadatas"][0].append({
                    "id": row["node_id"],
                    "name": row["name"],
                    "content": row["content"],
                    "description": row["description"],
                    "path": row["path"],
                    "language": row["language"],
                    "node_type": row["node_type"],
                    "order_in_file": str(row["order_in_file"]),
                    "declared_entities": row["declared_entities"],
                    "called_entities": row["called_entities"],
                })

            return results

        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            raise

    def __del__(self):
        """Clean up resources"""
        pass


# Factory function to create the appropriate CodeIndex
def CodeIndex(
    nodes: list,
    model_service,
    index_type: Literal['embedding-only', 'keyword-only', 'hybrid'] = 'hybrid',
    embedding_batch_size: int = 20,
    use_embed: bool = True,
    backend: Literal['weaviate', 'lancedb'] = 'weaviate',
    db_path: str = None,
    host: str = None,
    port: int = None,
    grpc_port: int = None
) -> BaseCodeIndex:
    """
    Factory function to create a CodeIndex instance.

    Args:
        nodes: List of nodes to index
        model_service: Service for embedding generation
        index_type: Type of search ('embedding-only', 'keyword-only', or 'hybrid')
        embedding_batch_size: Batch size for embedding generation
        use_embed: Whether to use pre-computed embeddings
        backend: Which backend to use ('weaviate' or 'lancedb')
        db_path: Path for LanceDB (only used with 'lancedb' backend)
        host: Weaviate host (only used with 'weaviate' backend)
        port: Weaviate port (only used with 'weaviate' backend)
        grpc_port: Weaviate gRPC port (only used with 'weaviate' backend)

    Returns:
        BaseCodeIndex: Either WeaviateCodeIndex or LanceDBCodeIndex instance
    """
    if backend == 'lancedb':
        return LanceDBCodeIndex(
            nodes=nodes,
            model_service=model_service,
            index_type=index_type,
            embedding_batch_size=embedding_batch_size,
            use_embed=use_embed,
            db_path=db_path
        )
    elif backend == 'weaviate':
        return WeaviateCodeIndex(
            nodes=nodes,
            model_service=model_service,
            index_type=index_type,
            embedding_batch_size=embedding_batch_size,
            use_embed=use_embed,
            host=host,
            port=port,
            grpc_port=grpc_port
        )
    else:  # default to weaviate
        return WeaviateCodeIndex(
            nodes=nodes,
            model_service=model_service,
            index_type=index_type,
            embedding_batch_size=embedding_batch_size,
            use_embed=use_embed,
            host=host,
            port=port,
            grpc_port=grpc_port
        )


