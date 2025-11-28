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
                 embedding_batch_size: int = 64, use_embed: bool = True):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.model_service = model_service
        self.index_type = index_type
        # Use larger batch size by default for better throughput
        self.embedding_batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', embedding_batch_size))
        self.use_embed = use_embed
        self.logger.info(f"CodeIndex initialized with batch_size={self.embedding_batch_size}, index_type={index_type}")

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
        self.logger.info(f"Weaviate indexing {len(chunk_nodes)} chunk nodes with batch_size={self.embedding_batch_size}")

        # Pre-generate embeddings in batches for better performance
        if self.index_type != 'keyword-only':
            # Identify nodes that need embeddings
            nodes_needing_embeddings = [
                node for node in chunk_nodes 
                if node.embedding is None or (isinstance(node.embedding, (list,)) and len(node.embedding) == 0) or not use_embed
            ]
            
            if nodes_needing_embeddings:
                total_batches = (len(nodes_needing_embeddings) + self.embedding_batch_size - 1) // self.embedding_batch_size
                self.logger.info(f'Batch embedding {len(nodes_needing_embeddings)} nodes in {total_batches} batches')
                
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
                    
                    # Log progress every 10 batches
                    batch_num = i // self.embedding_batch_size + 1
                    if batch_num % 10 == 0:
                        self.logger.info(f"Completed batch {batch_num}/{total_batches}")
                
                self.logger.info(f"Embedding complete: processed {len(nodes_needing_embeddings)} nodes")
            else:
                self.logger.info(f"Using existing embeddings for all {len(chunk_nodes)} nodes")

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
        self.logger.info(f"LanceDB indexing {len(chunk_nodes)} chunk nodes with batch_size={self.embedding_batch_size}")

        # -----------------------------------------------------------
        # Create embeddings IF using vector search
        # -----------------------------------------------------------
        if self.index_type != "keyword-only":
            # Find nodes that need embeddings
            # use_embed=True means we should USE existing embeddings if available
            # use_embed=False means we should regenerate all embeddings
            nodes_needing_embeddings = []
            for node in chunk_nodes:
                needs_embedding = False
                if not use_embed:
                    # Regenerate all embeddings
                    needs_embedding = True
                elif node.embedding is None:
                    needs_embedding = True
                elif isinstance(node.embedding, (list, np.ndarray)) and len(node.embedding) == 0:
                    needs_embedding = True
                
                if needs_embedding:
                    nodes_needing_embeddings.append(node)

            if nodes_needing_embeddings:
                total_batches = (len(nodes_needing_embeddings) + self.embedding_batch_size - 1) // self.embedding_batch_size
                self.logger.info(f"Embedding {len(nodes_needing_embeddings)} chunks in {total_batches} batches (batch_size={self.embedding_batch_size})...")
                
                for i in tqdm(range(0, len(nodes_needing_embeddings), self.embedding_batch_size),
                             desc="Batch embedding nodes"):
                    batch = nodes_needing_embeddings[i:i + self.embedding_batch_size]
                    texts = [n.get_field_to_embed() for n in batch]
                    batch_embeds = self.model_service.embed_chunk_code_batch(texts)

                    for n, emb in zip(batch, batch_embeds):
                        n.embedding = np.array(emb, dtype=np.float32)
                    
                    # Log progress every 10 batches
                    batch_num = i // self.embedding_batch_size + 1
                    if batch_num % 10 == 0:
                        self.logger.info(f"Completed batch {batch_num}/{total_batches}")
                
                self.logger.info(f"Embedding complete: processed {len(nodes_needing_embeddings)} chunks")
            else:
                self.logger.info(f"Using existing embeddings for all {len(chunk_nodes)} chunks")

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
        
        # Create full-text search index for keyword and hybrid search
        try:
            # use_tantivy=True is required to support multiple field names as a list
            self.table.create_fts_index(["content", "name", "description"], replace=True, use_tantivy=True)
            self.logger.info(f"Created FTS index on table: {self.table_name}")
        except Exception as e:
            self.logger.warning(f"Failed to create FTS index (keyword search may be slower): {e}")

    def query(self, query: str, n_results: int) -> dict:
        """
        Perform search based on index_type:
        - 'embedding-only': pure vector search
        - 'keyword-only': full-text search using LanceDB's native FTS
        - 'hybrid': combines vector similarity and full-text search with reranking
        """
        try:
            # ---------------------- KEYWORD ONLY ----------------------
            if self.index_type == "keyword-only":
                # Use LanceDB full-text search (requires FTS index on the table)
                try:
                    # Try full-text search first
                    df = self.table.search(query, query_type="fts").limit(n_results).to_pandas()
                except Exception as fts_error:
                    self.logger.warning(f"FTS search failed, falling back to scan: {fts_error}")
                    # Fallback: scan all rows and filter in Python
                    all_df = self.table.to_pandas()
                    query_lower = query.lower()
                    # Split query into words for more flexible matching
                    query_words = query_lower.split()
                    
                    def matches_query(row):
                        text = f"{row.get('content', '')} {row.get('name', '')} {row.get('description', '')}".lower()
                        # Match if any query word is found
                        return any(word in text for word in query_words)
                    
                    mask = all_df.apply(matches_query, axis=1)
                    df = all_df[mask].head(n_results)
                    # Add a dummy distance column
                    df = df.copy()
                    df['_distance'] = 0.0

            # ---------------------- VECTOR ONLY -----------------------
            elif self.index_type == "embedding-only":
                emb = np.array(self.model_service.embed_query(query), dtype=np.float32)
                df = self.table.search(
                    emb,
                    vector_column_name="vector"
                ).limit(n_results).to_pandas()

            # ---------------------- HYBRID ----------------------------
            else:
                # For hybrid search, we do vector search and optionally boost results
                # that also match keywords. This is more flexible than requiring both.
                emb = np.array(self.model_service.embed_query(query), dtype=np.float32)
                
                # Get more results from vector search to allow for reranking
                vector_limit = min(n_results * 3, 100)  # Get 3x results for reranking
                df = self.table.search(
                    emb,
                    vector_column_name="vector"
                ).limit(vector_limit).to_pandas()
                
                if not df.empty:
                    # Rerank results based on keyword matches
                    query_lower = query.lower()
                    query_words = query_lower.split()
                    
                    def compute_keyword_score(row):
                        """Compute a keyword match score (higher is better)"""
                        text = f"{row.get('content', '')} {row.get('name', '')} {row.get('description', '')}".lower()
                        score = 0
                        # Exact phrase match gets highest score
                        if query_lower in text:
                            score += 10
                        # Word matches
                        for word in query_words:
                            if word in text:
                                score += 1
                            # Bonus for word in name (more relevant)
                            if word in str(row.get('name', '')).lower():
                                score += 2
                        return score
                    
                    # Add keyword scores
                    df = df.copy()
                    df['_keyword_score'] = df.apply(compute_keyword_score, axis=1)
                    
                    # Normalize distance to a similarity score (lower distance = higher similarity)
                    max_dist = df['_distance'].max() if df['_distance'].max() > 0 else 1.0
                    df['_vector_score'] = 1.0 - (df['_distance'] / max_dist)
                    
                    # Combined score: weighted sum of vector similarity and keyword score
                    # Alpha controls the balance (higher alpha = more weight on vector search)
                    alpha = 0.7  # 70% vector, 30% keyword
                    max_keyword = df['_keyword_score'].max() if df['_keyword_score'].max() > 0 else 1.0
                    df['_combined_score'] = (
                        alpha * df['_vector_score'] + 
                        (1 - alpha) * (df['_keyword_score'] / max_keyword)
                    )
                    
                    # Sort by combined score (descending) and take top n_results
                    df = df.sort_values('_combined_score', ascending=False).head(n_results)

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


