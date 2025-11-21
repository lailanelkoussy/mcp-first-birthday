import logging
import uuid
import numpy as np
from tqdm import tqdm
from typing import Literal
import lancedb

from .utils.logger_utils import setup_logger


class CodeIndex:
    def __init__(
        self,
        nodes: list,
        model_service,
        index_type: Literal['embedding-only', 'keyword-only', 'hybrid'] = 'hybrid',
        embedding_batch_size: int = 20,
        use_embed: bool = True,
        db_path: str = "./local_code_index_db"
    ):
        setup_logger("CODE_INDEX_LOGGER")
        self.logger = logging.getLogger("CODE_INDEX_LOGGER")

        self.model_service = model_service
        self.index_type = index_type
        self.embedding_batch_size = embedding_batch_size
        self.use_embed = use_embed

        # Embedded DB
        self.db = lancedb.connect(db_path)
        self.table_name = f"code_chunks_{uuid.uuid4().hex}"
        self.table = None

        chunk_nodes = [node for node in nodes if node.node_type == "chunk"]

        # -----------------------------------------------------------
        # Create embeddings IF using vector search
        # -----------------------------------------------------------
        if index_type != "keyword-only":
            nodes_needing = [
                node for node in chunk_nodes
                if node.embedding is None or not use_embed
            ]

            if nodes_needing:
                self.logger.info(f"Embedding {len(nodes_needing)} chunks...")
                for i in tqdm(range(0, len(nodes_needing), embedding_batch_size)):
                    batch = nodes_needing[i:i + embedding_batch_size]
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
            if index_type != "keyword-only":
                row["vector"] = node.embedding

            rows.append(row)

        # Create table
        self.table = self.db.create_table(self.table_name, data=rows)
        self.logger.info(f"Created table: {self.table_name}")

    # ----------------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------------
    def query(self, query: str, n_results: int) -> dict:
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
                    vector_column="vector"
                )

            # ---------------------- HYBRID ----------------------------
            else:
                emb = np.array(self.model_service.embed_query(query), dtype=np.float32)
                q = (
                    self.table.search(emb, vector_column="vector")
                    .where(
                        f"content LIKE '%{query}%' OR name LIKE '%{query}%' OR description LIKE '%{query}%'",
                        prefilter=False
                    )
                )

            df = q.limit(n_results).to_pandas()

            # Build result format
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
        pass
