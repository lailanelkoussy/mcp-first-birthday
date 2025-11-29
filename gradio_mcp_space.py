"""
Simplified Gradio MCP Server for Knowledge Graphs loaded from HuggingFace datasets.
"""
import os
import sys
import argparse
import difflib
import fnmatch
import re
from typing import Optional, List
import gradio as gr

# Optional Langfuse integration
try:
    from langfuse import get_client, observe
    langfuse = get_client()
    LANGFUSE_ENABLED = langfuse.auth_check()
    if LANGFUSE_ENABLED:
        print("✓ Langfuse client is authenticated and ready!")
    else:
        print("⚠️ Langfuse authentication failed. Tracing disabled.")
except Exception as e:
    print(f"⚠️ Langfuse not available: {e}. Tracing disabled.")
    LANGFUSE_ENABLED = False
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pedagogia_graph_code_repo'))

from pedagogia_graph_code_repo.RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph

# Global knowledge graph instance
knowledge_graph = None


def initialize_knowledge_graph(
    hf_dataset: str,
    hf_token: Optional[str] = None,
    index_nodes: bool = True,
    code_index_kwargs: Optional[dict] = None
):
    """Initialize the knowledge graph from a HuggingFace dataset."""
    global knowledge_graph

    model_service_kwargs = {
        "embedder_type": "sentence-transformers",
        "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
    }

    print(f"Loading knowledge graph from HuggingFace dataset: {hf_dataset}")
    knowledge_graph = RepoKnowledgeGraph.from_hf_dataset(
        repo_id=hf_dataset,
        index_nodes=index_nodes,
        model_service_kwargs=model_service_kwargs,
        code_index_kwargs=code_index_kwargs,
        token=hf_token
    )


# ==================== Tool Functions ====================
@observe(as_type="tool")
def get_node_info(node_id: str) -> str:
    """
    Get detailed information about a node in the knowledge graph.

    Returns information including the node's type, name, description,
    declared/called entities, and type-specific details.

    Args:
        node_id: The ID of the node to retrieve information for

    Returns:
        str: A formatted string with node information
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        node = knowledge_graph.graph.nodes[node_id]['data']
        node_type = getattr(node, 'node_type', 'Unknown')
        node_class = node.__class__.__name__
        node_name = getattr(node, 'name', 'Unknown')
        description = getattr(node, 'description', None)

        result = f"Node Information:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result += f"Node ID: {node_id}\nClass: {node_class}\nName: {node_name}\nType: {node_type}\n"
        result += f"Description: {description or 'N/A'}\n"

        if node_class == 'EntityNode' or node_type == 'entity':
            entity_type = getattr(node, 'entity_type', 'Unknown')
            declaring_chunk_ids = getattr(node, 'declaring_chunk_ids', [])
            calling_chunk_ids = getattr(node, 'calling_chunk_ids', [])
            aliases = getattr(node, 'aliases', [])

            result += f"\nEntity Type: {entity_type}\n"
            result += f"Aliases: {', '.join(aliases) if aliases else 'None'}\n"
            result += f"Declared in {len(declaring_chunk_ids)} chunk(s):\n"
            for cid in declaring_chunk_ids[:5]:
                result += f"  - {cid}\n"
            if len(declaring_chunk_ids) > 5:
                result += f"  ... and {len(declaring_chunk_ids) - 5} more\n"
            result += f"Called in {len(calling_chunk_ids)} chunk(s):\n"
            for cid in calling_chunk_ids[:5]:
                result += f"  - {cid}\n"
            if len(calling_chunk_ids) > 5:
                result += f"  ... and {len(calling_chunk_ids) - 5} more\n"
            result += f"\nSummary: Entity {node_id} ({node_name}) — {entity_type} declared in {len(declaring_chunk_ids)} chunk(s) and called in {len(calling_chunk_ids)} chunk(s).\n"
        else:
            declared_entities = getattr(node, 'declared_entities', [])
            called_entities = getattr(node, 'called_entities', [])

            result += f"\nDeclared Entities ({len(declared_entities)}):\n"
            for entity in declared_entities[:10]:
                result += f"  - {entity}\n"
            if len(declared_entities) > 10:
                result += f"  ... and {len(declared_entities) - 10} more\n"

            result += f"\nCalled Entities ({len(called_entities)}):\n"
            for entity in called_entities[:10]:
                result += f"  - {entity}\n"
            if len(called_entities) > 10:
                result += f"  ... and {len(called_entities) - 10} more\n"

            # Add content preview for file/chunk nodes
            if node_type in ['file', 'chunk']:
                content = getattr(node, 'content', None)
                result += f"\nContent:\n{content or 'N/A'}\n"
                if hasattr(node, 'path'):
                    result += f"Path: {node.path}\n"
                if hasattr(node, 'language'):
                    result += f"Language: {node.language}\n"
                if node_type == 'chunk' and hasattr(node, 'order_in_file'):
                    result += f"Order in File: {node.order_in_file}\n"
            elif node_type == 'directory':
                if hasattr(node, 'path'):
                    result += f"Path: {node.path}\n"

            result += f"\nSummary: Node {node_id} ({node_name}) — {node_type} with {len(declared_entities)} declared and {len(called_entities)} called entities.\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_node_edges(node_id: str) -> str:
    """
    List all incoming and outgoing edges for a node.

    Shows relationships to other nodes in the knowledge graph.

    Args:
        node_id: The ID of the node whose edges to list

    Returns:
        str: A formatted string showing all edges
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        g = knowledge_graph.graph

        incoming = [
            {"source": src, "target": tgt, "relation": data.get("relation", "?")}
            for src, tgt, data in g.in_edges(node_id, data=True)
        ]
        outgoing = [
            {"source": src, "target": tgt, "relation": data.get("relation", "?")}
            for src, tgt, data in g.out_edges(node_id, data=True)
        ]

        result = f"""Node Edges for '{node_id}':
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Incoming Edges ({len(incoming)}):
"""
        for edge in incoming[:20]:
            result += f"  ← {edge['source']} [{edge['relation']}]\n"
        if len(incoming) > 20:
            result += f"  ... and {len(incoming) - 20} more\n"

        result += f"\nOutgoing Edges ({len(outgoing)}):\n"
        for edge in outgoing[:20]:
            result += f"  → {edge['target']} [{edge['relation']}]\n"
        if len(outgoing) > 20:
            result += f"  ... and {len(outgoing) - 20} more\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def search_nodes(query: str, limit: int = 10) -> str:
    """
    Search for nodes in the knowledge graph by query string.

    Uses semantic and keyword search via the code index.

    Args:
        query: The search string to match against code index
        limit: Maximum number of results to return (default: 10)

    Returns:
        str: A formatted string with search results
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string (MCP may pass strings)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        if limit <= 0:
            return "Error: limit must be a positive integer"

        results = knowledge_graph.code_index.query(query, n_results=limit)
        metadatas = results.get("metadatas", [[]])[0]

        if not metadatas:
            return f"No results found for '{query}'."

        result = f"Search Results for '{query}' ({len(metadatas)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, res in enumerate(metadatas, 1):
            result += f"{i}. ID: {res.get('id', 'N/A')}\n"
            content = res.get('content', '')
            if content:
                result += f"   Content: {content}\n"

            # Handle declared entities - parse JSON if it's a string
            declared = res.get('declared_entities', '')
            if declared and declared != '[]':
                try:
                    # Try to parse as JSON if it's a string
                    import json
                    if isinstance(declared, str):
                        declared = json.loads(declared)
                    # Extract entity names from the list of dicts
                    if isinstance(declared, list) and declared:
                        entity_names = [e.get('name', str(e)) if isinstance(e, dict) else str(e) for e in declared[:10]]
                        result += f"   Declared: {', '.join(entity_names)}\n"
                        if len(declared) > 10:
                            result += f"             ... and {len(declared) - 10} more\n"
                except (json.JSONDecodeError, AttributeError):
                    result += f"   Declared: {declared}\n"

            # Handle called entities - parse JSON if it's a string
            called = res.get('called_entities', '')
            if called and called != '[]':
                try:
                    # Try to parse as JSON if it's a string
                    import json
                    if isinstance(called, str):
                        called = json.loads(called)
                    # Extract entity names from the list of dicts
                    if isinstance(called, list) and called:
                        entity_names = [e.get('name', str(e)) if isinstance(e, dict) else str(e) for e in called[:10]]
                        result += f"   Called: {', '.join(entity_names)}\n"
                        if len(called) > 10:
                            result += f"             ... and {len(called) - 10} more\n"
                except (json.JSONDecodeError, AttributeError):
                    result += f"   Called: {called}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_graph_stats() -> str:
    """
    Get overall statistics about the knowledge graph.

    Includes node and edge counts, types, and relations.

    Returns:
        str: A formatted string with graph statistics
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        g = knowledge_graph.graph
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()

        node_types = {}
        for _, node_attrs in g.nodes(data=True):
            node_type = getattr(node_attrs['data'], 'node_type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        edge_relations = {}
        for _, _, attrs in g.edges(data=True):
            relation = attrs.get('relation', 'Unknown')
            edge_relations[relation] = edge_relations.get(relation, 0) + 1

        result = f"""Knowledge Graph Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Nodes: {num_nodes}
Total Edges: {num_edges}

Node Types:
"""
        for ntype, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            result += f"  - {ntype}: {count}\n"

        result += "\nEdge Relations:\n"
        for relation, count in sorted(edge_relations.items(), key=lambda x: x[1], reverse=True):
            result += f"  - {relation}: {count}\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def list_nodes_by_type(node_type: str, limit: int = 20) -> str:
    """
    List nodes of a specific type in the knowledge graph.

    Args:
        node_type: The type of nodes to list (e.g., 'function', 'class', 'file')
        limit: Maximum number of nodes to return (default: 20)

    Returns:
        str: A formatted string with matching nodes
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string (MCP may pass strings)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        g = knowledge_graph.graph
        matching_nodes = [
            {
                "id": node_id,
                "name": getattr(data['data'], 'name', 'Unknown')
            }
            for node_id, data in g.nodes(data=True)
            if getattr(data['data'], 'node_type', None) == node_type
        ][:limit]

        if not matching_nodes:
            return f"No nodes found of type '{node_type}'."

        result = f"Nodes of type '{node_type}' ({len(matching_nodes)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, node in enumerate(matching_nodes, 1):
            result += f"{i}. {node['name']}\n"
            result += f"   ID: {node['id']}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_neighbors(node_id: str) -> str:
    """
    Get all nodes directly connected to a given node.

    Shows neighboring nodes with their relationship types.

    Args:
        node_id: The ID of the node whose neighbors to retrieve

    Returns:
        str: A formatted string showing all neighbors
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        neighbors = knowledge_graph.get_neighbors(node_id)
        if not neighbors:
            return f"No neighbors found for node '{node_id}'"

        result = f"Neighbors of '{node_id}' ({len(neighbors)} total):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, neighbor in enumerate(neighbors[:20], 1):
            result += f"{i}. {neighbor.id}\n"
            result += f"   Name: {getattr(neighbor, 'name', 'Unknown')}\n"
            result += f"   Type: {neighbor.node_type}\n"

            if knowledge_graph.graph.has_edge(node_id, neighbor.id):
                edge_data = knowledge_graph.graph.get_edge_data(node_id, neighbor.id)
                result += f"   → Relation: {edge_data.get('relation', 'Unknown')}\n"
            elif knowledge_graph.graph.has_edge(neighbor.id, node_id):
                edge_data = knowledge_graph.graph.get_edge_data(neighbor.id, node_id)
                result += f"   ← Relation: {edge_data.get('relation', 'Unknown')}\n"
            result += "\n"

        if len(neighbors) > 20:
            result += f"... and {len(neighbors) - 20} more neighbors\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def go_to_definition(entity_name: str) -> str:
    """
    Find where an entity is declared or defined in the codebase.

    Locates the declaration point for functions, classes, variables, etc.

    Args:
        entity_name: The name of the entity to find the definition for

    Returns:
        str: A formatted string with definition locations
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if entity_name not in knowledge_graph.entities:
            return f"Error: Entity '{entity_name}' not found in knowledge graph"

        entity_info = knowledge_graph.entities[entity_name]
        declaring_chunks = entity_info.get('declaring_chunk_ids', [])

        if not declaring_chunks:
            return f"Entity '{entity_name}' found but no declarations identified."

        result = f"Definition(s) for '{entity_name}':\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        result += f"Type: {', '.join(entity_info.get('type', ['Unknown']))}\n"
        if entity_info.get('dtype'):
            result += f"Data Type: {entity_info['dtype']}\n"
        result += f"\nDeclared in {len(declaring_chunks)} location(s):\n\n"

        for i, chunk_id in enumerate(declaring_chunks[:5], 1):
            if chunk_id in knowledge_graph.graph:
                chunk = knowledge_graph.graph.nodes[chunk_id]['data']
                result += f"{i}. Chunk: {chunk_id}\n"
                result += f"   File: {chunk.path}\n"
                result += f"   Order: {chunk.order_in_file}\n"
                result += f"   Content:\n{chunk.content}\n\n"

        if len(declaring_chunks) > 5:
            result += f"... and {len(declaring_chunks) - 5} more locations\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def find_usages(entity_name: str, limit: int = 20) -> str:
    """
    Find all usages or calls of an entity in the codebase.

    Shows where functions, classes, variables, etc. are used.

    Args:
        entity_name: The name of the entity to find usages for
        limit: Maximum number of usages to return (default: 20)

    Returns:
        str: A formatted string with usage locations
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string (MCP may pass strings)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        if entity_name not in knowledge_graph.entities:
            return f"Error: Entity '{entity_name}' not found in knowledge graph"

        if limit <= 0:
            return "Error: limit must be a positive integer"

        entity_info = knowledge_graph.entities[entity_name]
        calling_chunks = entity_info.get('calling_chunk_ids', [])

        if not calling_chunks:
            return f"Entity '{entity_name}' found but no usages identified."

        result = f"Usages of '{entity_name}' ({len(calling_chunks)} total):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, chunk_id in enumerate(calling_chunks[:limit], 1):
            if chunk_id in knowledge_graph.graph:
                chunk = knowledge_graph.graph.nodes[chunk_id]['data']
                result += f"{i}. {chunk.path} (chunk {chunk.order_in_file})\n"
                result += f"   Content:\n{chunk.content}\n\n"

        if len(calling_chunks) > limit:
            result += f"... and {len(calling_chunks) - limit} more usages\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_file_structure(file_path: str) -> str:
    """
    Get an overview of the structure of a file.

    Shows chunks and declared entities within a specific file.

    Args:
        file_path: The path of the file to get the structure for

    Returns:
        str: A formatted string with file structure
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if file_path not in knowledge_graph.graph:
            return f"Error: File '{file_path}' not found in knowledge graph"

        file_node = knowledge_graph.graph.nodes[file_path]['data']
        chunks = knowledge_graph.get_chunks_of_file(file_path)

        result = f"File Structure: {file_node.name}\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        result += f"Path: {file_path}\n"
        result += f"Language: {getattr(file_node, 'language', 'Unknown')}\n"
        result += f"Total Chunks: {len(chunks)}\n\n"

        if hasattr(file_node, 'declared_entities') and file_node.declared_entities:
            result += f"Declared Entities ({len(file_node.declared_entities)}):\n"
            for entity in file_node.declared_entities[:15]:
                if isinstance(entity, dict):
                    result += f"  - {entity.get('name', '?')} ({entity.get('type', '?')})\n"
                else:
                    result += f"  - {entity}\n"
            if len(file_node.declared_entities) > 15:
                result += f"  ... and {len(file_node.declared_entities) - 15} more\n"

        result += f"\nChunks:\n"
        for chunk in chunks[:10]:
            result += f"  [{chunk.order_in_file}] {chunk.id}\n"
            if chunk.description:
                desc = chunk.description[:80] + "..." if len(chunk.description) > 80 else chunk.description
                result += f"      {desc}\n"

        if len(chunks) > 10:
            result += f"  ... and {len(chunks) - 10} more chunks\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_related_chunks(chunk_id: str, relation_type: str = "calls") -> str:
    """
    Get chunks related to a given chunk by a specific relationship.

    Find chunks connected via relationships like 'calls', 'contains', etc.

    Args:
        chunk_id: The ID of the chunk to find related chunks for
        relation_type: The type of relationship to filter by (default: 'calls')

    Returns:
        str: A formatted string with related chunks
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if chunk_id not in knowledge_graph.graph:
            return f"Error: Chunk '{chunk_id}' not found in knowledge graph"

        related = []
        for _, target, attrs in knowledge_graph.graph.out_edges(chunk_id, data=True):
            if attrs.get('relation') == relation_type:
                target_node = knowledge_graph.graph.nodes[target]['data']
                related.append({
                    "id": target,
                    "file_path": getattr(target_node, 'path', 'Unknown'),
                    "entity_name": attrs.get('entity_name')
                })

        if not related:
            return f"No chunks found with '{relation_type}' relationship from '{chunk_id}'"

        result = f"Chunks related to '{chunk_id}' via '{relation_type}' ({len(related)} total):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, chunk in enumerate(related[:15], 1):
            result += f"{i}. {chunk['id']}\n"
            result += f"   File: {chunk['file_path']}\n"
            if chunk['entity_name']:
                result += f"   Entity: {chunk['entity_name']}\n"
            result += "\n"

        if len(related) > 15:
            result += f"... and {len(related) - 15} more\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def list_all_entities(
    limit: int = 50,
    page: int = 1,
    entity_type: Optional[str] = None,
    declared_in_repo: Optional[bool] = None
) -> str:
    """
    List all entities tracked in the knowledge graph with filtering and pagination options.

    Shows entity types, declaration counts, and usage counts.

    Args:
        limit: Maximum number of entities to return per page (default: 50)
        page: Page number for pagination, 1-indexed (default: 1)
        entity_type: Filter by entity type ('class', 'function', 'method', 'variable', 'parameter', 'function_call', 'method_call')
        declared_in_repo: If True, only return entities with declarations. If False, only entities without declarations. If None, return all.

    Returns:
        str: A formatted string with all entities for the requested page
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string (MCP may pass strings)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        # Convert page to int if it's a string (MCP may pass strings)
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                return f"Error: 'page' must be an integer, got '{page}'"
        
        if page < 1:
            return "Error: 'page' must be a positive integer (1 or greater)"
        
        # Handle entity_type - empty string should be treated as None
        if entity_type == "" or entity_type == "null":
            entity_type = None
            
        # Handle declared_in_repo - convert string to bool if needed
        if isinstance(declared_in_repo, str):
            if declared_in_repo.lower() in ("true", "1", "yes"):
                declared_in_repo = True
            elif declared_in_repo.lower() in ("false", "0", "no"):
                declared_in_repo = False
            elif declared_in_repo.lower() in ("none", "null", "all", ""):
                declared_in_repo = None
        
        if not knowledge_graph.entities:
            return "No entities found in the knowledge graph."

        # Filter entities based on criteria
        filtered_entities = {}
        for entity_name, info in knowledge_graph.entities.items():
            # Filter by entity type if specified
            if entity_type is not None:
                entity_types = [t.lower() if t else '' for t in info.get('type', [])]
                if entity_type.lower() not in entity_types:
                    continue

            # Filter by declared_in_repo if specified
            if declared_in_repo is not None:
                has_declaration = len(info.get('declaring_chunk_ids', [])) > 0
                if declared_in_repo and not has_declaration:
                    continue
                if not declared_in_repo and has_declaration:
                    continue

            filtered_entities[entity_name] = info

        # Build the response with filtered entities
        if not filtered_entities:
            filter_desc = []
            if entity_type:
                filter_desc.append(f"type={entity_type}")
            if declared_in_repo is not None:
                filter_desc.append(f"declared_in_repo={declared_in_repo}")
            filter_text = f" (filtered by {', '.join(filter_desc)})" if filter_desc else ""
            return f"No entities found{filter_text}."

        # Calculate pagination
        total_entities = len(filtered_entities)
        total_pages = (total_entities + limit - 1) // limit  # Ceiling division
        
        if page > total_pages:
            return f"Error: Page {page} does not exist. Total pages: {total_pages} (with {total_entities} entities at {limit} per page)"
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Get the paginated slice of entities
        entity_items = list(filtered_entities.items())
        paginated_items = entity_items[start_idx:end_idx]

        result = f"All Entities (Page {page}/{total_pages}, {total_entities} total):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, (entity_name, info) in enumerate(paginated_items, start=start_idx + 1):
            result += f"{i}. {entity_name}\n"
            result += f"   Types: {', '.join(info.get('type', ['Unknown']))}\n"
            result += f"   Declarations: {len(info.get('declaring_chunk_ids', []))}\n"
            result += f"   Usages: {len(info.get('calling_chunk_ids', []))}\n\n"

        # Add pagination info
        result += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result += f"Showing {start_idx + 1}-{min(end_idx, total_entities)} of {total_entities} entities\n"
        result += f"Page {page} of {total_pages}\n"
        
        if page < total_pages:
            result += f"Use page={page + 1} to see the next page\n"

        # Add filter information
        if entity_type:
            result += f"\n(Filtered by type={entity_type})\n"
        if declared_in_repo is not None:
            result += f"(Filtered by declared_in_repo={declared_in_repo})\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def diff_chunks(node_id_1: str, node_id_2: str) -> str:
    """
    Show the diff between two code chunks or nodes.

    Compares the content of two nodes and shows differences.

    Args:
        node_id_1: The ID of the first node/chunk
        node_id_2: The ID of the second node/chunk

    Returns:
        str: A formatted string with the diff
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id_1 not in knowledge_graph.graph:
            return f"Error: Node '{node_id_1}' not found in knowledge graph"
        if node_id_2 not in knowledge_graph.graph:
            return f"Error: Node '{node_id_2}' not found in knowledge graph"

        g = knowledge_graph.graph
        content1 = getattr(g.nodes[node_id_1]['data'], 'content', None)
        content2 = getattr(g.nodes[node_id_2]['data'], 'content', None)

        if not content1 or not content2:
            return "Error: One or both nodes have no content."

        diff = list(difflib.unified_diff(
            content1.splitlines(), content2.splitlines(),
            fromfile=node_id_1, tofile=node_id_2, lineterm=""
        ))

        if not diff:
            return "No differences found between the two chunks."

        return "\n".join(diff)
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def print_tree(root_id: str = "root", max_depth: int = 3) -> str:
    """
    Show a tree view of the repository structure.

    Displays a hierarchical tree starting from a given node.

    Args:
        root_id: The node ID to start the tree from (default: 'root')
        max_depth: Maximum depth to show (default: 3)

    Returns:
        str: A formatted string with the tree structure
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert max_depth to int if it's a string (MCP may pass strings)
        if isinstance(max_depth, str):
            try:
                max_depth = int(max_depth)
            except ValueError:
                return f"Error: 'max_depth' must be an integer, got '{max_depth}'"
        
        g = knowledge_graph.graph

        if root_id not in g:
            # Try to find a suitable root
            roots = [n for n, d in g.nodes(data=True)
                    if getattr(d['data'], 'node_type', None) in ('repo', 'directory', 'file')]
            if roots:
                root_id = roots[0]
            else:
                return f"Error: Node '{root_id}' not found and no suitable root found"

        result = f"Tree View (starting from '{root_id}', max depth: {max_depth}):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        def format_node(node_id, depth):
            if depth > max_depth:
                return ""

            node = g.nodes[node_id]['data']
            name = getattr(node, 'name', node_id)
            node_type = getattr(node, 'node_type', '?')

            line = "  " * depth + f"- {name} ({node_type})\n"

            children = [t for s, t in g.out_edges(node_id)]
            for child in children[:20]:  # Limit children to prevent huge output
                line += format_node(child, depth + 1)

            if len(children) > 20:
                line += "  " * (depth + 1) + f"... and {len(children) - 20} more\n"

            return line

        result += format_node(root_id, 0)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def entity_relationships(node_id: str) -> str:
    """
    Show all relationships for a given entity or node.

    Displays incoming and outgoing relationships with their types.

    Args:
        node_id: The node/entity ID to explore relationships for

    Returns:
        str: A formatted string with all relationships
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        g = knowledge_graph.graph

        result = f"Relationships for '{node_id}':\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        incoming = list(g.in_edges(node_id, data=True))
        outgoing = list(g.out_edges(node_id, data=True))

        if incoming:
            result += f"Incoming Relationships ({len(incoming)}):\n"
            for source, target, data in incoming[:20]:
                result += f"  ← {source} [{data.get('relation', '?')}]\n"
            if len(incoming) > 20:
                result += f"  ... and {len(incoming) - 20} more\n"
            result += "\n"

        if outgoing:
            result += f"Outgoing Relationships ({len(outgoing)}):\n"
            for source, target, data in outgoing[:20]:
                result += f"  → {target} [{data.get('relation', '?')}]\n"
            if len(outgoing) > 20:
                result += f"  ... and {len(outgoing) - 20} more\n"

        if not incoming and not outgoing:
            result += "No relationships found.\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def search_by_type_and_name(node_type: str, name_query: str, limit: int = 10, fuzzy: bool = True) -> str:
    """
    Search for nodes/entities by type and name substring with fuzzy matching support.

    Filters nodes by type and searches for matching names. Supports partial/fuzzy matching
    so searching for 'Embedding' will find 'BertEmbeddings', 'LlamaRotaryEmbedding', etc.
    
    For entities, searches by entity_type (e.g., 'class', 'function', 'method').
    For other nodes, searches by node_type (e.g., 'file', 'chunk', 'directory').

    Args:
        node_type: Type of node/entity (e.g., 'function', 'class', 'file', 'chunk', 'directory')
        name_query: Substring to match in the name (case-insensitive, supports partial matches)
        limit: Maximum results to return (default: 10)
        fuzzy: Enable fuzzy/partial matching (default: True). If False, requires exact substring match.

    Returns:
        str: A formatted string with matching nodes
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string (MCP may pass strings)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        # Convert fuzzy to bool if it's a string
        if isinstance(fuzzy, str):
            fuzzy = fuzzy.lower() in ('true', '1', 'yes')
        
        if limit <= 0:
            return "Error: limit must be a positive integer"

        g = knowledge_graph.graph
        matches = []
        query_lower = name_query.lower()
        
        # Build regex pattern for fuzzy matching
        # This will match names containing all characters of the query in order
        if fuzzy:
            # Create pattern that matches query as substring or with characters spread out
            # e.g., "Embed" matches "Embedding", "BertEmbeddings", "EmbedLayer"
            fuzzy_pattern = '.*'.join(re.escape(c) for c in query_lower)
            fuzzy_regex = re.compile(fuzzy_pattern, re.IGNORECASE)
        
        for nid, n in g.nodes(data=True):
            node = n['data']
            node_name = getattr(node, 'name', '')
            
            if not node_name:
                continue
            
            # Check if name matches the query
            name_matches = False
            if fuzzy:
                # Fuzzy match: substring match OR regex pattern match
                if query_lower in node_name.lower() or fuzzy_regex.search(node_name):
                    name_matches = True
            else:
                # Exact substring match
                if query_lower in node_name.lower():
                    name_matches = True
            
            if not name_matches:
                continue
            
            # Check type based on node_type
            current_node_type = getattr(node, 'node_type', None)
            
            # For entity nodes, check entity_type instead of node_type
            if current_node_type == 'entity':
                entity_type = getattr(node, 'entity_type', '')
                
                # Fallback: if entity_type is empty, check the entities dictionary
                # This handles cases where EntityNode was created before the fix
                if not entity_type and nid in knowledge_graph.entities:
                    entity_types = knowledge_graph.entities[nid].get('type', [])
                    entity_type = entity_types[0] if entity_types else ''
                
                if entity_type and entity_type.lower() == node_type.lower():
                    # Calculate match score for sorting (exact matches first)
                    score = 0 if query_lower == node_name.lower() else (1 if query_lower in node_name.lower() else 2)
                    matches.append({
                        "id": nid,
                        "name": node_name,
                        "type": f"entity ({entity_type})",
                        "score": score
                    })
            # For other nodes, check node_type directly
            elif current_node_type == node_type:
                score = 0 if query_lower == node_name.lower() else (1 if query_lower in node_name.lower() else 2)
                matches.append({
                    "id": nid,
                    "name": node_name,
                    "type": current_node_type,
                    "score": score
                })
        
        # Sort by match score (best matches first) and limit results
        matches.sort(key=lambda x: (x['score'], x['name'].lower()))
        matches = matches[:limit]

        if not matches:
            return f"No matches for type '{node_type}' and name containing '{name_query}'."

        result = f"Matches for type '{node_type}' and name '{name_query}' ({len(matches)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, match in enumerate(matches, 1):
            result += f"{i}. {match['name']}\n"
            result += f"   ID: {match['id']}\n"
            result += f"   Type: {match['type']}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_chunk_context(node_id: str) -> str:
    """
    Get the full content of a code chunk along with its surrounding chunks.

    Returns the full content of the previous, current, and next chunks,
    organized by file and joined together.

    Args:
        node_id: The node/chunk ID to get context for

    Returns:
        str: The full content of surrounding code chunks
    """
    from pedagogia_graph_code_repo.RepoKnowledgeGraphLib.utils.chunk_utils import (
        organize_chunks_by_file_name, join_organized_chunks
    )
    
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        g = knowledge_graph.graph
        current_chunk = g.nodes[node_id]['data']
        previous_chunk = knowledge_graph.get_previous_chunk(node_id)
        next_chunk = knowledge_graph.get_next_chunk(node_id)

        # Collect all chunks (previous, current, next)
        chunks = []
        if previous_chunk:
            chunks.append(previous_chunk)
        chunks.append(current_chunk)
        if next_chunk:
            chunks.append(next_chunk)

        # Organize and join chunks
        organized = organize_chunks_by_file_name(chunks)
        full_content = join_organized_chunks(organized)

        return full_content
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_file_stats(path: str) -> str:
    """
    Get statistics for a file or directory.

    Shows number of entities, lines, chunks, etc.

    Args:
        path: The file or directory path to get statistics for

    Returns:
        str: A formatted string with file statistics
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        g = knowledge_graph.graph
        nodes = [n for n, d in g.nodes(data=True) if getattr(d['data'], 'path', None) == path]

        if not nodes:
            return f"No nodes found for path '{path}'."

        result = f"Statistics for '{path}':\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for node_id in nodes:
            node = g.nodes[node_id]['data']
            content = getattr(node, 'content', '')
            declared = getattr(node, 'declared_entities', [])
            called = getattr(node, 'called_entities', [])
            chunks = [t for s, t in g.out_edges(node_id)
                     if getattr(g.nodes[t]['data'], 'node_type', None) == 'chunk']

            result += f"Node: {node_id} ({getattr(node, 'node_type', '?')})\n"
            result += f"  Lines: {len(content.splitlines()) if content else 0}\n"
            result += f"  Declared entities: {len(declared)}\n"

            if declared:
                for entity in declared[:10]:
                    if isinstance(entity, dict):
                        result += f"    - {entity.get('name', '?')} ({entity.get('type', '?')})\n"
                    else:
                        result += f"    - {entity}\n"
                if len(declared) > 10:
                    result += f"    ... and {len(declared) - 10} more\n"

            result += f"  Called entities: {len(called)}\n"
            if called:
                for entity in called[:10]:
                    result += f"    - {entity}\n"
                if len(called) > 10:
                    result += f"    ... and {len(called) - 10} more\n"

            result += f"  Chunks: {len(chunks)}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def find_path(source_id: str, target_id: str, max_depth: int = 5) -> str:
    """
    Find the shortest path between two nodes in the knowledge graph.

    Uses graph traversal to find connections between nodes.

    Args:
        source_id: The ID of the source node
        target_id: The ID of the target node
        max_depth: Maximum depth to search for a path (default: 5)

    Returns:
        str: A formatted string showing the path
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert max_depth to int if it's a string (MCP may pass strings)
        if isinstance(max_depth, str):
            try:
                max_depth = int(max_depth)
            except ValueError:
                return f"Error: 'max_depth' must be an integer, got '{max_depth}'"
        
        path_result = knowledge_graph.find_path(source_id, target_id, max_depth)

        if "error" in path_result:
            return f"Error: {path_result['error']}"

        if not path_result.get("path"):
            return f"No path found from '{source_id}' to '{target_id}' within depth {max_depth}"

        result = f"Path from '{source_id}' to '{target_id}':\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        result += f"Length: {path_result['length']}\n\n"

        path = path_result['path']
        for i, node_id in enumerate(path):
            result += f"{i}. {node_id}\n"
            if i < len(path) - 1:
                result += "   ↓\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_subgraph(node_id: str, depth: int = 2, edge_types: Optional[str] = None) -> str:
    """
    Extract a subgraph around a node up to a specified depth.

    Optionally filters by edge types (comma-separated).

    Args:
        node_id: The ID of the central node
        depth: The depth/radius of the subgraph to extract (default: 2)
        edge_types: Optional comma-separated list of edge types (e.g., 'calls,contains')

    Returns:
        str: A formatted string describing the subgraph
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert depth to int if it's a string (MCP may pass strings)
        if isinstance(depth, str):
            try:
                depth = int(depth)
            except ValueError:
                return f"Error: 'depth' must be an integer, got '{depth}'"
        
        edge_types_list = edge_types.split(",") if edge_types else None
        subgraph_result = knowledge_graph.get_subgraph(node_id, depth, edge_types_list)

        if "error" in subgraph_result:
            return f"Error: {subgraph_result['error']}"

        result = f"Subgraph around '{node_id}' (depth: {depth}):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        result += f"Nodes: {subgraph_result['node_count']}\n"
        result += f"Edges: {subgraph_result['edge_count']}\n"

        if edge_types_list:
            result += f"Filtered by edge types: {', '.join(edge_types_list)}\n"

        result += "\nNodes in subgraph:\n"
        for node in subgraph_result['nodes'][:30]:
            result += f"  - {node}\n"

        if len(subgraph_result['nodes']) > 30:
            result += f"  ... and {len(subgraph_result['nodes']) - 30} more\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def list_files_in_directory(directory_path: str = "", pattern: str = "*", recursive: bool = True, limit: int = 50) -> str:
    """
    List files in a directory with optional glob pattern matching.

    This provides hierarchical file listing, showing files within directories
    rather than just top-level files. Supports glob patterns for filtering.

    Args:
        directory_path: Path to the directory to list (empty string for root/all files)
        pattern: Glob pattern to filter files (e.g., '*.py', 'test_*.py', '**/*.js')
        recursive: Whether to search recursively in subdirectories (default: True)
        limit: Maximum number of files to return (default: 50)

    Returns:
        str: A formatted string with matching files
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"
        
        # Convert recursive to bool if it's a string
        if isinstance(recursive, str):
            recursive = recursive.lower() in ('true', '1', 'yes')

        g = knowledge_graph.graph
        matching_files = []
        
        for nid, n in g.nodes(data=True):
            node = n['data']
            node_type = getattr(node, 'node_type', None)
            
            # Only look at file nodes
            if node_type != 'file':
                continue
            
            file_path = getattr(node, 'path', nid)
            file_name = getattr(node, 'name', '')
            
            # Filter by directory path if specified
            if directory_path:
                if recursive:
                    # Check if file is under the directory
                    if not file_path.startswith(directory_path.rstrip('/') + '/') and file_path != directory_path:
                        continue
                else:
                    # Check if file is directly in the directory (not in subdirectories)
                    parent_dir = '/'.join(file_path.rsplit('/', 1)[:-1]) if '/' in file_path else ''
                    if parent_dir != directory_path.rstrip('/'):
                        continue
            
            # Apply glob pattern matching
            if pattern and pattern != '*':
                # Match against both full path and filename
                if not (fnmatch.fnmatch(file_path, pattern) or 
                        fnmatch.fnmatch(file_name, pattern) or
                        fnmatch.fnmatch(file_path, f'**/{pattern}')):
                    continue
            
            language = getattr(node, 'language', 'Unknown')
            declared_entities = getattr(node, 'declared_entities', [])
            
            matching_files.append({
                'path': file_path,
                'name': file_name,
                'language': language,
                'entity_count': len(declared_entities)
            })
            
            if len(matching_files) >= limit:
                break
        
        # Sort by path for consistent ordering
        matching_files.sort(key=lambda x: x['path'])

        if not matching_files:
            filter_desc = f" in '{directory_path}'" if directory_path else ""
            pattern_desc = f" matching '{pattern}'" if pattern and pattern != '*' else ""
            return f"No files found{filter_desc}{pattern_desc}."

        result = f"Files"
        if directory_path:
            result += f" in '{directory_path}'"
        if pattern and pattern != '*':
            result += f" matching '{pattern}'"
        result += f" ({len(matching_files)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, f in enumerate(matching_files, 1):
            result += f"{i}. {f['path']}\n"
            result += f"   Language: {f['language']}, Entities: {f['entity_count']}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def find_classes_inheriting_from(base_class_name: str, limit: int = 20) -> str:
    """
    Find all classes that inherit from a given base class.

    Searches the knowledge graph for class entities that have the specified
    base class in their inheritance chain.

    Args:
        base_class_name: The name of the base class to find subclasses of
        limit: Maximum number of results to return (default: 20)

    Returns:
        str: A formatted string with classes inheriting from the base class
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"

        g = knowledge_graph.graph
        inheriting_classes = []
        base_lower = base_class_name.lower()
        
        # First, find all class entities
        for nid, n in g.nodes(data=True):
            node = n['data']
            node_type = getattr(node, 'node_type', None)
            entity_type = getattr(node, 'entity_type', '')
            
            if node_type != 'entity' or entity_type.lower() != 'class':
                continue
            
            class_name = getattr(node, 'name', '')
            
            # Check if this class has relationships indicating inheritance
            # Look for 'inherits', 'extends', or similar relationships
            for _, target, edge_data in g.out_edges(nid, data=True):
                relation = edge_data.get('relation', '').lower()
                target_node = g.nodes[target]['data']
                target_name = getattr(target_node, 'name', '')
                
                if relation in ('inherits', 'extends', 'inherits_from', 'base_class'):
                    if target_name.lower() == base_lower or base_lower in target_name.lower():
                        declaring_chunks = getattr(node, 'declaring_chunk_ids', [])
                        inheriting_classes.append({
                            'name': class_name,
                            'id': nid,
                            'base': target_name,
                            'file': declaring_chunks[0] if declaring_chunks else 'Unknown'
                        })
                        break
            
            # Also check called_entities for base class references
            # (Sometimes inheritance is tracked via calls relationship)
            called = getattr(node, 'called_entities', [])
            if any(base_lower in str(c).lower() for c in called):
                # Check if it's likely an inheritance pattern
                declaring_chunks = getattr(node, 'declaring_chunk_ids', [])
                if declaring_chunks:
                    chunk_id = declaring_chunks[0]
                    if chunk_id in g:
                        chunk_node = g.nodes[chunk_id]['data']
                        content = getattr(chunk_node, 'content', '')
                        # Look for class definition with inheritance pattern
                        class_pattern = rf'class\s+{re.escape(class_name)}\s*\([^)]*{re.escape(base_class_name)}'
                        if re.search(class_pattern, content, re.IGNORECASE):
                            if not any(c['name'] == class_name for c in inheriting_classes):
                                inheriting_classes.append({
                                    'name': class_name,
                                    'id': nid,
                                    'base': base_class_name,
                                    'file': chunk_id
                                })
            
            if len(inheriting_classes) >= limit:
                break

        if not inheriting_classes:
            return f"No classes found inheriting from '{base_class_name}'.\n\nTip: Try searching for the base class name in code content using search_nodes."

        result = f"Classes inheriting from '{base_class_name}' ({len(inheriting_classes)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, cls in enumerate(inheriting_classes, 1):
            result += f"{i}. {cls['name']}\n"
            result += f"   ID: {cls['id']}\n"
            result += f"   Inherits from: {cls['base']}\n"
            result += f"   Defined in: {cls['file']}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def find_files_importing(module_or_entity: str, limit: int = 30) -> str:
    """
    Find all files that import a specific module or entity.

    Searches for import statements and usage patterns across the codebase.

    Args:
        module_or_entity: The name of the module or entity to find imports of
        limit: Maximum number of results to return (default: 30)

    Returns:
        str: A formatted string with files that import the specified module/entity
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"

        g = knowledge_graph.graph
        importing_files = []
        search_term = module_or_entity.lower()
        
        # Search through file nodes
        for nid, n in g.nodes(data=True):
            node = n['data']
            node_type = getattr(node, 'node_type', None)
            
            if node_type != 'file':
                continue
            
            file_path = getattr(node, 'path', nid)
            called_entities = getattr(node, 'called_entities', [])
            
            # Check if the module/entity is in called entities
            found_in_calls = False
            matched_entities = []
            for entity in called_entities:
                entity_str = str(entity).lower() if not isinstance(entity, dict) else entity.get('name', '').lower()
                if search_term in entity_str:
                    found_in_calls = True
                    matched_entities.append(entity_str)
            
            if found_in_calls:
                importing_files.append({
                    'path': file_path,
                    'name': getattr(node, 'name', ''),
                    'matched_entities': matched_entities[:5],
                    'match_type': 'called_entity'
                })
                continue
            
            # Also check chunk contents for import statements
            chunks = knowledge_graph.get_chunks_of_file(file_path) if hasattr(knowledge_graph, 'get_chunks_of_file') else []
            for chunk in chunks[:3]:  # Check first few chunks (usually where imports are)
                content = getattr(chunk, 'content', '')
                # Look for import patterns
                import_patterns = [
                    rf'import\s+.*{re.escape(module_or_entity)}',
                    rf'from\s+.*{re.escape(module_or_entity)}.*\s+import',
                    rf'require\s*\(\s*["\'].*{re.escape(module_or_entity)}',
                    rf'use\s+.*{re.escape(module_or_entity)}',
                ]
                for pattern in import_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        if not any(f['path'] == file_path for f in importing_files):
                            importing_files.append({
                                'path': file_path,
                                'name': getattr(node, 'name', ''),
                                'matched_entities': [],
                                'match_type': 'import_statement'
                            })
                        break
            
            if len(importing_files) >= limit:
                break
        
        # Sort by path
        importing_files.sort(key=lambda x: x['path'])

        if not importing_files:
            return f"No files found importing '{module_or_entity}'.\n\nTip: Try searching for the module name in code content using search_nodes."

        result = f"Files importing '{module_or_entity}' ({len(importing_files)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, f in enumerate(importing_files, 1):
            result += f"{i}. {f['path']}\n"
            result += f"   Match type: {f['match_type']}\n"
            if f['matched_entities']:
                result += f"   Matched: {', '.join(f['matched_entities'][:3])}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


@observe(as_type="tool")
def get_concept_overview(concept: str, limit: int = 15) -> str:
    """
    Get a high-level overview of a concept across the codebase.

    Combines multiple search strategies to provide a comprehensive view of how
    a concept (like 'embeddings', 'authentication', 'caching') is implemented.

    Args:
        concept: The concept to search for (e.g., 'embedding', 'authentication', 'cache')
        limit: Maximum number of results per category (default: 15)

    Returns:
        str: A formatted overview of the concept across the codebase
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        # Convert limit to int if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                return f"Error: 'limit' must be an integer, got '{limit}'"

        g = knowledge_graph.graph
        concept_lower = concept.lower()
        
        # Categories to collect
        related_classes = []
        related_functions = []
        related_files = []
        related_chunks = []
        
        # Search through all nodes
        for nid, n in g.nodes(data=True):
            node = n['data']
            node_type = getattr(node, 'node_type', None)
            node_name = getattr(node, 'name', '')
            
            # Check if concept appears in name
            name_match = concept_lower in node_name.lower()
            
            if node_type == 'entity':
                entity_type = getattr(node, 'entity_type', '')
                if name_match:
                    if entity_type.lower() == 'class' and len(related_classes) < limit:
                        declaring = getattr(node, 'declaring_chunk_ids', [])
                        related_classes.append({
                            'name': node_name,
                            'id': nid,
                            'file': declaring[0] if declaring else 'Unknown'
                        })
                    elif entity_type.lower() in ('function', 'method') and len(related_functions) < limit:
                        declaring = getattr(node, 'declaring_chunk_ids', [])
                        related_functions.append({
                            'name': node_name,
                            'id': nid,
                            'type': entity_type,
                            'file': declaring[0] if declaring else 'Unknown'
                        })
            
            elif node_type == 'file' and len(related_files) < limit:
                # Check if concept in filename or path
                file_path = getattr(node, 'path', '')
                if concept_lower in file_path.lower() or name_match:
                    declared = getattr(node, 'declared_entities', [])
                    related_files.append({
                        'path': file_path,
                        'name': node_name,
                        'entity_count': len(declared)
                    })
            
            elif node_type == 'chunk' and len(related_chunks) < limit // 2:
                # Check if concept in chunk content or description
                content = getattr(node, 'content', '')
                description = getattr(node, 'description', '')
                if concept_lower in content.lower() or concept_lower in (description or '').lower():
                    file_path = getattr(node, 'path', '')
                    related_chunks.append({
                        'id': nid,
                        'file': file_path,
                        'content': content
                    })

        # Build the overview
        result = f"Concept Overview: '{concept}'\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Summary
        total = len(related_classes) + len(related_functions) + len(related_files) + len(related_chunks)
        result += f"Found {total} related items across the codebase.\n\n"
        
        if related_classes:
            result += f"📦 Related Classes ({len(related_classes)}):\n"
            for cls in related_classes[:10]:
                result += f"  • {cls['name']}\n"
                result += f"    File: {cls['file']}\n"
            if len(related_classes) > 10:
                result += f"  ... and {len(related_classes) - 10} more\n"
            result += "\n"
        
        if related_functions:
            result += f"⚡ Related Functions/Methods ({len(related_functions)}):\n"
            for func in related_functions[:10]:
                result += f"  • {func['name']} ({func['type']})\n"
                result += f"    File: {func['file']}\n"
            if len(related_functions) > 10:
                result += f"  ... and {len(related_functions) - 10} more\n"
            result += "\n"
        
        if related_files:
            result += f"📄 Related Files ({len(related_files)}):\n"
            for f in related_files[:10]:
                result += f"  • {f['path']}\n"
                result += f"    Entities: {f['entity_count']}\n"
            if len(related_files) > 10:
                result += f"  ... and {len(related_files) - 10} more\n"
            result += "\n"
        
        if related_chunks:
            result += f"📝 Code Snippets ({len(related_chunks)}):\n"
            for chunk in related_chunks[:5]:
                result += f"  • {chunk['id']}\n"
                result += f"    Content:\n{chunk['content']}\n\n"
            if len(related_chunks) > 5:
                result += f"  ... and {len(related_chunks) - 5} more\n"
        
        if total == 0:
            result += "No direct matches found.\n\n"
            result += "Suggestions:\n"
            result += f"  • Try searching with: search_nodes('{concept}')\n"
            result += f"  • Try partial name: search_by_type_and_name('class', '{concept[:4]}')\n"
            result += f"  • Check entity list: list_all_entities(entity_type='class')\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== Gradio App ====================

def create_gradio_app():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Transformers Knowledge Graph Explorer — Knowledge Graph MCP Server", theme=gr.themes.Soft()) as demo:
        # Helper to render tool docstrings in the UI
        def _tool_doc_md(func):
            doc = (func.__doc__ or "No description available.").strip()
            # Render as a fenced code block for readability
            return f"**Description:**\n\n```\n{doc}\n```"

        gr.Markdown("""
        # 🔍 Transformers Knowledge Graph Explorer

        Explore and query the Hugging Face Transformers library codebase using a knowledge graph.
        """)

        with gr.Tab("📊 Graph Overview"):
            stats_btn = gr.Button("Get Graph Statistics", variant="primary")
            stats_output = gr.Textbox(label="Statistics", lines=20, max_lines=30)
            stats_btn.click(fn=get_graph_stats, outputs=stats_output)
            gr.Markdown(_tool_doc_md(get_graph_stats))

        with gr.Tab("🔎 Search"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter search query...")
                    search_limit = gr.Slider(1, 50, value=10, step=1, label="Max Results")
                    search_btn = gr.Button("Search", variant="primary")
                with gr.Column():
                    search_output = gr.Textbox(label="Search Results", lines=20, max_lines=30)
            search_btn.click(fn=search_nodes, inputs=[search_query, search_limit], outputs=search_output)
            gr.Markdown(_tool_doc_md(search_nodes))

        with gr.Tab("📝 Node Info"):
            with gr.Row():
                with gr.Column():
                    node_id_input = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    node_info_btn = gr.Button("Get Node Info", variant="primary")
                    node_edges_btn = gr.Button("Get Node Edges", variant="secondary")
                with gr.Column():
                    node_output = gr.Textbox(label="Node Information", lines=20, max_lines=30)
            node_info_btn.click(fn=get_node_info, inputs=node_id_input, outputs=node_output)
            node_edges_btn.click(fn=get_node_edges, inputs=node_id_input, outputs=node_output)
            gr.Markdown(_tool_doc_md(get_node_info))
            gr.Markdown(_tool_doc_md(get_node_edges))

        with gr.Tab("🏗️ Structure"):
            gr.Markdown("### Repository Tree")
            with gr.Row():
                with gr.Column():
                    tree_root = gr.Textbox(label="Root Node ID", value="root", placeholder="root")
                    tree_depth = gr.Slider(1, 10, value=3, step=1, label="Max Depth")
                    tree_btn = gr.Button("Show Tree", variant="primary")
                with gr.Column():
                    tree_output = gr.Textbox(label="Tree View", lines=20, max_lines=40)
            tree_btn.click(fn=print_tree, inputs=[tree_root, tree_depth], outputs=tree_output)
            gr.Markdown(_tool_doc_md(print_tree))

            gr.Markdown("---")
            gr.Markdown("### File Structure")
            with gr.Row():
                with gr.Column():
                    file_path_input = gr.Textbox(label="File Path", placeholder="Enter file path...")
                    file_structure_btn = gr.Button("Get File Structure", variant="primary")
                with gr.Column():
                    file_structure_output = gr.Textbox(label="File Structure", lines=20, max_lines=30)
            file_structure_btn.click(fn=get_file_structure, inputs=file_path_input, outputs=file_structure_output)
            gr.Markdown(_tool_doc_md(get_file_structure))

        with gr.Tab("🎯 Entities"):
            gr.Markdown("### List All Entities")
            with gr.Row():
                with gr.Column():
                    entity_page = gr.Slider(1, 100, value=1, step=1, label="Page")
                    entity_limit = gr.Slider(10, 100, value=50, step=10, label="Per Page")
                    entity_type_filter = gr.Dropdown(
                        choices=["", "class", "function", "method", "variable", "parameter"],
                        label="Filter by Type (optional)", value=""
                    )
                    declared_in_repo = gr.Dropdown(
                        choices=["", "true", "false"],
                        label="Declared in Repo (optional)",
                        value=""
                    )
                    list_entities_btn = gr.Button("List Entities", variant="primary")
                with gr.Column():
                    list_entities_output = gr.Textbox(label="Entities", lines=20, max_lines=30)
            list_entities_btn.click(
                fn=list_all_entities,
                inputs=[entity_limit, entity_page, entity_type_filter, declared_in_repo],
                outputs=list_entities_output,
            )
            gr.Markdown(_tool_doc_md(list_all_entities))

            gr.Markdown("---")
            gr.Markdown("### Go to Definition")
            with gr.Row():
                with gr.Column():
                    entity_name_def = gr.Textbox(label="Entity Name", placeholder="Enter entity name...")
                    def_btn = gr.Button("Go to Definition", variant="primary")
                with gr.Column():
                    def_output = gr.Textbox(label="Definition", lines=15, max_lines=25)
            def_btn.click(fn=go_to_definition, inputs=entity_name_def, outputs=def_output)
            gr.Markdown(_tool_doc_md(go_to_definition))

            gr.Markdown("---")
            gr.Markdown("### Find Usages")
            with gr.Row():
                with gr.Column():
                    entity_name_usage = gr.Textbox(label="Entity Name", placeholder="Enter entity name...")
                    usage_limit = gr.Slider(1, 50, value=20, step=1, label="Max Results")
                    usage_btn = gr.Button("Find Usages", variant="primary")
                with gr.Column():
                    usage_output = gr.Textbox(label="Usages", lines=15, max_lines=25)
            usage_btn.click(fn=find_usages, inputs=[entity_name_usage, usage_limit], outputs=usage_output)
            gr.Markdown(_tool_doc_md(find_usages))

        with gr.Tab("🔬 Discovery"):
            gr.Markdown("### List Nodes by Type")
            with gr.Row():
                with gr.Column():
                    node_type_input = gr.Dropdown(
                        choices=["file", "directory", "chunk", "entity", "function", "class", "method"],
                        label="Node Type"
                    )
                    type_limit = gr.Slider(1, 100, value=20, step=1, label="Max Results")
                    type_btn = gr.Button("List Nodes", variant="primary")
                with gr.Column():
                    type_output = gr.Textbox(label="Results", lines=20, max_lines=30)
            type_btn.click(fn=list_nodes_by_type, inputs=[node_type_input, type_limit], outputs=type_output)
            gr.Markdown(_tool_doc_md(list_nodes_by_type))

            gr.Markdown("---")
            gr.Markdown("### Search by Type and Name")
            with gr.Row():
                with gr.Column():
                    search_type = gr.Dropdown(
                        choices=["file", "directory", "chunk", "entity", "function", "class", "method"],
                        label="Node Type"
                    )
                    search_name = gr.Textbox(label="Name Contains", placeholder="Enter partial name...")
                    search_type_btn = gr.Button("Search", variant="primary")
                with gr.Column():
                    search_type_output = gr.Textbox(label="Results", lines=20, max_lines=30)
            search_type_btn.click(fn=search_by_type_and_name, inputs=[search_type, search_name], outputs=search_type_output)
            gr.Markdown(_tool_doc_md(search_by_type_and_name))

        with gr.Tab("🔗 Relationships"):
            gr.Markdown("### Get Neighbors")
            with gr.Row():
                with gr.Column():
                    neighbor_node_id = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    neighbor_btn = gr.Button("Get Neighbors", variant="primary")
                with gr.Column():
                    neighbor_output = gr.Textbox(label="Neighbors", lines=20, max_lines=30)
            neighbor_btn.click(fn=get_neighbors, inputs=neighbor_node_id, outputs=neighbor_output)
            gr.Markdown(_tool_doc_md(get_neighbors))

            gr.Markdown("---")
            gr.Markdown("### Entity Relationships")
            with gr.Row():
                with gr.Column():
                    rel_node_id = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    rel_btn = gr.Button("Get Relationships", variant="primary")
                with gr.Column():
                    rel_output = gr.Textbox(label="Relationships", lines=20, max_lines=30)
            rel_btn.click(fn=entity_relationships, inputs=rel_node_id, outputs=rel_output)
            gr.Markdown(_tool_doc_md(entity_relationships))

            gr.Markdown("---")
            gr.Markdown("### Get Related Chunks")
            with gr.Row():
                with gr.Column():
                    related_chunk_id = gr.Textbox(label="Chunk ID", placeholder="Enter chunk ID...")
                    relation_type = gr.Dropdown(choices=["calls", "contains", "declares", "uses"], label="Relation Type", value="calls")
                    related_btn = gr.Button("Get Related Chunks", variant="primary")
                with gr.Column():
                    related_output = gr.Textbox(label="Related Chunks", lines=20, max_lines=30)
            related_btn.click(fn=get_related_chunks, inputs=[related_chunk_id, relation_type], outputs=related_output)
            gr.Markdown(_tool_doc_md(get_related_chunks))

            gr.Markdown("---")
            gr.Markdown("### Find Path Between Nodes")
            with gr.Row():
                with gr.Column():
                    path_source = gr.Textbox(label="Source Node ID", placeholder="Enter source node ID...")
                    path_target = gr.Textbox(label="Target Node ID", placeholder="Enter target node ID...")
                    path_depth = gr.Slider(1, 10, value=5, step=1, label="Max Depth")
                    path_btn = gr.Button("Find Path", variant="primary")
                with gr.Column():
                    path_output = gr.Textbox(label="Path", lines=20, max_lines=30)
            path_btn.click(fn=find_path, inputs=[path_source, path_target, path_depth], outputs=path_output)
            gr.Markdown(_tool_doc_md(find_path))

            gr.Markdown("---")
            gr.Markdown("### Find Classes Inheriting From")
            with gr.Row():
                with gr.Column():
                    base_class_input = gr.Textbox(label="Base Class Name", placeholder="Enter base class...")
                    inherit_btn = gr.Button("Find Subclasses", variant="primary")
                with gr.Column():
                    inherit_output = gr.Textbox(label="Inheriting Classes", lines=20, max_lines=30)
            inherit_btn.click(fn=find_classes_inheriting_from, inputs=base_class_input, outputs=inherit_output)
            gr.Markdown(_tool_doc_md(find_classes_inheriting_from))

        with gr.Tab("📖 Context"):
            gr.Markdown("### Get Chunk Context")
            with gr.Row():
                with gr.Column():
                    chunk_id_input = gr.Textbox(label="Chunk ID", placeholder="Enter chunk ID...")
                    context_btn = gr.Button("Get Context", variant="primary")
                with gr.Column():
                    context_output = gr.Textbox(label="Context", lines=25, max_lines=40)
            context_btn.click(fn=get_chunk_context, inputs=chunk_id_input, outputs=context_output)
            gr.Markdown(_tool_doc_md(get_chunk_context))

            gr.Markdown("---")
            gr.Markdown("### Concept Overview")
            with gr.Row():
                with gr.Column():
                    concept_input = gr.Textbox(label="Concept", placeholder="e.g., embedding, authentication...")
                    concept_btn = gr.Button("Get Overview", variant="primary")
                with gr.Column():
                    concept_output = gr.Textbox(label="Concept Overview", lines=25, max_lines=40)
            concept_btn.click(fn=get_concept_overview, inputs=concept_input, outputs=concept_output)
            gr.Markdown(_tool_doc_md(get_concept_overview))

            gr.Markdown("---")
            gr.Markdown("### Get Subgraph")
            with gr.Row():
                with gr.Column():
                    subgraph_node = gr.Textbox(label="Center Node ID", placeholder="Enter node ID...")
                    subgraph_depth = gr.Slider(1, 5, value=2, step=1, label="Depth")
                    subgraph_edge_types = gr.Textbox(label="Edge Types (comma-separated, optional)", placeholder="e.g., calls,contains")
                    subgraph_btn = gr.Button("Extract Subgraph", variant="primary")
                with gr.Column():
                    subgraph_output = gr.Textbox(label="Subgraph", lines=20, max_lines=30)
            subgraph_btn.click(fn=get_subgraph, inputs=[subgraph_node, subgraph_depth, subgraph_edge_types], outputs=subgraph_output)
            gr.Markdown(_tool_doc_md(get_subgraph))

        with gr.Tab("📁 Files"):
            gr.Markdown("### List Files in Directory")
            with gr.Row():
                with gr.Column():
                    dir_path = gr.Textbox(label="Directory Path (empty for root)", placeholder="e.g., src/")
                    file_pattern = gr.Textbox(label="Pattern", value="*", placeholder="e.g., *.py")
                    file_recursive = gr.Checkbox(label="Recursive", value=True)
                    file_limit = gr.Slider(10, 100, value=50, step=10, label="Max Results")
                    list_files_btn = gr.Button("List Files", variant="primary")
                with gr.Column():
                    list_files_output = gr.Textbox(label="Files", lines=20, max_lines=30)
            list_files_btn.click(fn=list_files_in_directory, inputs=[dir_path, file_pattern, file_recursive, file_limit], outputs=list_files_output)
            gr.Markdown(_tool_doc_md(list_files_in_directory))

            gr.Markdown("---")
            gr.Markdown("### Find Files Importing")
            with gr.Row():
                with gr.Column():
                    import_module = gr.Textbox(label="Module/Entity Name", placeholder="e.g., torch, numpy...")
                    import_limit = gr.Slider(10, 50, value=30, step=5, label="Max Results")
                    find_imports_btn = gr.Button("Find Files", variant="primary")
                with gr.Column():
                    find_imports_output = gr.Textbox(label="Importing Files", lines=20, max_lines=30)
            find_imports_btn.click(fn=find_files_importing, inputs=[import_module, import_limit], outputs=find_imports_output)
            gr.Markdown(_tool_doc_md(find_files_importing))

            gr.Markdown("---")
            gr.Markdown("### Get File Stats")
            with gr.Row():
                with gr.Column():
                    stats_path = gr.Textbox(label="File Path", placeholder="Enter file path...")
                    stats_btn = gr.Button("Get Stats", variant="primary")
                with gr.Column():
                    stats_output = gr.Textbox(label="Statistics", lines=20, max_lines=30)
            stats_btn.click(fn=get_file_stats, inputs=stats_path, outputs=stats_output)
            gr.Markdown(_tool_doc_md(get_file_stats))

        with gr.Tab("🔍 Analysis"):
            gr.Markdown("### Diff Chunks")
            with gr.Row():
                with gr.Column():
                    diff_node1 = gr.Textbox(label="First Node ID", placeholder="Enter first node ID...")
                    diff_node2 = gr.Textbox(label="Second Node ID", placeholder="Enter second node ID...")
                    diff_btn = gr.Button("Show Diff", variant="primary")
                with gr.Column():
                    diff_output = gr.Textbox(label="Diff Output", lines=25, max_lines=40)
            diff_btn.click(fn=diff_chunks, inputs=[diff_node1, diff_node2], outputs=diff_output)
            gr.Markdown(_tool_doc_md(diff_chunks))

    return demo


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph MCP Server from HuggingFace Dataset")
    
    # Required argument
    parser.add_argument("--hf-dataset", type=str, default=os.environ.get("HF_DATASET"), 
                        help="HuggingFace dataset repo ID (e.g., 'username/dataset-name')")
    
    # Optional HuggingFace auth (falls back to HF_TOKEN env var)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), 
                        help="HuggingFace API token for private datasets (or set HF_TOKEN env var)")
    
    # Server settings
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    # Index settings
    parser.add_argument("--no-index", action="store_true", help="Skip indexing nodes")
    parser.add_argument("--code-index-type", type=str, default="keyword-only", 
                        choices=["keyword-only", "embedding-only", "hybrid"],
                        help="Type of code index to use")
    parser.add_argument("--code-index-backend", type=str, default="lancedb", 
                        choices=["lancedb", "weaviate"],
                        help="Backend for code index")

    args = parser.parse_args()

    # Build code_index_kwargs
    code_index_kwargs = {
        "index_type": args.code_index_type,
        "backend": args.code_index_backend,
        "use_embed": args.code_index_type != "keyword-only",
    }

    # Initialize knowledge graph
    print("Initializing knowledge graph from HuggingFace dataset...")
    initialize_knowledge_graph(
        hf_dataset=args.hf_dataset,
        hf_token=args.hf_token,
        index_nodes=not args.no_index,
        code_index_kwargs=code_index_kwargs
    )
    print("Knowledge graph initialized!")

    # Create and launch app
    demo = create_gradio_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        mcp_server=True
    )


if __name__ == "__main__":
    main()
