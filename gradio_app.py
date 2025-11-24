import os
import sys
import argparse
import difflib
from typing import Optional
import gradio as gr

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pedagogia_graph_code_repo'))

from pedagogia_graph_code_repo.RepoKnowledgeGraphLib.RepoKnowledgeGraph import RepoKnowledgeGraph

# Global knowledge graph instance
knowledge_graph = None


def initialize_knowledge_graph(
    repo_path: Optional[str] = None,
    graph_file: Optional[str] = None,
    repo_url: Optional[str] = None,
    skip_dirs: Optional[str] = None,
    index_nodes: bool = True,
    describe_nodes: bool = False,
    extract_entities: bool = True
):
    """Initialize the knowledge graph from various sources."""
    global knowledge_graph

    skip_dirs_list = skip_dirs.split(",") if skip_dirs else []

    if graph_file:
        knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(
            graph_file,
            index_nodes=index_nodes,
            model_service_kwargs={
                "embedder_type": "sentence-transformers",
                "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
            },
        code_index_kwargs = {
            "index_type": "keyword-only",
        }
        )
    elif repo_url:
        knowledge_graph = RepoKnowledgeGraph.from_repo(
            repo_url=repo_url,
            index_nodes=index_nodes,
            describe_nodes=describe_nodes,
            extract_entities=extract_entities,
            skip_dirs=skip_dirs_list,
            model_service_kwargs={
                "embedder_type": "sentence-transformers",
                "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
            },
        code_index_kwargs = {
            "index_type": "keyword-only",
        }
        )
    elif repo_path:
        knowledge_graph = RepoKnowledgeGraph.from_path(
            repo_path,
            skip_dirs=skip_dirs_list,
            index_nodes=index_nodes,
            describe_nodes=describe_nodes,
            extract_entities=extract_entities,
            model_service_kwargs={
                "embedder_type": "sentence-transformers",
                "embed_model_name": "Salesforce/SFR-Embedding-Code-400M_R",
            },
        code_index_kwargs = {
            "index_type": "keyword-only",
        }
        )
    else:
        raise ValueError("Must provide either repo_path, graph_file, or repo_url")


def get_node_info(node_id: str) -> str:
    """
    Get detailed information about a node in the knowledge graph.

    Returns information including the node's type, name, description,
    declared and called entities, and a content preview.

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
        declared_entities = getattr(node, 'declared_entities', [])
        called_entities = getattr(node, 'called_entities', [])
        content = getattr(node, 'content', None)
        content_preview = content[:200] + "..." if content and len(content) > 200 else content

        result = f"""Node Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node ID: {node_id}
Class: {node.__class__.__name__}
Name: {getattr(node, 'name', 'Unknown')}
Type: {getattr(node, 'node_type', 'Unknown')}
Description: {getattr(node, 'description', 'N/A')}

Declared Entities ({len(declared_entities)}):
{chr(10).join(f"  - {entity}" for entity in declared_entities[:10])}
{f"  ... and {len(declared_entities) - 10} more" if len(declared_entities) > 10 else ""}

Called Entities ({len(called_entities)}):
{chr(10).join(f"  - {entity}" for entity in called_entities[:10])}
{f"  ... and {len(called_entities) - 10} more" if len(called_entities) > 10 else ""}

Content Preview:
{content_preview or 'N/A'}
"""
        return result
    except Exception as e:
        return f"Error: {str(e)}"


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
                preview = content[:150] + "..." if len(content) > 150 else content
                result += f"   Content: {preview}\n"

            declared = res.get('declared_entities', [])
            if declared:
                result += f"   Declared: {', '.join(str(e) for e in declared[:5])}\n"

            called = res.get('called_entities', [])
            if called:
                result += f"   Called: {', '.join(str(e) for e in called[:5])}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


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
                content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                result += f"{i}. Chunk: {chunk_id}\n"
                result += f"   File: {chunk.path}\n"
                result += f"   Order: {chunk.order_in_file}\n"
                result += f"   Content: {content_preview}\n\n"

        if len(declaring_chunks) > 5:
            result += f"... and {len(declaring_chunks) - 5} more locations\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


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
                content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                result += f"{i}. {chunk.path} (chunk {chunk.order_in_file})\n"
                result += f"   Content: {content_preview}\n\n"

        if len(calling_chunks) > limit:
            result += f"... and {len(calling_chunks) - limit} more usages\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


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


def list_all_entities(
    limit: int = 50,
    entity_type: Optional[str] = None,
    declared_in_repo: Optional[bool] = None
) -> str:
    """
    List all entities tracked in the knowledge graph with filtering options.

    Shows entity types, declaration counts, and usage counts.

    Args:
        limit: Maximum number of entities to return (default: 50)
        entity_type: Filter by entity type ('class', 'function', 'method', 'variable', 'parameter', 'function_call', 'method_call')
        declared_in_repo: If True, only return entities with declarations. If False, only entities without declarations. If None, return all.

    Returns:
        str: A formatted string with all entities
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
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

        result = f"All Entities ({len(filtered_entities)} total):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, (entity_name, info) in enumerate(list(filtered_entities.items())[:limit], 1):
            result += f"{i}. {entity_name}\n"
            result += f"   Types: {', '.join(info.get('type', ['Unknown']))}\n"
            result += f"   Declarations: {len(info.get('declaring_chunk_ids', []))}\n"
            result += f"   Usages: {len(info.get('calling_chunk_ids', []))}\n\n"

        if len(filtered_entities) > limit:
            result += f"... and {len(filtered_entities) - limit} more entities\n"

        # Add filter information
        if entity_type:
            result += f"\n(Filtered by type={entity_type})\n"
        if declared_in_repo is not None:
            result += f"(Filtered by declared_in_repo={declared_in_repo})\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


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


def search_by_type_and_name(node_type: str, name_query: str, limit: int = 10) -> str:
    """
    Search for nodes/entities by type and name substring.

    Filters nodes by type and searches for matching names.

    Args:
        node_type: Type of node/entity (e.g., 'function', 'class', 'file')
        name_query: Substring to match in the name
        limit: Maximum results to return (default: 10)

    Returns:
        str: A formatted string with matching nodes
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if limit <= 0:
            return "Error: limit must be a positive integer"

        g = knowledge_graph.graph
        matches = [
            {
                "id": nid,
                "name": getattr(n['data'], 'name', 'Unknown')
            }
            for nid, n in g.nodes(data=True)
            if getattr(n['data'], 'node_type', None) == node_type
            and name_query.lower() in getattr(n['data'], 'name', '').lower()
        ][:limit]

        if not matches:
            return f"No matches for type '{node_type}' and name containing '{name_query}'."

        result = f"Matches for type '{node_type}' and name '{name_query}' ({len(matches)} results):\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        for i, match in enumerate(matches, 1):
            result += f"{i}. {match['name']}\n"
            result += f"   ID: {match['id']}\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


def get_chunk_context(node_id: str) -> str:
    """
    Show the previous and next code chunk for a given chunk.

    Displays surrounding context for a specific code chunk.

    Args:
        node_id: The node/chunk ID to get context for

    Returns:
        str: A formatted string with chunk context
    """
    if knowledge_graph is None:
        return "Error: Knowledge graph not initialized"

    try:
        if node_id not in knowledge_graph.graph:
            return f"Error: Node '{node_id}' not found in knowledge graph"

        previous_chunk = knowledge_graph.get_previous_chunk(node_id)
        next_chunk = knowledge_graph.get_next_chunk(node_id)

        result = f"Context for chunk '{node_id}':\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if previous_chunk:
            prev_content = previous_chunk.content
            prev_preview = prev_content[:200] + '...' if len(prev_content) > 200 else prev_content
            result += f"Previous chunk ({previous_chunk}):\n{prev_preview}\n\n"
        else:
            result += "No previous chunk found.\n\n"

        if next_chunk:
            next_content = next_chunk.content
            next_preview = next_content[:200] + '...' if len(next_content) > 200 else next_content
            result += f"Next chunk ({next_chunk}):\n{next_preview}\n\n"
        else:
            result += "No next chunk found.\n\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


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


def create_gradio_app():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Knowledge Graph MCP Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Knowledge Graph MCP Server
        
        Explore and query your codebase knowledge graph through an intuitive interface.
        Each tool provides semantic understanding of your code structure, entities, and relationships.
        """)

        with gr.Tab("📊 Graph Overview"):
            gr.Markdown("### Get statistics and overview of the knowledge graph")

            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button("Get Graph Statistics", variant="primary")
                with gr.Column():
                    stats_output = gr.Textbox(label="Statistics", lines=20, max_lines=30)

            stats_btn.click(fn=get_graph_stats, outputs=stats_output)

        with gr.Tab("🔎 Search"):
            gr.Markdown("### Search nodes and entities")

            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter search query...")
                    search_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Result Limit")
                    search_btn = gr.Button("Search Nodes", variant="primary")
                with gr.Column():
                    search_output = gr.Textbox(label="Search Results", lines=20, max_lines=30)

            search_btn.click(
                fn=search_nodes,
                inputs=[search_query, search_limit],
                outputs=search_output
            )

            gr.Markdown("---")
            gr.Markdown("### Search by type and name")

            with gr.Row():
                with gr.Column():
                    search_type = gr.Textbox(label="Node Type", placeholder="e.g., function, class, file")
                    search_name = gr.Textbox(label="Name Query", placeholder="Enter name substring...")
                    search_type_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Result Limit")
                    search_type_btn = gr.Button("Search by Type & Name", variant="primary")
                with gr.Column():
                    search_type_output = gr.Textbox(label="Search Results", lines=20, max_lines=30)

            search_type_btn.click(
                fn=search_by_type_and_name,
                inputs=[search_type, search_name, search_type_limit],
                outputs=search_type_output
            )

        with gr.Tab("📝 Node Info"):
            gr.Markdown("### Get detailed information about nodes")

            with gr.Row():
                with gr.Column():
                    node_id = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    node_info_btn = gr.Button("Get Node Info", variant="primary")
                with gr.Column():
                    node_info_output = gr.Textbox(label="Node Information", lines=20, max_lines=30)

            node_info_btn.click(fn=get_node_info, inputs=node_id, outputs=node_info_output)

            gr.Markdown("---")
            gr.Markdown("### Get node edges")

            with gr.Row():
                with gr.Column():
                    edges_node_id = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    edges_btn = gr.Button("Get Node Edges", variant="primary")
                with gr.Column():
                    edges_output = gr.Textbox(label="Node Edges", lines=20, max_lines=30)

            edges_btn.click(fn=get_node_edges, inputs=edges_node_id, outputs=edges_output)

            gr.Markdown("---")
            gr.Markdown("### Get neighbors")

            with gr.Row():
                with gr.Column():
                    neighbors_node_id = gr.Textbox(label="Node ID", placeholder="Enter node ID...")
                    neighbors_btn = gr.Button("Get Neighbors", variant="primary")
                with gr.Column():
                    neighbors_output = gr.Textbox(label="Neighbors", lines=20, max_lines=30)

            neighbors_btn.click(fn=get_neighbors, inputs=neighbors_node_id, outputs=neighbors_output)

        with gr.Tab("🏗️ Structure"):
            gr.Markdown("### List nodes by type")

            with gr.Row():
                with gr.Column():
                    list_type = gr.Textbox(label="Node Type", placeholder="e.g., function, class, file")
                    list_limit = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Result Limit")
                    list_btn = gr.Button("List Nodes", variant="primary")
                with gr.Column():
                    list_output = gr.Textbox(label="Nodes", lines=20, max_lines=30)

            list_btn.click(fn=list_nodes_by_type, inputs=[list_type, list_limit], outputs=list_output)

            gr.Markdown("---")
            gr.Markdown("### Get file structure")

            with gr.Row():
                with gr.Column():
                    file_path = gr.Textbox(label="File Path", placeholder="Enter file path...")
                    file_structure_btn = gr.Button("Get File Structure", variant="primary")
                with gr.Column():
                    file_structure_output = gr.Textbox(label="File Structure", lines=20, max_lines=30)

            file_structure_btn.click(fn=get_file_structure, inputs=file_path, outputs=file_structure_output)

            gr.Markdown("---")
            gr.Markdown("### Print tree")

            with gr.Row():
                with gr.Column():
                    tree_root = gr.Textbox(label="Root Node ID", value="root", placeholder="Enter root node ID...")
                    tree_depth = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Depth")
                    tree_btn = gr.Button("Print Tree", variant="primary")
                with gr.Column():
                    tree_output = gr.Textbox(label="Tree Structure", lines=20, max_lines=30)

            tree_btn.click(fn=print_tree, inputs=[tree_root, tree_depth], outputs=tree_output)

        with gr.Tab("🎯 Entities"):
            gr.Markdown("### List all entities")

            with gr.Row():
                with gr.Column():
                    entities_limit = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Result Limit")
                    entities_type_filter = gr.Dropdown(
                        choices=["", "class", "function", "method", "variable", "parameter", "function_call", "method_call"],
                        value="",
                        label="Entity Type Filter (optional)",
                        info="Leave empty for all types"
                    )
                    entities_declared_filter = gr.Radio(
                        choices=[("All", "all"), ("Declared in repo", "true"), ("Not declared in repo", "false")],
                        value="all",
                        label="Declaration Filter"
                    )
                    entities_btn = gr.Button("List All Entities", variant="primary")
                with gr.Column():
                    entities_output = gr.Textbox(label="Entities", lines=20, max_lines=30)

            def list_entities_wrapper(limit, entity_type, declared_filter):
                # Convert filter values to appropriate types
                entity_type_val = entity_type if entity_type else None
                declared_val = None if declared_filter == "all" else (declared_filter == "true")
                return list_all_entities(limit, entity_type_val, declared_val)

            entities_btn.click(
                fn=list_entities_wrapper,
                inputs=[entities_limit, entities_type_filter, entities_declared_filter],
                outputs=entities_output
            )

            gr.Markdown("---")
            gr.Markdown("### Go to definition")

            with gr.Row():
                with gr.Column():
                    def_entity = gr.Textbox(label="Entity Name", placeholder="Enter entity name...")
                    def_btn = gr.Button("Go to Definition", variant="primary")
                with gr.Column():
                    def_output = gr.Textbox(label="Definition", lines=20, max_lines=30)

            def_btn.click(fn=go_to_definition, inputs=def_entity, outputs=def_output)

            gr.Markdown("---")
            gr.Markdown("### Find usages")

            with gr.Row():
                with gr.Column():
                    usage_entity = gr.Textbox(label="Entity Name", placeholder="Enter entity name...")
                    usage_limit = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Result Limit")
                    usage_btn = gr.Button("Find Usages", variant="primary")
                with gr.Column():
                    usage_output = gr.Textbox(label="Usages", lines=20, max_lines=30)

            usage_btn.click(fn=find_usages, inputs=[usage_entity, usage_limit], outputs=usage_output)

            gr.Markdown("---")
            gr.Markdown("### Entity relationships")

            with gr.Row():
                with gr.Column():
                    rel_node_id = gr.Textbox(label="Node/Entity ID", placeholder="Enter node/entity ID...")
                    rel_btn = gr.Button("Get Relationships", variant="primary")
                with gr.Column():
                    rel_output = gr.Textbox(label="Relationships", lines=20, max_lines=30)

            rel_btn.click(fn=entity_relationships, inputs=rel_node_id, outputs=rel_output)

        with gr.Tab("🔗 Relationships"):
            gr.Markdown("### Get related chunks")

            with gr.Row():
                with gr.Column():
                    rel_chunk_id = gr.Textbox(label="Chunk ID", placeholder="Enter chunk ID...")
                    rel_type = gr.Textbox(label="Relation Type", value="calls", placeholder="e.g., calls, contains")
                    rel_chunk_btn = gr.Button("Get Related Chunks", variant="primary")
                with gr.Column():
                    rel_chunk_output = gr.Textbox(label="Related Chunks", lines=20, max_lines=30)

            rel_chunk_btn.click(
                fn=get_related_chunks,
                inputs=[rel_chunk_id, rel_type],
                outputs=rel_chunk_output
            )

            gr.Markdown("---")
            gr.Markdown("### Find path between nodes")

            with gr.Row():
                with gr.Column():
                    path_source = gr.Textbox(label="Source Node ID", placeholder="Enter source node ID...")
                    path_target = gr.Textbox(label="Target Node ID", placeholder="Enter target node ID...")
                    path_depth = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Max Depth")
                    path_btn = gr.Button("Find Path", variant="primary")
                with gr.Column():
                    path_output = gr.Textbox(label="Path", lines=20, max_lines=30)

            path_btn.click(
                fn=find_path,
                inputs=[path_source, path_target, path_depth],
                outputs=path_output
            )

            gr.Markdown("---")
            gr.Markdown("### Get subgraph")

            with gr.Row():
                with gr.Column():
                    subgraph_node = gr.Textbox(label="Central Node ID", placeholder="Enter node ID...")
                    subgraph_depth = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Depth")
                    subgraph_edge_types = gr.Textbox(
                        label="Edge Types (comma-separated, optional)",
                        placeholder="e.g., calls,contains"
                    )
                    subgraph_btn = gr.Button("Get Subgraph", variant="primary")
                with gr.Column():
                    subgraph_output = gr.Textbox(label="Subgraph", lines=20, max_lines=30)

            subgraph_btn.click(
                fn=get_subgraph,
                inputs=[subgraph_node, subgraph_depth, subgraph_edge_types],
                outputs=subgraph_output
            )

        with gr.Tab("🔍 Analysis"):
            gr.Markdown("### Get chunk context")

            with gr.Row():
                with gr.Column():
                    context_chunk_id = gr.Textbox(label="Chunk ID", placeholder="Enter chunk ID...")
                    context_btn = gr.Button("Get Chunk Context", variant="primary")
                with gr.Column():
                    context_output = gr.Textbox(label="Chunk Context", lines=20, max_lines=30)

            context_btn.click(fn=get_chunk_context, inputs=context_chunk_id, outputs=context_output)

            gr.Markdown("---")
            gr.Markdown("### Get file statistics")

            with gr.Row():
                with gr.Column():
                    stats_path = gr.Textbox(label="File/Directory Path", placeholder="Enter path...")
                    stats_file_btn = gr.Button("Get File Stats", variant="primary")
                with gr.Column():
                    stats_file_output = gr.Textbox(label="File Statistics", lines=20, max_lines=30)

            stats_file_btn.click(fn=get_file_stats, inputs=stats_path, outputs=stats_file_output)

            gr.Markdown("---")
            gr.Markdown("### Diff chunks")

            with gr.Row():
                with gr.Column():
                    diff_node_1 = gr.Textbox(label="First Node ID", placeholder="Enter first node ID...")
                    diff_node_2 = gr.Textbox(label="Second Node ID", placeholder="Enter second node ID...")
                    diff_btn = gr.Button("Show Diff", variant="primary")
                with gr.Column():
                    diff_output = gr.Textbox(label="Diff", lines=20, max_lines=30)

            diff_btn.click(fn=diff_chunks, inputs=[diff_node_1, diff_node_2], outputs=diff_output)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph MCP Server Gradio App")
    parser.add_argument("--repo-path", type=str, help="Path to repository to analyze")
    parser.add_argument("--graph-file", type=str, help="Path to saved knowledge graph file")
    parser.add_argument("--repo-url", type=str, help="URL of repository to clone and analyze")
    parser.add_argument("--skip-dirs", type=str, help="Comma-separated list of directories to skip")
    parser.add_argument("--no-index", action="store_true", help="Skip indexing nodes")
    parser.add_argument("--describe-nodes", action="store_true", help="Generate node descriptions")
    parser.add_argument("--no-extract-entities", action="store_true", help="Skip entity extraction")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")

    args = parser.parse_args()

    # Initialize knowledge graph
    print("Initializing knowledge graph...")
    initialize_knowledge_graph(
        repo_path=args.repo_path,
        graph_file=args.graph_file,
        repo_url=args.repo_url,
        skip_dirs=args.skip_dirs,
        index_nodes=not args.no_index,
        describe_nodes=args.describe_nodes,
        extract_entities=not args.no_extract_entities
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

