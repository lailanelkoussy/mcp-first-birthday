import os
import re
import difflib
import fnmatch
from typing import Optional, Annotated, List
from fastmcp import FastMCP
from langfuse import get_client, observe

from .RepoKnowledgeGraph import RepoKnowledgeGraph


# Custom Exceptions
class MCPServerError(Exception):
    """Base exception for MCP server errors"""
    pass


class NodeNotFoundError(MCPServerError):
    """Raised when a node is not found"""
    pass


class EntityNotFoundError(MCPServerError):
    """Raised when an entity is not found"""
    pass


class InvalidInputError(MCPServerError):
    """Raised when input validation fails"""
    pass


class KnowledgeGraphMCPServer:
    """
    MCP Server for interacting with a codebase knowledge graph.

    Attributes:
        knowledge_graph (RepoKnowledgeGraph): The loaded knowledge graph object.
        app (FastMCP): The FastMCP application instance for tool registration and serving.
    """
    def __init__(self, knowledge_graph: Optional[RepoKnowledgeGraph] = None, knowledge_graph_path: Optional[str] = None, server_name: str = "knowledge-graph-mcp-server"):
        if knowledge_graph is not None:
            self.knowledge_graph = knowledge_graph
        else:
            if knowledge_graph_path is None:
                knowledge_graph_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
            self.knowledge_graph = RepoKnowledgeGraph.load_graph_from_file(knowledge_graph_path)
        self.langfuse = get_client()
        self.app = FastMCP(server_name)
        self.register_tools()

    def _validate_node_exists(self, node_id: str) -> bool:
        """Centralized node validation"""
        if node_id not in self.knowledge_graph.graph:
            raise NodeNotFoundError(f"Node '{node_id}' not found in knowledge graph")
        return True

    def _validate_entity_exists(self, entity_name: str) -> bool:
        """Centralized entity validation"""
        if entity_name not in self.knowledge_graph.entities:
            raise EntityNotFoundError(f"Entity '{entity_name}' not found in knowledge graph")
        return True

    def _validate_positive_int(self, value: int, param_name: str) -> bool:
        """Validate that an integer parameter is positive"""
        if value <= 0:
            raise InvalidInputError(f"{param_name} must be a positive integer, got {value}")
        return True

    def _sanitize_chunk_dict(self, chunk_dict: dict) -> dict:
        """Remove embedding data from chunk dictionary before returning to user"""
        sanitized = chunk_dict.copy()
        sanitized.pop('embedding', None)
        return sanitized

    def _sanitize_node_dict(self, node_dict: dict) -> dict:
        """Remove embedding data from node dictionary before returning to user"""
        sanitized = node_dict.copy()
        if 'data' in sanitized and isinstance(sanitized['data'], dict):
            sanitized['data'] = sanitized['data'].copy()
            sanitized['data'].pop('embedding', None)
        sanitized.pop('embedding', None)
        return sanitized

    def _handle_error(self, error: Exception, context: str = "") -> dict:
        """Centralized error handling with structured response"""
        if isinstance(error, NodeNotFoundError):
            return {
                "error": str(error),
                "error_type": "node_not_found",
                "context": context
            }
        elif isinstance(error, EntityNotFoundError):
            return {
                "error": str(error),
                "error_type": "entity_not_found",
                "context": context
            }
        elif isinstance(error, InvalidInputError):
            return {
                "error": str(error),
                "error_type": "invalid_input",
                "context": context
            }
        else:
            return {
                "error": str(error),
                "error_type": "internal_error",
                "context": context
            }

    @classmethod
    def from_path(cls, path: str, skip_dirs=None, index_nodes=True, describe_nodes=False, extract_entities=False, model_service_kwargs=None, code_index_kwargs=None, server_name: str = "knowledge-graph-mcp-server"):
        """
        Build a KnowledgeGraphMCPServer from a code repository path.
        """
        if skip_dirs is None:
            skip_dirs = []
        if model_service_kwargs is None:
            model_service_kwargs = {}
        kg = RepoKnowledgeGraph.from_path(path, skip_dirs=skip_dirs, index_nodes=index_nodes, describe_nodes=describe_nodes, extract_entities=extract_entities, model_service_kwargs=model_service_kwargs, code_index_kwargs=code_index_kwargs)
        return cls(knowledge_graph=kg, server_name=server_name)

    @classmethod
    def from_file(cls, filepath: str, index_nodes=True, use_embed=True, model_service_kwargs=None, code_index_kwargs = None, server_name: str = "knowledge-graph-mcp-server"):
        """
        Build a KnowledgeGraphMCPServer from a serialized knowledge graph file.
        """
        if model_service_kwargs is None:
            model_service_kwargs = {}
        kg = RepoKnowledgeGraph.load_graph_from_file(filepath, index_nodes=index_nodes, use_embed=use_embed, model_service_kwargs=model_service_kwargs, code_index_kwargs=code_index_kwargs)
        return cls(knowledge_graph=kg, server_name=server_name)

    @classmethod
    def from_repo(cls, repo_url: str, index_nodes=True, describe_nodes=False, model_service_kwargs=None, code_index_kwargs=None, server_name: str = "knowledge-graph-mcp-server", github_token=None, allow_unauthenticated_clone=True, skip_dirs=None, extract_entities=True):
        if model_service_kwargs is None:
            model_service_kwargs = {}
        kg = RepoKnowledgeGraph.from_repo(repo_url=repo_url, describe_nodes=describe_nodes, index_nodes=index_nodes, model_service_kwargs=model_service_kwargs, github_token=github_token, allow_unauthenticated_clone=allow_unauthenticated_clone, skip_dirs=skip_dirs, extract_entities=extract_entities, code_index_kwargs=code_index_kwargs)
        return cls(knowledge_graph=kg, server_name=server_name)


    def register_tools(self):
        @self.app.tool(
            description="""Retrieve comprehensive details about any node in the knowledge graph.

            PURPOSE:
            Use this tool to inspect the full metadata and content of a specific node when you need
            to understand what a particular code element contains, what entities it declares or calls,
            and how it fits into the codebase structure.

            WHEN TO USE:
            - After finding a node ID from search_nodes, list_nodes_by_type, or get_neighbors
            - To see the actual code content of a chunk node
            - To understand what entities (classes, functions, variables) are declared in a file or chunk
            - To examine entity metadata including aliases, declaration locations, and usage locations
            - To get file metadata like language and path information

            NODE TYPES SUPPORTED:
            - 'chunk': Code segments with content, declared/called entities, and file position
            - 'file': Source files with path, language, and entity summaries
            - 'directory': Folder nodes with path information
            - 'entity': Programming constructs (classes, functions, methods, variables) with declaration/usage tracking
            - 'repo': Repository root node

            TYPICAL WORKFLOW:
            1. search_nodes("attention mechanism") -> get node IDs
            2. get_node_info(node_id) -> see full content and metadata
            3. get_neighbors(node_id) or find_usages(entity_name) -> explore relationships
            """
        )
        @observe(as_type='tool')
        async def get_node_info(
                node_id: Annotated[str, "The unique identifier of the node (e.g., 'src/file.py::chunk_3' for chunks, or 'BertModel' for entities)"]
        ) -> dict:
            try:
                self._validate_node_exists(node_id)
                node = self.knowledge_graph.graph.nodes[node_id]['data']
                node_type = getattr(node, 'node_type', 'Unknown')
                node_class = node.__class__.__name__
                node_name = getattr(node, 'name', 'Unknown')
                description = getattr(node, 'description', None)

                # Handle entity nodes specially
                if node_class == 'EntityNode' or node_type == 'entity':
                    entity_type = getattr(node, 'entity_type', 'Unknown')
                    declaring_chunk_ids = getattr(node, 'declaring_chunk_ids', [])
                    calling_chunk_ids = getattr(node, 'calling_chunk_ids', [])
                    aliases = getattr(node, 'aliases', [])

                    return {
                        "node_id": node_id,
                        "class": node_class,
                        "name": node_name,
                        "type": node_type,
                        "entity_type": entity_type,
                        "description": description,
                        "aliases": aliases,
                        "declaring_chunk_ids": declaring_chunk_ids[:5],
                        "declaring_chunk_count": len(declaring_chunk_ids),
                        "calling_chunk_ids": calling_chunk_ids[:5],
                        "calling_chunk_count": len(calling_chunk_ids),
                        "text": f"Entity {node_id} ({node_name}) — {entity_type} declared in {len(declaring_chunk_ids)} chunk(s) and called in {len(calling_chunk_ids)} chunk(s)."
                    }
                else:
                    declared_entities = getattr(node, 'declared_entities', [])
                    called_entities = getattr(node, 'called_entities', [])
                    content = getattr(node, 'content', None)

                    result = {
                        "node_id": node_id,
                        "class": node_class,
                        "name": node_name,
                        "type": node_type,
                        "description": description,
                        "declared_entities": declared_entities[:10],
                        "declared_entities_count": len(declared_entities),
                        "called_entities": called_entities[:10],
                        "called_entities_count": len(called_entities),
                        "text": f"Node {node_id} ({node_name}) — {node_type} with {len(declared_entities)} declared and {len(called_entities)} called entities."
                    }

                    # Add content for file/chunk nodes
                    if node_type in ['file', 'chunk']:
                        result["content"] = content
                        if hasattr(node, 'path'):
                            result["path"] = node.path
                        if hasattr(node, 'language'):
                            result["language"] = node.language
                        if node_type == 'chunk' and hasattr(node, 'order_in_file'):
                            result["order_in_file"] = node.order_in_file
                    elif node_type == 'directory':
                        if hasattr(node, 'path'):
                            result["path"] = node.path

                    return result
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_node_info")
            except Exception as e:
                return self._handle_error(e, "get_node_info")

        @self.app.tool(
            description="""List all graph edges (relationships) connected to a specific node in the knowledge graph.

            PURPOSE:
            Use this tool to understand how a node is connected to other parts of the codebase.
            Reveals the dependency structure and relationships that link code elements together.

            WHEN TO USE:
            - To discover what code calls or depends on a specific function/class
            - To find parent-child relationships (e.g., which file contains a chunk)
            - To trace declaration and usage patterns through the codebase
            - To understand the connectivity of an entity in the dependency graph
            - When you need a raw view of all relationships without filtering

            EDGE TYPES YOU'LL SEE:
            - 'contains': Parent-child (file→chunk, directory→file, repo→directory)
            - 'calls': Entity usage relationships (chunk→entity it calls)
            - 'declares': Entity declaration relationships (chunk→entity it defines)

            DIRECTION MEANINGS:
            - Incoming edges (←): Other nodes pointing TO this node (e.g., "who calls me?")
            - Outgoing edges (→): This node pointing TO others (e.g., "what do I call?")

            COMPARISON WITH get_neighbors:
            - get_node_edges: Shows edge metadata and direction, raw relationship view
            - get_neighbors: Shows neighboring node details, easier for exploration
            """
        )
        @observe(as_type='tool')
        async def get_node_edges(
                node_id: Annotated[str, "The unique identifier of the node to inspect edges for"]
        ) -> dict:
            try:
                self._validate_node_exists(node_id)
                g = self.knowledge_graph.graph

                incoming = [
                    {"source": src, "target": tgt, "relation": data.get("relation", "?")}
                    for src, tgt, data in g.in_edges(node_id, data=True)
                ]
                outgoing = [
                    {"source": src, "target": tgt, "relation": data.get("relation", "?")}
                    for src, tgt, data in g.out_edges(node_id, data=True)
                ]

                text = f"Node '{node_id}' has {len(incoming)} incoming and {len(outgoing)} outgoing edges.\n\n"
                text += f"Incoming Edges ({len(incoming)}):\n"
                for edge in incoming[:20]:
                    text += f"  ← {edge['source']} [{edge['relation']}]\n"
                if len(incoming) > 20:
                    text += f"  ... and {len(incoming) - 20} more\n"
                text += f"\nOutgoing Edges ({len(outgoing)}):\n"
                for edge in outgoing[:20]:
                    text += f"  → {edge['target']} [{edge['relation']}]\n"
                if len(outgoing) > 20:
                    text += f"  ... and {len(outgoing) - 20} more\n"

                return {
                    "node_id": node_id,
                    "incoming": incoming[:20],
                    "outgoing": outgoing[:20],
                    "incoming_count": len(incoming),
                    "outgoing_count": len(outgoing),
                    "has_more_incoming": len(incoming) > 20,
                    "has_more_outgoing": len(outgoing) > 20,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_node_edges")
            except Exception as e:
                return self._handle_error(e, "get_node_edges")

        @self.app.tool(
            description="""Search the codebase using keyword matching against code content and metadata.

            PURPOSE:
            This is your PRIMARY SEARCH TOOL for exploring the codebase. Use it to find relevant
            code chunks based on natural language queries, function names, class names, comments,
            or any text that might appear in the source code.

            WHEN TO USE:
            - FIRST STEP when investigating any topic in the codebase
            - To find implementations of specific features (e.g., "rotary embeddings", "flash attention")
            - To locate code by function/class name when you don't have the exact node ID
            - To discover code related to a concept (e.g., "gradient checkpointing", "tokenization")
            - When you don't know where something is implemented

            SEARCH TIPS:
            - Use specific technical terms: "rope embedding" rather than just "embedding"
            - Include class/function names if known: "BertSelfAttention forward"
            - Try multiple related queries if first results aren't satisfactory
            - Results are ranked by relevance to your query

            TYPICAL WORKFLOW:
            1. search_nodes("attention mask handling") -> find relevant chunks
            2. get_node_info(chunk_id) -> examine the code content
            3. get_chunk_context(chunk_id) -> see surrounding code for fuller picture
            4. go_to_definition(entity_name) -> find where an entity is defined
            """
        )
        @observe(as_type='tool')
        async def search_nodes(
                query: Annotated[str, "Search terms to match against code content. Can be natural language, function names, class names, or code snippets."],
                limit: Annotated[int, "Results per page (default: 10, max recommended: 50)."] = 10,
                page: Annotated[int, "Page number starting from 1. Use pagination to browse through many results."] = 1
        ) -> dict:
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                # Fetch more results to support pagination
                max_fetch = limit * page
                results = self.knowledge_graph.code_index.query(query, n_results=max_fetch)
                metadatas = results.get("metadatas", [[]])[0]

                if not metadatas:
                    return {"query": query, "results": [], "count": 0, "page": page, "total_pages": 0, "text": f"No results found for '{query}'."}

                total = len(metadatas)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "search_nodes")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = metadatas[start_idx:end_idx]

                structured_results = [
                    {
                        "id": res.get("id"),
                        "content": res.get("content"),
                        "declared_entities": res.get("declared_entities"),
                        "called_entities": res.get("called_entities")
                    }
                    for res in page_slice
                ]

                text = f"Search Results for '{query}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, res in enumerate(structured_results, start=start_idx + 1):
                    text += f"{i}. ID: {res.get('id', 'N/A')}\n"
                    content = res.get('content', '')
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        text += f"   Content: {preview}\n"
                    text += "\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "query": query,
                    "count": len(structured_results),
                    "total": total,
                    "page": page,
                    "total_pages": total_pages,
                    "results": structured_results,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "search_nodes")
            except Exception as e:
                return self._handle_error(e, "search_nodes")

        @self.app.tool(
            description="""Get a comprehensive statistical overview of the knowledge graph.

            PURPOSE:
            Use this tool to understand the scope and structure of the knowledge graph.
            Provides counts and breakdowns of all node types, entity types, and relationship types.

            WHEN TO USE:
            - At the START of an exploration session to understand the codebase scope
            - To learn what types of entities and relationships are available for querying
            - To understand the terminology used in this knowledge graph (chunks, entities, edges)
            - When you need to report on the overall structure of the codebase

            WHAT YOU'LL LEARN:
            - Total number of nodes and edges in the graph
            - Breakdown of node types (chunks, files, directories, entities)
            - Entity type distribution (classes, functions, methods, variables, etc.)
            - Edge relationship types (contains, calls, declares)
            - Definitions of key concepts used throughout the tools

            GRAPH TERMINOLOGY:
            - Chunks: Logical code segments (a function body, a class definition, etc.)
            - Entities: Named programming constructs tracked across the codebase
            - Edges: Relationships connecting nodes (contains, calls, declares)
            """
        )
        @observe(as_type='tool')
        async def get_graph_stats() -> dict:
            g = self.knowledge_graph.graph
            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()

            # Count node types
            node_types = {}
            entity_breakdown = {}
            
            for _, node_attrs in g.nodes(data=True):
                node_type = getattr(node_attrs['data'], 'node_type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
                # For entity nodes, get entity_type breakdown
                if node_type == 'entity':
                    entity_type = getattr(node_attrs['data'], 'entity_type', 'Unknown')
                    
                    # Fallback: if entity_type is empty, check entities dictionary
                    if not entity_type:
                        node_id = node_attrs['data'].id if hasattr(node_attrs['data'], 'id') else None
                        if node_id and node_id in self.knowledge_graph.entities:
                            entity_types = self.knowledge_graph.entities[node_id].get('type', [])
                            entity_type = entity_types[0] if entity_types else 'Unknown'
                    
                    entity_breakdown[entity_type] = entity_breakdown.get(entity_type, 0) + 1

            # Count edge relations
            edge_relations = {}
            for _, _, attrs in g.edges(data=True):
                relation = attrs.get('relation', 'Unknown')
                edge_relations[relation] = edge_relations.get(relation, 0) + 1

            text = f"Knowledge Graph Statistics:\n"
            text += f"\n📊 Overview:\n"
            text += f"  Total Nodes: {num_nodes:,}\n"
            text += f"  Total Edges: {num_edges:,}\n\n"
            text += f"📦 Node Types:\n"
            
            for ntype, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                text += f"  • {ntype}: {count:,}\n"
                if ntype == 'entity' and entity_breakdown:
                    text += f"    └─ Entity Breakdown:\n"
                    for etype, ecount in sorted(entity_breakdown.items(), key=lambda x: x[1], reverse=True):
                        percentage = (ecount / count * 100) if count > 0 else 0
                        text += f"       ├─ {etype}: {ecount:,} ({percentage:.1f}%)\n"

            text += f"\n🔗 Edge Relations:\n"
            for relation, count in sorted(edge_relations.items(), key=lambda x: x[1], reverse=True):
                text += f"  • {relation}: {count:,}\n"

            return {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "node_types": node_types,
                "entity_breakdown": entity_breakdown,
                "edge_relations": edge_relations,
                "text": text
            }

        @self.app.tool(
            description="""List all nodes of a specific type in the knowledge graph with pagination.

            PURPOSE:
            Use this tool to browse and discover nodes by their type. Helpful when you want to
            see what classes, functions, files, or other constructs exist in the codebase.

            WHEN TO USE:
            - To get a list of all classes in the codebase: node_type='class'
            - To see all Python files: node_type='file'
            - To list all functions: node_type='function'
            - To browse all methods: node_type='method'
            - When you need to find node IDs for further exploration

            VALID node_type VALUES:
            For entities (programming constructs):
            - 'class': Class definitions
            - 'function': Standalone function definitions
            - 'method': Class method definitions
            - 'variable': Variable declarations
            - 'parameter': Function/method parameters

            For structural nodes:
            - 'file': Source code files
            - 'chunk': Code segments within files
            - 'directory': Folder structure nodes
            - 'repo': Repository root (typically one)

            COMPARISON WITH search_by_type_and_name:
            - list_nodes_by_type: Browse ALL nodes of a type (no name filter)
            - search_by_type_and_name: Filter by type AND search by name substring
            """
        )
        @observe(as_type='tool')
        async def list_nodes_by_type(
                node_type: Annotated[str, "The type to filter by. Use lowercase: 'class', 'function', 'method', 'file', 'chunk', 'directory'"],
                limit: Annotated[int, "Maximum results per page (default: 20)."] = 20,
                page: Annotated[int, "Page number starting from 1 for pagination."] = 1
        ) -> dict:
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                g = self.knowledge_graph.graph
                matching_nodes = []
                
                for node_id, data in g.nodes(data=True):
                    node = data['data']
                    current_node_type = getattr(node, 'node_type', None)
                    node_name = getattr(node, 'name', 'Unknown')
                    
                    # For entity nodes, check entity_type instead of node_type
                    if current_node_type == 'entity':
                        entity_type = getattr(node, 'entity_type', '')
                        
                        # Fallback: if entity_type is empty, check the entities dictionary
                        if not entity_type and node_id in self.knowledge_graph.entities:
                            entity_types = self.knowledge_graph.entities[node_id].get('type', [])
                            entity_type = entity_types[0] if entity_types else ''
                        
                        if entity_type and entity_type.lower() == node_type.lower():
                            matching_nodes.append({
                                "id": node_id,
                                "name": node_name,
                                "type": f"entity ({entity_type})"
                            })
                    # For other nodes, check node_type directly
                    elif current_node_type == node_type:
                        matching_nodes.append({
                            "id": node_id,
                            "name": node_name,
                            "type": current_node_type
                        })
                
                # Sort by name for consistent ordering
                matching_nodes.sort(key=lambda x: x['name'].lower())

                total = len(matching_nodes)
                if total == 0:
                    return {"node_type": node_type, "results": [], "count": 0, "page": page, "total_pages": 0, "text": f"No nodes found of type '{node_type}'."}

                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "list_nodes_by_type")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = matching_nodes[start_idx:end_idx]

                text = f"Nodes of type '{node_type}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, node in enumerate(page_slice, start=start_idx + 1):
                    text += f"{i}. {node['name']}\n"
                    text += f"   ID: {node['id']}\n"
                    text += f"   Type: {node['type']}\n\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "node_type": node_type,
                    "count": len(page_slice),
                    "total": total,
                    "page": page,
                    "total_pages": total_pages,
                    "results": page_slice,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "list_nodes_by_type")
            except Exception as e:
                return self._handle_error(e, "list_nodes_by_type")

        @self.app.tool(
            description="""Get all nodes directly connected to a given node with their relationship information.

            PURPOSE:
            Use this tool to explore the local neighborhood of any node in the knowledge graph.
            Shows what's connected to a node and how, making it easy to navigate the codebase structure.

            WHEN TO USE:
            - To explore what a node is connected to (files, chunks, entities)
            - To navigate from one code element to related elements
            - To understand the local structure around a specific node
            - After using get_node_info when you want to explore connected nodes
            - To discover related code without knowing exact names

            WHAT YOU'LL SEE:
            - Neighbor node IDs and names
            - Node types (chunk, file, entity, etc.)
            - Relationship direction (→ outgoing, ← incoming)
            - Relationship type (contains, calls, declares)

            TYPICAL NAVIGATION PATTERNS:
            - From a file: see its chunks and declared entities
            - From a chunk: see entities it declares/calls and its parent file
            - From an entity: see chunks that declare or call it
            - From a directory: see contained files and subdirectories

            COMPARISON WITH get_node_edges:
            - get_neighbors: Shows neighboring NODE details (name, type) - better for exploration
            - get_node_edges: Shows raw EDGE information - better for understanding relationships
            """
        )
        @observe(as_type='tool')
        async def get_neighbors(
            node_id: Annotated[str, "The ID of the node to explore neighbors for"],
            limit: Annotated[int, "Maximum neighbors to return per page (default: 20)"] = 20,
            page: Annotated[int, "Page number for pagination when node has many connections"] = 1
        ) -> dict:
            """Get all nodes directly connected to this node, with their relationship types."""
            try:
                self._validate_node_exists(node_id)
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                neighbors = self.knowledge_graph.get_neighbors(node_id)
                if not neighbors:
                    return {
                        "node_id": node_id,
                        "neighbors": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No neighbors found for node '{node_id}'"
                    }

                total = len(neighbors)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "get_neighbors")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = neighbors[start_idx:end_idx]

                neighbor_list = []
                for neighbor in page_slice:
                    neighbor_info = {
                        "id": neighbor.id,
                        "name": getattr(neighbor, 'name', 'Unknown'),
                        "type": neighbor.node_type,
                        "relation": None
                    }

                    if self.knowledge_graph.graph.has_edge(node_id, neighbor.id):
                        edge_data = self.knowledge_graph.graph.get_edge_data(node_id, neighbor.id)
                        neighbor_info["relation"] = edge_data.get('relation', 'Unknown')
                        neighbor_info["direction"] = "outgoing"
                    elif self.knowledge_graph.graph.has_edge(neighbor.id, node_id):
                        edge_data = self.knowledge_graph.graph.get_edge_data(neighbor.id, node_id)
                        neighbor_info["relation"] = edge_data.get('relation', 'Unknown')
                        neighbor_info["direction"] = "incoming"

                    neighbor_list.append(neighbor_info)

                text = f"Neighbors of '{node_id}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, neighbor in enumerate(neighbor_list, start=start_idx + 1):
                    text += f"{i}. {neighbor['id']}\n"
                    text += f"   Name: {neighbor['name']}\n"
                    text += f"   Type: {neighbor['type']}\n"
                    if neighbor['relation']:
                        arrow = "→" if neighbor.get('direction') == "outgoing" else "←"
                        text += f"   {arrow} Relation: {neighbor['relation']}\n"
                    text += "\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "node_id": node_id,
                    "total_neighbors": total,
                    "count": len(neighbor_list),
                    "page": page,
                    "total_pages": total_pages,
                    "neighbors": neighbor_list,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_neighbors")
            except Exception as e:
                return self._handle_error(e, "get_neighbors")

        @self.app.tool(
            description="""Jump to the source code location(s) where an entity is defined/declared.

            PURPOSE:
            Use this tool to find WHERE in the codebase a class, function, method, or variable
            is defined. Returns the actual code content of the definition along with file location.

            WHEN TO USE:
            - To see the implementation of a class like 'BertModel' or 'GPT2Attention'
            - To find where a function is defined when you know its name
            - To examine the source code of any entity found through search or listing
            - When you need to understand HOW something is implemented (not just WHERE it's used)
            - To get the actual code definition for analysis or explanation

            WHAT YOU'LL GET:
            - Entity type (class, function, method, variable)
            - Data type if available
            - List of all locations where the entity is declared (some entities may be defined in multiple places)
            - For each location: file path, chunk order, and FULL CODE CONTENT

            TYPICAL WORKFLOW:
            1. search_nodes("attention") -> find entity names
            2. go_to_definition("BertSelfAttention") -> see the class implementation
            3. find_usages("BertSelfAttention") -> see where it's used

            COMPARISON WITH find_usages:
            - go_to_definition: Shows WHERE entity is DEFINED (the implementation)
            - find_usages: Shows WHERE entity is USED/CALLED (the consumers)
            """
        )
        @observe(as_type='tool')
        async def go_to_definition(
            entity_name: Annotated[str, "Exact name of the entity (case-sensitive). Examples: 'BertModel', 'forward', 'attention_mask'"]
        ) -> dict:
            """Find where an entity is declared/defined in the codebase."""
            try:
                self._validate_entity_exists(entity_name)

                entity_info = self.knowledge_graph.entities[entity_name]
                declaring_chunks = entity_info.get('declaring_chunk_ids', [])

                if not declaring_chunks:
                    return {
                        "entity_name": entity_name,
                        "declarations": [],
                        "text": f"Entity '{entity_name}' found but no declarations identified."
                    }

                declarations = []
                for chunk_id in declaring_chunks[:5]:
                    if chunk_id in self.knowledge_graph.graph:
                        chunk = self.knowledge_graph.graph.nodes[chunk_id]['data']
                        declarations.append({
                            "chunk_id": chunk_id,
                            "file_path": chunk.path,
                            "order_in_file": chunk.order_in_file,
                            "content": chunk.content  # Full content, not preview
                        })

                text = f"Definition(s) for '{entity_name}':\n\n"
                text += f"Type: {', '.join(entity_info.get('type', ['Unknown']))}\n"
                if entity_info.get('dtype'):
                    text += f"Data Type: {entity_info['dtype']}\n"
                text += f"\nDeclared in {len(declaring_chunks)} location(s):\n\n"

                for i, decl in enumerate(declarations, 1):
                    text += f"{i}. Chunk: {decl['chunk_id']}\n"
                    text += f"   File: {decl['file_path']}\n"
                    text += f"   Order: {decl['order_in_file']}\n"
                    text += f"   Content:\n{decl['content']}\n\n"

                if len(declaring_chunks) > 5:
                    text += f"... and {len(declaring_chunks) - 5} more locations\n"

                return {
                    "entity_name": entity_name,
                    "type": entity_info.get('type', []),
                    "dtype": entity_info.get('dtype'),
                    "total_declarations": len(declaring_chunks),
                    "declarations": declarations,
                    "has_more": len(declaring_chunks) > 5,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "go_to_definition")
            except Exception as e:
                return self._handle_error(e, "go_to_definition")

        @self.app.tool(
            description="""Find all locations in the codebase where an entity is used or called.

            PURPOSE:
            Use this tool to understand the impact and usage patterns of any entity.
            Shows every place where a class is instantiated, a function is called,
            or a variable is referenced throughout the codebase.

            WHEN TO USE:
            - To understand how widely used a class or function is
            - To see usage examples of a particular API or function
            - To assess the impact of changing an entity (who depends on it?)
            - To learn how to use a class/function by seeing real examples
            - To trace data flow through the codebase

            WHAT YOU'LL GET:
            - Total count of usage locations
            - For each usage: file path, chunk position, and full code context showing the usage
            - Paginated results for entities with many usages

            TYPICAL WORKFLOWS:

            Impact Analysis:
            1. go_to_definition("deprecated_function") -> understand what it does
            2. find_usages("deprecated_function") -> see all code that needs updating

            Learning by Example:
            1. list_nodes_by_type('class') -> find interesting classes
            2. find_usages("BertModel") -> see how it's instantiated and used

            COMPARISON WITH go_to_definition:
            - find_usages: WHERE is this entity CALLED/USED (consumers)
            - go_to_definition: WHERE is this entity DEFINED (implementation)
            """
        )
        @observe(as_type='tool')
        async def find_usages(
            entity_name: Annotated[str, "Exact name of the entity to find usages for (case-sensitive)"],
            limit: Annotated[int, "Usages per page (default: 20). Many popular classes have 100+ usages."] = 20,
            page: Annotated[int, "Page number for pagination (starts at 1)"] = 1
        ) -> dict:
            """Find where an entity is used/called in the codebase."""
            try:
                self._validate_entity_exists(entity_name)
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                entity_info = self.knowledge_graph.entities[entity_name]
                calling_chunks = entity_info.get('calling_chunk_ids', [])

                if not calling_chunks:
                    return {
                        "entity_name": entity_name,
                        "usages": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"Entity '{entity_name}' found but no usages identified."
                    }

                total = len(calling_chunks)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "find_usages")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = calling_chunks[start_idx:end_idx]

                usages = []
                for chunk_id in page_slice:
                    if chunk_id in self.knowledge_graph.graph:
                        chunk = self.knowledge_graph.graph.nodes[chunk_id]['data']
                        usages.append({
                            "chunk_id": chunk_id,
                            "file_path": chunk.path,
                            "order_in_file": chunk.order_in_file,
                            "content": chunk.content  # Full content
                        })

                text = f"Usages of '{entity_name}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, usage in enumerate(usages, start=start_idx + 1):
                    text += f"{i}. {usage['file_path']} (chunk {usage['order_in_file']})\n"
                    text += f"   Content:\n{usage['content']}\n\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "entity_name": entity_name,
                    "total_usages": total,
                    "count": len(usages),
                    "page": page,
                    "total_pages": total_pages,
                    "usages": usages,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "find_usages")
            except Exception as e:
                return self._handle_error(e, "find_usages")

        @self.app.tool(
            description="""Get a structural overview of a source file showing its chunks and declared entities.

            PURPOSE:
            Use this tool to understand the organization of a specific file. Shows what classes,
            functions, and other entities are defined in the file, plus how the file is divided into chunks.

            WHEN TO USE:
            - To get a table of contents for a file before diving into specifics
            - To see what classes and functions a file defines
            - To understand how code is organized within a file
            - To find chunk IDs for further exploration with get_node_info or get_chunk_context
            - When you know the file path but need to understand its contents

            WHAT YOU'LL SEE:
            - File path and detected programming language
            - Total number of code chunks in the file
            - List of declared entities (classes, functions, methods, variables) with their types
            - Ordered list of chunks with their IDs and descriptions

            HOW TO GET FILE PATHS:
            - Use list_files_in_directory() to browse files
            - Use search_nodes() and look at file paths in results
            - Use list_nodes_by_type('file') to get file node IDs (which are the paths)

            TYPICAL WORKFLOW:
            1. list_files_in_directory('src/models/bert') -> find files
            2. get_file_structure('src/models/bert/modeling_bert.py') -> see structure
            3. get_node_info(chunk_id) -> examine specific code chunks
            """
        )
        @observe(as_type='tool')
        async def get_file_structure(
            file_path: Annotated[str, "The full path to the file. Must match exactly as stored in the knowledge graph."]
        ) -> dict:
            """Get an overview of chunks and entities in a specific file."""
            try:
                self._validate_node_exists(file_path)

                file_node = self.knowledge_graph.graph.nodes[file_path]['data']
                chunks = self.knowledge_graph.get_chunks_of_file(file_path)

                declared_entities = []
                if hasattr(file_node, 'declared_entities') and file_node.declared_entities:
                    for entity in file_node.declared_entities[:15]:
                        if isinstance(entity, dict):
                            declared_entities.append({
                                "name": entity.get('name', '?'),
                                "type": entity.get('type', '?')
                            })
                        else:
                            declared_entities.append({"name": str(entity), "type": "Unknown"})

                chunk_list = []
                for chunk in chunks[:10]:
                    chunk_list.append({
                        "id": chunk.id,
                        "order_in_file": chunk.order_in_file,
                        "description": chunk.description[:80] + "..." if chunk.description and len(chunk.description) > 80 else chunk.description
                    })

                text = f"File Structure: {file_node.name}\n"
                text += f"Path: {file_path}\n"
                text += f"Language: {getattr(file_node, 'language', 'Unknown')}\n"
                text += f"Total Chunks: {len(chunks)}\n\n"

                if declared_entities:
                    text += f"Declared Entities ({len(file_node.declared_entities)}):\n"
                    for entity in declared_entities:
                        text += f"  - {entity['name']} ({entity['type']})\n"
                    if len(file_node.declared_entities) > 15:
                        text += f"  ... and {len(file_node.declared_entities) - 15} more\n"

                text += f"\nChunks:\n"
                for chunk_info in chunk_list:
                    text += f"  [{chunk_info['order_in_file']}] {chunk_info['id']}\n"
                    if chunk_info['description']:
                        text += f"      {chunk_info['description']}\n"

                if len(chunks) > 10:
                    text += f"  ... and {len(chunks) - 10} more chunks\n"

                return {
                    "file_path": file_path,
                    "file_name": file_node.name,
                    "language": getattr(file_node, 'language', 'Unknown'),
                    "total_chunks": len(chunks),
                    "total_declared_entities": len(file_node.declared_entities) if hasattr(file_node, 'declared_entities') else 0,
                    "declared_entities": declared_entities,
                    "chunks": chunk_list,
                    "has_more_entities": hasattr(file_node, 'declared_entities') and len(file_node.declared_entities) > 15,
                    "has_more_chunks": len(chunks) > 10,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_file_structure")
            except Exception as e:
                return self._handle_error(e, "get_file_structure")

        @self.app.tool(
            description="""Find code chunks connected to a given chunk through a specific relationship type.

            PURPOSE:
            Use this tool to trace code dependencies by following relationship edges from a chunk.
            Helps understand what code a chunk depends on or what depends on it.

            WHEN TO USE:
            - To find what entities/code a chunk calls or uses (relation_type='calls')
            - To trace dependencies from a specific piece of code
            - To explore the call graph emanating from a chunk
            - When you have a chunk ID and want to see connected code

            RELATIONSHIP TYPES:
            - 'calls': Entities/chunks that this chunk calls or references (most common)
            - 'contains': Child nodes contained by this node (for files/directories)
            - 'declares': Entities declared by this chunk
            - 'all' or '': Get all outgoing relationships regardless of type

            TYPICAL WORKFLOW:
            1. search_nodes("BertAttention forward") -> find a chunk
            2. get_related_chunks(chunk_id, 'calls') -> see what it calls
            3. get_node_info(related_chunk_id) -> examine called code

            COMPARISON WITH OTHER TOOLS:
            - get_neighbors: All connected nodes (any direction, any type)
            - get_related_chunks: Outgoing edges only, filtered by relationship type
            - entity_relationships: Focused on entity nodes and their relationships
            """
        )
        @observe(as_type='tool')
        async def get_related_chunks(
            chunk_id: Annotated[str, "The ID of the chunk to explore from (e.g., 'src/file.py::chunk_5')"],
            relation_type: Annotated[str, "Filter by relationship type: 'calls', 'contains', 'declares', or 'all' for everything (default: 'calls')"] = "calls",
            limit: Annotated[int, "Maximum results per page (default: 20)"] = 20,
            page: Annotated[int, "Page number for pagination"] = 1
        ) -> dict:
            """Get chunks related to this chunk by a specific relationship (e.g., 'calls', 'contains')."""
            try:
                self._validate_node_exists(chunk_id)
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                related = []
                if relation_type == "" or relation_type == "all":
                    # Get all outgoing edges regardless of relation type
                    for _, target, attrs in self.knowledge_graph.graph.out_edges(chunk_id, data=True):
                        target_node = self.knowledge_graph.graph.nodes[target]['data']
                        related.append({
                            "id": target,
                            "file_path": getattr(target_node, 'path', 'Unknown'),
                            "entity_name": attrs.get('entity_name'),
                            "relation": attrs.get('relation', 'Unknown')
                        })
                else:
                    for _, target, attrs in self.knowledge_graph.graph.out_edges(chunk_id, data=True):
                        if attrs.get('relation') == relation_type:
                            target_node = self.knowledge_graph.graph.nodes[target]['data']
                            related.append({
                                "id": target,
                                "file_path": getattr(target_node, 'path', 'Unknown'),
                                "entity_name": attrs.get('entity_name'),
                                "relation": relation_type
                            })

                if not related:
                    return {
                        "chunk_id": chunk_id,
                        "relation_type": relation_type,
                        "related_chunks": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No chunks found with '{relation_type}' relationship from '{chunk_id}'"
                    }

                total = len(related)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "get_related_chunks")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = related[start_idx:end_idx]

                text = f"Chunks related to '{chunk_id}' via '{relation_type}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, chunk in enumerate(page_slice, start=start_idx + 1):
                    text += f"{i}. {chunk['id']}\n"
                    text += f"   File: {chunk['file_path']}\n"
                    if chunk['entity_name']:
                        text += f"   Entity: {chunk['entity_name']}\n"
                    if relation_type == "" or relation_type == "all":
                        text += f"   Relation: {chunk['relation']}\n"
                    text += "\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "chunk_id": chunk_id,
                    "relation_type": relation_type,
                    "total_related": total,
                    "count": len(page_slice),
                    "page": page,
                    "total_pages": total_pages,
                    "related_chunks": page_slice,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_related_chunks")
            except Exception as e:
                return self._handle_error(e, "get_related_chunks")

        @self.app.tool(
            description="""Browse all programming entities (classes, functions, methods, variables) tracked in the knowledge graph.

            PURPOSE:
            Use this tool to explore the full inventory of code entities in the codebase.
            Supports filtering by type and usage patterns, making it powerful for targeted exploration.

            WHEN TO USE:
            - To browse all classes, functions, or methods in the codebase
            - To find entities that are defined but never used (dead code analysis)
            - To find external entities that are called but not defined in the repo
            - To get an overview of entity distribution in the codebase
            - When you need entity names for use with go_to_definition or find_usages

            FILTERING OPTIONS:

            By entity_type:
            - 'class': Class definitions
            - 'function': Standalone functions
            - 'method': Class methods
            - 'variable': Variable declarations
            - 'parameter': Function/method parameters
            - None: All entity types

            By declaration status (declared_in_repo):
            - True: Only entities DEFINED in this repo (has source code)
            - False: Only external entities (imported from other packages)
            - None: All entities

            By usage status (called_in_repo):
            - True: Only entities that ARE USED somewhere in the code
            - False: Only entities that are NEVER USED (potential dead code)
            - None: All entities

            USEFUL FILTER COMBINATIONS:
            - All classes: entity_type='class'
            - Defined classes: entity_type='class', declared_in_repo=True
            - Unused functions: entity_type='function', called_in_repo=False
            - External dependencies: declared_in_repo=False, called_in_repo=True
            """
        )
        @observe(as_type='tool')
        async def list_all_entities(
            limit: Annotated[int, "Entities per page (default: 50). Use larger values for comprehensive listings."] = 50,
            page: Annotated[int, "Page number starting from 1 for pagination"] = 1,
            entity_type: Annotated[Optional[str], "Filter by type: 'class', 'function', 'method', 'variable', 'parameter', or None for all"] = None,
            declared_in_repo: Annotated[Optional[bool], "True=defined in repo, False=external only, None=all"] = None,
            called_in_repo: Annotated[Optional[bool], "True=has usages, False=never used, None=all"] = None
        ) -> dict:
            """List all entities tracked in the knowledge graph with their metadata."""
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")
                
                if not self.knowledge_graph.entities:
                    return {
                        "entities": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": "No entities found in the knowledge graph."
                    }

                # Filter entities based on criteria
                filtered_entities = {}
                for entity_name, info in self.knowledge_graph.entities.items():
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

                    # Filter by called_in_repo (usages) if specified
                    if called_in_repo is not None:
                        has_calls = len(info.get('calling_chunk_ids', [])) > 0
                        if called_in_repo and not has_calls:
                            continue
                        if not called_in_repo and has_calls:
                            continue

                    filtered_entities[entity_name] = info

                # Build the response with filtered entities
                if not filtered_entities:
                    filter_desc = []
                    if entity_type:
                        filter_desc.append(f"type={entity_type}")
                    if declared_in_repo is not None:
                        filter_desc.append(f"declared_in_repo={declared_in_repo}")
                    if called_in_repo is not None:
                        filter_desc.append(f"called_in_repo={called_in_repo}")
                    filter_text = f" (filtered by {', '.join(filter_desc)})" if filter_desc else ""
                    return {
                        "entities": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No entities found{filter_text}."
                    }

                # Calculate pagination
                total_entities = len(filtered_entities)
                total_pages = (total_entities + limit - 1) // limit
                
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "list_all_entities")
                
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                
                # Get the paginated slice of entities
                entity_items = list(filtered_entities.items())
                paginated_items = entity_items[start_idx:end_idx]

                entities = []
                for entity_name, info in paginated_items:
                    entities.append({
                        "name": entity_name,
                        "types": info.get('type', ['Unknown']),
                        "declaration_count": len(info.get('declaring_chunk_ids', [])),
                        "usage_count": len(info.get('calling_chunk_ids', []))
                    })

                text = f"All Entities (Page {page}/{total_pages}, {total_entities} total):\n\n"
                for i, entity in enumerate(entities, start=start_idx + 1):
                    text += f"{i}. {entity['name']}\n"
                    text += f"   Types: {', '.join(entity['types'])}\n"
                    text += f"   Declarations: {entity['declaration_count']}\n"
                    text += f"   Usages: {entity['usage_count']}\n\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                # Add filter information
                if entity_type or declared_in_repo is not None or called_in_repo is not None:
                    text += "\nFilters applied:\n"
                    if entity_type:
                        text += f"  - entity_type: {entity_type}\n"
                    if declared_in_repo is not None:
                        text += f"  - declared_in_repo: {declared_in_repo}\n"
                    if called_in_repo is not None:
                        text += f"  - called_in_repo: {called_in_repo}\n"

                return {
                    "total_entities": total_entities,
                    "count": len(entities),
                    "page": page,
                    "total_pages": total_pages,
                    "entities": entities,
                    "has_more": page < total_pages,
                    "filters": {
                        "entity_type": entity_type,
                        "declared_in_repo": declared_in_repo,
                        "called_in_repo": called_in_repo
                    },
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "list_all_entities")
            except Exception as e:
                return self._handle_error(e, "list_all_entities")

        # --- New Tools ---
        @self.app.tool(
            description="""Compare two code chunks and show their differences in unified diff format.

            PURPOSE:
            Use this tool to compare two pieces of code side-by-side. Shows exactly what's
            different between them using standard unified diff format (like git diff).

            WHEN TO USE:
            - To compare similar implementations (e.g., two attention mechanisms)
            - To understand differences between related classes or functions
            - To analyze variations in code patterns across the codebase
            - To compare two versions or implementations of similar functionality
            - When you suspect code duplication and want to see exact differences

            DIFF FORMAT:
            - Lines starting with '-' are only in the first chunk
            - Lines starting with '+' are only in the second chunk
            - Lines without prefix are common to both
            - @@ markers show line number context

            TYPICAL WORKFLOW:
            1. search_nodes("attention") -> find attention implementations
            2. Get chunk IDs from two different attention classes
            3. diff_chunks(chunk_id_1, chunk_id_2) -> compare implementations
            """
        )
        @observe(as_type='tool')
        async def diff_chunks(
            node_id_1: Annotated[str, "ID of the first chunk/node to compare"],
            node_id_2: Annotated[str, "ID of the second chunk/node to compare"]
        ) -> dict:
            try:
                import difflib
                self._validate_node_exists(node_id_1)
                self._validate_node_exists(node_id_2)

                g = self.knowledge_graph.graph
                content1 = getattr(g.nodes[node_id_1]['data'], 'content', None)
                content2 = getattr(g.nodes[node_id_2]['data'], 'content', None)

                if not content1 or not content2:
                    raise InvalidInputError("One or both nodes have no content.")

                diff = list(difflib.unified_diff(
                    content1.splitlines(), content2.splitlines(),
                    fromfile=node_id_1, tofile=node_id_2, lineterm=""
                ))

                diff_text = "\n".join(diff) if diff else "No differences."

                return {
                    "node_id_1": node_id_1,
                    "node_id_2": node_id_2,
                    "has_differences": bool(diff),
                    "diff": diff,
                    "text": diff_text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "diff_chunks")
            except Exception as e:
                return self._handle_error(e, "diff_chunks")

        @self.app.tool(
            description="""Display a hierarchical tree view of the repository structure starting from any node.

            PURPOSE:
            Use this tool to visualize the structure of the codebase. Shows parent-child relationships
            in a familiar tree format, helping you understand how files and directories are organized.

            WHEN TO USE:
            - To explore the directory structure of the repository
            - To see what's inside a specific directory (use directory as root_id)
            - To understand the file organization for a component
            - To get an overview of the codebase hierarchy
            - When you need to understand where files are located

            TREE VISUALIZATION:
            - Each level shows node name and type (repo, directory, file, chunk)
            - Indentation represents depth in the hierarchy
            - Children are limited to prevent overwhelming output

            TIPS:
            - Start with max_depth=2 for a high-level overview
            - Increase max_depth to see more detail (but output gets larger)
            - Use a directory path as root_id to focus on a specific area
            - Use list_files_in_directory for more detailed file listings
            """
        )
        @observe(as_type='tool')
        async def print_tree(
            root_id: Annotated[Optional[str], "Starting node ID. Use 'root' for repository root, or a directory/file path to start from a specific location."] = 'root',
            max_depth: Annotated[int, "How many levels deep to show (default: 3). Higher values show more detail but larger output."] = 3
        ) -> dict:
            try:
                g = self.knowledge_graph.graph

                def build_tree(node_id, depth, tree_data):
                    if depth > max_depth:
                        return
                    node = g.nodes[node_id]['data']
                    node_info = {
                        "id": node_id,
                        "name": getattr(node, 'name', node_id),
                        "type": getattr(node, 'node_type', '?'),
                        "depth": depth,
                        "children": []
                    }
                    tree_data.append(node_info)
                    children = [t for s, t in g.out_edges(node_id)]
                    for child in children:
                        build_tree(child, depth + 1, node_info["children"])

                def format_tree(tree_data):
                    result = ""
                    for node in tree_data:
                        result += "  " * node["depth"] + f"- {node['name']} ({node['type']})\n"
                        for child in node["children"]:
                            result += format_subtree(child)
                    return result

                def format_subtree(node):
                    result = "  " * node["depth"] + f"- {node['name']} ({node['type']})\n"
                    for child in node["children"]:
                        result += format_subtree(child)
                    return result

                if root_id is None:
                    roots = [n for n, d in g.nodes(data=True) if getattr(d['data'], 'node_type', None) in ('repo', 'directory', 'file')]
                    root_id = roots[0] if roots else list(g.nodes)[0]

                self._validate_node_exists(root_id)

                tree_data = []
                build_tree(root_id, 0, tree_data)

                return {
                    "root_id": root_id,
                    "max_depth": max_depth,
                    "tree": tree_data,
                    "text": format_tree(tree_data)
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "print_tree")
            except Exception as e:
                return self._handle_error(e, "print_tree")

        @self.app.tool(
            description="""Display all incoming and outgoing relationships for any node, with relationship types.

            PURPOSE:
            Use this tool to get a complete picture of how a node connects to the rest of the
            knowledge graph. Shows both what points TO this node and what this node points TO.

            WHEN TO USE:
            - To understand all dependencies of an entity
            - To see what declares or calls a specific entity
            - To trace the full relationship network around any node
            - When you need more detail than get_neighbors provides about relationship types
            - For entity-centric analysis (understanding a class or function's connections)

            WHAT YOU'LL SEE:
            - Incoming relationships: Other nodes that have edges pointing TO this node
              (e.g., chunks that CALL this function, files that CONTAIN this chunk)
            - Outgoing relationships: This node's edges pointing TO other nodes
              (e.g., entities this chunk CALLS, chunks this file CONTAINS)
            - Relationship types for each edge (calls, declares, contains)

            COMPARISON WITH SIMILAR TOOLS:
            - get_node_edges: Same information but different formatting
            - get_neighbors: Shows neighbor node details, not edge details
            - get_related_chunks: Filtered by relationship type, chunks only
            """
        )
        @observe(as_type='tool')
        async def entity_relationships(
            node_id: Annotated[str, "The ID of any node (entity, chunk, file, directory)"]
        ) -> dict:
            try:
                self._validate_node_exists(node_id)
                g = self.knowledge_graph.graph

                incoming = []
                outgoing = []

                for source, target, data in g.in_edges(node_id, data=True):
                    incoming.append({
                        "source": source,
                        "target": target,
                        "relation": data.get('relation', '?')
                    })

                for source, target, data in g.out_edges(node_id, data=True):
                    outgoing.append({
                        "source": source,
                        "target": target,
                        "relation": data.get('relation', '?')
                    })

                text = f"Relationships for '{node_id}':\n"
                for rel in incoming:
                    text += f"← {rel['source']} [{rel['relation']}]\n"
                for rel in outgoing:
                    text += f"→ {rel['target']} [{rel['relation']}]\n"

                if not incoming and not outgoing:
                    text = "No relationships found."

                return {
                    "node_id": node_id,
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "incoming_count": len(incoming),
                    "outgoing_count": len(outgoing),
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "entity_relationships")
            except Exception as e:
                return self._handle_error(e, "entity_relationships")

        @self.app.tool(
            description="""Search for nodes by combining type filtering with name pattern matching.

            PURPOSE:
            Use this tool for precise, targeted searches when you know the type of node you're looking
            for and have a partial name. More efficient than list_nodes_by_type when you have name hints.

            WHEN TO USE:
            - To find all classes containing 'Attention': search_by_type_and_name('class', 'Attention')
            - To find functions with 'forward' in name: search_by_type_and_name('function', 'forward')
            - To find files named 'config': search_by_type_and_name('file', 'config')
            - When you know the type AND have a partial name to search for
            - For pattern-based discovery of related components

            SEARCH BEHAVIOR:
            - Case-insensitive matching
            - partial_allowed=True (default): Fuzzy matching, finds 'BertEmbeddings' when searching 'Embed'
            - partial_allowed=False: Requires exact substring match
            - Results sorted by match quality (exact matches first, then substring, then fuzzy)

            VALID node_type VALUES:
            For entities: 'class', 'function', 'method', 'variable', 'parameter'
            For structural: 'file', 'chunk', 'directory'

            COMPARISON WITH SIMILAR TOOLS:
            - search_nodes: Full-text search in code content (doesn't filter by type)
            - list_nodes_by_type: Lists all of a type (no name filter)
            - search_by_type_and_name: Combines type filter + name search (best of both)
            """
        )
        @observe(as_type='tool')
        async def search_by_type_and_name(
            node_type: Annotated[str, "Type to filter by: 'class', 'function', 'method', 'file', 'chunk', 'directory', etc."],
            name_query: Annotated[str, "Name pattern to search for (case-insensitive). Can be partial."],
            limit: Annotated[int, "Results per page (default: 10)"] = 10,
            page: Annotated[int, "Page number for pagination"] = 1,
            partial_allowed: Annotated[bool, "Enable fuzzy matching (default: True). Set False for stricter matching."] = True
        ) -> dict:
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                g = self.knowledge_graph.graph
                matches = []
                query_lower = name_query.lower()
                
                # Build regex pattern for partial_allowed matching
                if partial_allowed:
                    partial_pattern = '.*'.join(re.escape(c) for c in query_lower)
                    partial_regex = re.compile(partial_pattern, re.IGNORECASE)
                
                for nid, n in g.nodes(data=True):
                    node = n['data']
                    node_name = getattr(node, 'name', '')
                    
                    if not node_name:
                        continue
                    
                    # Check if name matches the query
                    name_matches = False
                    if partial_allowed:
                        if query_lower in node_name.lower() or partial_regex.search(node_name):
                            name_matches = True
                    else:
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
                        if not entity_type and nid in self.knowledge_graph.entities:
                            entity_types = self.knowledge_graph.entities[nid].get('type', [])
                            entity_type = entity_types[0] if entity_types else ''
                        
                        if entity_type and entity_type.lower() == node_type.lower():
                            score = 0 if query_lower == node_name.lower() else (1 if query_lower in node_name.lower() else 2)
                            matches.append({
                                "id": nid,
                                "name": node_name,
                                "type": f"entity ({entity_type})",
                                "content": getattr(node, 'content', None),
                                "score": score
                            })
                    # For other nodes, check node_type directly
                    elif current_node_type == node_type:
                        score = 0 if query_lower == node_name.lower() else (1 if query_lower in node_name.lower() else 2)
                        matches.append({
                            "id": nid,
                            "name": node_name,
                            "type": current_node_type,
                            "content": getattr(node, 'content', None),
                            "score": score
                        })
                
                # Sort by match score (best matches first)
                matches.sort(key=lambda x: (x['score'], x['name'].lower()))

                total = len(matches)
                if total == 0:
                    return {
                        "node_type": node_type,
                        "name_query": name_query,
                        "matches": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No matches for type '{node_type}' and name containing '{name_query}'."
                    }

                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "search_by_type_and_name")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = matches[start_idx:end_idx]

                text = f"Matches for type '{node_type}' and name '{name_query}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, match in enumerate(page_slice, start=start_idx + 1):
                    text += f"{i}. {match['name']}\n"
                    text += f"   ID: {match['id']}\n"
                    text += f"   Type: {match['type']}\n\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "node_type": node_type,
                    "name_query": name_query,
                    "count": len(page_slice),
                    "total": total,
                    "page": page,
                    "total_pages": total_pages,
                    "matches": page_slice,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "search_by_type_and_name")
            except Exception as e:
                return self._handle_error(e, "search_by_type_and_name")

        @self.app.tool(
            description="""Get expanded code context by retrieving a chunk along with its previous and next chunks.

            PURPOSE:
            Use this tool when you need to see MORE CODE CONTEXT around a specific chunk.
            Chunks are logical code segments, but sometimes you need to see surrounding code
            to fully understand the implementation.

            WHEN TO USE:
            - After search_nodes or get_node_info when you need more surrounding context
            - When a chunk shows a partial function/class and you need the complete picture
            - To understand code flow across chunk boundaries
            - To see imports or setup code that precedes a chunk
            - To see what code follows after a chunk

            WHAT YOU'LL GET:
            - The previous chunk's content (if it exists)
            - The target chunk's content
            - The next chunk's content (if it exists)
            - All organized by file and joined together seamlessly

            CONTEXT EXPANSION:
            - Shows up to 3 consecutive chunks (prev + current + next)
            - Useful for understanding function bodies that span chunks
            - Helps see class context when looking at individual methods

            COMPARISON WITH get_node_info:
            - get_node_info: Single chunk content + full metadata
            - get_chunk_context: Expanded code view (prev + current + next chunks), less metadata
            """
        )
        @observe(as_type='tool')
        async def get_chunk_context(
            node_id: Annotated[str, "The chunk ID to get context for (e.g., 'src/file.py::chunk_5')"]
        ) -> dict:
            from .utils.chunk_utils import organize_chunks_by_file_name, join_organized_chunks
            try:
                self._validate_node_exists(node_id)

                g = self.knowledge_graph.graph
                current_chunk = g.nodes[node_id]['data']
                previous_chunk = self.knowledge_graph.get_previous_chunk(node_id)
                next_chunk = self.knowledge_graph.get_next_chunk(node_id)

                # Collect all chunks (previous, current, next)
                chunks = []
                prev_info = None
                next_info = None
                current_info = {
                    "id": node_id,
                    "content": getattr(current_chunk, 'content', '')
                }

                if previous_chunk:
                    prev_info = {
                        "id": previous_chunk.id,
                        "content": previous_chunk.content
                    }
                    chunks.append(previous_chunk)

                chunks.append(current_chunk)

                if next_chunk:
                    next_info = {
                        "id": next_chunk.id,
                        "content": next_chunk.content
                    }
                    chunks.append(next_chunk)

                # Organize and join chunks
                organized = organize_chunks_by_file_name(chunks)
                full_content = join_organized_chunks(organized)

                return {
                    "node_id": node_id,
                    "current_chunk": current_info,
                    "previous_chunk": prev_info,
                    "next_chunk": next_info,
                    "text": full_content
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_chunk_context")
            except Exception as e:
                return self._handle_error(e, "get_chunk_context")

        @self.app.tool(
            description="""Get detailed statistics and metrics for a specific file or directory.

            PURPOSE:
            Use this tool to get quantitative metrics about a file including line counts,
            entity counts, and chunk counts. Useful for understanding file complexity.

            WHEN TO USE:
            - To assess the size and complexity of a file
            - To see summary counts of entities declared and called
            - To understand how a file is chunked
            - For code metrics and analysis tasks
            - When deciding which files to explore further

            METRICS PROVIDED:
            - Line count (total lines in the file)
            - Declared entities count with a sample list
            - Called entities count with a sample list
            - Number of chunks the file is divided into

            COMPARISON WITH get_file_structure:
            - get_file_stats: Quantitative metrics (counts, numbers)
            - get_file_structure: Qualitative overview (entity names, chunk IDs)
            """
        )
        @observe(as_type='tool')
        async def get_file_stats(
            path: Annotated[str, "The file path to analyze. Must match the path as stored in the knowledge graph."]
        ) -> dict:
            try:
                g = self.knowledge_graph.graph
                nodes = [n for n, d in g.nodes(data=True) if getattr(d['data'], 'path', None) == path]

                if not nodes:
                    raise NodeNotFoundError(f"No nodes found for path '{path}'.")

                stats = []
                text = f"Statistics for '{path}':\n"

                for node_id in nodes:
                    node = g.nodes[node_id]['data']
                    content = getattr(node, 'content', '')
                    declared = getattr(node, 'declared_entities', [])
                    called = getattr(node, 'called_entities', [])
                    chunks = [t for s, t in g.out_edges(node_id) if getattr(g.nodes[t]['data'], 'node_type', None) == 'chunk']

                    declared_list = []
                    for entity in declared[:10]:
                        if isinstance(entity, dict):
                            declared_list.append({
                                "name": entity.get('name', '?'),
                                "type": entity.get('type', '?')
                            })
                        else:
                            declared_list.append({"name": str(entity), "type": "Unknown"})

                    called_list = [str(entity) for entity in called[:10]]

                    node_stats = {
                        "node_id": node_id,
                        "node_type": getattr(node, 'node_type', '?'),
                        "lines": len(content.splitlines()) if content else 0,
                        "declared_entities_count": len(declared),
                        "declared_entities": declared_list,
                        "called_entities_count": len(called),
                        "called_entities": called_list,
                        "chunks_count": len(chunks),
                        "has_more_declared": len(declared) > 10,
                        "has_more_called": len(called) > 10
                    }
                    stats.append(node_stats)

                    text += f"- Node: {node_id} ({node_stats['node_type']})\n"
                    text += f"  Lines: {node_stats['lines']}\n"

                    if declared_list:
                        text += f"  Declared entities ({len(declared)}):\n"
                        for entity in declared_list:
                            text += f"    - {entity['name']} ({entity['type']})\n"
                        if len(declared) > 10:
                            text += f"    ... and {len(declared) - 10} more\n"
                    else:
                        text += f"  Declared entities: 0\n"

                    if called_list:
                        text += f"  Called entities ({len(called)}):\n"
                        for entity in called_list:
                            text += f"    - {entity}\n"
                        if len(called) > 10:
                            text += f"    ... and {len(called) - 10} more\n"
                    else:
                        text += f"  Called entities: 0\n"

                    text += f"  Chunks: {len(chunks)}\n"

                return {
                    "path": path,
                    "nodes": stats,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_file_stats")
            except Exception as e:
                return self._handle_error(e, "get_file_stats")
        # --- End New Tools ---
        @self.app.tool(
            description="Search for file names in the repository using a regular expression pattern."
        )
        @observe(as_type='tool')
        async def search_file_names_by_regex(
            pattern: Annotated[str, "The regular expression pattern to match file names."]
        ) -> dict:
            """Search for file names matching a regex pattern."""
            import re
            g = self.knowledge_graph.graph

            try:
                regex = re.compile(pattern)
            except re.error as e:
                return {"error": f"Invalid regex pattern: {str(e)}"}

            matches = []
            for node_id, node_attrs in g.nodes(data=True):
                node = node_attrs['data']
                if getattr(node, 'node_type', None) == 'file':
                    file_name = getattr(node, 'name', '') or getattr(node, 'path', '')
                    if regex.search(file_name):
                        matches.append({
                            "node_id": node_id,
                            "file_name": file_name
                        })

            if not matches:
                return {
                    "pattern": pattern,
                    "matches": [],
                    "text": f"No file names matched the pattern: '{pattern}'"
                }

                text = f"Files matching pattern '{pattern}':\n"
                for match in matches[:20]:
                    text += f"- {match['file_name']} (node ID: {match['node_id']})\n"

                if len(matches) > 20:
                    text += f"... and {len(matches) - 20} more\n"

                return {
                    "pattern": pattern,
                    "count": len(matches),
                    "matches": matches[:20],
                    "has_more": len(matches) > 20,
                    "text": text
                }

        @self.app.tool(
            description="""Find the shortest path between two nodes in the knowledge graph.

            PURPOSE:
            Use this tool to discover how two code elements are connected through the graph.
            Reveals the chain of relationships linking two seemingly unrelated pieces of code.

            WHEN TO USE:
            - To understand how two classes/functions are related
            - To trace dependency chains between components
            - To discover indirect connections between code elements
            - To verify if two nodes are connected at all
            - For understanding code architecture and coupling

            WHAT YOU'LL GET:
            - Path length (number of hops)
            - Ordered list of nodes from source to target
            - Visual representation of the path

            LIMITATIONS:
            - max_depth limits search to avoid long computations
            - If no path found within max_depth, nodes may still be connected via longer path
            - Very distant nodes may require increasing max_depth
            """
        )
        @observe(as_type='tool')
        async def find_path(
            source_id: Annotated[str, "Starting node ID (any node type)"],
            target_id: Annotated[str, "Destination node ID (any node type)"],
            max_depth: Annotated[int, "Maximum path length to search (default: 5). Increase for distant nodes."] = 5
        ) -> dict:
            """Find shortest path between two nodes."""
            try:
                result = self.knowledge_graph.find_path(source_id, target_id, max_depth)
                
                if "error" in result:
                    return result

                if not result.get("path"):
                    return {
                        "source_id": source_id,
                        "target_id": target_id,
                        "path": [],
                        "length": 0,
                        "text": f"No path found from '{source_id}' to '{target_id}' within depth {max_depth}"
                    }

                path = result['path']
                text = f"Path from '{source_id}' to '{target_id}':\n\n"
                text += f"Length: {result['length']}\n\n"
                for i, node_id in enumerate(path):
                    text += f"{i}. {node_id}\n"
                    if i < len(path) - 1:
                        text += "   ↓\n"

                result["text"] = text
                return result
            except Exception as e:
                return self._handle_error(e, "find_path")

        @self.app.tool(
            description="""Extract a local subgraph around a node up to a specified depth.

            PURPOSE:
            Use this tool to get a bounded view of the graph neighborhood around any node.
            Shows all nodes reachable within a certain number of hops, optionally filtered by edge type.

            WHEN TO USE:
            - To understand the local network around a class or function
            - To extract a bounded region of the knowledge graph for analysis
            - To see all nodes within N hops of a target node
            - To analyze the dependency neighborhood of a component
            - When get_neighbors isn't enough and you need multi-hop exploration

            DEPTH EXPLANATION:
            - depth=1: Only immediate neighbors (same as get_neighbors)
            - depth=2: Neighbors and their neighbors (2 hops)
            - depth=3+: Larger neighborhood (exponentially more nodes)

            EDGE TYPE FILTERING:
            - Pass list of edge types to filter: ['calls', 'declares']
            - Common types: 'calls', 'contains', 'declares'
            - Leave empty or None for all edge types
            """
        )
        @observe(as_type='tool')
        async def get_subgraph(
            node_id: Annotated[str, "Central node to build subgraph around"],
            depth: Annotated[int, "Radius in hops from central node (default: 2). Higher = larger subgraph."] = 2,
            edge_types: Annotated[Optional[List[str]], "Optional list of edge types to include: ['calls', 'contains', 'declares'] or None for all"] = None
        ) -> dict:
            """Extract a subgraph around a node."""
            try:
                result = self.knowledge_graph.get_subgraph(node_id, depth, edge_types)
                
                if "error" in result:
                    return result

                text = f"Subgraph around '{node_id}' (depth: {depth}):\n\n"
                text += f"Nodes: {result.get('node_count', 0)}\n"
                text += f"Edges: {result.get('edge_count', 0)}\n"
                
                if edge_types:
                    text += f"Filtered by edge types: {', '.join(edge_types)}\n"
                
                nodes = result.get('nodes', [])
                text += "\nNodes in subgraph:\n"
                for node in nodes[:30]:
                    text += f"  - {node}\n"
                if len(nodes) > 30:
                    text += f"  ... and {len(nodes) - 30} more\n"

                result["text"] = text
                result["has_more"] = len(nodes) > 30
                return result
            except Exception as e:
                return self._handle_error(e, "get_subgraph")

        # --- New Tools from Gradio MCP Space ---
        @self.app.tool(
            description="""Browse and list files in the repository with flexible filtering options.

            PURPOSE:
            Use this tool to explore the file structure of the codebase.
            Supports directory scoping, glob patterns, and recursive/non-recursive modes.

            WHEN TO USE:
            - To see what files exist in a directory
            - To find files by pattern (e.g., all Python files, all test files)
            - To explore the repository structure directory by directory
            - To find specific file types in specific locations
            - When you need file paths for use with other tools

            FILTERING OPTIONS:

            directory_path:
            - Empty string '': Search all files in the repository
            - 'src/models': Only files under this directory
            - 'src/models/bert': Focus on a specific model

            pattern (glob patterns):
            - '*': All files (default)
            - '*.py': Python files only
            - 'test_*.py': Test files
            - '*config*': Files with 'config' in name
            - 'modeling_*.py': Modeling files

            recursive:
            - True (default): Include files in subdirectories
            - False: Only files directly in the specified directory

            COMPARISON WITH print_tree:
            - print_tree: Visual hierarchy, includes directories
            - list_files_in_directory: Flat file list with details, better for finding specific files
            """
        )
        @observe(as_type='tool')
        async def list_files_in_directory(
            directory_path: Annotated[str, "Directory to search in. Empty string for entire repository."] = "",
            pattern: Annotated[str, "Glob pattern for filename filtering (default: '*' matches all)"] = "*",
            recursive: Annotated[bool, "Search subdirectories (default: True)"] = True,
            limit: Annotated[int, "Files per page (default: 50)"] = 50,
            page: Annotated[int, "Page number for pagination"] = 1
        ) -> dict:
            """Browse and list files in the repository with flexible filtering options."""
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                g = self.knowledge_graph.graph
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
                
                # Sort by path for consistent ordering
                matching_files.sort(key=lambda x: x['path'])

                if not matching_files:
                    filter_desc = f" in '{directory_path}'" if directory_path else ""
                    pattern_desc = f" matching '{pattern}'" if pattern and pattern != '*' else ""
                    return {
                        "files": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No files found{filter_desc}{pattern_desc}."
                    }

                total = len(matching_files)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "list_files_in_directory")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = matching_files[start_idx:end_idx]

                text = "Files"
                if directory_path:
                    text += f" in '{directory_path}'"
                if pattern and pattern != '*':
                    text += f" matching '{pattern}'"
                text += f" (Page {page}/{total_pages}, {total} total):\n\n"

                for i, f in enumerate(page_slice, start=start_idx + 1):
                    text += f"{i}. {f['path']}\n"
                    text += f"   Language: {f['language']}, Entities: {f['entity_count']}\n\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "directory_path": directory_path,
                    "pattern": pattern,
                    "recursive": recursive,
                    "count": len(page_slice),
                    "total": total,
                    "page": page,
                    "total_pages": total_pages,
                    "files": page_slice,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "list_files_in_directory")
            except Exception as e:
                return self._handle_error(e, "list_files_in_directory")

        @self.app.tool(
            description="""Find all files that import or use a specific module, class, or function.

            PURPOSE:
            Use this tool to trace import dependencies and understand which parts of the
            codebase depend on a particular module or entity.

            WHEN TO USE:
            - To find all files that import a specific module (e.g., 'torch', 'numpy')
            - To trace dependencies on a class or function
            - To understand the impact scope of a module
            - To find usage patterns of external libraries
            - For dependency analysis and impact assessment

            SEARCH BEHAVIOR:
            - Searches through 'called_entities' metadata
            - Also scans code chunks for import statement patterns
            - Matches import, from...import, require, use patterns
            - Case-insensitive matching

            WHAT YOU'LL GET:
            - List of files that import/use the specified module or entity
            - Match type (called_entity or import_statement)
            - Matched entity names when applicable

            LIMITATIONS:
            - May not catch all dynamic imports
            - Pattern matching may have false positives/negatives
            - For comprehensive search, combine with search_nodes
            """
        )
        @observe(as_type='tool')
        async def find_files_importing(
            module_or_entity: Annotated[str, "Name of the module, class, or function to search for (case-insensitive)"],
            limit: Annotated[int, "Maximum results per page (default: 30)"] = 30,
            page: Annotated[int, "Page number for pagination"] = 1
        ) -> dict:
            """Find all files that import or use a specific module, class, or function."""
            try:
                self._validate_positive_int(limit, "limit")
                if page < 1:
                    raise InvalidInputError("page must be a positive integer (1 or greater)")

                g = self.knowledge_graph.graph
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
                    chunks = self.knowledge_graph.get_chunks_of_file(file_path) if hasattr(self.knowledge_graph, 'get_chunks_of_file') else []
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
                
                # Sort by path
                importing_files.sort(key=lambda x: x['path'])

                if not importing_files:
                    return {
                        "module_or_entity": module_or_entity,
                        "files": [],
                        "count": 0,
                        "page": page,
                        "total_pages": 0,
                        "text": f"No files found importing '{module_or_entity}'.\n\nTip: Try searching for the module name in code content using search_nodes."
                    }

                total = len(importing_files)
                # Pagination
                total_pages = (total + limit - 1) // limit
                if page > total_pages:
                    return self._handle_error(InvalidInputError(f"Page {page} does not exist. Total pages: {total_pages}"), "find_files_importing")

                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                page_slice = importing_files[start_idx:end_idx]

                text = f"Files importing '{module_or_entity}' (Page {page}/{total_pages}, {total} total):\n\n"
                for i, f in enumerate(page_slice, start=start_idx + 1):
                    text += f"{i}. {f['path']}\n"
                    text += f"   Match type: {f['match_type']}\n"
                    if f['matched_entities']:
                        text += f"   Matched: {', '.join(f['matched_entities'][:3])}\n"
                    text += "\n"

                if page < total_pages:
                    text += f"Use page={page + 1} to see the next page\n"

                return {
                    "module_or_entity": module_or_entity,
                    "count": len(page_slice),
                    "total": total,
                    "page": page,
                    "total_pages": total_pages,
                    "files": page_slice,
                    "has_more": page < total_pages,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "find_files_importing")
            except Exception as e:
                return self._handle_error(e, "find_files_importing")

        @self.app.tool(
            description="""Get a high-level overview of how a concept is implemented across the codebase.

            PURPOSE:
            Use this tool for broad exploration of a concept or feature. Aggregates related
            classes, functions, files, and code snippets into a single comprehensive view.
            Ideal for initial investigation of a topic.

            WHEN TO USE:
            - FIRST STEP when exploring a new concept (before detailed searches)
            - To understand how a feature is implemented across the codebase
            - To discover all components related to a concept
            - To get a bird's-eye view before diving into specifics
            - When you're not sure where to start investigating

            SEARCH STRATEGY:
            This tool combines multiple search approaches:
            - Searches entity names (classes, functions, methods) containing the concept
            - Searches file names and paths
            - Searches chunk content and descriptions
            - Aggregates results into categorized sections

            OUTPUT STRUCTURE:
            - Related Classes: Class definitions matching the concept
            - Related Functions/Methods: Functions matching the concept
            - Related Files: Files with concept in path/name
            - Code Snippets: Relevant code chunks
            """
        )
        @observe(as_type='tool')
        async def get_concept_overview(
            concept: Annotated[str, "The concept to explore (e.g., 'attention', 'embedding', 'generation', 'tokenizer')"],
            limit: Annotated[int, "Maximum items per category (default: 15)"] = 15
        ) -> dict:
            """Get a high-level overview of how a concept is implemented across the codebase."""
            try:
                self._validate_positive_int(limit, "limit")

                g = self.knowledge_graph.graph
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
                                'content_preview': content[:200] + "..." if len(content) > 200 else content
                            })

                # Build the overview
                total = len(related_classes) + len(related_functions) + len(related_files) + len(related_chunks)
                
                text = f"Concept Overview: '{concept}'\n"
                text += "=" * 50 + "\n\n"
                text += f"Found {total} related items across the codebase.\n\n"
                
                if related_classes:
                    text += f"📦 Related Classes ({len(related_classes)}):\n"
                    for cls in related_classes[:10]:
                        text += f"  • {cls['name']}\n"
                        text += f"    File: {cls['file']}\n"
                    if len(related_classes) > 10:
                        text += f"  ... and {len(related_classes) - 10} more\n"
                    text += "\n"
                
                if related_functions:
                    text += f"⚡ Related Functions/Methods ({len(related_functions)}):\n"
                    for func in related_functions[:10]:
                        text += f"  • {func['name']} ({func['type']})\n"
                        text += f"    File: {func['file']}\n"
                    if len(related_functions) > 10:
                        text += f"  ... and {len(related_functions) - 10} more\n"
                    text += "\n"
                
                if related_files:
                    text += f"📄 Related Files ({len(related_files)}):\n"
                    for f in related_files[:10]:
                        text += f"  • {f['path']}\n"
                        text += f"    Entities: {f['entity_count']}\n"
                    if len(related_files) > 10:
                        text += f"  ... and {len(related_files) - 10} more\n"
                    text += "\n"
                
                if related_chunks:
                    text += f"📝 Code Snippets ({len(related_chunks)}):\n"
                    for chunk in related_chunks[:5]:
                        text += f"  • {chunk['id']}\n"
                        text += f"    Preview: {chunk['content_preview'][:100]}...\n"
                    if len(related_chunks) > 5:
                        text += f"  ... and {len(related_chunks) - 5} more\n"
                
                if total == 0:
                    text += "No direct matches found.\n\n"
                    text += "Suggestions:\n"
                    text += f"  • Try searching with: search_nodes('{concept}')\n"
                    text += f"  • Try partial name: search_by_type_and_name('class', '{concept[:4]}')\n"
                    text += f"  • Check entity list: list_all_entities(entity_type='class')\n"

                return {
                    "concept": concept,
                    "total_items": total,
                    "related_classes": related_classes,
                    "related_functions": related_functions,
                    "related_files": related_files,
                    "related_chunks": related_chunks,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_concept_overview")
            except Exception as e:
                return self._handle_error(e, "get_concept_overview")

    def run(self, **kwargs):
        self.app.run(**kwargs)
