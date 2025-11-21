import os
from typing import Optional, Annotated
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
            description="Get detailed information about a node in the knowledge graph, including its type, name, description, declared and called entities, and a content preview."
        )
        @observe(as_type='tool')
        async def get_node_info(
                node_id: Annotated[str, "The ID of the node to retrieve information for."]
        ) -> dict:
            try:
                self._validate_node_exists(node_id)
                node = self.knowledge_graph.graph.nodes[node_id]['data']

                declared_entities = getattr(node, 'declared_entities', [])
                called_entities = getattr(node, 'called_entities', [])
                content = getattr(node, 'content', None)
                content_preview = content[:200] + "..." if content and len(content) > 200 else content

                return {
                    "node_id": node_id,
                    "class": node.__class__.__name__,
                    "name": getattr(node, 'name', 'Unknown'),
                    "type": getattr(node, 'node_type', 'Unknown'),
                    "description": getattr(node, 'description', None),
                    "declared_entities": declared_entities,
                    "called_entities": called_entities,
                    "content_preview": content_preview,
                    "text": f"Node {node_id} ({getattr(node, 'name', '?')}) — {getattr(node, 'node_type', '?')} with {len(declared_entities)} declared and {len(called_entities)} called entities."
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_node_info")
            except Exception as e:
                return self._handle_error(e, "get_node_info")

        @self.app.tool(
            description="List all incoming and outgoing edges for a node, showing relationships to other nodes."
        )
        @observe(as_type='tool')
        async def get_node_edges(
                node_id: Annotated[str, "The ID of the node whose edges to list."]
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

                return {
                    "node_id": node_id,
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "incoming_count": len(incoming),
                    "outgoing_count": len(outgoing),
                    "text": f"Node '{node_id}' has {len(incoming)} incoming and {len(outgoing)} outgoing edges."
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_node_edges")
            except Exception as e:
                return self._handle_error(e, "get_node_edges")

        @self.app.tool(
            description="Search for nodes in the knowledge graph by query string, using the code index semantic and keyword search."
        )
        @observe(as_type='tool')
        async def search_nodes(
                query: Annotated[str, "The search string to match against code index."],
                limit: Annotated[int, "Maximum number of results to return."] = 10
        ) -> dict:
            try:
                self._validate_positive_int(limit, "limit")

                results = self.knowledge_graph.code_index.query(query, n_results=limit)
                metadatas = results.get("metadatas", [[]])[0]

                if not metadatas:
                    return {"query": query, "results": [], "text": f"No results found for '{query}'."}

                structured_results = [
                    {
                        "id": res.get("id"),
                        "content": res.get("content"),
                        "declared_entities": res.get("declared_entities"),
                        "called_entities": res.get("called_entities")
                    }
                    for res in metadatas
                ]

                return {
                    "query": query,
                    "count": len(structured_results),
                    "results": structured_results,
                    "text": f"Found {len(structured_results)} result(s) for query '{query}'."
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "search_nodes")
            except Exception as e:
                return self._handle_error(e, "search_nodes")

        @self.app.tool(
            description="Get overall statistics about the knowledge graph, including node and edge counts, types, and relations."
        )
        @observe(as_type='tool')
        async def get_graph_stats() -> dict:
            g = self.knowledge_graph.graph
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

            return {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "node_types": node_types,
                "edge_relations": edge_relations,
                "text": f"Graph with {num_nodes} nodes, {num_edges} edges, {len(node_types)} node types, and {len(edge_relations)} relation types."
            }

        @self.app.tool(
            description="List nodes of a specific type in the knowledge graph."
        )
        @observe(as_type='tool')
        async def list_nodes_by_type(
                node_type: Annotated[str, "The type of nodes to list (e.g., 'function', 'class', 'file')."],
                limit: Annotated[int, "Maximum number of nodes to return."] = 20
        ) -> dict:
            g = self.knowledge_graph.graph
            matching_nodes = [
                {
                    "id": node_id,
                    "name": getattr(data['data'], 'name', 'Unknown')
                }
                for node_id, data in g.nodes(data=True)
                if getattr(data['data'], 'node_type', None) == node_type
            ][:limit]

            if not matching_nodes:
                return {"node_type": node_type, "results": [], "text": f"No nodes found of type '{node_type}'."}

            return {
                "node_type": node_type,
                "count": len(matching_nodes),
                "results": matching_nodes,
                "text": f"Found {len(matching_nodes)} node(s) of type '{node_type}'."
            }

        @self.app.tool(
            description="Get all nodes directly connected to a given node, including the relationship type."
        )
        @observe(as_type='tool')
        async def get_neighbors(
            node_id: Annotated[str, "The ID of the node whose neighbors to retrieve."]
        ) -> dict:
            """Get all nodes directly connected to this node, with their relationship types."""
            try:
                self._validate_node_exists(node_id)

                neighbors = self.knowledge_graph.get_neighbors(node_id)
                if not neighbors:
                    return {
                        "node_id": node_id,
                        "neighbors": [],
                        "text": f"No neighbors found for node '{node_id}'"
                    }

                neighbor_list = []
                for neighbor in neighbors[:20]:
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

                text = f"Neighbors of '{node_id}' ({len(neighbors)} total):\n\n"
                for neighbor in neighbor_list:
                    text += f"- {neighbor['id']}: {neighbor['name']} ({neighbor['type']})\n"
                    if neighbor['relation']:
                        arrow = "→" if neighbor['direction'] == "outgoing" else "←"
                        text += f"  {arrow} Relation: {neighbor['relation']}\n"

                if len(neighbors) > 20:
                    text += f"\n... and {len(neighbors) - 20} more neighbors\n"

                return {
                    "node_id": node_id,
                    "total_neighbors": len(neighbors),
                    "neighbors": neighbor_list,
                    "has_more": len(neighbors) > 20,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_neighbors")
            except Exception as e:
                return self._handle_error(e, "get_neighbors")

        @self.app.tool(
            description="Find where an entity (function, class, variable, etc.) is declared or defined in the codebase."
        )
        @observe(as_type='tool')
        async def go_to_definition(
            entity_name: Annotated[str, "The name of the entity to find the definition for."]
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
                        content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                        declarations.append({
                            "chunk_id": chunk_id,
                            "file_path": chunk.path,
                            "order_in_file": chunk.order_in_file,
                            "content_preview": content_preview
                        })

                text = f"Definition(s) for '{entity_name}':\n\n"
                text += f"Type: {', '.join(entity_info.get('type', ['Unknown']))}\n"
                if entity_info.get('dtype'):
                    text += f"Data Type: {entity_info['dtype']}\n"
                text += f"\nDeclared in {len(declaring_chunks)} location(s):\n\n"

                for decl in declarations:
                    text += f"- Chunk: {decl['chunk_id']}\n"
                    text += f"  File: {decl['file_path']}\n"
                    text += f"  Order: {decl['order_in_file']}\n"
                    text += f"  Content: {decl['content_preview']}\n\n"

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
            description="Find all usages or calls of an entity (function, class, variable, etc.) in the codebase."
        )
        @observe(as_type='tool')
        async def find_usages(
            entity_name: Annotated[str, "The name of the entity to find usages for."],
            limit: Annotated[int, "Maximum number of usages to return."] = 20
        ) -> dict:
            """Find where an entity is used/called in the codebase."""
            try:
                self._validate_entity_exists(entity_name)
                self._validate_positive_int(limit, "limit")

                entity_info = self.knowledge_graph.entities[entity_name]
                calling_chunks = entity_info.get('calling_chunk_ids', [])

                if not calling_chunks:
                    return {
                        "entity_name": entity_name,
                        "usages": [],
                        "text": f"Entity '{entity_name}' found but no usages identified."
                    }

                usages = []
                for chunk_id in calling_chunks[:limit]:
                    if chunk_id in self.knowledge_graph.graph:
                        chunk = self.knowledge_graph.graph.nodes[chunk_id]['data']
                        content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                        usages.append({
                            "chunk_id": chunk_id,
                            "file_path": chunk.path,
                            "order_in_file": chunk.order_in_file,
                            "content_preview": content_preview
                        })

                text = f"Usages of '{entity_name}' ({len(calling_chunks)} total):\n\n"
                for usage in usages:
                    text += f"- {usage['file_path']} (chunk {usage['order_in_file']})\n"
                    text += f"  Content: {usage['content_preview']}\n\n"

                if len(calling_chunks) > limit:
                    text += f"\n... and {len(calling_chunks) - limit} more usages\n"

                return {
                    "entity_name": entity_name,
                    "total_usages": len(calling_chunks),
                    "usages": usages,
                    "has_more": len(calling_chunks) > limit,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "find_usages")
            except Exception as e:
                return self._handle_error(e, "find_usages")

        @self.app.tool(
            description="Get an overview of the structure of a file, including its chunks and declared entities."
        )
        @observe(as_type='tool')
        async def get_file_structure(
            file_path: Annotated[str, "The path of the file to get the structure for."]
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
            description="Get chunks related to a given chunk by a specific relationship (e.g., 'calls', 'contains')."
        )
        @observe(as_type='tool')
        async def get_related_chunks(
            chunk_id: Annotated[str, "The ID of the chunk to find related chunks for."],
            relation_type: Annotated[str, "The type of relationship to filter by (e.g., 'calls', 'contains')."] = "calls"
        ) -> dict:
            """Get chunks related to this chunk by a specific relationship (e.g., 'calls', 'contains')."""
            try:
                self._validate_node_exists(chunk_id)

                related = []
                for _, target, attrs in self.knowledge_graph.graph.out_edges(chunk_id, data=True):
                    if attrs.get('relation') == relation_type:
                        target_node = self.knowledge_graph.graph.nodes[target]['data']
                        related.append({
                            "id": target,
                            "file_path": getattr(target_node, 'path', 'Unknown'),
                            "entity_name": attrs.get('entity_name')
                        })

                if not related:
                    return {
                        "chunk_id": chunk_id,
                        "relation_type": relation_type,
                        "related_chunks": [],
                        "text": f"No chunks found with '{relation_type}' relationship from '{chunk_id}'"
                    }

                text = f"Chunks related to '{chunk_id}' via '{relation_type}' ({len(related)} total):\n\n"
                for chunk in related[:15]:
                    text += f"- {chunk['id']}\n"
                    text += f"  File: {chunk['file_path']}\n"
                    if chunk['entity_name']:
                        text += f"  Entity: {chunk['entity_name']}\n"

                if len(related) > 15:
                    text += f"\n... and {len(related) - 15} more\n"

                return {
                    "chunk_id": chunk_id,
                    "relation_type": relation_type,
                    "total_related": len(related),
                    "related_chunks": related[:15],
                    "has_more": len(related) > 15,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_related_chunks")
            except Exception as e:
                return self._handle_error(e, "get_related_chunks")

        @self.app.tool(
            description="List all entities tracked in the knowledge graph, including their types, declaration, and usage counts."
        )
        @observe(as_type='tool')
        async def list_all_entities(
            limit: Annotated[int, "Maximum number of entities to return."] = 50
        ) -> dict:
            """List all entities tracked in the knowledge graph with their metadata."""
            if not self.knowledge_graph.entities:
                return {
                    "entities": [],
                    "text": "No entities found in the knowledge graph."
                }

            entities = []
            for entity_name, info in list(self.knowledge_graph.entities.items())[:limit]:
                entities.append({
                    "name": entity_name,
                    "types": info.get('type', ['Unknown']),
                    "declaration_count": len(info.get('declaring_chunk_ids', [])),
                    "usage_count": len(info.get('calling_chunk_ids', []))
                })

            text = f"All Entities ({len(self.knowledge_graph.entities)} total):\n\n"
            for i, entity in enumerate(entities, 1):
                text += f"{i}. {entity['name']}\n"
                text += f"   Types: {', '.join(entity['types'])}\n"
                text += f"   Declarations: {entity['declaration_count']}\n"
                text += f"   Usages: {entity['usage_count']}\n\n"

            if len(self.knowledge_graph.entities) > limit:
                text += f"... and {len(self.knowledge_graph.entities) - limit} more entities\n"

            return {
                "total_entities": len(self.knowledge_graph.entities),
                "entities": entities,
                "has_more": len(self.knowledge_graph.entities) > limit,
                "text": text
            }

        # --- New Tools ---
        @self.app.tool(
            description="Show the diff between two code chunks or nodes by their IDs."
        )
        @observe(as_type='tool')
        async def diff_chunks(
            node_id_1: Annotated[str, "The ID of the first node/chunk."],
            node_id_2: Annotated[str, "The ID of the second node/chunk."]
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
            description="Show a tree view of the repository or a subtree starting from a given node ID."
        )
        @observe(as_type='tool')
        async def print_tree(
            root_id: Annotated[Optional[str], "The node ID to start the tree from (default: repo root)."] = 'root',
            max_depth: Annotated[int, "Maximum depth to show."] = 3
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
            description="Show all relationships (calls, contains, etc.) for a given entity or node."
        )
        @observe(as_type='tool')
        async def entity_relationships(
            node_id: Annotated[str, "The node/entity ID to explore relationships for."]
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
            description="Search for nodes/entities by type and name substring."
        )
        @observe(as_type='tool')
        async def search_by_type_and_name(
            node_type: Annotated[str, "Type of node/entity (e.g., 'function', 'class', 'file')."],
            name_query: Annotated[str, "Substring to match in the name."],
            limit: Annotated[int, "Maximum results to return."] = 10
        ) -> dict:
            try:
                self._validate_positive_int(limit, "limit")

                g = self.knowledge_graph.graph
                matches = [
                    {
                        "id": nid,
                        "name": getattr(n['data'], 'name', 'Unknown'),
                        "content": getattr(n['data'], 'content', None)
                    }
                    for nid, n in g.nodes(data=True)
                    if getattr(n['data'], 'node_type', None) == node_type and name_query.lower() in getattr(n['data'], 'name', '').lower()
                ][:limit]

                if not matches:
                    return {
                        "node_type": node_type,
                        "name_query": name_query,
                        "matches": [],
                        "text": f"No matches for type '{node_type}' and name containing '{name_query}'."
                    }

                text = f"Matches for type '{node_type}' and name '{name_query}':\n"
                for match in matches:
                    text += f"- {match['id']}: {match['name']}\n"

                return {
                    "node_type": node_type,
                    "name_query": name_query,
                    "count": len(matches),
                    "matches": matches,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "search_by_type_and_name")
            except Exception as e:
                return self._handle_error(e, "search_by_type_and_name")

        @self.app.tool(
            description="Show the previous and next code chunk for a given chunk/node ID."
        )
        @observe(as_type='tool')
        async def get_chunk_context(
            node_id: Annotated[str, "The node/chunk ID to get context for."]
        ) -> dict:
            try:
                self._validate_node_exists(node_id)

                previous_chunk = self.knowledge_graph.get_previous_chunk(node_id)
                next_chunk = self.knowledge_graph.get_next_chunk(node_id)

                prev_info = None
                next_info = None

                text = f"Context for chunk '{node_id}':\n"

                if previous_chunk:
                    prev_content = previous_chunk.content
                    prev_preview = prev_content[:200] + '...' if len(prev_content) > 200 else prev_content
                    prev_info = {
                        "id": str(previous_chunk),
                        "content_preview": prev_preview
                    }
                    text += f"Previous chunk ({previous_chunk}):\n{prev_preview}\n"
                else:
                    text += "No previous chunk found.\n"

                if next_chunk:
                    next_content = next_chunk.content
                    next_preview = next_content[:200] + '...' if len(next_content) > 200 else next_content
                    next_info = {
                        "id": str(next_chunk),
                        "content_preview": next_preview
                    }
                    text += f"Next chunk ({next_chunk}):\n{next_preview}\n"
                else:
                    text += "No next chunk found.\n"

                return {
                    "node_id": node_id,
                    "previous_chunk": prev_info,
                    "next_chunk": next_info,
                    "text": text
                }
            except (NodeNotFoundError, InvalidInputError, EntityNotFoundError) as e:
                return self._handle_error(e, "get_chunk_context")
            except Exception as e:
                return self._handle_error(e, "get_chunk_context")

        @self.app.tool(
            description="Get statistics for a file or directory: number of entities, lines, chunks, etc."
        )
        @observe(as_type='tool')
        async def get_file_stats(
            path: Annotated[str, "The file or directory path to get statistics for."]
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
            description="Find the shortest path between two nodes in the knowledge graph."
        )
        @observe(as_type='tool')
        async def find_path(
            source_id: Annotated[str, "The ID of the source node."],
            target_id: Annotated[str, "The ID of the target node."],
            max_depth: Annotated[int, "Maximum depth to search for a path."] = 5
        ) -> dict:
            """Find shortest path between two nodes."""
            return self.knowledge_graph.find_path(source_id, target_id, max_depth)

        @self.app.tool(
            description="Extract a subgraph around a node up to a specified depth, optionally filtering by edge types."
        )
        @observe(as_type='tool')
        async def get_subgraph(
            node_id: Annotated[str, "The ID of the central node."],
            depth: Annotated[int, "The depth/radius of the subgraph to extract."] = 2,
            edge_types: Annotated[Optional[list], "Optional list of edge types to include (e.g., ['calls', 'contains'])."] = None
        ) -> dict:
            """Extract a subgraph around a node."""
            return self.knowledge_graph.get_subgraph(node_id, depth, edge_types)

    def run(self, **kwargs):
        self.app.run(**kwargs)
