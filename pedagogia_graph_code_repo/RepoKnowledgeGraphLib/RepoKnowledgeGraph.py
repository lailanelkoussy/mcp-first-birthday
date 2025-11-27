import networkx as nx
import json
import os
import asyncio
import nest_asyncio
import tqdm
# from pathlib import Path
import os.path
import tempfile
import subprocess
from typing import List, Optional, Dict
import logging
import urllib.parse

from .ModelService import create_model_service
from .Node import Node, DirectoryNode, FileNode, ChunkNode, EntityNode
from .CodeParser import CodeParser
from .EntityExtractor import HybridEntityExtractor
from .CodeIndex import CodeIndex
from .utils.logger_utils import setup_logger
from .utils.parsing_utils import read_directory_files_recursively, get_language_from_filename
from .utils.path_utils import prepare_input_path, build_entity_alias_map, resolve_entity_call
from .EntityChunkMapper import EntityChunkMapper

LOGGER_NAME = 'REPO_KNOWLEDGE_GRAPH_LOGGER'

MODEL_SERVICE_TYPES = ['openai', 'sentence-transformers']


# A RepoKnowledgeGraph is a weighted DAG based on a tree-structure with added edges
class RepoKnowledgeGraph:
    """
    RepoKnowledgeGraph builds a knowledge graph of a code repository.
    It parses source files, extracts code entities and relationships, and organizes them
    into a directed acyclic graph (DAG) with additional semantic edges.

    Use `from_path()` or `load_graph_from_file()` to create instances.
    """

    def __init__(self):
        """
        Private constructor. Use from_path() or load_graph_from_file() instead.
        """
        raise RuntimeError(
            "Cannot instantiate RepoKnowledgeGraph directly. "
            "Use RepoKnowledgeGraph.from_path() or RepoKnowledgeGraph.load_graph_from_file() instead."
        )

    def _initialize(self, model_service_kwargs: dict):
        """Internal initialization method."""
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.info('Initializing RepoKnowledgeGraph instance.')
        self.code_parser = CodeParser()
        self.model_service = create_model_service(**model_service_kwargs)
        self.entities = {}
        self.graph = nx.DiGraph()
        self.knowledge_graph = nx.DiGraph()
        self.code_index = None
        self.entity_extractor = HybridEntityExtractor()

    def __iter__(self):
        # Yield only the 'data' attribute from each node
        return (node_data['data'] for _, node_data in self.graph.nodes(data=True))

    def __getitem__(self, node_id):
        return self.graph.nodes[node_id]['data']
    

    @classmethod
    def from_path(cls, path: str, skip_dirs: Optional[list] = None, index_nodes: bool = True, describe_nodes=False,
                  extract_entities: bool = False, model_service_kwargs: Optional[dict] = None, code_index_kwargs: Optional[dict] = None):
        if skip_dirs is None:
            skip_dirs = []
        if model_service_kwargs is None:
            model_service_kwargs = {}
        """
        Alternative constructor to build a RepoKnowledgeGraph from a path, with options to skip directories
        and control entity extraction and node description.

        Args:
            path (str): Path to the root of the code repository.
            skip_dirs (list): List of directory names to skip.
            index_nodes (bool): Whether to build a code index.
            describe_nodes (bool): Whether to generate descriptions for code chunks.
            extract_entities (bool): Whether to extract entities from code.

        Returns:
            RepoKnowledgeGraph: The constructed knowledge graph.
        """
        instance = cls.__new__(cls)  # Create instance without calling __init__
        instance._initialize(model_service_kwargs=model_service_kwargs)

        instance.logger.info(f"Preparing to build knowledge graph from path: {path}")

        prepared_path = prepare_input_path(path)
        instance.logger.debug(f"Prepared input path: {prepared_path}")

        # Handle running event loop (e.g., in Jupyter)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            instance.logger.debug("Detected running event loop, applying nest_asyncio.")
            nest_asyncio.apply()
            task = instance._initial_parse_path_async(prepared_path, skip_dirs=skip_dirs, index_nodes=index_nodes,
                                                      describe_nodes=describe_nodes, extract_entities=extract_entities)
            loop.run_until_complete(task)
        else:
            instance.logger.debug("No running event loop, using asyncio.run.")
            asyncio.run(instance._initial_parse_path_async(prepared_path, skip_dirs=skip_dirs, index_nodes=index_nodes,
                                                           describe_nodes=describe_nodes,
                                                           extract_entities=extract_entities))

        instance.logger.info("Parsing files and building initial nodes...")
        instance.logger.info("Initial parse and node creation complete. Building relationships between nodes...")
        instance._build_relationships()

        if index_nodes:
            instance.logger.info("Building code index for all nodes in the graph...")
            instance.code_index = CodeIndex(list(instance), model_service=instance.model_service, **(code_index_kwargs or {}))

        instance.logger.info("Knowledge graph construction from path completed successfully.")
        return instance

    @classmethod
    def from_repo(
            cls,
            repo_url: str,
            skip_dirs: Optional[list] = None,
            index_nodes: bool = True,
            describe_nodes: bool = False,
            extract_entities: bool = False,
            model_service_kwargs: Optional[dict] = None,
            code_index_kwargs: Optional[dict]=None,
            github_token: Optional[str] = None,
            allow_unauthenticated_clone: bool = True,
    ):
        """
        Alternative constructor to build a RepoKnowledgeGraph from a remote git repository URL.

        Args:
            repo_url (str): Git repository URL (SSH or HTTPS).
            skip_dirs (list): List of directory names to skip.
            index_nodes (bool): Whether to build a code index.
            describe_nodes (bool): Whether to generate descriptions for code chunks.
            extract_entities (bool): Whether to extract entities from code.
            github_token (str, optional): Personal access token to access private GitHub repos.
                If not provided, the method will look for the `GITHUB_OAUTH_TOKEN` environment variable.
            allow_unauthenticated_clone (bool): If True, attempt to clone without a token when none is provided.
                If False, raise an error when no token is available.

        Returns:
            RepoKnowledgeGraph: The constructed knowledge graph.
        """
        if skip_dirs is None:
            skip_dirs = []
        if model_service_kwargs is None:
            model_service_kwargs = {}

        instance = cls.__new__(cls)
        instance._initialize(model_service_kwargs=model_service_kwargs)

        instance.logger.info(f"Starting knowledge graph build from remote repository: {repo_url}")

        # Determine token
        token = github_token or os.environ.get('GITHUB_OAUTH_TOKEN')

        with tempfile.TemporaryDirectory() as tmpdirname:
            clone_url = repo_url
            try:
                if repo_url.startswith('git@'):
                    # Convert git@github.com:owner/repo.git -> https://github.com/owner/repo.git
                    clone_url = repo_url.replace(':', '/').split('git@')[-1]
                    clone_url = f'https://{clone_url}'

                if token and clone_url.startswith('https://'):
                    encoded_token = urllib.parse.quote(token, safe='')
                    clone_url = clone_url.replace('https://', f'https://{encoded_token}@')
                elif not token and not allow_unauthenticated_clone:
                    raise ValueError(
                        "GitHub token not provided and unauthenticated clone is disabled. "
                        "Set allow_unauthenticated_clone=True or provide a token."
                    )

                instance.logger.debug(f"Running git clone: {clone_url} -> {tmpdirname}")
                subprocess.run(['git', 'clone', clone_url, tmpdirname], check=True)

            except Exception as e:
                instance.logger.error(f"Failed to clone repository {repo_url} using URL {clone_url}: {e}")
                raise

            instance.logger.info(f"Repository successfully cloned to: {tmpdirname}")

            return cls.from_path(
                tmpdirname,
                skip_dirs=skip_dirs,
                index_nodes=index_nodes,
                describe_nodes=describe_nodes,
                extract_entities=extract_entities,
                model_service_kwargs=model_service_kwargs,
                code_index_kwargs=code_index_kwargs
            )

    async def _initial_parse_path_async(self, path: str, skip_dirs: list, index_nodes=True, describe_nodes=True,
                                        extract_entities: bool = True):
        self.logger.info(f"Beginning async parsing of repository at path: {path}")
        """
        Orchestrates the parsing and graph construction process:
        1. Reads files and splits into chunks.
        2. Extracts entities and relationships.
        3. Builds chunk, file, directory, and root nodes.
        4. Aggregates entity information.

        Args:
            path (str): Root path to parse.
            skip_dirs (list): Directories to skip.
            index_nodes (bool): Whether to build code index.
            describe_nodes (bool): Whether to generate descriptions.
            extract_entities (bool): Whether to extract entities.
        """

        # --- Pass 1: Create ChunkNodes ---
        level1_node_contents = read_directory_files_recursively(
            path, skip_dirs=skip_dirs,
            skip_pattern=r"(?:\.log$|\.json$|(?:^|/)(?:\.git|\.idea|__pycache__|\.cache)(?:/|$)|(?:^|/)(?:changelog|ChangeLog)(?:\.[a-z0-9]+)?$|\.cache$)"
        )
        self.logger.debug(f"Found {len(level1_node_contents)} files to process.")
        self.logger.info("Chunk nodes creation step started.")
        chunk_info = await self._create_chunk_nodes(
            level1_node_contents, extract_entities, describe_nodes, index_nodes, root_path=path
        )
        self.logger.info("Chunk nodes creation step finished.")
        self.logger.info("File nodes creation step started.")
        file_info = self._create_file_nodes(
            chunk_info, level1_node_contents
        )
        self.logger.info("File nodes creation step finished.")
        self.logger.info("Directory nodes creation step started.")
        dir_agg = self._create_directory_nodes(
            file_info
        )
        self.logger.info("Directory nodes creation step finished.")
        self.logger.info("Aggregating all nodes to root node.")
        self._aggregate_to_root(dir_agg)
        self.logger.info("Async parse and node aggregation fully complete.")

    async def _create_chunk_nodes(self, level1_node_contents, extract_entities, describe_nodes, index_nodes, root_path=None):
        self.logger.info("Starting chunk node creation for all files.")
        accepted_extensions = {'.py', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.jsx', '.tsx', '.rs', '.html'}
        chunk_info = {}
        entity_mapper = EntityChunkMapper()

        # Use tqdm for progress bar over files
        for file_path in tqdm.tqdm(level1_node_contents, desc="Processing files for chunk nodes"):
            self.logger.info(f"Processing file for chunk nodes: {file_path}")
            full_path = os.path.normpath(file_path)
            parts = full_path.split(os.sep)
            _, ext = os.path.splitext(file_path)
            is_code_file = ext.lower() in accepted_extensions

            self.logger.debug(f"Parsing file: {file_path}")

            # Parse file into chunks
            parsed_content = self.code_parser.parse(file_name=file_path, file_content=level1_node_contents[file_path])
            self.logger.info(f"Parsed {len(parsed_content)} chunks from file: {file_path}")

            # Entity extraction logging
            if extract_entities and is_code_file:
                self.logger.info(f"Extracting entities from code file: {file_path}")
                try:
                    # Construct full path for entity extraction (needed for C/C++ include resolution)
                    extraction_file_path = os.path.join(root_path, file_path) if root_path else file_path
                    
                    file_declared_entities, file_called_entities = self.entity_extractor.extract_entities(
                        code=level1_node_contents[file_path], file_name=extraction_file_path)
                    self.logger.info(f"Extracted {len(file_declared_entities)} declared and {len(file_called_entities)} called entities from file: {file_path}")

                    chunk_declared_map, chunk_called_map = entity_mapper.map_entities_to_chunks(
                        file_declared_entities, file_called_entities, parsed_content, file_name=file_path)
                    self.logger.info(f"Mapped entities to {len(parsed_content)} chunks for file: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error extracting entities from {file_path}: {e}")
                    file_declared_entities, file_called_entities = [], []
                    chunk_declared_map = {i: [] for i in range(len(parsed_content))}
                    chunk_called_map = {i: [] for i in range(len(parsed_content))}
            else:
                self.logger.info(f"Skipping entity extraction for non-code file: {file_path}")
                file_declared_entities, file_called_entities = [], []
                chunk_declared_map = {i: [] for i in range(len(parsed_content))}
                chunk_called_map = {i: [] for i in range(len(parsed_content))}

            chunk_tasks = []
            for i, chunk in enumerate(parsed_content):
                chunk_id = f'{file_path}_{i}'
                self.logger.debug(f"Scheduling processing for chunk {chunk_id} of file {file_path}")

                async def process_chunk(i=i, chunk=chunk, chunk_id=chunk_id):
                    self.logger.info(f"Creating chunk node: {chunk_id}")
                    declared_entities = chunk_declared_map.get(i, [])
                    called_entities = chunk_called_map.get(i, [])

                    # FIRST PASS: Register all declared entities with aliases
                    # Build temporary alias map for checking existing entities
                    temp_alias_map = build_entity_alias_map(self.entities)

                    for entity in declared_entities:
                        name = entity.get("name")
                        if not name:
                            continue

                        # Check if this entity already exists under any of its aliases
                        entity_aliases = entity.get("aliases", [])
                        canonical_name = None

                        # First check if the name itself already exists or is an alias
                        if name in temp_alias_map:
                            canonical_name = temp_alias_map[name]
                            self.logger.debug(f"Entity '{name}' already exists as '{canonical_name}'")
                        else:
                            # Check if any of the entity's aliases match existing entities
                            for alias in entity_aliases:
                                if alias in temp_alias_map:
                                    canonical_name = temp_alias_map[alias]
                                    self.logger.debug(f"Entity '{name}' matches existing entity '{canonical_name}' via alias '{alias}'")
                                    break

                        # If we found a match, use the canonical name; otherwise use the entity name
                        if canonical_name:
                            entity_key = canonical_name
                        else:
                            entity_key = name
                            self.logger.info(f"Registering new declared entity '{name}' in chunk {chunk_id}")
                            self.entities[entity_key] = {
                                "declaring_chunk_ids": [],
                                "calling_chunk_ids": [],
                                "type": [],
                                "dtype": None,
                                "aliases": []
                            }
                            # Update temp alias map with new entity
                            temp_alias_map[entity_key] = entity_key

                        if chunk_id not in self.entities[entity_key]["declaring_chunk_ids"]:
                            self.entities[entity_key]["declaring_chunk_ids"].append(chunk_id)
                        entity_type = entity.get("type")
                        if entity_type and entity_type not in self.entities[entity_key]["type"]:
                            self.entities[entity_key]["type"].append(entity_type)
                        dtype = entity.get("dtype")
                        if dtype:
                            self.entities[entity_key]["dtype"] = dtype
                        # Store aliases (add new ones, avoiding duplicates)
                        for alias in [name] + entity_aliases:
                            if alias and alias not in self.entities[entity_key]["aliases"]:
                                self.entities[entity_key]["aliases"].append(alias)
                                temp_alias_map[alias] = entity_key  # Update temp map
                        self.logger.debug(f"Declared entity '{name}' registered as '{entity_key}' in chunk {chunk_id} with aliases: {self.entities[entity_key]['aliases']}")


                    # Logging for node creation
                    if describe_nodes:
                        self.logger.info(f"Generating description for chunk {chunk_id}")
                        try:
                            description = await self.model_service.query_async(
                                f'Summarize this {get_language_from_filename(file_path)} code chunk in a few sentences: {chunk}')
                        except Exception as e:
                            self.logger.error(f"Error generating description for chunk {chunk_id}: {e}")
                            description = ''
                    else:
                        self.logger.debug(f"No description requested for chunk {chunk_id}")
                        description = ''

                    chunk_node = ChunkNode(
                        id=chunk_id,
                        name=chunk_id,
                        path=file_path,
                        content=chunk,
                        order_in_file=i,
                        called_entities=called_entities,
                        declared_entities=declared_entities,
                        language=get_language_from_filename(file_path),
                        description=description,
                    )
                    self.logger.info(f"Chunk node created: {chunk_id}")

                    if index_nodes:
                        self.logger.info(f"Generating embedding for chunk {chunk_id}")
                        try:
                            embedding = await self.model_service.embed_async(text_to_embed=chunk_node.get_field_to_embed())
                        except Exception as e:
                            self.logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
                            embedding = []
                    else:
                        self.logger.debug(f"No embedding requested for chunk {chunk_id}")
                        embedding = []

                    chunk_node.embedding = embedding
                    return (chunk_id, chunk_node, declared_entities, called_entities)

                chunk_tasks.append(process_chunk())

            chunk_results = await asyncio.gather(*chunk_tasks)
            self.logger.info(f"Finished processing {len(chunk_results)} chunks for file {file_path}.")
            chunk_info[file_path] = {
                'chunk_results': chunk_results,
                'file_declared_entities': file_declared_entities,
                'file_called_entities': file_called_entities
            }
            self.logger.debug(f"Processed {len(chunk_results)} chunks for file {file_path}.")

        # SECOND PASS: Now that all declared entities are registered, resolve called entities
        self.logger.info("Starting second pass: resolving called entities using alias map...")
        alias_map = build_entity_alias_map(self.entities)
        self.logger.info(f"Built alias map with {len(alias_map)} entries for resolution")

        resolved_count = 0
        for file_path, file_data in chunk_info.items():
            chunk_results = file_data['chunk_results']
            for chunk_id, chunk_node, declared_entities, called_entities in chunk_results:
                for called_name in called_entities:
                    # Skip empty or whitespace-only names
                    if not called_name or not called_name.strip():
                        continue

                    # Try to resolve this called entity to an existing declared entity using aliases
                    resolved_name = resolve_entity_call(called_name, alias_map)

                    # Use the resolved name if found, otherwise check if called_name is already an alias
                    if resolved_name:
                        entity_key = resolved_name
                    elif called_name in alias_map:
                        # The called_name itself is an alias of an existing entity
                        entity_key = alias_map[called_name]
                    else:
                        # No match found, use the original called name
                        entity_key = called_name

                    if entity_key not in self.entities:
                        self.logger.debug(f"Registering new called entity '{entity_key}' (called as '{called_name}') in chunk {chunk_id}")
                        self.entities[entity_key] = {
                            "declaring_chunk_ids": [],
                            "calling_chunk_ids": [],
                            "type": [],
                            "dtype": None,
                            "aliases": []
                        }
                        # Add called_name as an alias if it's different from entity_key
                        if called_name != entity_key:
                            self.entities[entity_key]["aliases"].append(called_name)
                            alias_map[called_name] = entity_key  # Update alias map

                    if chunk_id not in self.entities[entity_key]["calling_chunk_ids"]:
                        self.entities[entity_key]["calling_chunk_ids"].append(chunk_id)

                    if resolved_name and resolved_name != called_name:
                        resolved_count += 1
                        self.logger.debug(f"Called entity '{called_name}' resolved to '{entity_key}' in chunk {chunk_id}")

        self.logger.info(f"Resolved {resolved_count} entity calls to existing declarations via aliases")
        self.logger.info("All chunk nodes have been created for all files.")
        return chunk_info

    def _create_file_nodes(self, chunk_info, level1_node_contents):
        self.logger.info("Starting file node creation.")
        """
        For each file, aggregate chunk information and create FileNode objects.
        This method remains mostly the same.
        """

        def merge_entities(target, source):
            # Merge entity lists, avoiding duplicates by (name, type)
            existing = set((e.get('name'), e.get('type')) for e in target)
            for e in source:
                k = (e.get('name'), e.get('type'))
                if k not in existing:
                    target.append(e)
                    existing.add(k)

        def merge_called_entities(target, source):
            # Merge called entity lists, avoiding duplicates
            existing = set(target)
            for e in source:
                if e not in existing:
                    target.append(e)
                    existing.add(e)

        file_info = {}
        for file_path, file_data in chunk_info.items():
            self.logger.info(f"Creating file node for: {file_path}")
            parts = os.path.normpath(file_path).split(os.sep)

            # Extract file-level entities and chunk results from the stored data
            chunk_results = file_data['chunk_results']
            file_declared_entities = list(file_data['file_declared_entities'])  # Use file-level entities directly
            file_called_entities = list(file_data['file_called_entities'])      # Use file-level entities directly
            chunk_ids = []

            for chunk_id, chunk_node, declared_entities, called_entities in chunk_results:
                self.logger.info(f"Adding chunk node {chunk_id} to graph for file {file_path}")
                self.graph.add_node(chunk_id, data=chunk_node, level=2)
                chunk_ids.append(chunk_id)
                # Note: We're using file-level entities for the FileNode, so we don't need to merge from chunks
                # The chunks already have their entities set correctly

            file_node = FileNode(
                id=file_path,
                name=parts[-1],
                path=file_path,
                node_type='file',
                content=level1_node_contents[file_path],
                declared_entities=file_declared_entities,
                called_entities=file_called_entities,
                language=get_language_from_filename(file_path),
            )

            self.logger.debug(f"Adding file node {file_path} to graph.")
            self.graph.add_node(file_path, data=file_node, level=1)
            for chunk_id in chunk_ids:
                self.graph.add_edge(file_path, chunk_id, relation='contains')

            file_info[file_path] = {
                'declared_entities': file_declared_entities,
                'called_entities': file_called_entities,
                'chunk_ids': chunk_ids,
                'parts': parts,
            }
            self.logger.info(f"File node {file_path} added to graph with {len(chunk_ids)} chunks.")

        self.logger.info("All file nodes have been created.")
        return file_info

    def _create_directory_nodes(self, file_info):
        self.logger.info("Starting directory node creation.")
        """
        For each directory, aggregate file information and create DirectoryNode objects.

        Args:
            file_info (dict): Mapping file_path -> file info dict.

        Returns:
            dict: Mapping dir_path -> aggregated entity info.
        """

        def merge_entities(target, source):
            # Merge entity lists, avoiding duplicates by (name, type)
            existing = set((e.get('name'), e.get('type')) for e in target)
            for e in source:
                k = (e.get('name'), e.get('type'))
                if k not in existing:
                    target.append(e)
                    existing.add(k)

        def merge_called_entities(target, source):
            # Merge called entity lists, avoiding duplicates
            existing = set(target)
            for e in source:
                if e not in existing:
                    target.append(e)
                    existing.add(e)

        dir_agg = {}
        for file_path, info in file_info.items():
            self.logger.info(f"Processing directory nodes for file: {file_path}")
            parts = os.path.normpath(file_path).split(os.sep)
            file_declared_entities = info['declared_entities']
            file_called_entities = info['called_entities']
            current_parent = 'root'
            path_accum = ''
            for part in parts[:-1]:  # Skip file itself
                path_accum = os.path.join(path_accum, part) if path_accum else part
                if path_accum not in self.graph:
                    self.logger.info(f"Adding new directory node: {path_accum}")
                    dir_node = DirectoryNode(id=path_accum, name=part, path=path_accum)
                    self.graph.add_node(path_accum, data=dir_node, level=1)
                    self.graph.add_edge(current_parent, path_accum, relation='contains')
                if path_accum not in dir_agg:
                    dir_agg[path_accum] = {'declared_entities': [], 'called_entities': []}
                merge_entities(dir_agg[path_accum]['declared_entities'], file_declared_entities)
                merge_called_entities(dir_agg[path_accum]['called_entities'], file_called_entities)
                current_parent = path_accum
            # Connect file to its parent directory
            self.graph.add_edge(current_parent, file_path, relation='contains')
        self.logger.info("All directory nodes created.")
        return dir_agg

    def _aggregate_to_root(self, dir_agg):
        self.logger.info("Aggregating directory information to root node.")
        """
        Aggregate all directory entity information to the root node.

        Args:
            dir_agg (dict): Mapping dir_path -> aggregated entity info.
        """

        def merge_entities(target, source):
            # Merge entity lists, avoiding duplicates by (name, type)
            existing = set((e.get('name'), e.get('type')) for e in target)
            for e in source:
                k = (e.get('name'), e.get('type'))
                if k not in existing:
                    target.append(e)
                    existing.add(k)

        def merge_called_entities(target, source):
            # Merge called entity lists, avoiding duplicates
            existing = set(target)
            for e in source:
                if e not in existing:
                    target.append(e)
                    existing.add(e)

        root_node = Node(id='root', name='root', node_type='root')
        self.graph.add_node('root', data=root_node, level=0)
        root_declared_entities = []
        root_called_entities = []
        for dir_path, agg in dir_agg.items():
            node = self.graph.nodes[dir_path]['data']
            if not hasattr(node, 'declared_entities'):
                node.declared_entities = []
            if not hasattr(node, 'called_entities'):
                node.called_entities = []
            merge_entities(node.declared_entities, agg['declared_entities'])
            merge_called_entities(node.called_entities, agg['called_entities'])
            merge_entities(root_declared_entities, agg['declared_entities'])
            merge_called_entities(root_called_entities, agg['called_entities'])
        if not hasattr(root_node, 'declared_entities'):
            root_node.declared_entities = []
        if not hasattr(root_node, 'called_entities'):
            root_node.called_entities = []
        merge_entities(root_node.declared_entities, root_declared_entities)
        merge_called_entities(root_node.called_entities, root_called_entities)
        self.logger.info("Aggregation to root node complete.")

    def _build_relationships(self):
        self.logger.info("Building relationships between chunk nodes based on entities.")
        """
        Build relationships between chunk nodes and entity nodes based on self.entities.
        For each entity in self.entities:
        1. Create an EntityNode with entity_name as the id
        2. Create edges from declaring chunks to entity node (declares relationship)
        3. Create edges from entity node to calling chunks (called_by relationship)
        4. Resolve called entity names using aliases for better matching
        """
        from .Node import EntityNode
        edges_created = 0
        entity_nodes_created = 0
        
        # Build alias map for quick lookups
        self.logger.info("Building entity alias map for call resolution...")
        alias_map = build_entity_alias_map(self.entities)
        self.logger.info(f"Built alias map with {len(alias_map)} entries")

        # First pass: Create all entity nodes
        for entity_name, info in self.entities.items():
            entity_type = info.get('entity_type', '')
            declaring_chunks = info.get('declaring_chunk_ids', [])
            calling_chunks = info.get('calling_chunk_ids', [])
            aliases = info.get('aliases', [])

            # Create EntityNode with entity_name as id
            entity_node = EntityNode(
                id=entity_name,
                name=entity_name,
                entity_type=entity_type,
                declaring_chunk_ids=declaring_chunks,
                calling_chunk_ids=calling_chunks,
                aliases=aliases
            )
            
            # Add entity node to graph
            self.graph.add_node(entity_name, data=entity_node, level=3)
            entity_nodes_created += 1
            
            # Log aliases for debugging
            if aliases:
                self.logger.debug(f"Created EntityNode '{entity_name}' with aliases: {aliases}")

            # Create edges from declaring chunks to entity node
            for declarer_id in declaring_chunks:
                if declarer_id in self.graph:
                    self.graph.add_edge(declarer_id, entity_name, relation='declares')
                    edges_created += 1
            
            # Create edges from entity node to calling chunks
            for caller_id in calling_chunks:
                if caller_id in self.graph and caller_id not in declaring_chunks:
                    self.graph.add_edge(entity_name, caller_id, relation='called_by')
                    edges_created += 1

        # Second pass: Resolve unmatched entity calls using alias matching
        self.logger.info("Resolving entity calls using alias matching...")
        resolved_calls = 0

        for entity_name, info in self.entities.items():
            # Skip entities that already have declarations (they were matched directly)
            if info.get('declaring_chunk_ids'):
                continue

            # Try to resolve this called entity to a declared entity using aliases
            resolved_name = resolve_entity_call(entity_name, alias_map)

            if resolved_name and resolved_name != entity_name:
                # Found a match! Update the calling_chunk_ids of the resolved entity
                calling_chunks = info.get('calling_chunk_ids', [])

                if resolved_name in self.entities:
                    for caller_id in calling_chunks:
                        if caller_id in self.graph:
                            # Add edge from resolved entity to calling chunk
                            if not self.graph.has_edge(resolved_name, caller_id):
                                self.graph.add_edge(resolved_name, caller_id, relation='called_by')
                                edges_created += 1
                                resolved_calls += 1
                                self.logger.debug(f"Resolved call: '{entity_name}' -> '{resolved_name}' in chunk {caller_id}")

        self.logger.info(f"_build_relationships: Created {entity_nodes_created} entity nodes, "
                        f"{edges_created} edges, and resolved {resolved_calls} entity calls using aliases.")

    def get_entity_by_alias(self, alias: str) -> Optional[str]:
        """
        Get the canonical entity name for a given alias.

        Args:
            alias: An alias of an entity (e.g., 'MyClass' or 'module.MyClass')

        Returns:
            Canonical entity name if found, None otherwise
        """
        alias_map = build_entity_alias_map(self.entities)
        return alias_map.get(alias)

    def resolve_entity_references(self) -> Dict[str, List[str]]:
        """
        Resolve all entity references in the knowledge graph using aliases.
        Returns a mapping of unresolved entity calls to their potential matches.

        Returns:
            Dictionary mapping called entity names to list of potential canonical matches
        """
        alias_map = build_entity_alias_map(self.entities)
        resolutions = {}

        for entity_name, info in self.entities.items():
            # Only look at entities that are called but not declared
            if not info.get('declaring_chunk_ids') and info.get('calling_chunk_ids'):
                resolved = resolve_entity_call(entity_name, alias_map)
                if resolved:
                    resolutions[entity_name] = resolved

        return resolutions

    def print_tree(self, max_depth=None, start_node_id='root', level=0, prefix=""):
        """
        Print the repository tree structure using the graph with 'contains' edges.

        Args:
            max_depth (int, optional): Maximum depth to print. None = unlimited.
            start_node_id (str): ID of the node to start from. Default is 'root'.
            level (int): Internal use only (used for recursion).
            prefix (str): Internal use only (used for formatting output).
        """
        if max_depth is not None and level > max_depth:
            self.logger.debug(f"Max depth {max_depth} reached at node {start_node_id}.")
            return

        if start_node_id not in self.graph:
            self.logger.warning(f"Start node '{start_node_id}' not found in graph.")
            return

        try:
            node_data = self[start_node_id]
        except KeyError as e:
            self.logger.error(f"KeyError when accessing node {start_node_id}: {e}")
            self.logger.error(f"Available node attributes: {list(self.graph.nodes[start_node_id].keys())}")
            # Use a fallback approach if 'data' is missing
            if 'data' not in self.graph.nodes[start_node_id]:
                self.logger.warning(f"Node {start_node_id} has no 'data' attribute, using node itself")
                # Create a fallback node if 'data' is missing
                if start_node_id == 'root':
                    # Create a default root node
                    node_data = Node(id='root', name='root', node_type='root')
                    # Update the graph node with the fallback data
                    self.graph.nodes[start_node_id]['data'] = node_data
                else:
                    # Try to infer node type from ID or structure
                    name = start_node_id.split('/')[-1] if '/' in start_node_id else start_node_id
                    if '_' in start_node_id and start_node_id.split('_')[-1].isdigit():
                        # Looks like a chunk ID
                        node_data = ChunkNode(id=start_node_id, name=name, node_type='chunk')
                    elif '.' in name:
                        # Looks like a file
                        node_data = FileNode(id=start_node_id, name=name, node_type='file', path=start_node_id)
                    else:
                        # Fallback to directory or generic node
                        node_data = DirectoryNode(id=start_node_id, name=name, node_type='directory',
                                                  path=start_node_id)
                    # Update the graph node with the fallback data
                    self.graph.nodes[start_node_id]['data'] = node_data
            return

        # Choose icon based on node type
        if node_data.node_type == 'file':
            node_symbol = "📄"
        elif node_data.node_type == 'chunk':
            node_symbol = "📝"
        elif node_data.node_type == 'root':
            node_symbol = "📁"
        elif node_data.node_type == 'directory':
            node_symbol = "📂"
        else:
            node_symbol = "📦"

        if level == 0:
            print(f"{node_symbol} {node_data.name} ({node_data.node_type})")
        else:
            print(f"{prefix}└── {node_symbol} {node_data.name} ({node_data.node_type})")

        # Get children via 'contains' edges
        children = [
            child for child in self.graph.successors(start_node_id)
            if self.graph.edges[start_node_id, child].get('relation') == 'contains'
        ]

        child_count = len(children)
        for i, child_id in enumerate(children):
            is_last = i == child_count - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.print_tree(max_depth, start_node_id=child_id, level=level + 1, prefix=new_prefix)

    def to_dict(self):
        self.logger.info("Serializing graph to dictionary.")
        from .Node import EntityNode
        graph_data = {
            'nodes': [],
            'edges': []
        }

        for node_id, node_attrs in self.graph.nodes(data=True):
            if 'data' not in node_attrs:
                self.logger.warning(f"Node {node_id} has no 'data' attribute, skipping in serialization")
                continue

            node = node_attrs['data']
            node_dict = {
                'id': node.id or node_id,
                'class': node.__class__.__name__,
                'data': {
                    'id': node.id or node_id,
                    'name': node.name,
                    'node_type': node.node_type,
                    'description': getattr(node, 'description', ''),
                    'declared_entities': list(getattr(node, 'declared_entities', [])),
                    'called_entities': list(getattr(node, 'called_entities', [])),
                }
            }

            # FileNode-specific
            if isinstance(node, FileNode):
                node_dict['data']['path'] = node.path
                node_dict['data']['content'] = node.content
                node_dict['data']['language'] = getattr(node, 'language', '')

            # ChunkNode-specific
            if isinstance(node, ChunkNode):
                node_dict['data']['order_in_file'] = getattr(node, 'order_in_file', 0)
                node_dict['data']['embedding'] = getattr(node, 'embedding', None)
            
            # EntityNode-specific
            if isinstance(node, EntityNode):
                node_dict['data']['entity_type'] = getattr(node, 'entity_type', '')
                node_dict['data']['declaring_chunk_ids'] = list(getattr(node, 'declaring_chunk_ids', []))
                node_dict['data']['calling_chunk_ids'] = list(getattr(node, 'calling_chunk_ids', []))
                node_dict['data']['aliases'] = list(getattr(node, 'aliases', []))

            graph_data['nodes'].append(node_dict)

        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'relation': attrs.get('relation', '')
            }
            if 'entities' in attrs:
                edge_data['entities'] = list(attrs['entities'])
            graph_data['edges'].append(edge_data)

        self.logger.info("Serialization complete.")
        return graph_data

    @classmethod
    def from_dict(cls, data_dict, index_nodes: bool = True, use_embed: bool = True,
                  model_service_kwargs: Optional[dict] = None, code_index_kwargs: Optional[dict] = None):
        # ...existing code...
        instance = cls.__new__(cls)  # bypass __init__
        instance._initialize(model_service_kwargs=model_service_kwargs)

        instance.logger.info("Deserializing graph from dictionary.")

        
        node_classes = {
            'Node': Node,
            'FileNode': FileNode,
            'ChunkNode': ChunkNode,
            'DirectoryNode': DirectoryNode,
            'EntityNode': EntityNode,
        }

        # Create a root node if not present in the data
        root_found = any(node_data['id'] == 'root' for node_data in data_dict['nodes'])
        if not root_found:
            instance.logger.warning("Root node not found in the data, creating one")
            root_node = Node(id='root', name='root', node_type='root')
            instance.graph.add_node('root', data=root_node, level=0)

        # --- Rebuild nodes ---
        for node_data in data_dict['nodes']:
            cls_name = node_data['class']
            node_cls = node_classes.get(cls_name, Node)
            kwargs = node_data['data']

            # Ensure ID is properly set
            if not kwargs.get('id'):
                kwargs['id'] = node_data['id']

            # Always use lists for declared_entities and called_entities
            kwargs['declared_entities'] = list(kwargs.get('declared_entities', []))
            kwargs['called_entities'] = list(kwargs.get('called_entities', []))

            # FileNode-specific
            if node_cls in (FileNode, ChunkNode):
                kwargs.setdefault('path', '')
                kwargs.setdefault('content', '')
                kwargs.setdefault('language', '')
            if node_cls == ChunkNode:
                kwargs.setdefault('order_in_file', 0)
                kwargs.setdefault('embedding', [])
            # EntityNode-specific
            if node_cls == EntityNode:
                kwargs.setdefault('entity_type', '')
                kwargs.setdefault('declaring_chunk_ids', [])
                kwargs.setdefault('calling_chunk_ids', [])
                kwargs.setdefault('aliases', [])

            node_instance = node_cls(**kwargs)
            instance.graph.add_node(node_data['id'], data=node_instance, level=instance._infer_level(node_instance))

        # --- Rebuild edges ---
        for edge in data_dict['edges']:
            source = edge['source']
            target = edge['target']
            if source in instance.graph and target in instance.graph:
                edge_kwargs = {'relation': edge.get('relation', '')}
                if 'entities' in edge:
                    edge_kwargs['entities'] = list(edge['entities'])
                instance.graph.add_edge(source, target, **edge_kwargs)
            else:
                instance.logger.warning(f"Cannot add edge {source} -> {target}, nodes don't exist")

        # --- Rebuild instance.entities ---
        instance.entities = {}
        for node_id, node_attrs in instance.graph.nodes(data=True):
            node = node_attrs['data']
            declared_entities = getattr(node, 'declared_entities', [])
            called_entities = getattr(node, 'called_entities', [])
            for entity in declared_entities:
                if isinstance(entity, dict):
                    name = entity.get('name')
                else:
                    name = entity
                if not name:
                    continue
                if name not in instance.entities:
                    instance.entities[name] = {
                        "declaring_chunk_ids": [],
                        "calling_chunk_ids": [],
                        "type": [],
                        "dtype": None
                    }
                # Only add node_id if it is a ChunkNode
                if node_id not in instance.entities[name]["declaring_chunk_ids"]:
                    if node_id in instance.graph and isinstance(instance.graph.nodes[node_id]["data"], ChunkNode):
                        instance.entities[name]["declaring_chunk_ids"].append(node_id)
                if isinstance(entity, dict):
                    entity_type = entity.get("type")
                    if entity_type and entity_type not in instance.entities[name]["type"]:
                        instance.entities[name]["type"].append(entity_type)
                    dtype = entity.get("dtype")
                    if dtype:
                        instance.entities[name]["dtype"] = dtype
            for called_name in called_entities:
                if not called_name:
                    continue
                if called_name not in instance.entities:
                    instance.entities[called_name] = {
                        "declaring_chunk_ids": [],
                        "calling_chunk_ids": [],
                        "type": [],
                        "dtype": None
                    }
                if node_id not in instance.entities[called_name]["calling_chunk_ids"]:
                    if node_id in instance.graph and isinstance(instance.graph.nodes[node_id]["data"], ChunkNode):
                        instance.entities[called_name]["calling_chunk_ids"].append(node_id)

        if index_nodes:
            instance.logger.info("Building code index after deserialization.")
            # Merge use_embed with code_index_kwargs, avoiding duplicate keyword arguments
            code_idx_kwargs = code_index_kwargs or {}
            if 'use_embed' not in code_idx_kwargs:
                code_idx_kwargs['use_embed'] = use_embed
            instance.code_index = CodeIndex(list(instance), model_service=instance.model_service, **code_idx_kwargs)

        instance.logger.info("Deserialization complete.")
        return instance

    def _infer_level(self, node):
        """Infer the level of a node based on its type"""
        if node.node_type == 'root':
            return 0
        elif node.node_type in ('file', 'directory'):
            return 1
        elif node.node_type == 'chunk':
            return 2
        return 1  # Default level

    def save_graph_to_file(self, filepath: str):
        self.logger.info(f"Saving graph to file: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        self.logger.info("Graph saved successfully.")

    @classmethod
    def load_graph_from_file(cls, filepath: str, index_nodes=True, use_embed: bool = True,
                             model_service_kwargs: Optional[dict] = None, code_index_kwargs: Optional[dict] = None):
        if model_service_kwargs is None:
            model_service_kwargs = {}
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.getLogger(LOGGER_NAME).info(f"Loaded graph data from file: {filepath}")
        return cls.from_dict(data, use_embed=use_embed, index_nodes=index_nodes,
                             model_service_kwargs=model_service_kwargs, code_index_kwargs=code_index_kwargs)

    def get_neighbors(self, node_id):
        self.logger.debug(f"Getting neighbors for node: {node_id}")
        # Return all nodes that are directly connected to node_id (successors and predecessors) for any edge type
        neighbors = set()
        for n in self.graph.successors(node_id):
            neighbors.add(n)
        for n in self.graph.predecessors(node_id):
            neighbors.add(n)
        # Also include nodes connected by any edge (not just 'contains')
        for u, v in self.graph.edges(node_id):
            if u == node_id:
                neighbors.add(v)
            else:
                neighbors.add(u)
        for u, v in self.graph.in_edges(node_id):
            if v == node_id:
                neighbors.add(u)
            else:
                neighbors.add(v)
        return [self.graph.nodes[n]['data'] for n in neighbors if 'data' in self.graph.nodes[n]]

    def get_previous_chunk(self, node_id: str) -> ChunkNode:
        self.logger.debug(f"Getting previous chunk for node: {node_id}")
        node = self[node_id]
        # Check if node is of type ChunkNode
        if not isinstance(node, ChunkNode):
            raise Exception(f'Cannot get previous chunk on node of type {type(node)}')

        if node.order_in_file == 0:
            self.logger.warning(f'Cannot get previous chunk for first node')
            return None

        file_path = node.path
        previous_chunk_id = f'{file_path}_{node.order_in_file - 1}'

        if previous_chunk_id not in self.graph:
            raise Exception(f'Previous chunk {previous_chunk_id} not found in graph')

        previous_chunk = self[previous_chunk_id]
        return previous_chunk

    def get_next_chunk(self, node_id: str) -> ChunkNode:
        self.logger.debug(f"Getting next chunk for node: {node_id}")
        node = self[node_id]
        # Check if node is of type ChunkNode
        if not isinstance(node, ChunkNode):
            raise Exception(f'Cannot get previous chunk on node of type {type(node)}')

        file_path = node.path
        next_chunk_id = f'{file_path}_{node.order_in_file + 1}'

        if next_chunk_id not in self.graph:
            self.logger.warning(f'Next chunk {next_chunk_id} not found in graph, it might be the last chunk')
            return None
        previous_chunk = self[next_chunk_id]
        return previous_chunk

    def get_all_chunks(self) -> List[ChunkNode]:
        self.logger.debug("Getting all chunk nodes.")
        chunk_nodes = []
        for node in self:
            if isinstance(node, ChunkNode):
                chunk_nodes.append(node)
        return chunk_nodes

    def get_all_files(self) -> List[FileNode]:
        self.logger.debug("Getting all file nodes.")
        """
        Get all FileNodes in the knowledge graph.

        Returns:
            List[FileNode]: A list of FileNodes in the graph.
        """
        file_nodes = []
        for node in self.graph.nodes(data=True):
            node_data = node[1]['data']
            # Check for exact FileNode type, not ChunkNode (which inherits from FileNode)
            if isinstance(node_data, FileNode) and node_data.node_type == 'file':
                file_nodes.append(node_data)
        return file_nodes

    def get_chunks_of_file(self, file_node_id: str) -> List[ChunkNode]:
        self.logger.debug(f"Getting chunks for file node: {file_node_id}")
        """
        Get all ChunkNodes associated with a specific FileNode.

        Args:
            file_node (FileNode): The file node to get chunks for.

        Returns:
            List[ChunkNode]: A list of ChunkNodes associated with the file.
        """
        chunk_nodes = []
        for node in self.graph.neighbors(file_node_id):
            # Only include ChunkNodes that are connected by a 'contains' edge
            edge_data = self.graph.get_edge_data(file_node_id, node)
            node_data = self.graph.nodes[node]['data']
            if (
                    isinstance(node_data, ChunkNode)
                    and node_data.node_type == 'chunk'
                    and edge_data is not None
                    and edge_data.get('relation') == 'contains'
            ):
                chunk_nodes.append(node_data)
        return chunk_nodes

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> dict:
        """
        Find the shortest path between two nodes in the knowledge graph.

        Args:
            source_id (str): The ID of the source node.
            target_id (str): The ID of the target node.
            max_depth (int): Maximum depth to search for a path. Defaults to 5.

        Returns:
            dict: A dictionary containing path information or error message.
        """
        self.logger.debug(f"Finding path from {source_id} to {target_id} with max_depth={max_depth}")
        g = self.graph

        if source_id not in g:
            return {"error": f"Source node '{source_id}' not found."}
        if target_id not in g:
            return {"error": f"Target node '{target_id}' not found."}

        try:
            path = nx.shortest_path(g, source=source_id, target=target_id)

            if len(path) - 1 > max_depth:
                return {
                    "source_id": source_id,
                    "target_id": target_id,
                    "path": [],
                    "length": len(path) - 1,
                    "text": f"Path exists but exceeds max_depth of {max_depth} (actual length: {len(path) - 1})"
                }

            # Build detailed path information
            path_details = []
            for i, node_id in enumerate(path):
                node = g.nodes[node_id]['data']
                node_info = {
                    "node_id": node_id,
                    "name": getattr(node, 'name', 'Unknown'),
                    "type": getattr(node, 'node_type', 'Unknown'),
                    "step": i
                }

                # Add edge information for all but the last node
                if i < len(path) - 1:
                    next_node_id = path[i + 1]
                    edge_data = g.get_edge_data(node_id, next_node_id)
                    node_info["edge_to_next"] = edge_data.get('relation', 'Unknown') if edge_data else 'Unknown'

                path_details.append(node_info)

            # Format text output
            text = f"Path from '{source_id}' to '{target_id}' (length: {len(path) - 1}):\n\n"
            for i, node_info in enumerate(path_details):
                text += f"{i}. {node_info['name']} ({node_info['type']})\n"
                text += f"   Node ID: {node_info['node_id']}\n"
                if 'edge_to_next' in node_info:
                    text += f"   --[{node_info['edge_to_next']}]--> \n"

            return {
                "source_id": source_id,
                "target_id": target_id,
                "path": path_details,
                "length": len(path) - 1,
                "text": text
            }

        except nx.NetworkXNoPath:
            return {
                "source_id": source_id,
                "target_id": target_id,
                "path": [],
                "length": -1,
                "text": f"No path found between '{source_id}' and '{target_id}'"
            }
        except Exception as e:
            self.logger.error(f"Error finding path: {str(e)}")
            return {"error": f"Error finding path: {str(e)}"}

    def get_subgraph(self, node_id: str, depth: int = 2, edge_types: Optional[List[str]] = None) -> dict:
        """
        Extract a subgraph around a node up to a specified depth.

        Args:
            node_id (str): The ID of the central node.
            depth (int): The depth/radius of the subgraph to extract. Defaults to 2.
            edge_types (Optional[List[str]]): Optional list of edge types to include (e.g., ['calls', 'contains']).

        Returns:
            dict: A dictionary containing subgraph information or error message.
        """
        self.logger.debug(f"Getting subgraph for node {node_id} with depth={depth}, edge_types={edge_types}")
        g = self.graph

        if node_id not in g:
            return {"error": f"Node '{node_id}' not found."}

        # Collect nodes within specified depth
        nodes_at_depth = {node_id}
        all_nodes = {node_id}

        for d in range(depth):
            next_level = set()
            for n in nodes_at_depth:
                # Get all neighbors (both incoming and outgoing)
                for neighbor in g.successors(n):
                    if edge_types is None:
                        next_level.add(neighbor)
                    else:
                        edge_data = g.get_edge_data(n, neighbor)
                        if edge_data and edge_data.get('relation') in edge_types:
                            next_level.add(neighbor)

                for neighbor in g.predecessors(n):
                    if edge_types is None:
                        next_level.add(neighbor)
                    else:
                        edge_data = g.get_edge_data(neighbor, n)
                        if edge_data and edge_data.get('relation') in edge_types:
                            next_level.add(neighbor)

            nodes_at_depth = next_level - all_nodes
            all_nodes.update(next_level)

        # Extract subgraph
        subgraph = g.subgraph(all_nodes).copy()

        # Build node details
        nodes = []
        for n in subgraph.nodes():
            node = subgraph.nodes[n]['data']
            nodes.append({
                "node_id": n,
                "name": getattr(node, 'name', 'Unknown'),
                "type": getattr(node, 'node_type', 'Unknown')
            })

        # Build edge details
        edges = []
        for source, target, data in subgraph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "relation": data.get('relation', 'Unknown')
            })

        # Format text output
        text = f"Subgraph around '{node_id}' (depth: {depth}):\n"
        if edge_types:
            text += f"Edge types filter: {', '.join(edge_types)}\n"
        text += f"\nNodes: {len(nodes)}\n"
        text += f"Edges: {len(edges)}\n\n"

        # Group nodes by type
        nodes_by_type = {}
        for node in nodes:
            node_type = node['type']
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        for node_type, type_nodes in nodes_by_type.items():
            text += f"{node_type} ({len(type_nodes)}):\n"
            for node in type_nodes[:5]:
                text += f"  - {node['name']} ({node['node_id']})\n"
            if len(type_nodes) > 5:
                text += f"  ... and {len(type_nodes) - 5} more\n"
            text += "\n"

        # Show edge statistics
        edge_by_relation = {}
        for edge in edges:
            relation = edge['relation']
            edge_by_relation[relation] = edge_by_relation.get(relation, 0) + 1

        if edge_by_relation:
            text += "Edge types:\n"
            for relation, count in edge_by_relation.items():
                text += f"  - {relation}: {count}\n"

        return {
            "center_node_id": node_id,
            "depth": depth,
            "edge_types_filter": edge_types,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
            "nodes_by_type": nodes_by_type,
            "edge_by_relation": edge_by_relation,
            "text": text
        }
