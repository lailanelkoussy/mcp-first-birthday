from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

from .Entity import Entity

@dataclass
class Node:
    id: str = ''
    name: str = ''
    node_type: str = ''
    description: Optional[str] = None
    declared_entities: List[dict] = field(default_factory=list)  # Classes, functions, variables
    called_entities: List[str] = field(  default_factory=list)  # Classes, functions, variables, but also external libraries

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

@dataclass
class DirectoryNode(Node):
    path: str = ''
    node_type: str = 'directory'


@dataclass
class FileNode(Node):
    path: str = ''
    content: str = ''
    node_type: str = 'file'
    language : str = ''


@dataclass
class ChunkNode(FileNode):
    node_type: str = 'chunk'
    order_in_file: int = field(default_factory=int)
    embedding : list = None

    def get_field_to_embed(self) -> Optional[str]:
        # Use description if available, otherwise fall back to content
        # This ensures we always have something meaningful to embed
        if self.description and self.description.strip():
            return self.description
        return self.content


@dataclass 
class EntityNode(Node):
    entity_type: str = ''
    declaring_chunk_ids: List[str] = field(default_factory=list)
    calling_chunk_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)  # All possible aliases for this entity
    node_type: str = 'entity'

    def __post_init__(self):
        # Use entity_name (stored in name field) as the id if id is not set
        if not self.id and self.name:
            self.id = self.name

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

    def get_field_to_embed(self) -> Optional[str]:
        return self.name