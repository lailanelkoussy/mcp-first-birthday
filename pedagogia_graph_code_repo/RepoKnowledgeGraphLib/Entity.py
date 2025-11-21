from typing import Optional, Dict, List, Type, Any
from dataclasses import dataclass, field, asdict, fields, is_dataclass

# Helper for dynamic class lookup
ENTITY_TYPE_MAP = {}

def register_entity(cls):
    ENTITY_TYPE_MAP[cls.__name__] = cls
    return cls

def _entity_to_dict(obj):
    if isinstance(obj, list):
        return [_entity_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {(_entity_to_dict(k) if isinstance(k, Entity) else k): _entity_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Entity):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj

def _entity_from_dict(data):
    if isinstance(data, list):
        return [_entity_from_dict(item) for item in data]
    elif isinstance(data, dict) and 'entity_type' in data:
        cls = ENTITY_TYPE_MAP.get(data['entity_type'].capitalize(), Entity)
        return cls.from_dict(data)
    else:
        return data

@register_entity
@dataclass
class Entity:
    entity_type: str
    entity_name: str
    defined_chunk_id: str
    entity_dtype: str

    def to_dict(self):
        d = asdict(self)
        d['entity_type'] = self.entity_type
        d['__class__'] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, data):
        # Remove __class__ if present
        data = dict(data)
        data.pop('__class__', None)
        return cls(**data)

@register_entity
@dataclass
class Variable(Entity):
    entity_type = 'variable'

    def to_dict(self):
        d = super().to_dict()
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        return super().from_dict(data)

@register_entity
@dataclass
class Parameter(Entity):
    entity_type = 'parameter'
    entity_dtype: str

    def to_dict(self):
        d = super().to_dict()
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        return super().from_dict(data)

@register_entity
@dataclass
class Method(Entity):
    entity_type = 'method'
    parameters: List['Parameter'] = field(default_factory=list)
    associated_class: Optional['Class'] = None

    def to_dict(self):
        d = super().to_dict()
        d['parameters'] = _entity_to_dict(self.parameters)
        d['associated_class'] = self.associated_class.to_dict() if self.associated_class else None
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        params = [_entity_from_dict(p) for p in data.get('parameters', [])]
        assoc_cls = Class.from_dict(data['associated_class']) if data.get('associated_class') else None
        base = {k: v for k, v in data.items() if k not in ['parameters', 'parameters_pairs', 'associated_class']}
        return cls(parameters=params, associated_class=assoc_cls, **base)

@register_entity
@dataclass
class Class(Entity):
    entity_type = 'class'
    defined_methods: List['Method'] = field(default_factory=list)

    def to_dict(self):
        d = super().to_dict()
        d['defined_methods'] = _entity_to_dict(self.defined_methods)
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        methods = [_entity_from_dict(m) for m in data.get('defined_methods', [])]
        base = {k: v for k, v in data.items() if k != 'defined_methods'}
        return cls(defined_methods=methods, **base)

@register_entity
@dataclass
class Function(Entity):
    entity_type = 'function'
    parameters: List[Parameter] = field(default_factory=list)
    parameters_pairs: List[tuple] = field(default_factory=list)  # List of (Parameter, Variable)

    def to_dict(self):
        d = super().to_dict()
        d['parameters'] = _entity_to_dict(self.parameters)
        d['parameters_pairs'] = [ (p.to_dict(), v.to_dict()) for p, v in self.parameters_pairs ]
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        params = [_entity_from_dict(p) for p in data.get('parameters', [])]
        parameters_pairs = [(Parameter.from_dict(p), Variable.from_dict(v)) for p, v in data.get('parameters_pairs', [])]
        base = {k: v for k, v in data.items() if k not in ['parameters', 'parameters_pairs']}
        return cls(parameters=params, parameters_pairs=parameters_pairs, **base)

@register_entity
@dataclass
class FunctionCall(Entity):
    entity_type: str = 'function_call'
    entity_name: str = ''
    defined_chunk_id: str = ''
    entity_dtype: str = ''
    arguments: List[tuple] = field(default_factory=list)  # List of (Parameter, Variable)
    associated_functions: Optional[Function] = field(default_factory=list)


    def to_dict(self):
        d = super().to_dict()
        d['arguments'] = [ (p.to_dict(), v.to_dict()) for p, v in self.arguments ]
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        arguments = [(Parameter.from_dict(p), Variable.from_dict(v)) for p, v in data.get('arguments', [])]
        base = {k: v for k, v in data.items() if k != 'arguments'}
        return cls(arguments=arguments, **base)

@register_entity
@dataclass
class MethodCall(Entity):
    entity_type: str = 'method_call'
    entity_name: str = ''
    defined_chunk_id: str = ''
    entity_dtype: str = ''
    arguments: List[tuple] = field(default_factory=list)  # List of (Parameter, Variable)
    associated_class: Optional[Class] = None
    associated_method: Optional[Method] = None

    def to_dict(self):
        d = super().to_dict()
        d['arguments'] = [ (p.to_dict(), v.to_dict()) for p, v in self.arguments ]
        d['associated_class'] = self.associated_class.to_dict() if self.associated_class else None
        d['entity_type'] = self.entity_type
        return d

    @classmethod
    def from_dict(cls, data):
        arguments = [(Parameter.from_dict(p), Variable.from_dict(v)) for p, v in data.get('arguments', [])]
        assoc_cls = Class.from_dict(data['associated_class']) if data.get('associated_class') else None
        base = {k: v for k, v in data.items() if k not in ['arguments', 'associated_class']}
        return cls(arguments=arguments, associated_class=assoc_cls, **base)