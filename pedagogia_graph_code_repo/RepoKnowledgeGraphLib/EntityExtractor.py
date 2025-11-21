import ast
import os
import logging
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from clang import cindex
import javalang
import javalang.tree as T
import esprima
from bs4 import BeautifulSoup
import tree_sitter_rust as ts_rust
from tree_sitter import Language, Parser
import re
from .utils.path_utils import generate_entity_aliases



LOGGER_NAME = "AST_ENTITY_EXTRACTOR"
logger = logging.getLogger(LOGGER_NAME)


class BaseASTEntityExtractor:
    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract entities from source code.

        Args:
            code: Source code as string
            file_path: Optional path to the source file (for better context and include resolution)

        Returns:
            Tuple of (declared_entities, called_entities)
        """
        raise NotImplementedError


    # Add a reset contract so extractors can be reused safely
    def reset(self) -> None:
        """
        Reset internal state so the extractor instance can be reused.
        Concrete extractors should override this to clear their buffers.
        """
        raise NotImplementedError

class HTMLEntityExtractor(BaseASTEntityExtractor):
    """
    Hybrid HTML AST-based entity extractor.

    Responsibilities:
      • Parse HTML into a tree
      • Extract declared DOM entities (ids, names, classes)
      • Extract JavaScript calls from inline event handlers
      • Extract JS entities from <script> tags
      • Integrate cleanly with the hybrid AST graph linker
    """

    EVENT_ATTR_PREFIX = "on"  # e.g., onclick, onsubmit, etc.

    def __init__(self):
        self.js_extractor = JavaScriptEntityExtractor()
        self.reset()

    # --------------------------------------
    # Core interface
    # --------------------------------------
    def reset(self):
        self.declared_entities: List[Dict[str, str]] = []
        self.called_entities: List[str] = []

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, str]], List[str]]:
        """Main entry point: parse HTML and extract entities."""
        self.reset()
        try:
            soup = BeautifulSoup(code, "html.parser")
        except Exception as e:
            print(f"[HTMLEntityExtractor] Parsing error: {e}")
            return [], []

        # --- DOM element declarations ---
        for tag in soup.find_all(True):
            self._handle_tag_declaration(tag)
            self._handle_event_attributes(tag)

        # --- <script> tags (inline + external) ---
        for script in soup.find_all("script"):
            self._handle_script(script)

        # --- Deduplication ---
        self.declared_entities = self._deduplicate_dicts(self.declared_entities)
        self.called_entities = self._deduplicate_list(self.called_entities)

        return self.declared_entities, self.called_entities

    # --------------------------------------
    # Tag & attribute handlers
    # --------------------------------------
    def _handle_tag_declaration(self, tag):
        """Extract declared DOM elements (id, name, class)."""
        if tag.has_attr("id"):
            self.declared_entities.append({"name": tag["id"], "type": "element"})

        if tag.has_attr("name"):
            self.declared_entities.append({"name": tag["name"], "type": "element"})

        if tag.has_attr("class"):
            classes = tag["class"]
            if isinstance(classes, list):
                for c in classes:
                    self.declared_entities.append({"name": c, "type": "class"})
            elif isinstance(classes, str):
                self.declared_entities.append({"name": classes, "type": "class"})

    def _handle_event_attributes(self, tag):
        """Extract JS calls from inline event attributes."""
        if not self.js_extractor:
            return
        for attr, value in tag.attrs.items():
            if attr.lower().startswith(self.EVENT_ATTR_PREFIX) and isinstance(value, str):
                try:
                    _, called = self.js_extractor.extract_entities(value)
                    self.called_entities.extend(called)
                except Exception as e:
                    print(f"[HTMLEntityExtractor] JS parse error in {attr}: {e}")

    def _handle_script(self, script):
        """Extract JS entities from <script> blocks or src attributes."""
        if script.has_attr("src"):
            src = script["src"]
            self.called_entities.append(src)
            return

        if not self.js_extractor:
            return

        js_code = (script.string or "").strip()
        if js_code:
            try:
                declared, called = self.js_extractor.extract_entities(js_code)
                self.declared_entities.extend(declared)
                self.called_entities.extend(called)
            except Exception as e:
                print(f"[HTMLEntityExtractor] JS parse error in <script>: {e}")

    # --------------------------------------
    # Helpers
    # --------------------------------------
    @staticmethod
    def _deduplicate_dicts(dicts: List[Dict]) -> List[Dict]:
        seen = set()
        result = []
        for d in dicts:
            key = tuple(sorted(d.items()))
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    @staticmethod
    def _deduplicate_list(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for i in items:
            if i not in seen:
                seen.add(i)
                result.append(i)
        return result


class JavaEntityExtractor(BaseASTEntityExtractor):
    """
    Extract declared and called entities from Java code using javalang.
    Produces the same (declared_entities, called_entities) structure as other extractors.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.declared_entities: List[Dict[str, Any]] = []
        self.called_entities: List[str] = []
        self.current_package: Optional[str] = None
        self.scope_stack: List[str] = []
        self.api_endpoints: List[Dict[str, Any]] = []  # Track API endpoint definitions
        self.current_class_base_path: Optional[str] = None  # For @RequestMapping on class

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def _qualified(self, name: str) -> str:
        if not name:
            return ""
        scope = "::".join(self.scope_stack)
        return f"{scope}::{name}" if scope else name

    def _walk_type(self, t):
        """Return string representation of a type node."""
        if not t:
            return "unknown"
        if isinstance(t, str):
            return t
        if hasattr(t, "name"):
            name = t.name
            if getattr(t, "arguments", None):
                args = [self._walk_type(a.type) for a in t.arguments if hasattr(a, "type")]
                name += "<" + ", ".join(args) + ">"
            return name
        return "unknown"

    # -----------------------------------------------------------
    # Main AST traversal
    # -----------------------------------------------------------

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.reset()

        try:
            tree = javalang.parse.parse(code)
        except javalang.parser.JavaSyntaxError as e:
            logger.error(f"Syntax error in Java code: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Error parsing Java code: {e}", exc_info=True)
            return [], []

        # --- Package ---
        if tree.package:
            self.current_package = tree.package.name

        # --- Imports ---
        for imp in tree.imports:
            self.called_entities.append(imp.path)

        # --- Types (classes, interfaces, enums) ---
        for type_decl in tree.types:
            self._visit_type(type_decl)

        # Deduplicate
        seen_decl = set()
        unique_declared = []
        for e in self.declared_entities:
            key = (e.get("name"), e.get("type"), e.get("dtype"))
            if key not in seen_decl:
                unique_declared.append(e)
                seen_decl.add(key)

        unique_called = list(dict.fromkeys(self.called_entities))
        return unique_declared, unique_called

    # -----------------------------------------------------------
    # Visitors for different node types
    # -----------------------------------------------------------

    def _visit_type(self, node):
        if isinstance(node, javalang.tree.ClassDeclaration):
            self._visit_class(node)
        elif isinstance(node, javalang.tree.InterfaceDeclaration):
            self._visit_interface(node)
        elif isinstance(node, javalang.tree.EnumDeclaration):
            self._visit_enum(node)

    def _visit_class(self, node):
        full_name = node.name
        if self.current_package:
            full_name = f"{self.current_package}.{node.name}"
        qualified = self._qualified(full_name)

        self.declared_entities.append({"name": qualified, "type": "class"})

        # Check for REST controller annotations and extract base path
        old_base_path = self.current_class_base_path
        if node.annotations:
            for annotation in node.annotations:
                if annotation.name in {'RestController', 'Controller'}:
                    # Mark as REST controller
                    pass
                elif annotation.name == 'RequestMapping':
                    # Extract base path from class-level @RequestMapping
                    self.current_class_base_path = self._extract_path_from_annotation(annotation)

        # Inheritance
        if node.extends:
            self.called_entities.append(self._walk_type(node.extends))
        for impl in node.implements or []:
            self.called_entities.append(self._walk_type(impl))

        self.scope_stack.append(full_name)
        for member in node.body:
            self._visit_member(member)
        self.scope_stack.pop()

        # Restore the previous base path
        self.current_class_base_path = old_base_path

    def _visit_interface(self, node):
        full_name = node.name
        if self.current_package:
            full_name = f"{self.current_package}.{node.name}"
        qualified = self._qualified(full_name)
        self.declared_entities.append({"name": qualified, "type": "interface"})

        for impl in node.extends or []:
            self.called_entities.append(self._walk_type(impl))

        self.scope_stack.append(full_name)
        for member in node.body:
            self._visit_member(member)
        self.scope_stack.pop()

    def _visit_enum(self, node):
        full_name = node.name
        if self.current_package:
            full_name = f"{self.current_package}.{node.name}"
        qualified = self._qualified(full_name)
        self.declared_entities.append({"name": qualified, "type": "enum"})

    def _visit_member(self, node):

        # --- Method ---
        if isinstance(node, T.MethodDeclaration):
            method_name = self._qualified(node.name)

            # Check for API endpoint annotations
            api_info = self._extract_api_endpoint_from_annotations(node)
            if api_info:
                self.declared_entities.append({
                    "name": method_name,
                    "type": "api_endpoint",
                    "endpoint": api_info.get("endpoint"),
                    "methods": api_info.get("methods")
                })
                self.api_endpoints.append({**api_info, "function": method_name})
            else:
                self.declared_entities.append({"name": method_name, "type": "method"})

            for param in node.parameters:
                ptype = self._walk_type(param.type)
                pname = f"{method_name}.{param.name}"
                self.declared_entities.append({
                    "name": pname,
                    "type": "variable",
                    "dtype": ptype
                })

            # Look for method calls in the body
            if node.body:
                self._find_calls(node.body)

        # --- Constructor ---
        elif isinstance(node, T.ConstructorDeclaration):
            ctor_name = self._qualified(node.name)
            self.declared_entities.append({"name": ctor_name, "type": "constructor"})
            for param in node.parameters:
                ptype = self._walk_type(param.type)
                pname = f"{ctor_name}.{param.name}"
                self.declared_entities.append({
                    "name": pname,
                    "type": "variable",
                    "dtype": ptype
                })
            if node.body:
                self._find_calls(node.body)

        # --- Field ---
        elif isinstance(node, T.FieldDeclaration):
            dtype = self._walk_type(node.type)
            for decl in node.declarators:
                var_name = self._qualified(decl.name)
                self.declared_entities.append({
                    "name": var_name,
                    "type": "variable",
                    "dtype": dtype
                })

        # --- Nested class/interface ---
        elif isinstance(node, (T.ClassDeclaration, T.InterfaceDeclaration)):
            self._visit_type(node)

    # -----------------------------------------------------------
    # API Endpoint Detection
    # -----------------------------------------------------------

    def _extract_api_endpoint_from_annotations(self, method) -> Optional[Dict[str, Any]]:
        """
        Extract API endpoint information from Spring Boot method annotations.
        Handles: @GetMapping, @PostMapping, @RequestMapping, etc.
        """
        if not method.annotations:
            return None

        for annotation in method.annotations:
            annotation_name = annotation.name

            if annotation_name in {'GetMapping', 'PostMapping', 'PutMapping', 'PatchMapping', 'DeleteMapping'}:
                # Extract HTTP method from annotation name
                http_method = annotation_name.replace('Mapping', '').upper()
                path = self._extract_path_from_annotation(annotation)

                if path:
                    # Combine with class-level base path if present
                    full_path = self._combine_paths(self.current_class_base_path, path)
                    return {
                        "endpoint": full_path,
                        "methods": [http_method],
                        "type": "api_endpoint_definition"
                    }

            elif annotation_name == 'RequestMapping':
                # @RequestMapping can specify multiple methods
                path = self._extract_path_from_annotation(annotation)
                methods = self._extract_methods_from_annotation(annotation)

                if path:
                    full_path = self._combine_paths(self.current_class_base_path, path)
                    return {
                        "endpoint": full_path,
                        "methods": methods if methods else ['GET'],  # Default to GET
                        "type": "api_endpoint_definition"
                    }

        return None

    def _extract_path_from_annotation(self, annotation) -> Optional[str]:
        """Extract path/value from Spring annotation."""
        if not annotation.element:
            return None

        # Handle @GetMapping("/path") - single value
        if isinstance(annotation.element, T.Literal):
            return annotation.element.value.strip('"')

        # Handle @RequestMapping(value = "/path") or @RequestMapping(path = "/path")
        if isinstance(annotation.element, list):
            for elem in annotation.element:
                if isinstance(elem, T.ElementValuePair):
                    if elem.name in {'value', 'path'}:
                        if isinstance(elem.value, T.Literal):
                            return elem.value.value.strip('"')
                        elif isinstance(elem.value, T.ElementArrayValue):
                            # Handle array: value = {"/path1", "/path2"}
                            if elem.value.values:
                                first_val = elem.value.values[0]
                                if isinstance(first_val, T.Literal):
                                    return first_val.value.strip('"')

        return None

    def _extract_methods_from_annotation(self, annotation) -> List[str]:
        """Extract HTTP methods from @RequestMapping annotation."""
        methods = []

        if isinstance(annotation.element, list):
            for elem in annotation.element:
                if isinstance(elem, T.ElementValuePair):
                    if elem.name == 'method':
                        # Handle method = RequestMethod.GET or method = {RequestMethod.GET, RequestMethod.POST}
                        if hasattr(elem.value, 'member'):
                            # Single method: RequestMethod.GET
                            methods.append(elem.value.member)
                        elif isinstance(elem.value, T.ElementArrayValue):
                            # Multiple methods: {RequestMethod.GET, RequestMethod.POST}
                            for val in elem.value.values:
                                if hasattr(val, 'member'):
                                    methods.append(val.member)

        return methods

    def _combine_paths(self, base_path: Optional[str], path: str) -> str:
        """Combine base path from class annotation with method path."""
        if not base_path:
            return path

        # Normalize paths
        base = base_path.rstrip('/')
        path = path.lstrip('/')

        return f"{base}/{path}" if path else base

    # -----------------------------------------------------------
    # Find method invocations
    # -----------------------------------------------------------

    def _find_calls(self, statements):
        """Recursively find method and constructor calls inside Java AST nodes."""

        def _recurse(node):
            if isinstance(node, T.MethodInvocation):
                if node.qualifier:
                    self.called_entities.append(f"{node.qualifier}.{node.member}")
                else:
                    self.called_entities.append(node.member)
            elif isinstance(node, T.ClassCreator):
                self.called_entities.append(self._walk_type(node.type))

            # Recurse into all children
            if hasattr(node, '__dict__'):
                for attr, val in vars(node).items():
                    if isinstance(val, list):
                        for child in val:
                            if isinstance(child, T.Node):
                                _recurse(child)
                    elif isinstance(val, T.Node):
                        _recurse(val)

        if not statements:
            return

        if isinstance(statements, list):
            for stmt in statements:
                _recurse(stmt)
        else:
            _recurse(statements)


class JavaScriptEntityExtractor(BaseASTEntityExtractor):
    """
    Extract declared and called entities from JavaScript code using esprima.
    Handles ES6+ syntax including classes, arrow functions, imports/exports.
    Also detects API endpoint calls (fetch, axios, etc.).
    """

    # Common HTTP methods to detect
    HTTP_METHODS = {'get', 'post', 'put', 'patch', 'delete', 'head', 'options'}

    # API call patterns to detect
    API_PATTERNS = {
        'fetch',           # fetch('/api/users')
        'axios',           # axios.get('/api/users')
        '$http',           # Angular $http
        'request',         # request library
        'superagent',      # superagent library
    }

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.declared_entities: List[Dict[str, Any]] = []
        self.called_entities: List[str] = []
        self.scope_stack: List[str] = []
        self.api_calls: List[Dict[str, Any]] = []  # Track API endpoint calls

    def _qualified(self, name: str) -> str:
        """Return fully qualified name using current scope stack."""
        if not name:
            return ""
        scope = ".".join(self.scope_stack)
        return f"{scope}.{name}" if scope else name

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from various function node types."""
        if hasattr(node, 'id') and node.id:
            return node.id.name
        return None

    def _walk_node(self, node):
        """Recursively walk the AST and extract entities."""
        if not node or not hasattr(node, 'type'):
            return

        node_type = node.type

        # --- Function Declaration ---
        if node_type == 'FunctionDeclaration':
            func_name = self._get_function_name(node)
            if func_name:
                qualified = self._qualified(func_name)
                self.declared_entities.append({"name": qualified, "type": "function"})

                # Extract parameters
                if hasattr(node, 'params'):
                    for param in node.params:
                        param_name = self._extract_pattern_name(param)
                        if param_name:
                            self.declared_entities.append({
                                "name": f"{qualified}.{param_name}",
                                "type": "variable",
                                "dtype": "unknown"
                            })

                self.scope_stack.append(func_name)
                if hasattr(node, 'body'):
                    self._walk_node(node.body)
                self.scope_stack.pop()

        # --- Arrow Function Expression ---
        elif node_type == 'ArrowFunctionExpression':
            # Arrow functions are typically assigned, handle in VariableDeclarator
            if hasattr(node, 'params'):
                for param in node.params:
                    param_name = self._extract_pattern_name(param)
                    # Note: can't fully qualify without parent context
            if hasattr(node, 'body'):
                self._walk_node(node.body)

        # --- Function Expression ---
        elif node_type == 'FunctionExpression':
            func_name = self._get_function_name(node)
            if func_name:
                qualified = self._qualified(func_name)
                self.declared_entities.append({"name": qualified, "type": "function"})
                self.scope_stack.append(func_name)

            if hasattr(node, 'params'):
                for param in node.params:
                    param_name = self._extract_pattern_name(param)
                    if param_name and func_name:
                        self.declared_entities.append({
                            "name": f"{self._qualified(func_name)}.{param_name}",
                            "type": "variable",
                            "dtype": "unknown"
                        })

            if hasattr(node, 'body'):
                self._walk_node(node.body)

            if func_name:
                self.scope_stack.pop()

        # --- Class Declaration ---
        elif node_type == 'ClassDeclaration':
            class_name = node.id.name if hasattr(node, 'id') and node.id else None
            if class_name:
                qualified = self._qualified(class_name)
                self.declared_entities.append({"name": qualified, "type": "class"})

                # Handle inheritance
                if hasattr(node, 'superClass') and node.superClass:
                    if hasattr(node.superClass, 'name'):
                        self.called_entities.append(node.superClass.name)

                self.scope_stack.append(class_name)
                if hasattr(node, 'body') and hasattr(node.body, 'body'):
                    for method in node.body.body:
                        self._walk_node(method)
                self.scope_stack.pop()

        # --- Method Definition ---
        elif node_type == 'MethodDefinition':
            method_name = node.key.name if hasattr(node, 'key') and hasattr(node.key, 'name') else None
            if method_name:
                qualified = self._qualified(method_name)
                self.declared_entities.append({"name": qualified, "type": "method"})

                if hasattr(node, 'value') and hasattr(node.value, 'params'):
                    for param in node.value.params:
                        param_name = self._extract_pattern_name(param)
                        if param_name:
                            self.declared_entities.append({
                                "name": f"{qualified}.{param_name}",
                                "type": "variable",
                                "dtype": "unknown"
                            })

                if hasattr(node, 'value'):
                    self._walk_node(node.value)

        # --- Variable Declaration ---
        elif node_type == 'VariableDeclaration':
            if hasattr(node, 'declarations'):
                for decl in node.declarations:
                    self._walk_node(decl)

        # --- Variable Declarator ---
        elif node_type == 'VariableDeclarator':
            var_name = self._extract_pattern_name(node.id) if hasattr(node, 'id') else None
            if var_name:
                qualified = self._qualified(var_name)

                # Check if it's a function assignment
                if hasattr(node, 'init') and node.init:
                    if node.init.type in ('FunctionExpression', 'ArrowFunctionExpression'):
                        self.declared_entities.append({"name": qualified, "type": "function"})
                        self.scope_stack.append(var_name)
                        self._walk_node(node.init)
                        self.scope_stack.pop()
                    else:
                        self.declared_entities.append({
                            "name": qualified,
                            "type": "variable",
                            "dtype": "unknown"
                        })
                        self._walk_node(node.init)
                else:
                    self.declared_entities.append({
                        "name": qualified,
                        "type": "variable",
                        "dtype": "unknown"
                    })

        # --- Call Expression ---
        elif node_type == 'CallExpression':
            callee_name = self._extract_callee_name(node.callee) if hasattr(node, 'callee') else None
            if callee_name:
                self.called_entities.append(callee_name)

                # Detect API endpoint calls
                self._detect_api_call(node, callee_name)

            # Walk arguments
            if hasattr(node, 'arguments'):
                for arg in node.arguments:
                    self._walk_node(arg)

        # --- Member Expression ---
        elif node_type == 'MemberExpression':
            # Don't record as call, just traverse
            if hasattr(node, 'object'):
                self._walk_node(node.object)
            if hasattr(node, 'property'):
                self._walk_node(node.property)

        # --- Import/Export ---
        elif node_type == 'ImportDeclaration':
            if hasattr(node, 'source') and hasattr(node.source, 'value'):
                self.called_entities.append(node.source.value)

        elif node_type == 'ExportNamedDeclaration':
            if hasattr(node, 'declaration'):
                self._walk_node(node.declaration)

        elif node_type == 'ExportDefaultDeclaration':
            if hasattr(node, 'declaration'):
                self._walk_node(node.declaration)

        # --- Recursive traversal for other nodes ---
        else:
            if hasattr(node, '__dict__'):
                for attr, val in vars(node).items():
                    if isinstance(val, list):
                        for item in val:
                            if hasattr(item, 'type'):
                                self._walk_node(item)
                    elif hasattr(val, 'type'):
                        self._walk_node(val)

    def _extract_pattern_name(self, pattern) -> Optional[str]:
        """Extract name from various pattern types (Identifier, ObjectPattern, etc.)."""
        if not pattern:
            return None
        if hasattr(pattern, 'type'):
            if pattern.type == 'Identifier':
                return pattern.name if hasattr(pattern, 'name') else None
            elif pattern.type == 'RestElement':
                return self._extract_pattern_name(pattern.argument) if hasattr(pattern, 'argument') else None
        return None

    def _extract_callee_name(self, callee) -> Optional[str]:
        """Extract the name of the function being called."""
        if not callee:
            return None

        if hasattr(callee, 'type'):
            if callee.type == 'Identifier':
                return callee.name if hasattr(callee, 'name') else None
            elif callee.type == 'MemberExpression':
                obj = self._extract_callee_name(callee.object) if hasattr(callee, 'object') else ""
                prop = callee.property.name if hasattr(callee, 'property') and hasattr(callee.property, 'name') else ""
                if obj and prop:
                    return f"{obj}.{prop}"
                return prop or obj
        return None

    def _detect_api_call(self, call_node, callee_name: str):
        """
        Detect API endpoint calls in JavaScript code.
        Handles patterns like:
        - fetch('/api/users')
        - axios.get('/api/users')
        - axios.post('/api/users', data)
        - request.get('/api/users')
        """
        if not callee_name or not hasattr(call_node, 'arguments'):
            return

        # Split callee name to check for patterns
        parts = callee_name.split('.')
        base = parts[0]
        method = parts[-1].lower() if len(parts) > 1 else None

        # Check if this is an API call
        is_api_call = False
        http_method = 'unknown'

        # Pattern 1: fetch('/api/...')
        if base == 'fetch':
            is_api_call = True
            http_method = 'GET'  # Default for fetch

        # Pattern 2: axios.get('/api/...'), request.post(...), etc.
        elif base in self.API_PATTERNS and method in self.HTTP_METHODS:
            is_api_call = True
            http_method = method.upper()

        # Pattern 3: axios('/api/...', {method: 'POST'})
        elif base in self.API_PATTERNS and method is None:
            is_api_call = True
            http_method = 'GET'  # Default

        if not is_api_call:
            return

        # Extract the endpoint URL from arguments
        if call_node.arguments:
            first_arg = call_node.arguments[0]
            endpoint = self._extract_string_literal(first_arg)

            if endpoint:
                # Store as a called entity with special type
                self.called_entities.append(f"API:{http_method}:{endpoint}")

                # Also track in api_calls for easier filtering
                self.api_calls.append({
                    "endpoint": endpoint,
                    "method": http_method,
                    "type": "api_call"
                })

    def _extract_string_literal(self, node) -> Optional[str]:
        """Extract string value from a Literal/TemplateLiteral node."""
        if not node or not hasattr(node, 'type'):
            return None

        if node.type == 'Literal' and isinstance(node.value, str):
            return node.value
        elif node.type == 'TemplateLiteral':
            # For template literals, we try to extract the quasi parts
            # e.g., `/api/${version}/users` -> /api/{version}/users
            if hasattr(node, 'quasis'):
                parts = []
                for i, quasi in enumerate(node.quasis):
                    if hasattr(quasi, 'value') and hasattr(quasi.value, 'raw'):
                        parts.append(quasi.value.raw)
                    if i < len(node.quasis) - 1:
                        parts.append('{param}')
                return ''.join(parts)

        return None

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.reset()

        try:
            tree = esprima.parseScript(code, {'tolerant': True, 'loc': False})
        except Exception as e:
            # Try parsing as module if script fails
            try:
                tree = esprima.parseModule(code, {'tolerant': True, 'loc': False})
            except Exception as e2:
                logger.error(f"Failed to parse JavaScript code: {e2}")
                return [], []

        if hasattr(tree, 'body'):
            for node in tree.body:
                self._walk_node(node)

        # Deduplicate
        seen_decl = set()
        unique_declared = []
        for e in self.declared_entities:
            key = (e.get("name"), e.get("type"), e.get("dtype"))
            if key not in seen_decl:
                unique_declared.append(e)
                seen_decl.add(key)

        unique_called = list(dict.fromkeys(self.called_entities))
        return unique_declared, unique_called


class CEntityExtractor(BaseASTEntityExtractor):
    """
    Extract declared and called entities from C code using clang.cindex (libclang),
    with filtering to ignore system headers.
    """

    def __init__(self):
        self.index = cindex.Index.create()

    def reset(self) -> None:
        """No persistent state to reset, but method provided for interface consistency."""
        pass

    def _walk_cursor(self, cursor, declared, called, source_file):
        """Recursively walk a clang Cursor, restricted to the main file."""
        for c in cursor.get_children():
            # --- Include directives ---
            # Note: INCLUSION_DIRECTIVE nodes are at the root level and need special handling
            if c.kind == cindex.CursorKind.INCLUSION_DIRECTIVE:
                # Get the included file name
                included_file = c.displayname
                if included_file:
                    called.append(included_file)
                continue

            loc = c.location
            if not loc.file or not source_file:
                continue

            # Skip system / external headers for other nodes
            if os.path.abspath(loc.file.name) != os.path.abspath(source_file):
                continue

            # --- Declarations ---
            if c.kind.is_declaration():
                if c.kind in (cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.FUNCTION_TEMPLATE):
                    name = c.spelling or c.displayname
                    declared.append({"name": name, "type": "function"})
                    for p in c.get_arguments():
                        declared.append({
                            "name": f"{name}.{p.spelling}",
                            "type": "variable",
                            "dtype": p.type.spelling
                        })
                elif c.kind == cindex.CursorKind.VAR_DECL:
                    declared.append({
                        "name": c.spelling,
                        "type": "variable",
                        "dtype": c.type.spelling
                    })

                    # Add the variable's type to called entities
                    # This captures struct references like "struct Point p;"
                    if c.type.spelling:
                        # Extract the base type name (remove const, &, *, struct keyword, etc.)
                        type_name = c.type.spelling.strip()
                        # Remove common qualifiers and keywords
                        type_name = type_name.replace('const', '').replace('&', '').replace('*', '').replace('struct', '').strip()
                        if type_name and not type_name in ['int', 'float', 'double', 'char', 'bool', 'void', 'long', 'short', 'unsigned', 'signed', 'size_t']:
                            called.append(type_name)
                elif c.kind == cindex.CursorKind.STRUCT_DECL:
                    declared.append({"name": c.spelling or c.displayname, "type": "struct"})
                elif c.kind == cindex.CursorKind.TYPEDEF_DECL:
                    declared.append({"name": c.spelling, "type": "typedef"})

            # --- Calls ---
            if c.kind == cindex.CursorKind.CALL_EXPR:
                callee = None
                for child in c.get_children():
                    if child.kind in (cindex.CursorKind.DECL_REF_EXPR, cindex.CursorKind.MEMBER_REF_EXPR):
                        callee = child.spelling
                        break
                if callee:
                    called.append(callee)
                else:
                    called.append(c.displayname or c.spelling)

            # --- Recurse ---
            self._walk_cursor(c, declared, called, source_file)

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        declared, called = [], []

        # If file_path is provided, use it directly for better include resolution
        # Otherwise, create a temporary file
        tf_name = None
        temp_file = False

        if file_path and os.path.exists(file_path):
            tf_name = file_path
            temp_file = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w+", delete=False) as tf:
                tf_name = tf.name
                tf.write(code)
                tf.flush()
            temp_file = True

        # Get the directory containing the file for include paths
        include_dir = os.path.dirname(tf_name) if tf_name else None
        args = ['-std=c11']
        if include_dir:
            args.append(f'-I{include_dir}')

        try:
            tu = self.index.parse(
                tf_name,
                args=args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
        except Exception as e:
            raise RuntimeError(f"libclang failed to parse translation unit: {e}")

        self._walk_cursor(tu.cursor, declared, called, tf_name)

        # Deduplicate
        seen_decl = set()
        unique_declared = []
        for e in declared:
            key = (e.get("name"), e.get("type"), e.get("dtype", None))
            if key not in seen_decl:
                unique_declared.append(e)
                seen_decl.add(key)

        unique_called = list(dict.fromkeys(called))

        # Only delete if we created a temp file
        if temp_file:
            try:
                os.unlink(tf_name)
            except Exception:
                pass

        return unique_declared, unique_called


class CppEntityExtractor(BaseASTEntityExtractor):
    """
    Extract declared and called entities from C++ code using clang.cindex (libclang),
    including classes, namespaces, and methods.
    """

    def __init__(self):
        self.index = cindex.Index.create()
        self.reset()

    def reset(self) -> None:
        self.declared_entities = []
        self.called_entities = []
        self.scope_stack = []

    def _qualified(self, name: str) -> str:
        """Return fully qualified name using current scope stack."""
        if not name:
            return ""
        if not self.scope_stack:
            return name
        return "::".join(self.scope_stack + [name])

    def _walk_cursor(self, cursor, source_file: str):
        for c in cursor.get_children():
            # --- Include directives ---
            # Note: INCLUSION_DIRECTIVE nodes are at the root level and need special handling
            if c.kind == cindex.CursorKind.INCLUSION_DIRECTIVE:
                # Get the included file name
                included_file = c.displayname
                if included_file:
                    self.called_entities.append(included_file)
                continue

            kind = c.kind

            # --- Namespace --- (process before location check)
            if kind == cindex.CursorKind.NAMESPACE:
                if c.spelling:  # Only add non-empty namespace names
                    self.scope_stack.append(c.spelling)
                self._walk_cursor(c, source_file)
                if c.spelling:
                    self.scope_stack.pop()
                continue

            # Check location for other node types
            loc = c.location
            # Skip nodes from other files, but allow nodes without location info
            if loc.file and os.path.abspath(loc.file.name) != os.path.abspath(source_file):
                continue

            # --- Class / Struct ---
            if kind in (cindex.CursorKind.CLASS_DECL, cindex.CursorKind.STRUCT_DECL):
                # Only process if it has a name
                if c.spelling:
                    # Check if it's a definition (not a forward declaration)
                    is_def = c.is_definition() if hasattr(c, 'is_definition') else True
                    if is_def:
                        full_name = self._qualified(c.spelling)
                        self.declared_entities.append({"name": full_name, "type": "class"})

                        # Handle base classes (inheritance)
                        for base in c.get_children():
                            if base.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                                if base.spelling:
                                    self.called_entities.append(base.spelling)

                        self.scope_stack.append(c.spelling)
                        self._walk_cursor(c, source_file)
                        self.scope_stack.pop()
                continue

            # --- Methods ---
            if kind in (cindex.CursorKind.CXX_METHOD, cindex.CursorKind.CONSTRUCTOR, cindex.CursorKind.DESTRUCTOR):
                if c.spelling:  # Only process if it has a name
                    full_name = self._qualified(c.spelling)
                    self.declared_entities.append({"name": full_name, "type": "method"})

                    for p in c.get_arguments():
                        if p.spelling:  # Only add parameters with names
                            self.declared_entities.append({
                                "name": f"{full_name}.{p.spelling}",
                                "type": "variable",
                                "dtype": p.type.spelling
                            })

                self._walk_cursor(c, source_file)
                continue

            # --- Free functions ---
            if kind == cindex.CursorKind.FUNCTION_DECL:
                if c.spelling:  # Only process if it has a name
                    full_name = self._qualified(c.spelling)
                    self.declared_entities.append({"name": full_name, "type": "function"})
                    for p in c.get_arguments():
                        if p.spelling:  # Only add parameters with names
                            self.declared_entities.append({
                                "name": f"{full_name}.{p.spelling}",
                                "type": "variable",
                                "dtype": p.type.spelling
                            })
                self._walk_cursor(c, source_file)
                continue

            # --- Variables ---
            if kind == cindex.CursorKind.VAR_DECL:
                full_name = self._qualified(c.spelling)
                self.declared_entities.append({
                    "name": full_name,
                    "type": "variable",
                    "dtype": c.type.spelling
                })

                # Look for TYPE_REF children which explicitly reference the type
                # This is more reliable than c.type.spelling when includes aren't resolved
                type_ref_found = False
                for child in c.get_children():
                    if child.kind == cindex.CursorKind.TYPE_REF:
                        # TYPE_REF.spelling gives us the fully qualified type name
                        # It may have 'class ' or 'struct ' prefix, so strip it
                        if child.spelling:
                            type_name = child.spelling.replace('class ', '').replace('struct ', '').strip()
                            if type_name:
                                # TYPE_REF gives us the canonical name from the definition,
                                # which includes namespace qualifiers if present.
                                # We only add this canonical name and rely on alias resolution
                                # to match unqualified usage (e.g., 'Calculator' -> 'math::Calculator')
                                self.called_entities.append(type_name)
                                type_ref_found = True
                                break

                # Fallback: use c.type.spelling if no TYPE_REF found
                # Note: c.type.spelling may give us the name as written in source code,
                # which could be unqualified even if it refers to a namespaced type
                if not type_ref_found and c.type.spelling:
                    # Extract the base type name (remove const, &, *, etc.)
                    type_name = c.type.spelling.strip()
                    # Remove common qualifiers
                    type_name = type_name.replace('const', '').replace('&', '').replace('*', '').strip()
                    if type_name and not type_name in ['int', 'float', 'double', 'char', 'bool', 'void', 'long', 'short', 'unsigned', 'signed']:
                        # Only add if not already added via TYPE_REF
                        # c.type.spelling might give unqualified name even for namespaced types
                        # We'll add it and let alias resolution handle it
                        self.called_entities.append(type_name)

            # --- Calls ---
            if kind == cindex.CursorKind.CALL_EXPR:
                callee = None
                for child in c.get_children():
                    if child.kind in (cindex.CursorKind.DECL_REF_EXPR, cindex.CursorKind.MEMBER_REF_EXPR):
                        callee = child.spelling
                        break
                if callee:
                    self.called_entities.append(callee)
                else:
                    self.called_entities.append(c.displayname or c.spelling)

            # Recurse
            self._walk_cursor(c, source_file)

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        self.reset()

        # If file_path is provided, use it directly for better include resolution
        # Otherwise, create a temporary file
        tf_name = None
        temp_file = False

        if file_path and os.path.exists(file_path):
            tf_name = file_path
            temp_file = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w+", delete=False) as tf:
                tf_name = tf.name
                tf.write(code)
                tf.flush()
            temp_file = True

        # Get the directory containing the file for include paths
        include_dir = os.path.dirname(tf_name) if tf_name else None
        args = ['-std=c++17', '-xc++']
        if include_dir:
            args.append(f'-I{include_dir}')

        try:
            tu = self.index.parse(
                tf_name,
                args=args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
        except Exception as e:
            raise RuntimeError(f"libclang failed to parse C++ translation unit: {e}")

        self._walk_cursor(tu.cursor, tf_name)

        # Deduplicate
        seen_decl = set()
        unique_declared = []
        for e in self.declared_entities:
            key = (e.get("name"), e.get("type"), e.get("dtype", None))
            if key not in seen_decl:
                unique_declared.append(e)
                seen_decl.add(key)

        unique_called = list(dict.fromkeys(self.called_entities))

        # Only delete if we created a temp file
        if temp_file:
            try:
                os.unlink(tf_name)
            except Exception:
                pass

        return unique_declared, unique_called


class RustEntityExtractor(BaseASTEntityExtractor):
    """
    Extract declared and called entities from Rust code using tree-sitter.
    Handles structs, enums, traits, functions, methods, and modules.
    Also detects API endpoint definitions (Actix-web, Rocket, Axum, Warp).
    """

    # HTTP method route macros for Rust web frameworks
    ROUTE_MACROS = {
        'get', 'post', 'put', 'patch', 'delete', 'head', 'options',  # Actix-web, Rocket
        'Get', 'Post', 'Put', 'Patch', 'Delete', 'Head', 'Options',  # Alternative casing
    }

    # Route-related macros and functions
    ROUTE_PATTERNS = {
        'route',        # Generic route macro
        'web::get', 'web::post', 'web::put', 'web::delete',  # Actix-web with web::
    }

    def __init__(self):

        self.parser = Parser()
        self.parser.language = Language(ts_rust.language())
        self.reset()

    def reset(self) -> None:
        self.declared_entities = []
        self.called_entities = []
        self.scope_stack = []
        self.api_endpoints: List[Dict[str, Any]] = []  # Track API endpoint definitions

    def _qualified(self, name: str) -> str:
        """Return fully qualified name using current scope stack."""
        if not name:
            return ""
        if not self.scope_stack:
            return name
        return "::".join(self.scope_stack + [name])

    def _get_node_text(self, node, code_bytes: bytes) -> str:
        """Extract text content of a node."""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    def _extract_api_endpoint_from_attributes(self, node, code_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Extract API endpoint information from Rust function attributes.
        Handles patterns like:
        - #[get("/users")]                    # Actix-web, Rocket
        - #[post("/users")]                   # Actix-web, Rocket
        - #[route("/users", method="GET")]    # Generic route

        Note: In tree-sitter Rust AST, attributes appear as PREVIOUS SIBLINGS
        of the function_item node, not as children.
        """


        # Get the parent node to access siblings
        parent = node.parent
        if not parent:
            return None

        # Find the index of current node in parent's children
        node_index = None
        for i, child in enumerate(parent.children):
            if child == node:
                node_index = i
                break

        if node_index is None:
            return None

        # Look backwards through previous siblings for attribute_item nodes
        for i in range(node_index - 1, -1, -1):
            sibling = parent.children[i]

            # Stop if we hit a non-attribute node (except comments/whitespace)
            if sibling.type not in ['attribute_item', 'line_comment', 'block_comment']:
                break

            if sibling.type == 'attribute_item':
                attr_text = self._get_node_text(sibling, code_bytes)

                # Match HTTP method macros: #[get("/path")], #[post("/path")], #[post("/path", data = "<var>")], etc.
                # The pattern now allows optional additional parameters after the path
                method_pattern = r'#\[(get|post|put|patch|delete|head|options)\s*\(\s*"([^"]+)"(?:\s*,.*?)?\s*\)\]'
                match = re.search(method_pattern, attr_text, re.IGNORECASE)

                if match:
                    http_method = match.group(1).upper()
                    endpoint_path = match.group(2)
                    return {
                        "endpoint": endpoint_path,
                        "methods": [http_method],
                        "type": "api_endpoint_definition"
                    }

                # Match generic route macro: #[route("/path", method="GET")]
                route_pattern = r'#\[route\s*\(\s*"([^"]+)"(?:.*?method\s*=\s*"([^"]+)")?\s*\)\]'
                match = re.search(route_pattern, attr_text, re.IGNORECASE)

                if match:
                    endpoint_path = match.group(1)
                    http_method = match.group(2).upper() if match.group(2) else "GET"
                    return {
                        "endpoint": endpoint_path,
                        "methods": [http_method],
                        "type": "api_endpoint_definition"
                    }

        return None

    def _walk_tree(self, node, code_bytes: bytes):
        """Recursively walk the tree-sitter AST."""
        node_type = node.type

        # --- Module declarations ---
        if node_type == 'mod_item':
            # mod my_module { ... }
            name_node = node.child_by_field_name('name')
            if name_node:
                mod_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(mod_name)
                self.declared_entities.append({"name": qualified, "type": "module"})

                self.scope_stack.append(mod_name)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        self._walk_tree(child, code_bytes)
                self.scope_stack.pop()
                return

        # --- Struct declarations ---
        elif node_type == 'struct_item':
            name_node = node.child_by_field_name('name')
            if name_node:
                struct_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(struct_name)
                self.declared_entities.append({"name": qualified, "type": "struct"})

                # Check for generic parameters
                type_params = node.child_by_field_name('type_parameters')
                if type_params:
                    self._walk_tree(type_params, code_bytes)

                self.scope_stack.append(struct_name)
                # Process fields
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        if child.type == 'field_declaration':
                            field_name_node = child.child_by_field_name('name')
                            field_type_node = child.child_by_field_name('type')
                            if field_name_node:
                                field_name = self._get_node_text(field_name_node, code_bytes)
                                field_type = self._get_node_text(field_type_node, code_bytes) if field_type_node else "unknown"
                                self.declared_entities.append({
                                    "name": f"{qualified}.{field_name}",
                                    "type": "field",
                                    "dtype": field_type
                                })
                self.scope_stack.pop()
                return

        # --- Enum declarations ---
        elif node_type == 'enum_item':
            name_node = node.child_by_field_name('name')
            if name_node:
                enum_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(enum_name)
                self.declared_entities.append({"name": qualified, "type": "enum"})

                self.scope_stack.append(enum_name)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        if child.type == 'enum_variant':
                            variant_name_node = child.child_by_field_name('name')
                            if variant_name_node:
                                variant_name = self._get_node_text(variant_name_node, code_bytes)
                                self.declared_entities.append({
                                    "name": f"{qualified}::{variant_name}",
                                    "type": "enum_variant"
                                })
                self.scope_stack.pop()
                return

        # --- Trait declarations ---
        elif node_type == 'trait_item':
            name_node = node.child_by_field_name('name')
            if name_node:
                trait_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(trait_name)
                self.declared_entities.append({"name": qualified, "type": "trait"})

                self.scope_stack.append(trait_name)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        self._walk_tree(child, code_bytes)
                self.scope_stack.pop()
                return

        # --- Implementation blocks ---
        elif node_type == 'impl_item':
            # impl MyStruct { ... } or impl Trait for MyStruct { ... }
            type_node = node.child_by_field_name('type')
            trait_node = node.child_by_field_name('trait')

            impl_name = None
            if type_node:
                impl_name = self._get_node_text(type_node, code_bytes)

            if trait_node:
                trait_name = self._get_node_text(trait_node, code_bytes)
                self.called_entities.append(trait_name)

            if impl_name:
                self.scope_stack.append(impl_name)

            body = node.child_by_field_name('body')
            if body:
                for child in body.children:
                    self._walk_tree(child, code_bytes)

            if impl_name:
                self.scope_stack.pop()
            return

        # --- Function declarations ---
        elif node_type == 'function_item':
            name_node = node.child_by_field_name('name')
            if name_node:
                func_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(func_name)

                # Check for API endpoint attributes (e.g., #[get("/users")])
                api_info = self._extract_api_endpoint_from_attributes(node, code_bytes)

                if api_info:
                    # This is an API endpoint handler
                    self.declared_entities.append({
                        "name": qualified,
                        "type": "api_endpoint",
                        "endpoint": api_info.get("endpoint"),
                        "methods": api_info.get("methods")
                    })
                    self.api_endpoints.append({**api_info, "function": qualified})
                    entity_type = "api_endpoint"
                else:
                    # Determine if this is a method (inside impl block) or free function
                    entity_type = "method" if len(self.scope_stack) > 0 else "function"
                    self.declared_entities.append({"name": qualified, "type": entity_type})

                # Extract parameters
                params = node.child_by_field_name('parameters')
                if params:
                    for child in params.children:
                        if child.type == 'parameter':
                            pattern = child.child_by_field_name('pattern')
                            type_node = child.child_by_field_name('type')
                            if pattern:
                                param_name = self._get_node_text(pattern, code_bytes)
                                param_type = self._get_node_text(type_node, code_bytes) if type_node else "unknown"
                                # Skip 'self' parameters
                                if param_name not in ['self', '&self', '&mut self', 'mut self']:
                                    self.declared_entities.append({
                                        "name": f"{qualified}.{param_name}",
                                        "type": "variable",
                                        "dtype": param_type
                                    })

                # Walk the function body to find calls
                body = node.child_by_field_name('body')
                if body:
                    self._walk_tree(body, code_bytes)
                return

        # --- Type alias ---
        elif node_type == 'type_item':
            name_node = node.child_by_field_name('name')
            if name_node:
                type_name = self._get_node_text(name_node, code_bytes)
                qualified = self._qualified(type_name)
                self.declared_entities.append({"name": qualified, "type": "type_alias"})
                return

        # --- Constant declarations ---
        elif node_type == 'const_item':
            name_node = node.child_by_field_name('name')
            type_node = node.child_by_field_name('type')
            if name_node:
                const_name = self._get_node_text(name_node, code_bytes)
                const_type = self._get_node_text(type_node, code_bytes) if type_node else "unknown"
                qualified = self._qualified(const_name)
                self.declared_entities.append({
                    "name": qualified,
                    "type": "constant",
                    "dtype": const_type
                })

        # --- Static declarations ---
        elif node_type == 'static_item':
            name_node = node.child_by_field_name('name')
            type_node = node.child_by_field_name('type')
            if name_node:
                static_name = self._get_node_text(name_node, code_bytes)
                static_type = self._get_node_text(type_node, code_bytes) if type_node else "unknown"
                qualified = self._qualified(static_name)
                self.declared_entities.append({
                    "name": qualified,
                    "type": "static",
                    "dtype": static_type
                })

        # --- Let bindings (local variables) ---
        elif node_type == 'let_declaration':
            pattern = node.child_by_field_name('pattern')
            type_node = node.child_by_field_name('type')
            if pattern and pattern.type == 'identifier':
                var_name = self._get_node_text(pattern, code_bytes)
                var_type = self._get_node_text(type_node, code_bytes) if type_node else "unknown"
                # Only track top-level or module-level variables, not function-local ones
                # For now, we skip local variables to avoid clutter

        # --- Use declarations (imports) ---
        elif node_type == 'use_declaration':
            # Extract imported items
            use_text = self._get_node_text(node, code_bytes)
            self.called_entities.append(use_text)

        # --- Call expressions ---
        elif node_type == 'call_expression':
            function = node.child_by_field_name('function')
            if function:
                func_text = self._get_node_text(function, code_bytes)
                # Clean up function call to get just the name/path
                # Handle method calls like obj.method() and path calls like std::vec::Vec::new()
                self.called_entities.append(func_text)

        # --- Macro invocations ---
        elif node_type == 'macro_invocation':
            macro_node = node.child_by_field_name('macro')
            if macro_node:
                macro_name = self._get_node_text(macro_node, code_bytes)
                self.called_entities.append(f"{macro_name}!")

        # --- Field expressions (method calls or field access) ---
        elif node_type == 'field_expression':
            field = node.child_by_field_name('field')
            if field:
                field_name = self._get_node_text(field, code_bytes)
                # This could be a field access or method call, record it
                # We don't have full context here, so just record the field name

        # Recursively walk all children
        for child in node.children:
            self._walk_tree(child, code_bytes)

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extract entities from Rust code using tree-sitter."""
        self.reset()

        code_bytes = code.encode('utf8')
        tree = self.parser.parse(code_bytes)

        # Walk the AST
        self._walk_tree(tree.root_node, code_bytes)

        # Deduplicate
        seen_decl = set()
        unique_declared = []
        for e in self.declared_entities:
            key = (e.get("name"), e.get("type"), e.get("dtype", None))
            if key not in seen_decl:
                unique_declared.append(e)
                seen_decl.add(key)

        unique_called = list(dict.fromkeys(self.called_entities))

        return unique_declared, unique_called


class PythonASTEntityExtractor(ast.NodeVisitor, BaseASTEntityExtractor):
    """
    AST-based entity extractor for Python code.
    Also detects API endpoint definitions (FastAPI, Flask, Django REST Framework).
    """

    # Common HTTP decorators/patterns for Python web frameworks
    API_DECORATORS = {
        'route',           # Flask @app.route
        'get', 'post', 'put', 'patch', 'delete', 'head', 'options',  # FastAPI/Flask methods
        'api_view',        # DRF @api_view
    }

    def __init__(self):
        self.declared_entities: List[Dict[str, Any]] = []
        self.called_entities: List[str] = []
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.api_endpoints: List[Dict[str, Any]] = []  # Track API endpoint definitions

    def reset(self) -> None:
        """Clear previous extraction state including context"""
        self.declared_entities = []
        self.called_entities = []
        self.current_class = None
        self.current_function = None
        self.api_endpoints = []

    def _get_type_annotation(self, node: ast.AST) -> str:
        """Extract type annotation from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.Attribute):
            return f"{self._get_type_annotation(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            # Handle generic types like List[str], Dict[str, int]
            base = self._get_type_annotation(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = [self._get_type_annotation(elt) for elt in node.slice.elts]
                return f"{base}[{', '.join(args)}]"
            else:
                arg = self._get_type_annotation(node.slice)
                return f"{base}[{arg}]"
        return "unknown"

    def _infer_type_from_value(self, node: ast.AST) -> str:
        """Infer type from assigned value"""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id  # Constructor call
            elif isinstance(node.func, ast.Attribute):
                return "unknown"
        elif isinstance(node, ast.Name):
            return "unknown"  # Reference to another variable
        return "unknown"

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions"""
        old_class = self.current_class
        self.current_class = node.name

        # Add class to declared entities
        self.declared_entities.append({
            "name": node.name,
            "type": "class"
        })

        # Record base classes as called entities
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.called_entities.append(base.id)
            elif isinstance(base, ast.Attribute):
                self.called_entities.append(self._get_type_annotation(base))

        # Continue visiting child nodes
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function/method definitions and detect API endpoints"""
        old_function = self.current_function

        if self.current_class:
            # This is a method
            full_name = f"{self.current_class}.{node.name}"
            entity_type = "method"
        else:
            # This is a function
            full_name = node.name
            entity_type = "function"

        self.current_function = full_name

        # Check for API endpoint decorators
        api_info = self._extract_api_endpoint_from_decorators(node.decorator_list, full_name)
        if api_info:
            # Mark this as an API endpoint
            self.declared_entities.append({
                "name": full_name,
                "type": "api_endpoint",
                "endpoint": api_info.get("endpoint"),
                "methods": api_info.get("methods")
            })
            self.api_endpoints.append(api_info)
        else:
            self.declared_entities.append({
                "name": full_name,
                "type": entity_type
            })

        # Process parameters
        for arg in node.args.args:
            if arg.arg == 'self' and self.current_class:
                continue  # Skip self parameter

            dtype = "unknown"
            if arg.annotation:
                dtype = self._get_type_annotation(arg.annotation)

            param_name = f"{full_name}.{arg.arg}" if entity_type == "method" else arg.arg
            self.declared_entities.append({
                "name": param_name,
                "type": "variable",
                "dtype": dtype
            })

        # Continue visiting child nodes
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function/method definitions"""
        # Treat async functions the same as regular functions
        self.visit_FunctionDef(node)

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statements"""
        # Infer type from the assigned value
        dtype = self._infer_type_from_value(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple variable assignment
                var_name = target.id
                if self.current_class and self.current_function and self.current_function.startswith(self.current_class):
                    # Local variable in method
                    pass  # Could add local variables if needed
                else:
                    # Module-level variable
                    self.declared_entities.append({
                        "name": var_name,
                        "type": "variable",
                        "dtype": dtype
                    })

            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                # Attribute assignment like self.name = value
                if target.value.id == 'self' and self.current_class:
                    attr_name = f"{self.current_class}.{target.attr}"
                    self.declared_entities.append({
                        "name": attr_name,
                        "type": "variable",
                        "dtype": dtype
                    })

        # Continue visiting to catch function calls in the assignment
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment statements (PEP 526)"""
        if isinstance(node.target, ast.Name):
            dtype = self._get_type_annotation(node.annotation)
            var_name = node.target.id

            self.declared_entities.append({
                "name": var_name,
                "type": "variable",
                "dtype": dtype
            })

        elif isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name):
            if node.target.value.id == 'self' and self.current_class:
                dtype = self._get_type_annotation(node.annotation)
                attr_name = f"{self.current_class}.{node.target.attr}"
                self.declared_entities.append({
                    "name": attr_name,
                    "type": "variable",
                    "dtype": dtype
                })

        # Continue visiting
        if node.value:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Visit import statements"""
        for alias in node.names:
            # Record the imported module/package
            self.called_entities.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statements"""
        if node.module:
            # Record the module being imported from
            self.called_entities.append(node.module)
            # Optionally, also record specific imports as module.name
            for alias in node.names:
                if alias.name != '*':
                    self.called_entities.append(f"{node.module}.{alias.name}")
        else:
            # Relative imports without module (from . import x)
            for alias in node.names:
                if alias.name != '*':
                    self.called_entities.append(alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Visit function/method calls"""
        if isinstance(node.func, ast.Name):
            # Simple function call
            self.called_entities.append(node.func.id)

        elif isinstance(node.func, ast.Attribute):
            # Method call or attribute access
            if isinstance(node.func.value, ast.Name):
                # obj.method() - we need to infer the class of obj
                # For now, just record the method name
                method_name = node.func.attr
                # Try to find the variable type from our declared entities
                obj_name = node.func.value.id
                obj_class = self._find_variable_type(obj_name)
                if obj_class and obj_class != "unknown":
                    self.called_entities.append(f"{obj_class}.{method_name}")
                else:
                    # Fallback: just record the method call
                    self.called_entities.append(method_name)

            elif isinstance(node.func.value, ast.Attribute):
                # Nested attribute access like module.Class.method()
                full_name = self._get_type_annotation(node.func)
                self.called_entities.append(full_name)

        # Continue visiting child nodes
        self.generic_visit(node)

    def _find_variable_type(self, var_name: str) -> str:
        """Find the type of a variable from declared entities"""
        for entity in self.declared_entities:
            if entity["name"] == var_name and entity["type"] == "variable":
                return entity.get("dtype", "unknown")
        return "unknown"

    def _extract_api_endpoint_from_decorators(self, decorators: List[ast.expr], function_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract API endpoint information from function decorators.
        Handles patterns like:
        - @app.route("/api/users", methods=["GET", "POST"])  # Flask
        - @app.get("/api/users")                              # FastAPI
        - @router.post("/api/users")                          # FastAPI with router
        - @api_view(['GET', 'POST'])                          # Django REST Framework
        """
        for decorator in decorators:
            # Handle @app.route(...) or @app.get(...)
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    # e.g., app.route, app.get, router.post
                    method_name = decorator.func.attr.lower()

                    if method_name in self.API_DECORATORS:
                        endpoint = None
                        http_methods = []

                        # Extract endpoint from first positional argument
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            endpoint = decorator.args[0].value

                        # For FastAPI-style decorators (@app.get, @app.post)
                        if method_name in {'get', 'post', 'put', 'patch', 'delete', 'head', 'options'}:
                            http_methods = [method_name.upper()]

                        # For Flask-style @app.route with methods kwarg
                        elif method_name == 'route':
                            for keyword in decorator.keywords:
                                if keyword.arg == 'methods':
                                    if isinstance(keyword.value, ast.List):
                                        http_methods = [
                                            elt.value for elt in keyword.value.elts
                                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                        ]
                            if not http_methods:
                                http_methods = ['GET']  # Flask default

                        # For DRF @api_view(['GET', 'POST'])
                        elif method_name == 'api_view':
                            if decorator.args and isinstance(decorator.args[0], ast.List):
                                http_methods = [
                                    elt.value for elt in decorator.args[0].elts
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                ]

                        if endpoint:
                            return {
                                "function": function_name,
                                "endpoint": endpoint,
                                "methods": http_methods,
                                "type": "api_endpoint_definition"
                            }

        return None

    def extract_entities(self, code: str, file_path: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract entities from Python code using AST parsing

        Args:
            code: Python source code as string
            file_path: Optional path to the source file (for context)

        Returns:
            Tuple of (declared_entities, called_entities)
        """
        # Ensure fresh state on each extraction
        self.reset()

        try:
            tree = ast.parse(code)
            self.visit(tree)

            # Remove duplicates while preserving order
            seen_declared = set()
            unique_declared = []
            for entity in self.declared_entities:
                key = (entity["name"], entity["type"], entity.get("dtype"))
                if key not in seen_declared:
                    unique_declared.append(entity)
                    seen_declared.add(key)

            unique_called = list(dict.fromkeys(self.called_entities))  # Remove duplicates

            return unique_declared, unique_called

        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Error parsing Python code: {e}", exc_info=True)
            return [], []


class HybridEntityExtractor:
    """
    Hybrid entity extractor that uses AST for known languages,
    falls back to LLM for unknown ones
    """

    def __init__(self):
        self.extractors = {
            'py': PythonASTEntityExtractor(),
            'c': CEntityExtractor(),
            'h': CppEntityExtractor(),  # C/C++ headers
            'cpp': CppEntityExtractor(),
            'cc': CppEntityExtractor(),
            'cxx': CppEntityExtractor(),
            'hpp': CppEntityExtractor(),
            'hxx': CppEntityExtractor(),
            'hh': CppEntityExtractor(),
            'java': JavaEntityExtractor(),
            'js': JavaScriptEntityExtractor(),  # ✅ NEW
            'jsx': JavaScriptEntityExtractor(),  # ✅ NEW
            'ts': JavaScriptEntityExtractor(),  # TypeScript uses similar AST
            'tsx': JavaScriptEntityExtractor(),  # TSX similar to JSX
            'rs': RustEntityExtractor(),
            'html': HTMLEntityExtractor()
        }

    def _get_language_from_filename(self, file_name: str) -> str:
        ext = file_name.split('.')[-1].lower()
        return ext

    def extract_entities(self, code: str, file_name: str):

        lang = self._get_language_from_filename(file_name)
        extractor = self.extractors.get(lang)

        if extractor:
            # Reset the shared extractor instance to ensure no state is carried over
            try:
                extractor.reset()
            except Exception:
                # If extractor doesn't implement reset for some reason, ignore and proceed
                pass

            logger.info(f"Using AST extraction for {lang.upper()} file: {file_name}")
            try:
                # Try to pass file_name if the extractor supports it (C++ extractor does)
                try:
                    declared_entities, called_entities = extractor.extract_entities(code, file_path=file_name)
                except TypeError:
                    # Fallback for extractors that don't accept file_path parameter
                    declared_entities, called_entities = extractor.extract_entities(code)

                # Add aliases to each declared entity based on file path
                for entity in declared_entities:
                    entity_name = entity.get('name', '')
                    if entity_name:
                        aliases = generate_entity_aliases(entity_name, file_name)
                        entity['aliases'] = aliases
                        logger.debug(f"Generated aliases for entity '{entity_name}': {aliases}")

                return declared_entities, called_entities
            except Exception as e:
                logger.error(f"Error during AST extraction for file {file_name}: {e}", exc_info=True)
                return [], []
        else:
            raise Exception(f"Using LLM extraction for unsupported language: {file_name}")
