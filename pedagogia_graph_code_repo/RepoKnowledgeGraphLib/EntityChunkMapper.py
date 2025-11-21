import logging
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from enum import Enum


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    C = "c"
    CPP = "cpp"
    JAVA = "java"


class EntityChunkMapper:
    """Maps entities from file-level extraction back to their respective chunks"""

    def __init__(self):
        self.logger = logging.getLogger("ENTITY_CHUNK_MAPPER")
        self.extension_to_language = {
            'py': Language.PYTHON,
            'pyw': Language.PYTHON,
            'c': Language.C,
            'h': Language.C,
            'cpp': Language.CPP,
            'cc': Language.CPP,
            'cxx': Language.CPP,
            'hpp': Language.CPP,
            'hh': Language.CPP,
            'hxx': Language.CPP,
            'java': Language.JAVA,
        }

    def _detect_language(self, file_name: Optional[str] = None) -> Language:
        """
        Detect the programming language from file extension

        Args:
            file_name: Name of the file (optional)

        Returns:
            Language enum value, defaults to PYTHON if not detected
        """
        if file_name:
            extension = file_name.split('.')[-1].lower()
            return self.extension_to_language.get(extension, Language.PYTHON)
        return Language.PYTHON

    def _is_comment_or_docstring(self, line: str, in_docstring: bool, language: Language) -> Tuple[bool, bool]:
        """
        Check if a line is a comment or part of a docstring/multi-line comment

        Args:
            line: The line to check
            in_docstring: Whether we're currently inside a docstring/multi-line comment
            language: The programming language

        Returns:
            Tuple of (is_comment_or_docstring, new_in_docstring_state)
        """
        stripped = line.strip()

        if language == Language.PYTHON:
            # Check for single-line comments
            if stripped.startswith('#'):
                return True, in_docstring

            # Check for docstring delimiters (""" or ''')
            triple_double = '"""'
            triple_single = "'''"

            # Count occurrences of triple quotes
            if triple_double in stripped or triple_single in stripped:
                # Check if it's a single-line docstring
                if (stripped.count(triple_double) >= 2 or
                    stripped.count(triple_single) >= 2):
                    # Single-line docstring
                    return True, in_docstring
                else:
                    # Toggle docstring state
                    return True, not in_docstring

            # If we're in a docstring, this line is part of it
            if in_docstring:
                return True, in_docstring

        elif language in [Language.C, Language.CPP, Language.JAVA]:
            # Check for single-line comments
            if stripped.startswith('//'):
                return True, in_docstring

            # Check for multi-line comment delimiters /* */
            if '/*' in line and '*/' in line:
                # Single-line multi-line comment
                return True, in_docstring
            elif '/*' in line:
                # Start of multi-line comment
                return True, True
            elif '*/' in line:
                # End of multi-line comment
                return True, False

            # If we're in a multi-line comment
            if in_docstring:
                return True, in_docstring

        return False, in_docstring

    def _get_code_lines(self, chunk_lines: List[str], language: Language) -> List[str]:
        """
        Filter out comments and docstrings from chunk lines

        Args:
            chunk_lines: List of lines in the chunk
            language: The programming language

        Returns:
            List of lines that are actual code (not comments or docstrings)
        """
        code_lines = []
        in_docstring = False

        for line in chunk_lines:
            is_doc, in_docstring = self._is_comment_or_docstring(line, in_docstring, language)
            if not is_doc:
                code_lines.append(line)

        return code_lines

    def _is_valid_identifier_match(self, text: str, identifier: str, position: int) -> bool:
        """
        Check if an identifier match at a position is valid (not part of another word)

        Args:
            text: The text containing the identifier
            identifier: The identifier to check
            position: The position where the identifier was found

        Returns:
            True if this is a valid standalone identifier match
        """
        # Check character before (if exists)
        if position > 0:
            char_before = text[position - 1]
            if char_before.isalnum() or char_before == '_':
                return False

        # Check character after (if exists)
        end_pos = position + len(identifier)
        if end_pos < len(text):
            char_after = text[end_pos]
            if char_after.isalnum() or char_after == '_':
                return False

        return True

    def _contains_identifier(self, line: str, identifier: str) -> bool:
        """
        Check if a line contains an identifier as a standalone word (not part of another word)

        Args:
            line: The line to check
            identifier: The identifier to find

        Returns:
            True if the identifier appears as a standalone word
        """
        # Use word boundary regex for precise matching
        pattern = r'\b' + re.escape(identifier) + r'\b'
        return bool(re.search(pattern, line))


    def find_entity_in_chunks(self, entity_name: str, chunks: List[str], entity_type: str = None,
                            file_name: Optional[str] = None) -> Set[int]:
        """
        Find which chunks contain a specific entity declaration or call

        Args:
            entity_name: Name of the entity to find
            chunks: List of code chunks
            entity_type: Type of entity (class, function, method, variable)
            file_name: Name of the file to detect language (optional)

        Returns:
            Set of chunk indices that contain this entity
        """
        matching_chunks = set()
        language = self._detect_language(file_name)

        # Split the entity name to handle nested entities like "ClassName.method"
        # For Java/C++, also handle :: separator
        if '::' in entity_name:
            parts = entity_name.split('::')
        else:
            parts = entity_name.split('.')
        base_name = parts[-1]  # The actual identifier

        for chunk_idx, chunk in enumerate(chunks):
            chunk_lines = chunk.strip().split('\n')

            # Look for different patterns based on entity type
            if self._entity_appears_in_chunk(entity_name, base_name, chunk, chunk_lines, entity_type, language):
                matching_chunks.add(chunk_idx)

        return matching_chunks

    def _entity_appears_in_chunk(self, full_name: str, base_name: str, chunk: str, chunk_lines: List[str],
                                 entity_type: str, language: Language) -> bool:
        """Check if an entity appears in a specific chunk (excluding comments and docstrings)"""

        # Filter out comments and docstrings
        code_lines = self._get_code_lines(chunk_lines, language)

        # If no code lines remain, entity doesn't appear in actual code
        if not code_lines:
            return False

        # Language-specific entity matching
        if language == Language.PYTHON:
            return self._entity_appears_in_python(full_name, base_name, code_lines, entity_type)
        elif language in [Language.C, Language.CPP]:
            return self._entity_appears_in_c_cpp(full_name, base_name, code_lines, entity_type)
        elif language == Language.JAVA:
            return self._entity_appears_in_java(full_name, base_name, code_lines, entity_type)

        return False

    def _entity_appears_in_python(self, full_name: str, base_name: str, code_lines: List[str],
                                  entity_type: str) -> bool:
        """Check if entity appears in Python code"""

        if entity_type == "class":
            # Look for class definition
            for line in code_lines:
                stripped = line.strip()
                if re.match(rf'class\s+{re.escape(base_name)}[\s:(]', stripped):
                    return True

        elif entity_type == "api_endpoint":
            # Look for API endpoint definition - the function decorated with @app.get, @app.post, etc.
            # We look for the function definition itself
            for line in code_lines:
                stripped = line.strip()
                # Match the function definition with the endpoint name
                if re.match(rf'(async\s+)?def\s+{re.escape(base_name)}\s*\(', stripped):
                    return True
                # Also check for decorators that might reference the endpoint
                if re.search(rf'@\w+\.(get|post|put|delete|patch|options|head)\s*\(', stripped):
                    return True

        elif entity_type == "function":
            # Look for function definition (not method)
            for line in code_lines:
                stripped = line.strip()
                # Check it's not indented (not a method)
                if not line.startswith("    ") and not line.startswith("\t"):
                    if re.match(rf'(async\s+)?def\s+{re.escape(base_name)}\s*\(', stripped):
                        return True

        elif entity_type == "method":
            # Look for method definition (indented def)
            method_name = full_name.split('.')[-1]
            for line in code_lines:
                stripped = line.strip()
                # Check it's indented (is a method)
                if line.startswith("    ") or line.startswith("\t"):
                    if re.match(rf'(async\s+)?def\s+{re.escape(method_name)}\s*\(', stripped):
                        return True

        elif entity_type == "variable":
            # Look for variable assignment or usage
            if "." in full_name:
                parts = full_name.split('.')
                attr_name = parts[-1]
                for line in code_lines:
                    if re.search(rf'\.\s*{re.escape(attr_name)}\b', line):
                        return True
            else:
                for line in code_lines:
                    stripped = line.strip()
                    if re.match(rf'{re.escape(base_name)}\s*[=:]', stripped):
                        return True

        # For called entities, look for usage patterns
        if entity_type in ["function", "method"] or entity_type is None:
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\s*\(', line):
                    return True

        if entity_type == "class" or entity_type is None:
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\s*\(', line):
                    return True

        # General usage as identifier
        if entity_type is None or entity_type == "variable":
            for line in code_lines:
                if self._contains_identifier(line, base_name):
                    return True

        return False

    def _extract_using_namespace_directives(self, code_lines: List[str]) -> List[str]:
        """
        Extract using namespace directives from C++ code.
        Returns a list of namespace names that are being imported.
        """
        namespaces = []
        for line in code_lines:
            stripped = line.strip()
            # Match "using namespace <name>;"
            match = re.match(r'using\s+namespace\s+([a-zA-Z_][a-zA-Z0-9_:]*)\s*;', stripped)
            if match:
                namespaces.append(match.group(1))
        return namespaces

    def _entity_appears_in_c_cpp(self, full_name: str, base_name: str, code_lines: List[str],
                                 entity_type: str) -> bool:
        """Check if entity appears in C/C++ code"""

        # Extract using namespace directives
        using_namespaces = self._extract_using_namespace_directives(code_lines)
        
        # Check if the full_name matches any imported namespace + base_name
        # e.g., if full_name is "math::Calculator" and we have "using namespace math",
        # then "Calculator" in code should match
        namespace_match = False
        if '::' in full_name:
            for ns in using_namespaces:
                # Check if full_name starts with this namespace
                if full_name.startswith(ns + '::'):
                    namespace_match = True
                    break

        if entity_type == "class":
            # Look for class/struct definition
            for line in code_lines:
                stripped = line.strip()
                if re.match(rf'(class|struct)\s+{re.escape(base_name)}[\s:{{]', stripped):
                    return True

        elif entity_type == "function":
            # Look for function definition or declaration
            for line in code_lines:
                stripped = line.strip()
                # Match function patterns: return_type function_name(
                # Also handle constructors and destructors
                if (re.search(rf'\b{re.escape(base_name)}\s*\(', stripped) and
                    not stripped.startswith('//')):
                    # Additional check: likely a function if followed by parameters
                    return True

        elif entity_type == "method":
            # Look for method definition (with class scope)
            method_name = full_name.split('::')[-1] if '::' in full_name else full_name.split('.')[-1]
            for line in code_lines:
                stripped = line.strip()
                # Match ClassName::methodName( or just methodName( inside class
                if re.search(rf'\b{re.escape(method_name)}\s*\(', stripped):
                    return True

        elif entity_type == "variable":
            # Look for variable declaration or usage
            for line in code_lines:
                stripped = line.strip()
                # Match variable declarations and assignments
                if re.search(rf'\b{re.escape(base_name)}\b', stripped):
                    return True

        # For called entities, look for usage patterns
        if entity_type in ["function", "method"] or entity_type is None:
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\s*\(', line):
                    return True

        if entity_type == "class" or entity_type is None:
            # Look for instantiation or usage
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\b', line):
                    # If we found base_name and there's a namespace match, this is a match
                    if namespace_match:
                        return True
                    # If full_name doesn't have a namespace, it's a direct match
                    if '::' not in full_name:
                        return True

        # General usage as identifier
        if entity_type is None or entity_type == "variable":
            for line in code_lines:
                if self._contains_identifier(line, base_name):
                    # If we found base_name and there's a namespace match, this is a match
                    if namespace_match:
                        return True
                    # If full_name doesn't have a namespace, it's a direct match
                    if '::' not in full_name:
                        return True

        return False

    def _entity_appears_in_java(self, full_name: str, base_name: str, code_lines: List[str],
                                entity_type: str) -> bool:
        """Check if entity appears in Java code"""

        if entity_type == "class":
            # Look for class/interface/enum definition
            for line in code_lines:
                stripped = line.strip()
                if re.match(rf'(public|private|protected)?\s*(class|interface|enum)\s+{re.escape(base_name)}[\s<{{]', stripped):
                    return True
                # Without modifier
                if re.match(rf'(class|interface|enum)\s+{re.escape(base_name)}[\s<{{]', stripped):
                    return True

        elif entity_type == "api_endpoint":
            # Look for API endpoint definition - the method with Spring annotations
            # Extract just the method name from the full qualified name (e.g., "com.example.Controller::method" -> "method")
            method_name = base_name.split('::')[-1] if '::' in base_name else base_name
            for line in code_lines:
                stripped = line.strip()
                # Match the method definition
                if re.search(rf'\b{re.escape(method_name)}\s*\(', stripped):
                    return True
                # Also check for Spring annotations
                if re.search(r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)', stripped):
                    return True

        elif entity_type == "function":
            # In Java, functions are methods
            for line in code_lines:
                stripped = line.strip()
                # Match method signature patterns
                if re.search(rf'\b{re.escape(base_name)}\s*\(', stripped):
                    return True

        elif entity_type == "method":
            # Look for method definition
            method_name = full_name.split('.')[-1]
            for line in code_lines:
                stripped = line.strip()
                if re.search(rf'\b{re.escape(method_name)}\s*\(', stripped):
                    return True

        elif entity_type == "variable":
            # Look for variable declaration or usage
            for line in code_lines:
                stripped = line.strip()
                if re.search(rf'\b{re.escape(base_name)}\b', stripped):
                    return True

        # For called entities, look for usage patterns
        if entity_type in ["function", "method"] or entity_type is None:
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\s*\(', line):
                    return True

        if entity_type == "class" or entity_type is None:
            # Look for instantiation (new ClassName) or usage
            for line in code_lines:
                if re.search(rf'\b{re.escape(base_name)}\b', line):
                    return True

        # General usage as identifier
        if entity_type is None or entity_type == "variable":
            for line in code_lines:
                if self._contains_identifier(line, base_name):
                    return True

        return False

    def map_entities_to_chunks(self, declared_entities: List[Dict[str, Any]],
                               called_entities: List[str],
                               chunks: List[str],
                               file_name: Optional[str] = None) -> Tuple[Dict[int, List[Dict[str, Any]]],
    Dict[int, List[str]]]:
        """
        Map file-level entities back to their respective chunks

        Args:
            declared_entities: List of declared entities from file-level extraction
            called_entities: List of called entities from file-level extraction
            chunks: List of code chunks
            file_name: Name of the file to detect language (optional)

        Returns:
            Tuple of (chunk_declared_entities, chunk_called_entities)
            - chunk_declared_entities: Dict mapping chunk_index -> list of declared entities
            - chunk_called_entities: Dict mapping chunk_index -> list of called entities
        """
        chunk_declared = {}
        chunk_called = {}

        # Initialize empty lists for all chunks
        for i in range(len(chunks)):
            chunk_declared[i] = []
            chunk_called[i] = []

        # Map declared entities to chunks
        for entity in declared_entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")

            matching_chunks = self.find_entity_in_chunks(entity_name, chunks, entity_type, file_name)

            # Add entity to matching chunks
            for chunk_idx in matching_chunks:
                chunk_declared[chunk_idx].append(entity)

        # Map called entities to chunks
        for called_entity in called_entities:
            matching_chunks = self.find_entity_in_chunks(called_entity, chunks, None, file_name)

            # Add called entity to matching chunks
            for chunk_idx in matching_chunks:
                if called_entity not in chunk_called[chunk_idx]:
                    chunk_called[chunk_idx].append(called_entity)

        return chunk_declared, chunk_called
