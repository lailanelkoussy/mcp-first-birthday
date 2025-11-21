import os
import tempfile
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _extract_zip(path: Path) -> str:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


def _extract_tgz(path: Path) -> str:
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(path, 'r:gz') as tar_ref:
        tar_ref.extractall(temp_dir)
    return temp_dir


def prepare_input_path(path: str) -> str:
    """Handles different input types: directories, files, zip or tgz archives."""
    path_obj = Path(path)
    if path_obj.is_dir():
        return str(path_obj)

    if path_obj.suffix == '.zip':
        return _extract_zip(path_obj)
    elif path_obj.suffix in {'.tgz', '.tar.gz'}:
        return _extract_tgz(path_obj)
    elif path_obj.is_file():
        # Copy single file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        shutil.copy(path_obj, temp_dir)
        return temp_dir
    else:
        raise ValueError(f"Unsupported path type or extension: {path}")


def file_path_to_module_path(file_path: str) -> str:
    """
    Convert a file path to a module path by replacing path separators with dots
    and removing the file extension.

    Examples:
        path/to/repo/python_script.py -> path.to.repo.python_script
        src/utils/helper.py -> src.utils.helper
        module.py -> module

    Args:
        file_path: File path string

    Returns:
        Module path with dots instead of slashes
    """
    # Normalize path separators
    normalized = file_path.replace('\\', '/').replace(os.sep, '/')

    # Remove file extension
    without_ext = os.path.splitext(normalized)[0]

    # Replace / with .
    module_path = without_ext.replace('/', '.')

    return module_path


def generate_entity_aliases(entity_name: str, file_path: str) -> list:
    """
    Generate all possible aliases for an entity based on its name and file path.

    For example, if a file 'path/to/repo/python_script.py' defines 'Class_1',
    the aliases would be:
    - Class_1 (simple name)
    - path.to.repo.python_script.Class_1 (fully qualified from file path)

    For C++ namespaced entities like 'math::Calculator':
    - math::Calculator (fully qualified name)
    - Calculator (unqualified name, for use with 'using namespace')
    - math.calculator.math::Calculator (module-based fully qualified)

    For temporary paths like '.tmp.tmptqky4yk4..pyinstaller.run_astropy_tests.pos':
    - pos (simple name)
    - .run_astropy_tests.pos (progressive path removal)
    - pyinstaller.run_astropy_tests.pos (further removal)
    - .tmp.tmptqky4yk4..pyinstaller.run_astropy_tests.pos (full path)

    Args:
        entity_name: The name of the entity (e.g., 'Class_1', 'my_function', 'math::Calculator')
        file_path: The file path where the entity is defined

    Returns:
        List of alias strings
    """
    aliases = []

    # Always include the simple entity name
    aliases.append(entity_name)

    # For C++/C-style namespaced entities (using ::), add the unqualified name
    if '::' in entity_name:
        # Extract the unqualified name (last part after ::)
        unqualified_name = entity_name.split('::')[-1]
        if unqualified_name != entity_name:
            aliases.append(unqualified_name)

    # Generate module-based alias
    module_path = file_path_to_module_path(file_path)

    # If entity_name already contains scope separators (., ::),
    # it might be a nested entity (e.g., 'MyClass.my_method')
    # In this case, add the module path before the entire qualified name
    fully_qualified = f"{module_path}.{entity_name}"
    
    # Generate progressive path aliases by removing temporary/noise components
    # Split the module path into components
    components = module_path.split('.')
    
    # Filter out components that look like temporary directories or UUIDs
    def is_temp_component(component: str) -> bool:
        """Check if a path component looks like a temporary directory."""
        if not component:
            return True
        # Check for common temp directory patterns
        if component.startswith('tmp') and len(component) > 3:
            return True
        if component.startswith('.tmp'):
            return True
        # Check for UUID-like patterns (long alphanumeric strings)
        if len(component) > 8 and component.replace('_', '').replace('-', '').isalnum():
            # If it's mostly lowercase and has mix of letters and numbers, likely a temp ID
            if sum(c.islower() for c in component) > len(component) / 2:
                if sum(c.isdigit() for c in component) > 2:
                    return True
        return False
    
    # Generate aliases by progressively including more path components
    # Start from the rightmost meaningful components and work backwards
    clean_components = []
    for component in components:
        if not is_temp_component(component):
            clean_components.append(component)
    
    # Generate aliases with increasing path depth from meaningful components
    if clean_components:
        for i in range(1, len(clean_components) + 1):
            # Take the last i components
            partial_path = '.'.join(clean_components[-i:])
            partial_alias = f".{partial_path}.{entity_name}"
            if partial_alias != entity_name and partial_alias not in aliases:
                aliases.append(partial_alias)
            
            # Also add without leading dot for the full clean path
            if i == len(clean_components):
                no_dot_alias = f"{partial_path}.{entity_name}"
                if no_dot_alias != entity_name and no_dot_alias not in aliases:
                    aliases.append(no_dot_alias)
    
    # Always add the fully qualified path at the end (even if it contains temp components)
    if fully_qualified != entity_name and fully_qualified not in aliases:
        aliases.append(fully_qualified)

    return aliases


def normalize_include_path(include_path: str) -> str:
    """
    Normalize an include path from #include directive to a module-like path.

    Examples:
        <vector> -> vector
        <iostream> -> iostream
        "myheader.h" -> myheader
        "utils/helper.h" -> utils.helper
        <boost/algorithm/string.hpp> -> boost.algorithm.string

    Args:
        include_path: The include path from #include directive

    Returns:
        Normalized module-like path
    """
    # Remove angle brackets and quotes
    path = include_path.strip('<>"')

    # Convert to module path
    module_path = file_path_to_module_path(path)

    return module_path


def build_entity_alias_map(entities: Dict[str, Dict]) -> Dict[str, str]:
    """
    Build a mapping from all entity aliases to their canonical entity names.
    This allows quick lookup when matching called entities to their definitions.

    Args:
        entities: Dictionary of entity info keyed by canonical entity name

    Returns:
        Dictionary mapping alias -> canonical entity name
    """
    alias_map = {}

    for entity_name, info in entities.items():
        # Map the canonical name to itself
        alias_map[entity_name] = entity_name

        # Map all aliases to the canonical name
        aliases = info.get('aliases', [])
        for alias in aliases:
            if alias and alias not in alias_map:
                alias_map[alias] = entity_name

    return alias_map


def resolve_entity_call(called_name: str, alias_map: Dict[str, str],
                        imports: List[str] = None) -> Optional[str]:
    """
    Resolve a called entity name to its canonical definition using aliases.

    This handles cases like:
    - Direct call: 'MyClass' -> 'MyClass'
    - Qualified call: 'module.MyClass' -> 'MyClass' (if alias exists)
    - Imported call: 'helper' -> 'utils.helper' (if imported)
    - Simple name to qualified: 'Calculator' -> 'utils::Calculator'

    Args:
        called_name: The name of the called entity
        alias_map: Mapping from aliases to canonical entity names
        imports: List of import paths (optional, for context)

    Returns:
        Canonical entity name if found, None otherwise
    """
    # Don't try to resolve empty strings
    if not called_name or not called_name.strip():
        return None

    # Direct match
    if called_name in alias_map:
        return alias_map[called_name]

    # Try partial matches if imports are provided
    if imports:
        for import_path in imports:
            # Try combining import path with called name
            qualified = f"{import_path}.{called_name}"
            if qualified in alias_map:
                return alias_map[qualified]

            # Try with :: separator (C++/Rust style)
            qualified_cpp = f"{import_path}::{called_name}"
            if qualified_cpp in alias_map:
                return alias_map[qualified_cpp]

    # Try fuzzy matching - look for canonical names that end with the called name
    # This helps match 'Calculator' to 'utils::Calculator' or 'MyClass' to 'module.MyClass'
    simple_name = extract_simple_name(called_name)
    candidates = []

    for alias, canonical in alias_map.items():
        alias_simple = extract_simple_name(alias)
        # If the simple names match, this could be a match
        if alias_simple == simple_name:
            candidates.append(canonical)

    # If we found exactly one candidate, return it
    if len(candidates) == 1:
        return candidates[0]

    # If we have multiple candidates, prefer the shortest qualified name
    # (most likely to be the direct definition rather than an alias)
    if len(candidates) > 1:
        return min(candidates, key=lambda x: len(x))

    return None


def extract_simple_name(qualified_name: str) -> str:
    """
    Extract the simple name from a qualified name.

    Examples:
        'namespace::MyClass' -> 'MyClass'
        'module.MyClass' -> 'MyClass'
        'MyClass' -> 'MyClass'

    Args:
        qualified_name: Fully or partially qualified name

    Returns:
        Simple name without namespace/module prefix
    """
    # Handle C++ style namespace separator
    if '::' in qualified_name:
        return qualified_name.split('::')[-1]

    # Handle Python/JS style module separator
    if '.' in qualified_name:
        return qualified_name.split('.')[-1]

    return qualified_name

