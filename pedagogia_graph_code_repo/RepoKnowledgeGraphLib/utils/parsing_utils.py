import os
import re

def read_directory_files_recursively(directory_path: str, skip_dirs:list, skip_pattern: str = None) -> dict:
    """
    Recursively reads all files in a directory and its subdirectories.
    Skips files and directories that match the given regex pattern or are in skip_dirs.

    Args:
        directory_path (str): The path to start reading files from.
        skip_dirs (list): List of directory names to skip.
        skip_pattern (str, optional): Regex pattern to skip files/directories.

    Returns:
        dict: A dictionary where keys are relative file paths and values are file contents.
    """
    file_contents = {}
    compiled_pattern = re.compile(skip_pattern) if skip_pattern else None

    for root, dirs, files in os.walk(directory_path):
        # Skip directories listed in skip_dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs and not (compiled_pattern and compiled_pattern.search(os.path.join(root, d)))]

        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory_path)

            # Skip matching files
            if compiled_pattern and compiled_pattern.search(relative_path):
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    file_contents[relative_path] = f.read()
            except (UnicodeDecodeError, OSError) as e:
                print(f'Failed to read {relative_path}: {e}')
                continue
                #file_contents[relative_path] = f"<<Error reading file: {e}>>"

    return file_contents



def get_language_from_filename(file_name:str) -> str:
    file_extension = file_name.split('.')[-1]
    extension_mapping = {
        'c': 'c',
        'h': 'c',
        'cpp': 'c++',
        'cc': 'c++',
        'cxx': 'c++',
        'hpp': 'c++',
        'hh': 'c++',
        'hxx': 'c++',
        'go': 'go',
        'java': 'java',
        'py': 'python',
        'pyc': 'python',
        'pyw':'python',
        'js': 'javascript',
        'mjs': 'javascript',
        'cjs': 'javascript',
    }
    # Throws error if language not defined
    return extension_mapping.get(file_extension, file_extension)