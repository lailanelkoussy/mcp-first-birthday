from ..Node import ChunkNode
from typing import List, Dict

def dict_to_chunknode(d: dict) -> ChunkNode:
    """
    Converts a dictionary to a ChunkNode instance.
    """
    return ChunkNode(**d)

def extract_filename_from_chunk(chunk:ChunkNode) -> str:
    """
    Extracts the file name from a chunk.

    Args:
        chunk (str): The chunk from which to extract the file name.

    Returns:
        str: The extracted file name.
    """
    if isinstance(chunk, dict):
        chunk = dict_to_chunknode(chunk)
    return '_'.join(chunk.id.split('_')[:-1])
    

def order_chunks_by_order_in_file(chunks:List[ChunkNode]) -> list:
    """
    Orders a list of chunks by their order in the file.

    Args:
        chunks (list): The list of chunks to order.

    Returns:
        list: The ordered list of chunks.
    """
    # Convert dicts to ChunkNode if needed
    chunks = [dict_to_chunknode(c) if isinstance(c, dict) else c for c in chunks]
    return sorted(chunks, key=lambda x: int(x.order_in_file))

def organize_chunks_by_file_name(chunks: List[ChunkNode]) -> Dict[str, List[ChunkNode]]: 
    """
    Organizes a list of chunks by their file names.

    Args:
        chunks (list): The list of chunks to organize.

    Returns:
        dict: A dictionary mapping file names to lists of chunks.
    """
    # Convert dicts to ChunkNode if needed
    chunks = [dict_to_chunknode(c) if isinstance(c, dict) else c for c in chunks]
    organized_chunks = {}
    for chunk in chunks:
        file_name = extract_filename_from_chunk(chunk)
        if file_name not in organized_chunks:
            organized_chunks[file_name] = []
        organized_chunks[file_name].append(chunk)
    for file_name in organized_chunks:
        organized_chunks[file_name] = order_chunks_by_order_in_file(organized_chunks[file_name])
    return organized_chunks

def join_organized_chunks(organized_chunks: Dict[str, List[ChunkNode]]) -> str:
    """
    Joins organized chunks into a single string.

    Args:
        organized_chunks (dict): The dictionary of organized chunks.

    Returns:
        str: The joined string of organized chunks.
    """
    joined_chunks_list = []
    separator = "=" * 48
    for filename in organized_chunks:
        joined_chunks_list.append(separator)
        joined_chunks_list.append(f"File: {filename}")
        joined_chunks_list.append(separator)
        # Convert dicts to ChunkNode if needed
        chunks = [dict_to_chunknode(c) if isinstance(c, dict) else c for c in organized_chunks[filename]]
        if len(chunks) == 0:
            continue
        if int(chunks[0].order_in_file) > 0:
            joined_chunks_list.append("\n[...]")
        for i, chunk in enumerate(chunks):
            joined_chunks_list.append(chunk.content)
            if i < len(chunks) - 1:
                if int(chunks[i+1].order_in_file) - int(chunk.order_in_file) > 1:
                    joined_chunks_list.append("\n[...]")
    return "\n".join(joined_chunks_list)
