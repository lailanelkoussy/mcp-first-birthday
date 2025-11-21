
def flatten_list(my_list: list) -> list:
    """
    Args:
        my_list: list composed of lists (of lists of lists...)

    Returns: flattened list

    """
    flattened_list = []
    for item in my_list:
        if isinstance(item, list) and len(item) > 0:
            print(item)
            flattened_list += flatten_list(item)
        elif not isinstance(item, list):
            flattened_list.append(item)

    return flattened_list