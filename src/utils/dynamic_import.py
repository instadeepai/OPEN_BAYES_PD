import importlib


def import_class_from_string(class_path: str):
    """
    Import a class from a string.

    Parameters:
    -----------
    class_path: str
        The path to the class to import.

    Returns:
    --------
    class: The imported class.

    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
