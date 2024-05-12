"""
Module for creating singleton instances of classes.
"""
from typing import Any, Dict


class SingletonMeta(type):
    """Metaclass for creating singleton instances of classes.

    This metaclass ensures that only one instance of each class using
    it is created. If an instance of the class already exists, it
    returns the existing instance.

    Args:
        cls: The class to create a singleton instance of.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The singleton instance of the class.
    """

    _instances: dict = {}

    def __call__(cls, *args, **kwargs) -> Dict:
        if cls not in cls._instances:
            instance: Any = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
