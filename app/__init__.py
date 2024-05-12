# app/__init__.py

import builtins
import sys

from .config import Config
from .types import SingletonMeta

# Register globals
globals()["SingletonMeta"] = SingletonMeta
globals()["Config"] = Config

package_name = __name__.split(".", maxsplit=1)[0]
sys.modules[package_name].__dict__["SingletonMeta"] = SingletonMeta
sys.modules[package_name].__dict__["Config"] = Config

# Register builtins
builtins.SingletonMeta = SingletonMeta
