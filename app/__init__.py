# app/__init__.py
import sys

from app.config import Config
from app.types import SingletonMeta

# Register globals
globals()["SingletonMeta"] = SingletonMeta
globals()["Config"] = Config

package_name: str = __name__.split(sep=".", maxsplit=1)[0]
sys.modules[package_name].__dict__["SingletonMeta"] = SingletonMeta
sys.modules[package_name].__dict__["Config"] = Config
