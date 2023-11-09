"""
The core is a 2D dense matrix.
"""

from ._core import *
import importlib as _importlib

submodules = ["sparse" , "linalg"]

def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'PMatrix.{name}')
    else:
        raise AttributeError(
            f"Module 'PMatrix' has no attribute '{name}'"
            )