import importlib
import pytest

MODULES = [
    'mt.analysis',
    'mt.benchmark',
    'mt.config',
    'mt.data_generation',
    'mt.file_io',
    'mt.plotting',
    'mt.tracking',
    'mt.utils',
]

def test_import_submodules():
    for module in MODULES:
        importlib.import_module(module)

def test_import_mt():
    import mt

