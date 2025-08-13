import importlib
import pytest

MODEL_MODULES = [
    # 'mt.benchmark.models.anystar',
    'mt.benchmark.models.base',
    'mt.benchmark.models.cellpose_sam',
    'mt.benchmark.models.cellsam',
    'mt.benchmark.models.drift',
    'mt.benchmark.models.factory',
    'mt.benchmark.models.fiesta',
    'mt.benchmark.models.musam',
    'mt.benchmark.models.sifine',
    'mt.benchmark.models.soax',
    # 'mt.benchmark.models.stardist',
]

def test_import_model_modules():
    for module in MODEL_MODULES:
        print(f"Testing import of module: {module}")
        importlib.import_module(module)

