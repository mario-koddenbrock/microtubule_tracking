from typing import Type, Dict, List

from mt.benchmark.models.base import BaseModel

from .sam import SAM

# from .anystar import AnyStar
# from .cellsam import CellSAM
from .cellpose_sam import CellposeSAM
from .drift import DRIFT
from .fiesta import FIESTA

# from .micro_sam import MicroSAM
from .sifine import SIFINE
from .soax import SOAX

# from .stardist import StarDist


class ModelFactory:
    """A factory for creating model instances."""

    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}

    def register_model(self, model_class: Type[BaseModel]):
        """
        Registers a model class with the factory.
        The model name is retrieved from the class's `model_name` property.
        """
        model_instance = model_class()
        name = model_instance.model_name
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")
        self._models[name] = model_class

    def create_model(self, name: str) -> BaseModel:
        """
        Creates an instance of a registered model by its name.

        Args:
            name: The name of the model to create.

        Returns:
            An instance of the specified model.
        """
        model_class = self._models.get(name)
        if not model_class:
            raise ValueError(
                f"Model '{name}' not registered. Available models: {self.get_available_models()}"
            )
        return model_class()

    def get_available_models(self) -> List[str]:
        """Returns a list of all registered model names."""
        return sorted(list(self._models.keys()))


def setup_model_factory() -> ModelFactory:
    """Initializes and registers all models with the factory."""
    factory = ModelFactory()
    model_classes = [
        FIESTA,
        SOAX,
        SIFINE,
        DRIFT,
        SAM,
        # CellSAM,
        # AnyStar,
        # MicroSAM,
        CellposeSAM,
        # StarDist,
    ]
    for model_class in model_classes:
        factory.register_model(model_class)
    return factory
