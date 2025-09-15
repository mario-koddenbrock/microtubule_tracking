from typing import Type, Dict, List

from mt.benchmark.models.base import BaseModel

from .sam import SAM
from .sam2 import SAM2

from .anystar import AnyStar
from .cellsam import CellSAM
from .cellpose_sam import CellposeSAM
from .drift import DRIFT
from .fiesta import FIESTA

from .micro_sam import MicroSAM
from .sifne import SIFNE
from .soax import SOAX

from .stardist import StarDist


class ModelFactory:
    """A factory for creating model instances."""

    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}

    def register_model(self, model_class: Type[BaseModel]):
        """
        Registers a model class with the factory.
        The model name is retrieved from the class's `get_model_name` method.
        """
        name = model_class.__name__
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")
        self._models[name] = model_class

    def create_model(self, name: str, **kwargs) -> BaseModel:
        """
        Creates an instance of a registered model by its name.

        Args:
            name: The name of the model to create.
            **kwargs: Additional keyword arguments to pass to the model's constructor.

        Returns:
            An instance of the specified model.
        """
        model_class = self._models.get(name)
        if not model_class:
            raise ValueError(
                f"Model '{name}' not registered. Available models: {self.get_available_models()}"
            )
        return model_class(**kwargs)

    def get_available_models(self) -> List[str]:
        """Returns a list of all registered model names."""
        return sorted(list(self._models.keys()))


def setup_model_factory() -> ModelFactory:
    """Initializes and registers all models with the factory."""
    factory = ModelFactory()
    model_classes = [
        FIESTA,
        SOAX,
        SIFNE,
        DRIFT,
        SAM,
        SAM2,
        CellSAM,
        AnyStar,
        MicroSAM,
        CellposeSAM,
        StarDist,
    ]
    for model_class in model_classes:
        factory.register_model(model_class)
    return factory
