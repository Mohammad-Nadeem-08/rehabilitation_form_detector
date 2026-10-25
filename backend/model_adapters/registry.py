"""
Central Model Registry for RehabVision AI.

Provides registration, discovery, compatibility lookup, and health inspection
for all machine learning model adapters.
"""

from typing import Dict, List, Optional, Any
from model_adapters.base import BaseModelAdapter, ModelMetadata
from model_adapters.existing_mocap_lstm import ExistingMocapLSTMAdapter
from model_adapters.example_teammate_adapter import ExampleTeammateModelAdapter


class ModelRegistry:
    """
    Registry for model adapters.
    Maintains clean separation between exercise movement strategies and machine learning models.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._defaults_by_exercise: Dict[str, str] = {}
        self._register_default_adapters()

    def _register_default_adapters(self):
        """Pre-registers the built-in system adapters."""
        # 1. Existing Mocap LSTM for standing right-arm shoulder abduction
        mocap_adapter = ExistingMocapLSTMAdapter()
        self.register(mocap_adapter, is_default=True)

        # 2. Example template adapter for seated right bicep curl
        bicep_adapter = ExampleTeammateModelAdapter()
        self.register(bicep_adapter, is_default=True)

    def register(self, adapter: BaseModelAdapter, is_default: bool = False):
        """
        Registers a model adapter.
        Optionally sets it as the default adapter for its supported exercises.
        """
        meta = adapter.get_metadata()
        self._adapters[meta.model_id] = adapter

        for ex_id in meta.supported_exercise_ids:
            if is_default or ex_id not in self._defaults_by_exercise:
                self._defaults_by_exercise[ex_id] = meta.model_id

    def get(self, model_id: str) -> Optional[BaseModelAdapter]:
        """Retrieves an adapter by its unique model_id."""
        return self._adapters.get(model_id)

    def get_for_exercise(self, exercise_id: str) -> List[BaseModelAdapter]:
        """Returns all adapters that declare support for the given exercise_id."""
        matches = []
        for adapter in self._adapters.values():
            if exercise_id in adapter.get_metadata().supported_exercise_ids:
                matches.append(adapter)
        return matches

    def get_default_for_exercise(self, exercise_id: str) -> Optional[BaseModelAdapter]:
        """
        Returns the primary default model adapter for an exercise, if registered.
        Returns None if the exercise has no registered model (e.g., statistical-only).
        """
        model_id = self._defaults_by_exercise.get(exercise_id)
        if model_id:
            return self.get(model_id)

        candidates = self.get_for_exercise(exercise_id)
        return candidates[0] if candidates else None

    def list_all(self) -> List[Dict[str, Any]]:
        """Returns structured metadata summaries for all registered adapters."""
        summaries = []
        for model_id, adapter in self._adapters.items():
            meta = adapter.get_metadata().to_dict()
            meta["is_default_for"] = [
                ex for ex, d_id in self._defaults_by_exercise.items() if d_id == model_id
            ]
            meta["health"] = adapter.health_check()
            summaries.append(meta)
        return summaries

    def reset_defaults(self):
        """Resets the registry to initial default state."""
        self._adapters.clear()
        self._defaults_by_exercise.clear()
        self._register_default_adapters()


# Global Singleton Registry Instance
model_registry = ModelRegistry()
