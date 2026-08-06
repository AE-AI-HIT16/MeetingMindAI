"""Central registry for all MeetASR components."""

import logging
import inspect
from dataclasses import dataclass, field


@dataclass
class RegisterTables:
    """Registry tables for all swappable MeetASR components."""

    model_classes: dict = field(default_factory=dict)
    frontend_classes: dict = field(default_factory=dict)
    tokenizer_classes: dict = field(default_factory=dict)
    llm_classes: dict = field(default_factory=dict)

    def register(self, registry_name: str, key: str | None = None):
        """Decorator to register a class into a named registry table.

        Args:
            registry_name: Name of the registry (e.g. "model_classes").
            key: Registry key. Defaults to class name.

        Returns:
            Decorator function.

        Example:
            @tables.register("model_classes", key="fsmn-vad")
            class FsmnVAD(AbsVAD): ...
        """
        def decorator(cls):
            if not hasattr(self, registry_name):
                raise AttributeError(
                    f"Registry '{registry_name}' does not exist. "
                    f"Valid registries: {list(vars(self).keys())}"
                )
            registry = getattr(self, registry_name)
            reg_key = key if key is not None else cls.__name__
            if reg_key in registry:
                logging.debug(
                    f"Re-registering '{reg_key}' in {registry_name} "
                    f"(was {registry[reg_key].__name__}, now {cls.__name__})"
                )
            registry[reg_key] = cls
            logging.debug(f"Registered {cls.__name__} as '{reg_key}' in {registry_name}")
            return cls

        return decorator

    def list_registered(self, registry_name: str) -> list[str]:
        """Return sorted list of registered keys for a registry.

        Args:
            registry_name: Name of the registry table.

        Returns:
            Sorted list of registered key strings.
        """
        registry = getattr(self, registry_name, {})
        return sorted(registry.keys())


tables = RegisterTables()
