from importlib.metadata import entry_points
from typing import Any, Dict

class RegistryError(RuntimeError):
    pass

def _item_name(item: Any) -> str:
    module = getattr(item, "__module__", None)
    name = getattr(item, "__name__", repr(item))
    if module is None:
        return name

    return f"{module}.{name}"

def load_entry_points(group: str, registry: Dict[str, Any]):
    for entry_point in entry_points(group=group):
        if entry_point.name in registry:
            registered = registry[entry_point.name]
            raise RegistryError(f"Plugin entry point '{entry_point.name}' in group "
                                f"'{group}' conflicts with registered item "
                                f"'{_item_name(registered)}'.")
        try:
            registry[entry_point.name] = entry_point.load()
        except Exception as exc:
            raise RegistryError(f"Failed to load plugin entry point "
                                f"'{entry_point.name}' in group '{group}'.") from exc

def registered_names(registry: Dict[str, Any]) -> str:
    return ", ".join(sorted(registry.keys()))
