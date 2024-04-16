from typing import Protocol, Dict, Any, TypeVar

T = TypeVar("T")

class ModelCache(Protocol[T]):
    def set(self, model_args: Dict[str, Any], model: T):
        ...

    def get(self, model_args: Dict[str, Any]) -> T:
        ...

    def has(self, model_args: Dict[str, Any]) -> bool:
        ...
