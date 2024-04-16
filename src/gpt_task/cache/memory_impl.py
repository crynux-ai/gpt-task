import hashlib
import json
from typing import Any, Dict


def generate_key(model_args: Dict[str, Any]) -> str:
    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    key = hashlib.md5(model_args_str.encode("utf-8")).hexdigest()
    return key


class MemoryModelCache(object):
    def __init__(self) -> None:
        self._cache = {}

    def set(self, model_args: Dict[str, Any], model: Any):
        key = generate_key(model_args)
        self._cache[key] = model

    def get(self, model_args: Dict[str, Any]):
        key = generate_key(model_args)
        return self._cache[key]

    def has(self, model_args: Dict[str, Any]):
        key = generate_key(model_args)
        return key in self._cache
