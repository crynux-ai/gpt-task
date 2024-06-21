import hashlib
import json
from typing import Any, Dict

from gpt_task import models


def generate_model_key(args: models.GPTTaskArgs) -> str:
    model_args: Dict[str, Any] = {"model": args.model, "dtype": args.dtype}
    if args.quantize_bits is not None:
        model_args["quantize_bits"] = args.quantize_bits

    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    key = hashlib.md5(model_args_str.encode("utf-8")).hexdigest()
    return key
