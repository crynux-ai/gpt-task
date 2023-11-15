from typing import List, Union

import torch
from transformers import pipeline, set_seed

from gpt_task import models



def run_task(args: models.GPTTaskArgs) -> Union[List[str], List[List[str]]]:
    dtype = "auto"
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16

    pipe = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=args.model,
        torch_dtype=dtype,
        device=0,
        trust_remote_code=True,
    )

    set_seed(args.seed)

    if args.generation_config is not None:
        generation_config = args.generation_config.model_dump(exclude_defaults=True, exclude_none=True)
    else:
        generation_config = {}

    output = pipe(
        args.prompts,
        return_full_text=False,
        **generation_config,
    )
    assert output is not None

    res = []
    for single_output in output:
        if isinstance(single_output, dict):
            # output is a list of dict, res should be a list of str
            res.append(single_output["generated_text"])
        elif isinstance(single_output, list):
            # output is a list of list of dict, res should be a list of list of str
            res.append([d["generated_text"] for d in single_output])

    return res
