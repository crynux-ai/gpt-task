from __future__ import annotations

from typing import Any, List, Literal, Mapping, Sequence, Union

import torch
from pydantic import TypeAdapter
from transformers import AutoTokenizer, pipeline, set_seed

from gpt_task import models


def run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
) -> Union[str, List[str]]:
    if args is None:
        args = models.GPTTaskArgs.model_validate(
            {
                "model": model,
                "messages": messages,
                "generation_config": generation_config,
                "seed": seed,
                "dtype": dtype,
                "quantize_bits": quantize_bits,
            }
        )

    set_seed(args.seed)

    torch_dtype = None
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model_kwargs = {}
    if args.quantize_bits == 4:
        model_kwargs["load_in_4bit"] = True
    elif args.quantize_bits == 8:
        model_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=args.model,
        config=args.model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        use_fast=False,
        device_map="auto",
        torch_dtype=torch_dtype,
        model_kwargs=dict(
            offload_folder="offload",
            offload_state_dict=True,
            **model_kwargs,
        ),
    )

    generation_config = {"num_return_sequences": 1, "max_new_tokens": 256}
    if args.generation_config is not None:
        customer_config = TypeAdapter(models.GPTGenerationConfig).dump_python(
            args.generation_config,
            exclude_none=True,
            exclude_unset=True,
        )
        for k, v in customer_config.items():
            if v is not None:
                generation_config[k] = v

    chats = [dict(**m) for m in args.messages]
    if tokenizer.chat_template is not None:
        inputs = tokenizer.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=True
        )
    else:
        inputs = "\n".join(c["content"] for c in chats)

    output = pipe(
        inputs,
        return_full_text=False,
        clean_up_tokenization_spaces=True,
        **generation_config,
    )
    assert output is not None
    assert isinstance(output, list)

    res = []
    for single in output:
        assert isinstance(single, dict)
        res.append(single["generated_text"])

    if len(res) == 1:
        return res[0]
    return res
