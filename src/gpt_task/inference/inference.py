from __future__ import annotations

from typing import Any, List, Literal, Mapping, Sequence

import torch
from pydantic import TypeAdapter
from transformers import AutoTokenizer, pipeline, set_seed

from gpt_task import models
from gpt_task.config import Config

from .utils import load_model_kwargs, use_deterministic_mode
from .errors import wrap_error

use_deterministic_mode()


def _find_prompt_tokens(input_tokens: List[int], output_tokens: List[int]) -> int:
    start = output_tokens.index(input_tokens[0])
    if start == -1:
        return 0
    end = output_tokens.index(input_tokens[-1], start + len(input_tokens) - 1)
    if end == -1:
        return 0
    return end + 1


@wrap_error
def run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
) -> models.GPTTaskResponse:
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

    model_kwargs = load_model_kwargs(config=config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True, **model_kwargs
    )

    if args.quantize_bits == 4:
        model_kwargs["load_in_4bit"] = True
    elif args.quantize_bits == 8:
        model_kwargs["load_in_8bit"] = True

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
        return_tensors=True,
        **generation_config,
    )
    assert output is not None
    assert isinstance(output, list)

    res_token_ids = []
    for single in output:
        assert isinstance(single, dict)
        res_token_ids.append(single["generated_token_ids"])

    assert len(res_token_ids) > 0

    input_tokens = tokenizer.encode(inputs, add_special_tokens=False)
    prompt_tokens = _find_prompt_tokens(input_tokens, res_token_ids[0])

    completion_tokens = 0
    output_texts = []
    finish_reasons = []
    for token_ids in res_token_ids:
        # when the last token is eos token, finish reason is stop, otherwise is length
        if token_ids[-1] == tokenizer.eos_token_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        finish_reasons.append(finish_reason)

        completion_tokens += len(token_ids) - prompt_tokens

        text = tokenizer.decode(
            token_ids[prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        output_texts.append(text)

    usage: models.Usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    choices: List[models.ResponseChoice] = [
        {
            "finish_reason": reason,
            "message": {"role": "assistant", "content": text},
            "index": i,
        }
        for i, (reason, text) in enumerate(zip(finish_reasons, output_texts))
    ]

    resp: models.GPTTaskResponse = {
        "model": args.model,
        "choices": choices,
        "usage": usage,
    }
    return resp
