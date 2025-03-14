from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union, Generator

import torch
from pydantic import TypeAdapter
from transformers import AutoConfig, AutoTokenizer, pipeline, set_seed
from transformers.generation.streamers import BaseStreamer

from gpt_task import models
from gpt_task.config import Config
from gpt_task.cache import ModelCache

from .errors import wrap_error
from .utils import load_model_kwargs, use_deterministic_mode
from .key import generate_model_key

_logger = logging.getLogger(__name__)


class TokenStreamer(BaseStreamer):
    """Streamer that yields tokens as they are generated."""

    def __init__(self, tokenizer, input_tokens: List[int]):
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens  # Store the actual input tokens
        self.tokens = []
        self.is_eos = False
        self.completion_tokens = 0
        self.is_done = False
        self.text_queue = []
        self.found_prompt_end = False  # Flag to track if we've found the end of the prompt
        self.first_token = True  # Flag to track if this is the first token being returned
        self.prompt_tokens = len(input_tokens)  # Initialize with input length, will be updated when prompt end is found

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        token_list = value.tolist()

        for token in token_list:
            self.tokens.append(token)

            # Always check for prompt end first
            if not self.found_prompt_end:
                if len(self.tokens) >= len(self.input_tokens):
                    # Try to find the end of the input sequence
                    prompt_end = _find_prompt_tokens(self.input_tokens, self.tokens)
                    if prompt_end > 0:
                        self.found_prompt_end = True
                        self.prompt_tokens = prompt_end  # Update prompt tokens count to match non-streaming mode
                        self.tokens = self.tokens[prompt_end:]  # Keep only the new tokens
                        _logger.debug(f"Found prompt end at position {prompt_end}")

                        # Check if we've already collected any completion tokens
                        for completion_token in self.tokens:
                            self.completion_tokens += 1
                            # Process each completion token (decode, etc.)
                            new_text = self.tokenizer.decode(
                                [completion_token],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True
                            )
                            if new_text:
                                self.text_queue.append(new_text)

                            # Check for EOS in completion tokens
                            if completion_token == self.tokenizer.eos_token_id:
                                self.is_eos = True
                                break
                        continue

            # For tokens after prompt identification
            if self.found_prompt_end:
                # Check for EOS after we've found the prompt
                if token == self.tokenizer.eos_token_id:
                    self.is_eos = True
                    break

                self.completion_tokens += 1

                # Decode the new token
                new_text = self.tokenizer.decode(
                    [token],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                if new_text:
                    self.text_queue.append(new_text)

    def end(self):
        self.is_done = True

    def get_text(self) -> str:
        if not self.text_queue:
            return ""
        # Return text if we've found prompt end OR if generation is complete
        if self.found_prompt_end or self.is_done:
            text = self.text_queue.pop(0)
            if self.first_token:
                text = text.lstrip()
                self.first_token = False
            return text
        return ""

    def get_finish_reason(self) -> Literal["stop", "length"]:
        return "stop" if self.is_eos else "length"

    def get_usage(self) -> models.Usage:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }


def _find_prompt_tokens(input_tokens: List[int], output_tokens: List[int]) -> int:

    _logger.debug(f"Finding prompt tokens: input_tokens={input_tokens}")
    _logger.debug(f"Finding prompt tokens: output_tokens={output_tokens}")

    try:
        start = output_tokens.index(input_tokens[0])
        _logger.debug(f"Finding prompt tokens: start={start}")
        end = output_tokens.index(input_tokens[-1], start + len(input_tokens) - 1)
        _logger.debug(f"Finding prompt tokens: end={end}")
        return end + 1
    except ValueError:
        _logger.debug(f"Finding prompt tokens: ValueError")
        return 0


@wrap_error
def run_task(
    args: models.GPTTaskArgs | None = None,
    *,
    model: str | None = None,
    messages: Sequence[models.Message | Mapping[str, Any]] | None = None,
    tools: Sequence[Dict[str, Any]] | None = None,
    generation_config: models.GPTGenerationConfig | Mapping[str, Any] | None = None,
    stream: bool = False,
    seed: int = 0,
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
    quantize_bits: Literal[4, 8] | None = None,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
) -> Union[models.GPTTaskResponse, models.GPTTaskStreamResponse]:
    if args is None:
        args = models.GPTTaskArgs.model_validate(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "generation_config": generation_config,
                "stream": stream,
                "seed": seed,
                "dtype": dtype,
                "quantize_bits": quantize_bits,
            }
        )

    _logger.info("Task starts")
    _logger.debug(f"task args: {args}")

    use_deterministic_mode()

    set_seed(args.seed)

    model_key = generate_model_key(args)

    def model_loader():
        _logger.info("Start loading pipeline")

        torch_dtype = None
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "float32":
            torch_dtype = torch.float32
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model_kwargs = load_model_kwargs(config=config)
        _logger.debug(f"model kwargs: {model_kwargs}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=False, trust_remote_code=True, **model_kwargs
        )
        model_config = AutoConfig.from_pretrained(
            args.model,
            _from_pipeline="text-generation",
            trust_remote_code=True,
            **model_kwargs,
        )

        if args.quantize_bits == 4:
            model_kwargs["load_in_4bit"] = True
        elif args.quantize_bits == 8:
            model_kwargs["load_in_8bit"] = True

        pipe = pipeline(
            "text-generation",
            model=args.model,
            config=model_config,
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

        _logger.info("Loading pipeline completes")
        return pipe

    if model_cache is not None:
        pipe = model_cache.load(model_key, model_loader)
    else:
        pipe = model_loader()

    tokenizer = pipe.tokenizer

    _logger.info("Start text generation")

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

    # Check if model supports chat templates
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    _logger.debug(f"Model has chat template: {has_chat_template}")

    # Warn if tools are requested but model doesn't support them
    if args.tools and not has_chat_template:
        _logger.warning("Tools were provided but model does not support chat template. Tool calling will be disabled.")
        args.tools = None  # Disable tools since they won't work

    if has_chat_template:
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True
        }

        if args.tools is not None:
            template_args["tools"] = [dict(**t) for t in args.tools]
            _logger.debug(f"Adding tools to chat template: {template_args['tools']}")

        inputs = tokenizer.apply_chat_template(chats,**template_args)
        _logger.debug("Applied chat template for input formatting")
    else:
        _logger.debug("No chat template available, falling back to basic formatting")
        inputs = "\n".join(c["content"] for c in chats)

    _logger.debug(f"Generation config: {generation_config}")
    _logger.debug(f"Input text: {inputs}")

    input_tokens = tokenizer.encode(inputs, add_special_tokens=False)

    if args.stream:
        streamer = TokenStreamer(tokenizer, input_tokens)  # Pass both tokenizer and input tokens
        generation_config["streamer"] = streamer
        generation_config["pad_token_id"] = tokenizer.eos_token_id
        generation_config["use_cache"] = True

        def stream_generator() -> Generator[models.StreamResponse, None, None]:
            # Set up streaming generation
            _logger.debug("Starting streaming generation")
            pipe(
                inputs,
                **generation_config,
            )
            _logger.debug("Generation initiated, starting to yield chunks")

            _logger.debug(f"Streamer: is_done={streamer.is_done}, text_queue={streamer.text_queue}")

            # Keep getting text until we're done and no more text in queue
            while not streamer.is_done or streamer.text_queue:
                _logger.debug(f"Stream loop: is_done={streamer.is_done}, queue_size={len(streamer.text_queue)}")
                new_text = streamer.get_text()
                _logger.debug(f"Got new text: '{new_text}'")
                if new_text:
                    yield {
                        "model": args.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": new_text},
                            "finish_reason": None
                        }],
                        "usage": streamer.get_usage()
                    }

                    _logger.debug("Yielded chunk")
                    _logger.debug(f"Streamer: is_done={streamer.is_done}, text_queue={streamer.text_queue}")

            # Send final chunk with finish_reason
            finish_reason = streamer.get_finish_reason()
            _logger.debug(f"Sending final chunk with finish_reason={finish_reason}")
            yield {
                "model": args.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": finish_reason
                }],
                "usage": streamer.get_usage()
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _logger.info("Text generation completes")

        return stream_generator()

    output = pipe(
        inputs,
        return_tensors=True,
        **generation_config,
    )
    assert output is not None
    assert isinstance(output, list)

    _logger.debug(f"Raw output: {output}")

    res_token_ids = []
    for single in output:
        assert isinstance(single, dict)
        res_token_ids.append(single["generated_token_ids"])

    assert len(res_token_ids) > 0

    del output

    prompt_tokens = _find_prompt_tokens(input_tokens, res_token_ids[0])

    del input_tokens

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
        ).strip()

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

    del res_token_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _logger.info("Text generation completes")
    return resp
