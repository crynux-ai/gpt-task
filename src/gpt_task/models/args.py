from typing import List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict

from .utils import NonEmptyString


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class GPTGenerationConfig(TypedDict, total=False):
    max_new_tokens: int

    do_sample: bool
    num_beams: int

    temperature: float
    typical_p: float
    top_k: int
    top_p: float
    repetition_penalty: float

    num_return_sequences: int


class GPTTaskArgs(BaseModel):
    model: NonEmptyString
    messages: List[Message]
    generation_config: Optional[GPTGenerationConfig] = None

    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    quantize_bits: Optional[Literal[4, 8]] = None


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseChoice(TypedDict):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class GPTTaskResponse(TypedDict):
    model: NonEmptyString
    choices: List[ResponseChoice]
    usage: Usage
