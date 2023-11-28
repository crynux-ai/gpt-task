from typing import List, Literal, Optional

from pydantic import BaseModel

from .utils import NonEmptyString


class GPTGenerationConfig(BaseModel):
    max_new_tokens: Optional[int] = None

    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None

    temperature: Optional[float] = None
    typical_p: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    num_return_sequences: Optional[int] = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GPTTaskArgs(BaseModel):
    model: NonEmptyString
    messages: List[Message]
    generation_config: Optional[GPTGenerationConfig] = None

    seed: int = 0
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    quantize_bits: Optional[Literal[4, 8]] = None
