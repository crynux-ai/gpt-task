from typing import List, Optional, Tuple, Union, Literal

from pydantic import BaseModel

from .utils import NonEmptyString


class GPTGenerationConfig(BaseModel):
    max_length: int = 50
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: Union[str, bool] = False
    max_time: Optional[float] = None

    # Parameters that control the generation strategy used
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    penalty_alpha: Optional[float] = None
    use_cache: bool = True

    # Parameters for manipulation of the model output logits
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    encoder_repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Union[None, List[List[int]], List[List[List[int]]]] = None
    renormalize_logits: bool = False
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Union[None, int, List[int]] = None
    remove_invalid_values: bool = False
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    forced_decoder_ids: Optional[List[List[int]]] = None

    # Parameters that define the output variables of `generate`
    num_return_sequences: int = 1

    # Special tokens that can be used at generation time
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Generation parameters exclusive to encoder-decoder models
    encoder_no_repeat_ngram_size: int = 0
    decoder_start_token_id: Optional[int] = None


class GPTTaskArgs(BaseModel):
    model: NonEmptyString
    prompts: Union[NonEmptyString, List[NonEmptyString]]
    generation_config: Optional[GPTGenerationConfig] = None
    seed: int = 42
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
