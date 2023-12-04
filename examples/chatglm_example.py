from gpt_task.inference import run_task
from gpt_task.config import Config, ProxyConfig

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


res = run_task(
    model="THUDM/chatglm3-6b",
    messages=messages,
    generation_config={
        "repetition_penalty": 1.1,
        "do_sample": True,
        "temperature": 0.3
    },
    seed=42,
    dtype="float16",
    quantize_bits=4,
)
print(res)
