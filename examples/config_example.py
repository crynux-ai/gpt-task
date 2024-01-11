import logging

from gpt_task.config import Config, ProxyConfig
from gpt_task.inference import run_task

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [
    {"role": "user", "content": "I want to create a chat bot. Any suggestions?"}
]


res = run_task(
    model="gpt2",
    messages=messages,
    seed=42,
    config=Config(hf_cache_dir="./.cache", proxy=ProxyConfig(host="http://localhost")),
)
print(res)
