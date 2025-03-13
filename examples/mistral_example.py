import logging

from gpt_task.inference import run_task

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

_logger = logging.getLogger(__name__)

messages = [
    {"role": "user", "content": "Write a short story about a magical forest."}
]

print("Starting generation...", flush=True)

try:
    # Non-streaming generation
    response = run_task(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=messages,
        generation_config={
            "repetition_penalty": 1.1,
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 100
        },
        seed=42,
        dtype="float16"
    )

    _logger.debug(f"Received response: {response}")

    # Get the generated content
    content = response["choices"][0]["message"]["content"]
    finish_reason = response["choices"][0]["finish_reason"]

    # Print the generated content
    print(content)

    if finish_reason:
        _logger.debug(f"Generation finished with reason: {finish_reason}")

except Exception as e:
    _logger.exception("Error during generation")
    raise

print("\nGeneration complete!", flush=True)
