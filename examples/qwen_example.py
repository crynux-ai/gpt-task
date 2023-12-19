from gpt_task.inference import run_task

messages = [
    {"role": "user", "content": "I want to create a chat bot. Any suggestions?"}
]


res = run_task(
    model="Qwen/Qwen-1_8B",
    messages=messages,
    generation_config={
        "repetition_penalty": 1.1,
        "do_sample": True,
        "temperature": 0.3,
    },
    seed=42,
    quantize_bits=4,
)
print(res)
