from gpt_task.inference import run_task

messages = [
    {"role": "user", "content": "I want to create a chat bot. Any suggestions?"}
]


res = run_task(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=messages,
    seed=42,
    quantize_bits=4,
)
print(res)
