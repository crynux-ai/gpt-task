from gpt_task.inference import run_task

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


res = run_task(
    model="gpt2",
    messages=messages,
    seed=42,
)
print(res)
