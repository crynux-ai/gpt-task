## Stable Diffusion Task

A general framework to define and execute the llm text generation task.


### Features

* Unified task definition for various different large language model
* Apply model specific chat templates to input prompts automatically
* Model quantizing (INT4 or INT8)
* Fine grained control text generation arguments
* ChatGPT style response


### Example

Here is an example of the gpt2 text generation:

```python
import logging
from gpt_task.inference import run_task


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


res = run_task(
    model="gpt2",
    messages=messages,
    seed=42,
)
print(res)
```


### Get started

Create and activate the virtual environment:
```shell
$ python -m venv ./venv
$ source ./venv/bin/activate
```

Install the dependencies and the library:
```shell
(venv) $ pip install -r requirments.txt && pip install -e .
```

Check and run the examples:
```shell
(venv) $ python ./examples/gpt2_example.py
```

More explanations can be found in the doc:

[https://docs.crynux.ai/application-development/gpt-task](https://docs.crynux.ai/application-development/gpt-task)

### Task Definition

The complete task definition is `GPTTaskArgs` in the file [```./src/gpt_task/models/args.py```](src/gpt_task/models/args.py)

### Task Response

The task response definition is `GPTTaskResponse` in the file [```./src/gpt_task/models/args.py```](src/gpt_task/models/args.py)

### JSON Schema

The JSON schemas for the tasks could be used to validate the task arguments by other projects.
The schemas are given under [```./schema```](./schema). Projects could use the URL to load the JSON schema files directly.
