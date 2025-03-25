from gpt_task.models import GPTTaskArgs
import json

if __name__ == '__main__':
    dest_file = "./schema/gpt-inference-task.json"

    schema = GPTTaskArgs.model_json_schema()

    with open(dest_file, "w") as f:
        json.dump(schema, f)

    print("json schema output successfully")
