import os

def load_prompt(name: str) -> str:
    base_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(base_dir, "..", "prompts", name)
    with open(prompt_path, "r") as file:
        return file.read()
