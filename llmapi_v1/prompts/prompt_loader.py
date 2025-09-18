import json, os
from functools import lru_cache
from typing import Any, Dict

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

class PromptNotFoundError(FileNotFoundError):
    pass

@lru_cache(maxsize=64)
def load_prompt(name: str) -> Dict[str, Any]:
    filename = f"{name}.json" if not name.endswith('.json') else name
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        raise PromptNotFoundError(f"Prompt file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt(name: str, **kwargs: Any) -> Dict[str, str]:
    data = load_prompt(name)
    system = data.get('system', '')
    raw_user = data.get('user_template', '')
    try:
        user = raw_user.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing placeholder {e.args[0]} for prompt '{name}'") from e
    meta = {k: v for k, v in data.items() if k not in {'system', 'user_template'}}
    return {'id': data.get('id', name), 'system': system, 'user': user, 'meta': meta}
