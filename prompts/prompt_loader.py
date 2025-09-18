import json
import os
from functools import lru_cache
from typing import Any, Dict

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

class PromptNotFoundError(FileNotFoundError):
    pass

@lru_cache(maxsize=64)
def load_prompt(name: str) -> Dict[str, Any]:
    """Load a prompt JSON by stem name (without .json) from the prompts directory.

    Args:
        name: filename stem e.g. "rephrase_v1".
    Returns:
        Dict with keys from JSON (id, system, user_template, etc.)
    Raises:
        PromptNotFoundError: if the JSON file doesn't exist.
        json.JSONDecodeError: if the JSON is invalid.
    """
    filename = f"{name}.json" if not name.endswith('.json') else name
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        raise PromptNotFoundError(f"Prompt file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt(name: str, **kwargs: Any) -> Dict[str, str]:
    """Return formatted prompt parts for a given template.

    Performs simple `{placeholder}` substitution on the user_template.

    Args:
        name: prompt stem (e.g. 'rephrase_v1')
        **kwargs: placeholders to substitute.

    Returns:
        Dict with keys: id, system, user (formatted), meta (remaining metadata sans templates)
    """
    data = load_prompt(name)
    system = data.get('system', '')
    raw_user = data.get('user_template', '')
    try:
        user = raw_user.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing placeholder '{missing}' for prompt '{name}'") from e

    meta = {k: v for k, v in data.items() if k not in {'system', 'user_template'}}
    return {
        'id': data.get('id', name),
        'system': system,
        'user': user,
        'meta': meta,
    }

__all__ = [
    'load_prompt',
    'format_prompt',
    'PromptNotFoundError'
]
