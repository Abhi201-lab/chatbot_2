import os
from dotenv import load_dotenv

# Ensure .env is loaded as soon as this module is imported
load_dotenv()

def load_env():
    # kept for explicit calls from other modules if desired
    load_dotenv()

def env(key: str, default: str | None = None, required: bool = True) -> str | None:
    """
    Return the environment variable value or default.
    If required is True and no value/default is provided, raise EnvironmentError.
    """
    v = os.getenv(key, None)
    if v is None or v == "":
        if default is not None:
            return default
        if required:
            raise EnvironmentError(f"Missing env var: {key}")
        return None
    return v
