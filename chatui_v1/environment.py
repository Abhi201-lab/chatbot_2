import os
from dotenv import load_dotenv

def load_env():
    # Load from local .env if present
    load_dotenv()

def env(key: str, default: str | None = None, required: bool = True) -> str:
    val = os.getenv(key, default)
    if required and val is None:
        raise EnvironmentError(f"Missing env var: {key}")
    return val
