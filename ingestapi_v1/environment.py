import os
from dotenv import load_dotenv

# ensure .env is loaded when this module is imported
load_dotenv()

def load_env():
    load_dotenv()

def env(key: str, default: str | None = None, required: bool = True) -> str:
    v = os.getenv(key, default)
    if required and (v is None or v == ""):
        raise EnvironmentError(f"Missing env var: {key}")
    return v
