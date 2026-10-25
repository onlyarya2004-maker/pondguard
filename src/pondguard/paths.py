"""Project paths and local-only cache setup.

PondGuard keeps everything inside the "AI team project" folder:

    AI team project/
        pondguard/          <- this git repo (code, config, data, results)
        models/             <- LM Studio GGUF files (outside the repo)
        cache/huggingface/  <- bge-small embedding model (HF_HOME)
        cache/phoenix/      <- local Phoenix trace store (PHOENIX_WORKING_DIR)

`configure_local_env()` runs when the `pondguard` package is imported, before any
library that reads these variables (sentence-transformers, Phoenix, ChromaDB) is
imported. It only fills in variables that are not already set, so a value in the
shell or in `.env` always wins. Relative paths in `.env` are resolved against the
repo root, not the current directory.

Assumes an editable install (`pip install -e .`), so this file lives in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "config"

_DEFAULT_DIRS = {
    "HF_HOME": PROJECT_ROOT / "cache" / "huggingface",
    "PHOENIX_WORKING_DIR": PROJECT_ROOT / "cache" / "phoenix",
}

# Local-only rule: turn off library usage telemetry that would phone home.
_OFFLINE_FLAGS = {
    "ANONYMIZED_TELEMETRY": "False",  # ChromaDB
    "HF_HUB_DISABLE_TELEMETRY": "1",  # Hugging Face Hub
    "PHOENIX_TELEMETRY_ENABLED": "false",  # Arize Phoenix
    "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",  # Phoenix: no external downloads/CDNs
    "LANGSMITH_TRACING": "false",  # LangSmith is an online service; never send traces
    "LANGCHAIN_TRACING_V2": "false",  # older name for the same switch
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def configure_local_env() -> None:
    """Load `.env` and point every cache at a folder inside the project."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    for name, default in _DEFAULT_DIRS.items():
        path = _resolve(os.environ[name]) if os.environ.get(name) else default
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name, value in _OFFLINE_FLAGS.items():
        os.environ.setdefault(name, value)
