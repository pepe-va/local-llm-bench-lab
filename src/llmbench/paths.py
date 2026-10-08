"""Locations outside the repository. Everything heavy lives under `$LLB_HOME`."""

import os
from pathlib import Path


def llb_home() -> Path:
    if env := os.environ.get("LLB_HOME"):
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "~/.local/share")
    return Path(xdg).expanduser() / "llmbench"


def builds_dir() -> Path:
    return llb_home() / "builds"


def llama_src_dir() -> Path:
    return llb_home() / "src" / "llama.cpp"


def runs_dir() -> Path:
    return llb_home() / "runs"


def models_dir() -> Path:
    """Favorites folder. Defaults to `./models` (inside the project, gitignored)."""
    return Path(os.environ.get("LLB_MODELS_DIR", "models")).expanduser()
