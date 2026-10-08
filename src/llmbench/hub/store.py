"""Model source resolution: `models/` favorites -> local path. (HF cache arrives in Phase 2.)"""

from pathlib import Path

from llmbench.config.experiment import ModelSpec
from llmbench.paths import models_dir


class ModelNotFoundError(Exception):
    pass


def resolve_model(spec: ModelSpec) -> Path:
    if spec.scheme == "hf":
        raise ModelNotFoundError(
            f"model {spec.id!r}: hf:// sources are not supported yet (planned for Phase 2); "
            "download the GGUF into models/ and use a file path meanwhile"
        )
    if spec.scheme != "file":
        raise ModelNotFoundError(f"model {spec.id!r}: unsupported source scheme {spec.scheme!r}")

    raw = spec.source.removeprefix("file://")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, models_dir() / path]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise ModelNotFoundError(f"model {spec.id!r}: file not found (tried {tried})")
