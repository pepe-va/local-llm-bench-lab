from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    """Immutable config node that rejects unknown keys (typos fail loudly)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
