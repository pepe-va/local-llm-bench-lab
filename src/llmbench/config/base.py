from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    """Immutable config node that rejects unknown keys (typos fail loudly).

    Attribute docstrings become field descriptions in the JSON schema (the lab's editor
    shows them as help)."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)
