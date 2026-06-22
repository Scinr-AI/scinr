from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtractionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )
