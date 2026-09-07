"""Shared base model for everything the API returns.

Pydantic marks a field with a default as *optional* in the generated JSON
schema, on the reasonable grounds that a client need not send it. For a
**response** model that is simply inaccurate: a `Paper` always carries an
`entities` list and an `also_found_in` list, because the server serializes
them whether they are empty or not.

The consequence is not cosmetic. The frontend generates its types from
this schema, so every defaulted field arrived as `T[] | undefined` and
strict TypeScript demanded a null check for a value that can never be
absent — pushing thirty pointless guards into the UI to describe a
situation that does not exist.

``json_schema_serialization_defaults_required`` makes the serialization
schema say what the server actually sends. Fields that are genuinely
nullable are still typed `T | null`, because those really can be absent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base for models that appear in an API response."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)
