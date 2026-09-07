"""Write the OpenAPI schema to a file.

Run without starting a server, so the frontend's types can be regenerated
in CI. The alternative - hand-written interfaces mirroring the Pydantic
models - drifts the moment a field is added, and drifts silently.

Usage::

    python -m scripts.dump_openapi ../frontend/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import Settings
from app.main import create_app

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "frontend" / "openapi.json"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    output = Path(args[0]) if args else DEFAULT_OUTPUT
    # Settings only, no lifespan: nothing here needs a model loaded or a
    # database opened just to describe the API surface.
    schema = create_app(Settings(environment="schema")).openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(schema.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
