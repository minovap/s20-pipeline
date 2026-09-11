"""File identity and atomic receipts. Captures are always read-only."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_destination(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        protected = source if source.is_dir() else source.parent
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"Output and source must be separate: {protected}")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    return output
