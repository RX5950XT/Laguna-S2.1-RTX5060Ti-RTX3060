"""Stable UTF-8 JSON report output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def dumps_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON report after flushing it to disk."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(dumps_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def require_within(path: Path, root: Path) -> Path:
    """Resolve an artifact path and reject paths outside the project root."""
    destination = path.resolve()
    boundary = root.resolve()
    try:
        destination.relative_to(boundary)
    except ValueError as error:
        raise ValueError(f"輸出路徑必須位於專案內：{boundary}") from error
    return destination
