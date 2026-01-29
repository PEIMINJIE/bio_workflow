from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def chunk_text(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current = []
    current_len = 0
    for line in text.splitlines():
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> List[Any]:
    return [item for sub in list_of_lists for item in sub]
