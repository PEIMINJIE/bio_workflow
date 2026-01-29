from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    project: Dict[str, Any]
    search: Dict[str, Any]
    pdf: Dict[str, Any]
    llm: Dict[str, Any]
    extract: Dict[str, Any]
    scoring: Dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return Path(self.project.get("output_dir", "data/output"))

    @property
    def cache_dir(self) -> Path:
        return Path(self.project.get("cache_dir", "data/cache"))

    @property
    def pdf_input_dir(self) -> Path:
        return Path(self.pdf.get("input_dir", "data/input/pdfs"))


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        project=data.get("project", {}),
        search=data.get("search", {}),
        pdf=data.get("pdf", {}),
        llm=data.get("llm", {}),
        extract=data.get("extract", {}),
        scoring=data.get("scoring", {}),
    )
