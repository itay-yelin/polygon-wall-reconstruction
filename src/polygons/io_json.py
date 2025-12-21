from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    id: str | None = None


def load_lines_from_json(path: str) -> list[Line]:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    entities = data.get("entities", [])
    lines: list[Line] = []

    for e in entities:
        if e.get("type") != "line":
            continue

        sp = e["start_point"]
        ep = e["end_point"]
        line_id = e.get("id")

        lines.append(
            Line(
                start=Point(float(sp["x"]), float(sp["y"])),
                end=Point(float(ep["x"]), float(ep["y"])),
                id=str(line_id) if line_id is not None else None,
            )
        )

    return lines
