from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Any
from shapely.geometry import LineString

@dataclass(frozen=True)
class Point:
    x: float
    y: float

@dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    id: str | None = None

    def to_shapely(self) -> LineString:
        """
        Converts this domain Line object directly into a Shapely LineString.
        This encapsulates the geometry creation logic.
        """
        return LineString([(self.start.x, self.start.y), (self.end.x, self.end.y)])

def load_lines_from_json(path: str) -> List[Line]:
    """
    Parses a JSON file (DXF-style structure) into a list of Line objects.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    entities = data.get("entities", [])
    
    for i, e in enumerate(entities):
        if e.get("type") == "line":
            sp = e["start_point"]
            ep = e["end_point"]
            
            # Use provided ID or fallback to index if missing
            lid = e.get("id")
            
            lines.append(Line(
                Point(float(sp["x"]), float(sp["y"])),
                Point(float(ep["x"]), float(ep["y"])),
                id=str(lid) if lid is not None else None
            ))
            
    return lines