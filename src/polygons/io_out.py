import json
import os
from typing import Any


def write_polygons_json(polygons: list[dict[str, Any]], outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "polygons.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"polygons": polygons}, f, indent=2)
    return out_path
