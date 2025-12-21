import json
import sys
from pathlib import Path


def test_cli_creates_outputs(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    # src_path = project_root / "src"
    # sys.path.insert(0, str(src_path))

    from polygons.cli import main

    input_path = tmp_path / "input.json"
    outdir = tmp_path / "out"

    payload = {
        "name": "test_layer",
        "entities": [
            {
                "type": "line",
                "start_point": {"x": 0, "y": 0},
                "end_point": {"x": 10, "y": 0},
            },
            {
                "type": "line",
                "start_point": {"x": 10, "y": 0},
                "end_point": {"x": 10, "y": 10},
            },
        ],
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    argv = [
        "polygons.cli",
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = main()
    assert rc == 0

    json_out = outdir / "polygons.json"
    png_out = outdir / "polygons.png"

    assert json_out.exists()
    assert png_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert "polygons" in data
    assert isinstance(data["polygons"], list)
