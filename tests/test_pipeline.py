import sys
import json
from pathlib import Path

def test_cli_creates_outputs(tmp_path, monkeypatch):
    """
    Simulate running the CLI via arguments and checking output generation.
    """
    from polygons.cli import main

    input_path = tmp_path / "input.json"
    outdir = tmp_path / "out"

    # Create dummy input
    payload = {
        "entities": [
            {"type": "line", "start_point": {"x":0, "y":0}, "end_point": {"x":10, "y":0}},
            {"type": "line", "start_point": {"x":10, "y":0}, "end_point": {"x":10, "y":10}},
            {"type": "line", "start_point": {"x":10, "y":10}, "end_point": {"x":0, "y":10}},
            {"type": "line", "start_point": {"x":0, "y":10}, "end_point": {"x":0, "y":0}}
        ]
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    # Mock argv
    argv = ["polygons.cli", "--input", str(input_path), "--outdir", str(outdir)]
    monkeypatch.setattr(sys, "argv", argv)

    # Run
    rc = main()
    assert rc == 0

    # Verify outputs
    assert (outdir / "polygons.json").exists()
    assert (outdir / "polygons.png").exists()