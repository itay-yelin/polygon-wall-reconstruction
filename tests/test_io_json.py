import pytest
from polygons.io_json import load_lines_from_json, Line

def test_load_lines_smoke(tmp_path):
    """Test loading lines from a temporary JSON file."""
    f = tmp_path / "test_input.json"
    f.write_text("""
    {
        "entities": [
            {"type": "line", "start_point": {"x":0,"y":0}, "end_point": {"x":10,"y":0}, "id": "1"}
        ]
    }
    """, encoding="utf-8")
    
    lines = load_lines_from_json(str(f))
    assert len(lines) == 1
    assert isinstance(lines[0], Line)
    assert lines[0].id == "1"