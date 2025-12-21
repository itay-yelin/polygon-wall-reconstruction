from polygons.io_json import load_lines_from_json

def test_load_lines_smoke():
    lines = load_lines_from_json("data/a_wall_layer.json")
    assert len(lines) > 0
