from polygons.io_json import load_lines_from_json
from polygons.detect import detect_polygons

def test_detect_single_square(tmp_path):
    input_path = tmp_path / "square.json"

    input_path.write_text(
        """
        {
          "entities": [
            { "type": "line", "start_point": { "x": 0, "y": 0 }, "end_point": { "x": 10, "y": 0 } },
            { "type": "line", "start_point": { "x": 10, "y": 0 }, "end_point": { "x": 10, "y": 10 } },
            { "type": "line", "start_point": { "x": 10, "y": 10 }, "end_point": { "x": 0, "y": 10 } },
            { "type": "line", "start_point": { "x": 0, "y": 10 }, "end_point": { "x": 0, "y": 0 } }
          ]
        }
        """,
        encoding="utf-8",
    )

    lines = load_lines_from_json(str(input_path))
    polys, _ = detect_polygons(lines)

    assert len(polys) == 1

    pts = polys[0]["points"]
    assert len(pts) >= 4

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    assert min(xs) == 0
    assert max(xs) == 10
    assert min(ys) == 0
    assert max(ys) == 10
