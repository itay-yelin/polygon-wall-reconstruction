from polygons.detect import detect_polygons
from polygons.io_json import Line, Point


def test_open_shape_returns_no_polygons():
    lines = [
        Line(Point(0, 0), Point(10, 0)),
        Line(Point(10, 0), Point(10, 10)),
        Line(Point(10, 10), Point(0, 10))
        # missing closing edge
    ]

    polys, _ = detect_polygons(lines)
    assert polys == []
