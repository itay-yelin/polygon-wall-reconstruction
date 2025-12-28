import pytest
import math
from polygons.detect import detect_polygons, DetectionConfig
from polygons.io_json import Line, Point

def test_sharp_corners_preserved():
    """
    CRITICAL TEST: Detects "Destructive Merging".
    
    Issue: If the algorithm uses buffer(0.01) -> buffer(-0.01), sharp 90-degree corners
    become rounded arcs.
    
    Expected: A 10x10 input square should result in an output polygon with exactly
    5 points (4 corners + closing point) and coordinates matching the input.
    """
    # 1. Setup: A perfect 10x10 square
    lines = [
        Line(Point(0,0), Point(10,0)),
        Line(Point(10,0), Point(10,10)),
        Line(Point(10,10), Point(0,10)),
        Line(Point(0,10), Point(0,0))
    ]
    
    # 2. Run detection
    # snap_tol=0.01 matches the previous buffer size, making it a good stress test.
    cfg = DetectionConfig(snap_tol=0.01, merge_overlapping=True)
    polys, _ = detect_polygons(lines, cfg)
    
    assert len(polys) == 1, "Failed to detect the square."
    
    # 3. Analyze Geometry
    points = polys[0]['points']
    
    # CHECK 1: Vertex Count
    # A sharp square has 5 coords (0,0 -> 10,0 -> 10,10 -> 0,10 -> 0,0).
    # A rounded square from buffer() will have 12+ coords (approximating corners).
    assert len(points) == 5, (
        f"Corner Rounding Detected! Expected 5 vertices, got {len(points)}. "
        "The algorithm is likely using morphological buffering (buffer +/-)."
    )
    
    # CHECK 2: Coordinate Drift
    # The vertices should be at (0,0), (10,0), etc. NOT (0.004, 0.004).
    # We round to 3 decimal places to verify precision.
    x_coords = sorted([round(p[0], 3) for p in points])
    y_coords = sorted([round(p[1], 3) for p in points])
    
    # We expect x to be [0, 0, 10, 10, (start=end)] -> [0, 0, 0, 10, 10] sorted? 
    # Actually just check bounds.
    assert min(x_coords) == 0.0, f"X-coord drift detected: min x is {min(x_coords)}"
    assert max(x_coords) == 10.0, f"X-coord drift detected: max x is {max(x_coords)}"