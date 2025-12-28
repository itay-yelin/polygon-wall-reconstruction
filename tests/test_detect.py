import pytest
import math
from shapely.geometry import Polygon, LineString
from polygons.io_json import Line, Point
from polygons.detect import (
    detect_polygons,
    DetectionConfig,
    _intersect_ray_segment_optimized,
    _extend_undershoots_bulk,
    _bridge_gaps_bulk,
    _postprocess_polygons,
    _calculate_coverage_fast
)

# =========================================================================
# UNIT TESTS: GEOMETRY HELPERS
# =========================================================================

class TestGeometryHelpers:
    def test_intersect_ray_segment_strict_crossing(self):
        """Ray crosses segment at (2,0)."""
        hit = _intersect_ray_segment_optimized((0,0), (1,0), ((2,-1), (2,1)))
        assert hit is not None
        assert math.isclose(hit[0], 2.0)

    def test_intersect_ray_segment_parallel(self):
        """Parallel ray should not hit."""
        hit = _intersect_ray_segment_optimized((0,0), (1,0), ((0,1), (10,1)))
        assert hit is None

    def test_intersect_ray_segment_endpoint_hit(self):
        """Ray hits exactly on the endpoint."""
        hit = _intersect_ray_segment_optimized((0,0), (1,0), ((2,0), (2,5)))
        assert hit is not None
        assert math.isclose(hit[0], 2.0)

class TestExtensionLogic:
    def test_extend_simple_t_junction(self):
        """Horizontal line should extend to hit vertical line."""
        lines = [LineString([(0,5), (8,5)]), LineString([(10,0), (10,10)])]
        extended = _extend_undershoots_bulk(lines, max_dist=3.0)
        # Check if the first line (horizontal) ends at x=10
        assert math.isclose(extended[0].coords[1][0], 10.0)

    def test_extend_closest_hit_only(self):
        """Ray should stop at the FIRST wall, not shoot through to the second."""
        lines = [LineString([(0,5), (8,5)]), LineString([(10,0), (10,10)]), LineString([(20,0), (20,10)])]
        extended = _extend_undershoots_bulk(lines, max_dist=50.0)
        assert math.isclose(extended[0].coords[1][0], 10.0, abs_tol=1e-5)

class TestBridgingLogic:
    def test_bridge_double_wall_cap(self):
        """Side-by-side parallel lines (Double Wall Cap) should be bridged."""
        lines = [LineString([(0,0), (10,0)]), LineString([(0,1), (10,1)])]
        result = _bridge_gaps_bulk(lines, bridge_dist=1.5)
        # 2 original + 2 bridges = 4
        assert len(result) >= 4

    def test_dont_bridge_connected_points(self):
        """Already connected points should not be bridged."""
        lines = [LineString([(0,0), (5,0)]), LineString([(5,0), (5,5)])]
        result = _bridge_gaps_bulk(lines, bridge_dist=1.0)
        assert len(result) == 2 

    def test_ignore_parallel_hallway(self):
        """Ignore side-by-side if gap is too large (Hallway vs Wall Cap)."""
        lines = [LineString([(0, 0), (10, 0)]), LineString([(0, 0.5), (10, 0.5)])]
        # Gap is 0.5, but we enforce strict bridging
        result = _bridge_gaps_bulk(lines, bridge_dist=0.4) 
        assert len(result) == 2

    def test_bridge_valid_corner(self):
        """Bridge perpendicular lines (L-Junction)."""
        lines = [LineString([(0, 0), (5, 0)]), LineString([(5.2, 0.2), (5.2, 5)])]
        result = _bridge_gaps_bulk(lines, bridge_dist=0.75)
        assert len(result) == 3

    def test_bridge_collinear_gap(self):
        """Bridge collinear facing lines."""
        lines = [LineString([(0, 0), (4, 0)]), LineString([(4.5, 0), (10, 0)])]
        result = _bridge_gaps_bulk(lines, bridge_dist=0.75)
        assert len(result) == 3

    def test_ignore_diverging_corners(self):
        """Ignore lines forming a sharp V (Rabbit Ears) pointing away."""
        lines = [
            LineString([(0, 0), (-1, 2)]), 
            LineString([(0.5, 0), (1.5, 2)])
        ]
        result = _bridge_gaps_bulk(lines, bridge_dist=0.75)
        assert len(result) == 2

class TestMetricsAndPostProcessing:
    def test_coverage_perfect_square(self):
        lines = [
            LineString([(0,0), (10,0)]), LineString([(10,0), (10,10)]),
            LineString([(10,10), (0,10)]), LineString([(0,10), (0,0)])
        ]
        poly = Polygon([(0,0), (10,0), (10,10), (0,10)])
        cov, _ = _calculate_coverage_fast(lines, [poly], buffer_dist=0.1)
        assert math.isclose(cov, 100.0)

    def test_postprocess_area_filtering(self):
        """Should remove tiny polygons."""
        cfg = DetectionConfig(min_area=10.0)
        p_tiny = Polygon([(0,0), (1,0), (0,1)]) 
        p_big = Polygon([(0,0), (10,0), (10,10), (0,10)]) 
        result = _postprocess_polygons([p_tiny, p_big], cfg)
        assert len(result) == 1

# =========================================================================
# INTEGRATION TESTS: FULL PIPELINE
# =========================================================================

class TestDetectPipeline:
    def test_detect_single_square(self):
        """Simple closed square."""
        lines = [
            Line(Point(0,0), Point(10,0)), Line(Point(10,0), Point(10,10)),
            Line(Point(10,10), Point(0,10)), Line(Point(0,10), Point(0,0))
        ]
        polys, stats = detect_polygons(lines)
        assert len(polys) == 1
        assert stats['coverage_pct'] > 99.0

    def test_open_shape_returns_no_polygons(self):
        """U-shape should not return a polygon."""
        lines = [
            Line(Point(0, 0), Point(10, 0)),
            Line(Point(10, 0), Point(10, 10)),
            Line(Point(10, 10), Point(0, 10))
            # Missing closing edge
        ]
        polys, _ = detect_polygons(lines)
        assert polys == []

    def test_detect_open_corner_recovery(self):
        """Pipeline should fix small gaps in a square."""
        lines = [
            Line(Point(0,0), Point(10,0)), 
            Line(Point(10,0), Point(10,9.5)), # Short
            Line(Point(9.5,10), Point(0,10)), # Short
            Line(Point(0,10), Point(0,0))
        ]
        cfg = DetectionConfig(bridge_dist=1.0, min_area=50.0)
        polys, stats = detect_polygons(lines, cfg)
        assert len(polys) == 1
        assert stats['coverage_pct'] > 95.0

    def test_detect_double_walls_separated(self):
        """
        Concentric double walls should remain separate if merge_overlapping=False.
        """
        inner = [Line(Point(0,0), Point(10,0)), Line(Point(10,0), Point(10,10)),
                 Line(Point(10,10), Point(0,10)), Line(Point(0,10), Point(0,0))]
        outer = [Line(Point(-1,-1), Point(11,-1)), Line(Point(11,-1), Point(11,11)),
                 Line(Point(11,11), Point(-1,11)), Line(Point(-1,11), Point(-1,-1))]
        
        cfg = DetectionConfig(
            snap_tol=0.01, extension_dist=0.1,
            area_ratio_thresh=0.0, merge_overlapping=False
        )
        polys, stats = detect_polygons(inner + outer, cfg)
        assert len(polys) >= 2