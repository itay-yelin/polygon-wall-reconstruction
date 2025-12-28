from __future__ import annotations

from typing import Sequence, Any
from math import hypot
import math

from collections import defaultdict
from typing import Sequence, Tuple, List, Optional
from .io_json import Line, Point

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union

try:
    # Shapely 2.x
    from shapely import set_precision as _set_precision
except Exception:
    _set_precision = None

# ----------------------------
# parameters
# ----------------------------

EPS_POINT = 0.1
MIN_AREA = 2.0

AREA_RATIO_THRESH = 1.35
CONTAIN_TOL = 1e-6
MIN_COMPACTNESS = 0.0


# ----------------------------
# Helpers
# ----------------------------
def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _is_close(p1: Point, p2: Point, tol=1e-5) -> bool:
    return _distance(p1, p2) < tol

def _get_intersection(l1: Line, l2: Line, tol=1e-5) -> Optional[Point]:
    """
    Finds intersection between two line segments. 
    Returns None if parallel or if intersection is outside segments.
    """
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Parallel lines (denom is 0)
    if abs(denom) < tol:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # Check if intersection is STRICTLY within the segments 
    # (0 <= u <= 1 means it touches. 0 < u < 1 means strict crossing).
    # We use a small epsilon to handle "almost endpoint" hits.
    if tol < ua < 1 - tol and tol < ub < 1 - tol:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None

def _planarize_lines(lines: Sequence[Line]) -> List[Line]:
    """
    Breaks lines at intersection points so that no two lines cross.
    Resulting lines only touch at endpoints.
    """
    # map: line_index -> list of cut points
    cut_points = defaultdict(list)
    
    # 1. Detect all intersections (O(N^2))
    # Note: For N > 1000, you would want a Sweep Line algorithm here.
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = _get_intersection(lines[i], lines[j])
            if intersection:
                cut_points[i].append(intersection)
                cut_points[j].append(intersection)

    # 2. Reconstruct segments
    final_segments = []

    for i, line in enumerate(lines):
        start, end = line
        
        # Start with original endpoints
        points_on_line = [start, end]
        
        # Add any detected cuts
        points_on_line.extend(cut_points[i])
        
        # 3. Sort points along the line to ensure correct segment order
        # We project points onto the X (or Y) axis of the line to sort them.
        # Simple distance from 'start' works well.
        points_on_line.sort(key=lambda p: _distance(start, p))
        
        # 4. Create new segments between consecutive points
        for k in range(len(points_on_line) - 1):
            p1 = points_on_line[k]
            p2 = points_on_line[k+1]
            
            # Avoid creating zero-length segments (if multiple cuts were close)
            if not is_close(p1, p2):
                final_segments.append((p1, p2))

    return final_segments

def _dist(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)

def _quantize_linework(mls: MultiLineString, grid_size: float):
    """
    Quantize coordinates to a fixed precision grid for topological stability.
    Uses shapely.set_precision if available (Shapely 2.x).
    Falls back to manual rounding if not available.
    """
    if _set_precision is not None:
        return _set_precision(mls, grid_size=grid_size)

    # Fallback: manual grid snapping by rounding coordinates
    def q(v: float) -> float:
        return round(v / grid_size) * grid_size

    snapped = []
    for geom in mls.geoms:
        (x1, y1), (x2, y2) = geom.coords[0], geom.coords[-1]
        snapped.append(LineString([(q(x1), q(y1)), (q(x2), q(y2))]))
    return MultiLineString(snapped)



def _compactness(p: Polygon) -> float:
    if p.length <= 0.0:
        return 0.0
    return (4.0 * math.pi * p.area) / (p.length * p.length)


def _dedup_by_wkt(polys: list[Polygon]) -> list[Polygon]:
    uniq: dict[str, Polygon] = {}
    for p in polys:
        # simplify(0.0) keeps geometry but normalizes representation a bit
        key = p.simplify(0.0).wkt
        if key not in uniq:
            uniq[key] = p
    return list(uniq.values())


def _remove_nested_shells(
    polys: list[Polygon],
    area_ratio_thresh: float,
    tol: float,
) -> list[Polygon]:
    """
    If outer contains inner and outer is only slightly bigger,
    treat outer as the 'other side of the wall' and drop it.
    """
    polys_sorted = sorted(polys, key=lambda p: p.area)
    removed: set[int] = set()

    for i, inner in enumerate(polys_sorted):
        if i in removed:
            continue

        for j in range(i + 1, len(polys_sorted)):
            if j in removed:
                continue

            outer = polys_sorted[j]

            if not outer.buffer(tol).contains(inner):
                continue

            ratio = outer.area / max(inner.area, 1e-9)
            if ratio < area_ratio_thresh:
                removed.add(j)

            # Stop at the first container (smallest outer) for this inner
            break

    return [p for idx, p in enumerate(polys_sorted) if idx not in removed]


def _filter_thin(polys: list[Polygon], min_compactness: float) -> list[Polygon]:
    return [p for p in polys if _compactness(p) >= min_compactness]


def _merge_touching_polys(polys: list[Polygon]) -> list[Polygon]:
    """
    Unify polygons that are touching or overlapping.
    """
    if not polys:
        return []
    
    # union_all is available in recent Shapely, unary_union is older compat
    merged = unary_union(polys)
    
    # Result can be Polygon or MultiPolygon
    if merged.is_empty:
        return []
    if merged.geom_type == 'Polygon':
        return [merged]
    elif merged.geom_type == 'MultiPolygon':
        return list(merged.geoms)
    return []


def _postprocess_polys(
    polys: list[Polygon],
    min_area: float,
    area_ratio_thresh: float,
    contain_tol: float,
    min_compactness: float,
) -> list[Polygon]:
    # 1) Fix invalid, drop empty
    fixed: list[Polygon] = []
    for p in polys:
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        fixed.append(p)

    # 2) Area filter
    fixed = [p for p in fixed if p.area >= min_area]

    # 3) Dedup
    fixed = _dedup_by_wkt(fixed)

    # 4) Remove nested wall shells
    fixed = _remove_nested_shells(fixed, area_ratio_thresh=area_ratio_thresh, tol=contain_tol)

    # 5) Remove thin wall-like polygons
    fixed = _filter_thin(fixed, min_compactness=min_compactness)
    
    # 6) Merge touching polygons (Unify)
    # We buffer slightly to ensure shared edges merge
    buffered = [p.buffer(0.01) for p in fixed]
    merged = _merge_touching_polys(buffered)
    # Debuffer to restore roughly original size (though corners might round slightly)
    # Using join_style=2 (mitre) might help preserve corners, but default is round
    fixed = [p.buffer(-0.01) for p in merged]

    return fixed



def _snap_endpoints(lines: Sequence[Line], tol: float) -> list[Line]:
    """
    Snap endpoints of lines to each other if they are within tolerance.
    This helps close gaps without global grid quantization which distorts geometry.
    """
    # 1. Collect all points
    points = []
    for ln in lines:
        points.append([ln.start.x, ln.start.y])
        points.append([ln.end.x, ln.end.y])

    # 2. Simple clustering: if point B is within tol of point A, move B to A
    # Iterative approach (naive but effective for small N)
    snapped_map = {} # (x,y) -> (new_x, new_y)

    # We iterate and essentially "weld" points.
    # For a production system with large N, use a spatial index (KDTree).
    # Here N is small (< 1000).
    
    # Pre-sort to help? No, just straightforward O(N^2) pass or close to it.
    # Actually, we can just use a list of "canonical" points.
    canonical: list[tuple[float, float]] = []

    for pt in points:
        px, py = pt
        found = False
        for (cx, cy) in canonical:
            if math.hypot(cx - px, cy - py) <= tol:
                snapped_map[(px, py)] = (cx, cy)
                found = True
                break
        if not found:
            canonical.append((px, py))
            snapped_map[(px, py)] = (px, py)

    # 3. Rebuild lines
    new_lines = []
    for ln in lines:
        s = snapped_map.get((ln.start.x, ln.start.y), (ln.start.x, ln.start.y))
        e = snapped_map.get((ln.end.x, ln.end.y), (ln.end.x, ln.end.y))
        new_lines.append(Line(Point(s[0], s[1]), Point(e[0], e[1]), ln.id))
    
    return new_lines


    return fixed


def _calculate_coverage(lines: Sequence[Line], polys: list[Polygon]) -> tuple[float, list[list[tuple[float, float]]]]:
    """
    Calculate the percentage of input line length that is contained within the output polygons.
    Returns: (coverage_pct, missing_segments)
    missing_segments is a list of lines (start, end) that are NOT covered.
    
    The 'missing' parts are calculated by subtracting the INFLATED polygons from the lines.
    Coverage is calculated as 100 * (1 - len(missing) / len(total)).
    """
    if not lines:
        return 0.0, []

    total_len = 0.0
    missing_len = 0.0
    missing_segs = []

    # Create a single geometry for the polygons to speed up intersection
    # Inflate polygons by fixed amount (buffer) to ensure we capture walls and avoid grazing issues
    # Using 5.0 as a robust wall thickness inclusion radius
    if not polys:
        poly_union = Polygon()
    else:
        # Buffer by 0.5 (consistent with snap tolerance) to capture wall thickness
        inflated_polys = [p.buffer(0.5) for p in polys]
        poly_union = unary_union(inflated_polys)
    
    for ln in lines:
        ls = LineString([(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)])
        l_len = ls.length
        total_len += l_len
        
        if l_len > 0:
            if poly_union.is_empty:
                 missing_segs.append(list(ls.coords))
                 missing_len += l_len
            else:
                try:
                    # Difference (missing part)
                    diff = ls.difference(poly_union)
                    if not diff.is_empty:
                        d_len = diff.length
                        missing_len += d_len
                        
                        if diff.geom_type == 'LineString':
                            missing_segs.append(list(diff.coords))
                        elif diff.geom_type == 'MultiLineString':
                            for g in diff.geoms:
                                missing_segs.append(list(g.coords))
                except Exception:
                    # Fallback for topology errors: assume missing
                     missing_segs.append(list(ls.coords))
                     missing_len += l_len

    coverage = 0.0
    if total_len > 0:
        coverage = (1.0 - (missing_len / total_len)) * 100.0
        # Clamp to 0..100 just in case
        coverage = max(0.0, min(100.0, coverage))
    
    return coverage, missing_segs


def detect_polygons(lines: Sequence[Line]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    
    # 0.1 go over all lines: for each line if it intersects with another line, break it into segments
    lines = _planarize_lines(lines)

    # 0.2) Pre-snap endpoints to close small gaps
    # Use 0.5 as tolerance (half wall thickness usually)
    lines = _snap_endpoints(lines, tol=0.5)

    # 0.3) Remove lines that are too short
    lines = [ln for ln in lines if ln.length > MIN_LINE_LEN]



    # 1) Build linework for shapely
    segs = [
        LineString([(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)])
        for ln in lines
    ]
    if not segs:
        return [], {"coverage_pct": 0.0, "missing_lines": []}

    # 2) Quantize to a precision grid (replaces manual snapping)
    mls = MultiLineString(segs)
    mls = _quantize_linework(mls, grid_size=EPS_POINT)

    # 3) Node intersections + polygonize
    merged = unary_union(MultiLineString(mls))
    polys = list(polygonize(merged))
    if not polys:
        return [], {"coverage_pct": 0.0, "missing_lines": []}

    # 4) Postprocess (refactored block)
    polys = _postprocess_polys(
        polys=polys,
        min_area=MIN_AREA,
        area_ratio_thresh=AREA_RATIO_THRESH,
        contain_tol=CONTAIN_TOL,
        min_compactness=MIN_COMPACTNESS,
    )

    # 5) Calc Stats
    coverage, missing_lines = _calculate_coverage(lines, polys)

    # 6) Convert to expected output format for visualize/io_out
    out: list[dict[str, Any]] = []

    def _add_poly(p: Polygon):
        if p.is_empty:
            return
        # Exterior only for now
        coords = list(p.exterior.coords)
        out.append({"points": [(float(x), float(y)) for (x, y) in coords]})

    for geom in polys:
        if geom.geom_type == 'Polygon':
            _add_poly(geom)
        elif geom.geom_type == 'MultiPolygon':
            for p in geom.geoms:
                _add_poly(p)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                if g.geom_type == 'Polygon':
                    _add_poly(g)
                elif g.geom_type == 'MultiPolygon':
                    for p in g.geoms:
                        _add_poly(p)

    return out, {"coverage_pct": coverage, "missing_lines": missing_lines}
