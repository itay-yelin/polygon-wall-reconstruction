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
MIN_LINE_LEN = 0.5

# ----------------------------
# Helpers
# ----------------------------
def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _is_close(p1: Point, p2: Point, tol=1e-5) -> bool:
    return _distance(p1, p2) < tol

def _get_intersection(l1_coords: tuple, l2_coords: tuple, tol=1e-5) -> Optional[Point]:
    """
    Finds intersection between two line segments (passed as coordinate tuples).
    l1_coords: ((x1, y1), (x2, y2))
    """
    (x1, y1), (x2, y2) = l1_coords
    (x3, y3), (x4, y4) = l2_coords

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if abs(denom) < tol:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if tol < ua < 1 - tol and tol < ub < 1 - tol:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None
def _planarize_lines(lines: Sequence[Line]) -> List[Line]:
    """
    Breaks lines at intersection points so that no two lines cross.
    Handles 'Line' objects correctly.
    """
    cut_points = defaultdict(list)
    
    # Pre-convert all lines to coordinate tuples for math operations
    # shape: [ ((x1, y1), (x2, y2)), ... ]
    line_coords = [
        ((ln.start.x, ln.start.y), (ln.end.x, ln.end.y)) 
        for ln in lines
    ]

    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            # Pass simple tuples to the math function
            intersection = _get_intersection(line_coords[i], line_coords[j])
            if intersection:
                cut_points[i].append(intersection)
                cut_points[j].append(intersection)

    final_segments = []

    for i, line in enumerate(lines):
        # Unpack coordinates from the Line object
        start = (line.start.x, line.start.y)
        end = (line.end.x, line.end.y)
        
        points_on_line = [start, end]
        points_on_line.extend(cut_points[i])
        
        # Sort by distance from start
        points_on_line.sort(key=lambda p: _distance(start, p))
        
        for k in range(len(points_on_line) - 1):
            p1 = points_on_line[k]
            p2 = points_on_line[k+1]
            
            if not _is_close(p1, p2):
                # Reconstruct a proper Line object (inheriting the ID)
                # Assuming Point(x, y) constructor exists
                new_line = Line(Point(p1[0], p1[1]), Point(p2[0], p2[1]), line.id)
                final_segments.append(new_line)

    return final_segments

def _intersect_ray_segment(ray_origin: Point, ray_dir: Point, seg_p1: Point, seg_p2: Point) -> Optional[Point]:
    """
    Finds intersection between a RAY (origin + t * dir) and a SEGMENT (p1-p2).
    Returns the point if it hits, otherwise None.
    """
    rx, ry = ray_origin
    dx, dy = ray_dir
    sx1, sy1 = seg_p1
    sx2, sy2 = seg_p2

    # Vector form of segment: S(u) = p1 + u * (p2 - p1)
    # Vector form of ray: R(t) = origin + t * dir
    # We solve for t and u:
    # origin + t * dir = p1 + u * (p2 - p1)
    
    # Cross product 2D analog to solve linear system
    r_cross_s = dx * (sy2 - sy1) - dy * (sx2 - sx1)
    
    # Parallel check (cross product near 0)
    if abs(r_cross_s) < 1e-9:
        return None

    # Solve for t (distance along ray) and u (position along segment)
    # t = (p1 - origin) x (p2 - p1) / (dir x (p2 - p1))
    # u = (p1 - origin) x dir / (dir x (p2 - p1))
    
    diff_x = sx1 - rx
    diff_y = sy1 - ry
    
    t = (diff_x * (sy2 - sy1) - diff_y * (sx2 - sx1)) / r_cross_s
    u = (diff_x * dy - diff_y * dx) / r_cross_s

    # Conditions for intersection:
    # t > 0: Ray must move forward (not backward)
    # 0 <= u <= 1: Intersection must be WITHIN the target segment
    if t > 1e-9 and 0 <= u <= 1:
        # Intersection point
        return (rx + t * dx, ry + t * dy)
        
    return None

def _extend_undershoots(lines: Sequence[Line], max_distance: float = 0.5) -> List[Line]:
    """
    Extends line endpoints along their vector if they are close to intersecting.
    """
    # 1. Create a mutable list of coordinates: [ [start_tuple, end_tuple], ... ]
    mutable_coords = [
        [(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)] 
        for ln in lines
    ]
    
    for i in range(len(mutable_coords)):
        p1, p2 = mutable_coords[i]
        
        # --- Check Endpoint P2 (Forward) ---
        ray_dir = (p2[0] - p1[0], p2[1] - p1[1])
        best_point = None
        min_dist = float('inf')
        
        for j in range(len(mutable_coords)):
            if i == j: continue
            target_p1, target_p2 = mutable_coords[j]
            
            # Using existing _intersect_ray_segment helper
            hit = _intersect_ray_segment(p2, ray_dir, target_p1, target_p2)
            if hit:
                d = _distance(p2, hit) # Using _distance helper
                if d < max_distance and d < min_dist:
                    min_dist = d
                    best_point = hit
        
        if best_point:
            mutable_coords[i][1] = best_point # Update P2
            
        # --- Check Endpoint P1 (Backward) ---
        # Re-read P2 (it might have changed)
        p1, p2 = mutable_coords[i]
        ray_dir = (p1[0] - p2[0], p1[1] - p2[1])
        best_point = None
        min_dist = float('inf')
        
        for j in range(len(mutable_coords)):
            if i == j: continue
            target_p1, target_p2 = mutable_coords[j]
            
            hit = _intersect_ray_segment(p1, ray_dir, target_p1, target_p2)
            if hit:
                d = _distance(p1, hit)
                if d < max_distance and d < min_dist:
                    min_dist = d
                    best_point = hit
                    
        if best_point:
            mutable_coords[i][0] = best_point # Update P1

    # 2. Convert back to Line objects
    result_lines = []
    for i, (start, end) in enumerate(mutable_coords):
        original_id = lines[i].id
        result_lines.append(
            Line(Point(start[0], start[1]), Point(end[0], end[1]), original_id)
        )

    return result_lines

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

# Add this parameter at the top with the others if it's missing
MIN_LINE_LEN = 0.5 

def detect_polygons(lines: Sequence[Line]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    
    # 0) Preprocessing
    # 0.1 Extend lines a bit (fix undershoots)
    lines = _extend_undershoots(lines)
    
    # 0.2 Planarize: break lines at intersections
    lines = _planarize_lines(lines)

    # 0.3) Pre-snap endpoints to close small gaps
    lines = _snap_endpoints(lines, tol=0.5)

    # 0.4) Remove lines that are too short (FIXED)
    # We calculate length manually: hypot(dx, dy)
    valid_lines = []
    for ln in lines:
        length = math.hypot(ln.start.x - ln.end.x, ln.start.y - ln.end.y)
        if length > MIN_LINE_LEN:
            valid_lines.append(ln)
    lines = valid_lines

    # 1) Build linework for shapely
    segs = [
        LineString([(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)])
        for ln in lines
    ]
    if not segs:
        return [], {"coverage_pct": 0.0, "missing_lines": []}

    # 2) Quantize to a precision grid
    mls = MultiLineString(segs)
    mls = _quantize_linework(mls, grid_size=EPS_POINT)

    # 3) Node intersections + polygonize
    merged = unary_union(MultiLineString(mls))
    polys = list(polygonize(merged))
    if not polys:
        return [], {"coverage_pct": 0.0, "missing_lines": []}

    # 4) Postprocess
    polys = _postprocess_polys(
        polys=polys,
        min_area=MIN_AREA,
        area_ratio_thresh=AREA_RATIO_THRESH,
        contain_tol=CONTAIN_TOL,
        min_compactness=MIN_COMPACTNESS,
    )

    # 5) Calc Stats
    coverage, missing_lines = _calculate_coverage(lines, polys)

    # 6) Convert to output format
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