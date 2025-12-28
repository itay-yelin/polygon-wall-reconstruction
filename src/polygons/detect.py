from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Any, Optional, List

# Optimized Shapely imports
from shapely import LineString, MultiLineString, Polygon, Point as ShapelyPoint, set_precision, get_coordinates
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from .io_json import Line, Point

@dataclass
class DetectionConfig:
    min_area: float = 0.5           
    min_line_len: float = 0.01      
    snap_tol: float = 0.01           
    extension_dist: float = 10.0     
    bridge_dist: float = 0.75
    gap_bridge_dist: float = 2.5    
    eps_point: float = 0.1          
    area_ratio_thresh: float = 1.35 
    contain_tol: float = 1e-6
    min_compactness: float = 0.0
    coverage_buffer: float = 3.0    
    merge_overlapping: bool = True

def _intersect_ray_segment_optimized(ray_origin, ray_dir, segment_coords):
    rx, ry = ray_origin
    dx, dy = ray_dir
    (sx1, sy1), (sx2, sy2) = segment_coords

    r_cross_s = dx * (sy2 - sy1) - dy * (sx2 - sx1)
    if abs(r_cross_s) < 1e-9: return None

    diff_x = sx1 - rx
    diff_y = sy1 - ry
    t = (diff_x * (sy2 - sy1) - diff_y * (sx2 - sx1)) / r_cross_s
    u = (diff_x * dy - diff_y * dx) / r_cross_s

    if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return (rx + t * dx, ry + t * dy)
    return None

def _extend_undershoots_bulk(geoms: List[LineString], max_dist: float) -> List[LineString]:
    if not geoms: return []
    
    # OPTIMIZATION: Use Shapely 2.0 fast coordinate access
    # get_coordinates returns (N*2, 2). Reshape to (N, 4) -> x1, y1, x2, y2
    coords = get_coordinates(geoms).reshape(-1, 4)
    
    tree = STRtree(geoms)
    
    # Create points for spatial query
    p1_pts = [ShapelyPoint(c[0], c[1]) for c in coords]
    p2_pts = [ShapelyPoint(c[2], c[3]) for c in coords]
    
    # Bulk Query
    p1_indices = tree.query(p1_pts, predicate="dwithin", distance=max_dist)
    p2_indices = tree.query(p2_pts, predicate="dwithin", distance=max_dist)

    best_p2 = {}
    best_p1 = {}

    # Helper to process hits
    def process_hits(query_indices, target_indices, is_p1: bool):
        updates = best_p1 if is_p1 else best_p2
        col_idx_x, col_idx_y = (0, 1) if is_p1 else (2, 3)
        
        for q_idx, t_idx in zip(query_indices, target_indices):
            if q_idx == t_idx: continue
            
            x_src, y_src = coords[q_idx][col_idx_x], coords[q_idx][col_idx_y]
            
            # Vector logic depending on which end we are extending
            if is_p1:
                dx, dy = x_src - coords[q_idx][2], y_src - coords[q_idx][3]
            else:
                dx, dy = x_src - coords[q_idx][0], y_src - coords[q_idx][1]

            target_seg = ((coords[t_idx][0], coords[t_idx][1]), (coords[t_idx][2], coords[t_idx][3]))
            hit = _intersect_ray_segment_optimized((x_src, y_src), (dx, dy), target_seg)
            
            if hit:
                d = math.hypot(hit[0] - x_src, hit[1] - y_src)
                if d < max_dist:
                    if q_idx not in updates or d < updates[q_idx][0]:
                        updates[q_idx] = (d, hit)

    process_hits(p1_indices[0], p1_indices[1], is_p1=True)
    process_hits(p2_indices[0], p2_indices[1], is_p1=False)

    # Apply updates
    new_geoms = []
    for i, c in enumerate(coords):
        p1 = best_p1[i][1] if i in best_p1 else (c[0], c[1])
        p2 = best_p2[i][1] if i in best_p2 else (c[2], c[3])
        
        # Only create new object if changed
        if i in best_p1 or i in best_p2:
            new_geoms.append(LineString([p1, p2]))
        else:
            new_geoms.append(geoms[i])
            
    return new_geoms

def _bridge_gaps_bulk(geoms: List[LineString], bridge_dist: float) -> List[LineString]:
    if not geoms: return []
    
    # Extract start/end points for every line
    # indices 2*i and 2*i+1
    coords = get_coordinates(geoms).reshape(-1, 4)
    points = []
    for c in coords:
        points.append(ShapelyPoint(c[0], c[1]))
        points.append(ShapelyPoint(c[2], c[3]))

    tree = STRtree(points)
    # query returns pairs of [query_idx, result_idx]
    pairs = tree.query(points, predicate="dwithin", distance=bridge_dist)
    
    new_bridges = []
    seen_pairs = set()

    for i, j in zip(pairs[0], pairs[1]):
        if i >= j: continue 
        
        line_i, line_j = i // 2, j // 2
        if line_i == line_j: continue # Don't bridge self
        
        pair_key = tuple(sorted((i, j)))
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)
        
        p1, p2 = points[i], points[j]
        if p1.distance(p2) < 1e-3: continue

        # --- Vector Logic ---
        # Calculate line vectors (pointing AWAY from the endpoint)
        def get_out_vector(c_arr, endpoint_idx):
            # endpoint_idx 0 = start, 1 = end
            if endpoint_idx == 0: 
                dx, dy = c_arr[0] - c_arr[2], c_arr[1] - c_arr[3]
            else: 
                dx, dy = c_arr[2] - c_arr[0], c_arr[3] - c_arr[1]
            l = math.hypot(dx, dy)
            return (dx/l, dy/l) if l > 1e-9 else (0,0)

        v1 = get_out_vector(coords[line_i], i % 2)
        v2 = get_out_vector(coords[line_j], j % 2)
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        
        should_bridge = False
        
        # 1. Facing (Collinear opposite)
        if dot < -0.9:
            # displacement vector
            dx, dy = p2.x - p1.x, p2.y - p1.y
            l = math.hypot(dx, dy)
            if l > 1e-9:
                # check alignment
                if (v1[0]*(dx/l) + v1[1]*(dy/l)) > 0.9:
                    should_bridge = True
        
        # 2. Perpendicular
        elif abs(dot) < 0.2:
            should_bridge = True
            
        # 3. Side-by-Side (Parallel vectors, perpendicular bridge)
        elif dot > 0.9:
            dx, dy = p2.x - p1.x, p2.y - p1.y
            l = math.hypot(dx, dy)
            if l > 1e-9:
                # Bridge is perpendicular to wall direction
                if abs(v1[0]*(dx/l) + v1[1]*(dy/l)) < 0.2:
                    should_bridge = True

        if should_bridge:
            new_bridges.append(LineString([p1, p2]))
        
    return geoms + new_bridges

def _postprocess_polygons(polys: List[Polygon], cfg: DetectionConfig) -> List[Polygon]:
    # 1. Area Filter
    polys = [p for p in polys if p.area >= cfg.min_area]
    if not polys: return []

    # 2. Simplification & sorting
    uniq = {p.simplify(0.0).wkt: p for p in polys}
    polys = sorted(list(uniq.values()), key=lambda p: p.area)

    # 3. Optimized Containment Check (STRtree)
    # Replaces O(N^2) loop with O(N log N)
    tree = STRtree(polys)
    
    # Query: which polygons contain which?
    # returns [indices_of_inner, indices_of_outer]
    contains_indices = tree.query(polys, predicate="contains")
    
    removed = set()
    for inner_idx, outer_idx in zip(contains_indices[0], contains_indices[1]):
        if inner_idx == outer_idx: continue
        
        inner = polys[inner_idx]
        outer = polys[outer_idx]
        
        # Check buffer tolerance
        if outer.buffer(cfg.contain_tol).contains(inner):
            ratio = outer.area / max(inner.area, 1e-9)
            if ratio < cfg.area_ratio_thresh:
                removed.add(outer_idx) # Remove duplicate wall (outer)

    polys = [p for i, p in enumerate(polys) if i not in removed]

    # 4. Compactness
    def compactness(p): 
        return (4.0 * math.pi * p.area) / (p.length**2) if p.length > 0 else 0
    polys = [p for p in polys if compactness(p) >= cfg.min_compactness]

    if not polys: return []

    # 5. Non-Destructive Merge
    if cfg.merge_overlapping:
        merged = unary_union(polys)
        if merged.is_empty: return []
        
        if merged.geom_type == 'Polygon': 
            return [merged]
        elif merged.geom_type == 'MultiPolygon': 
            return list(merged.geoms)
        elif merged.geom_type == 'GeometryCollection':
            return [g for g in merged.geoms if g.geom_type == 'Polygon']
        return []
    
    return polys

def _calculate_coverage_fast(geoms: List[LineString], polys: List[Polygon], buffer_dist: float) -> tuple[float, list[list]]:
    if not geoms or not polys: return 0.0, []
    
    mask = unary_union([p.buffer(buffer_dist) for p in polys])
    all_lines_geom = MultiLineString(geoms)
    
    if all_lines_geom.is_empty: return 0.0, []
    total_len = all_lines_geom.length
    if total_len == 0: return 0.0, []

    missing_geom = all_lines_geom.difference(mask)
    
    missing_segs = []
    if missing_geom.geom_type == 'LineString':
        if missing_geom.length > 0.1: missing_segs.append(list(missing_geom.coords))
    elif missing_geom.geom_type == 'MultiLineString':
        for g in missing_geom.geoms:
            if g.length > 0.1: missing_segs.append(list(g.coords))

    coverage = ((total_len - missing_geom.length) / total_len) * 100.0
    return max(0.0, min(100.0, coverage)), missing_segs

def detect_polygons(lines: Sequence[Line], config: Optional[DetectionConfig] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config is None: config = DetectionConfig()
    
    # Pre-convert to Shapely LineStrings once
    # Assumption: Line object has .to_shapely() or we convert manually here if not.
    # To be safe and dependency-free on Line changes:
    shapely_lines = [LineString([(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)]) for ln in lines]

    if not shapely_lines: return [], {"coverage_pct": 0.0, "missing_lines": []}

    # 1. Ray Casting
    shapely_lines = _extend_undershoots_bulk(shapely_lines, config.extension_dist)

    # 2. Directional Bridging
    shapely_lines = _bridge_gaps_bulk(shapely_lines, config.bridge_dist)

    # 3. Planarization
    processed_geom = unary_union(set_precision(MultiLineString(shapely_lines), grid_size=config.snap_tol))

    # 4. Polygonize
    raw_polys = list(polygonize(processed_geom))

    # 5. Post-Process
    clean_polys = _postprocess_polygons(raw_polys, config)

    # 6. Stats
    coverage, missing = _calculate_coverage_fast(shapely_lines, clean_polys, config.coverage_buffer)

    out_json = []
    def _extract_coords(geom):
        if geom.is_empty: return
        if geom.geom_type == 'Polygon':
            out_json.append({"points": list(geom.exterior.coords)})
        elif geom.geom_type == 'MultiPolygon':
            for part in geom.geoms: _extract_coords(part)

    for p in clean_polys: _extract_coords(p)

    return out_json, {"coverage_pct": coverage, "missing_lines": missing}