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

# ----------------------------
# 1. Configuration & Constants
# ----------------------------

DOT_PARALLEL = 0.9  # Threshold for ~25 degrees
DOT_PERP = 0.2      # Threshold for ~78-102 degrees

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

# ----------------------------
# 2. Math Helpers (Stateless)
# ----------------------------

def _intersect_ray_segment_optimized(ray_origin, ray_dir, segment_coords):
    """Calculates intersection between a ray and a segment."""
    rx, ry = ray_origin
    dx, dy = ray_dir
    (sx1, sy1), (sx2, sy2) = segment_coords

    r_cross_s = dx * (sy2 - sy1) - dy * (sx2 - sx1)
    if abs(r_cross_s) < 1e-9: return None

    diff_x = sx1 - rx
    diff_y = sy1 - ry
    t = (diff_x * (sy2 - sy1) - diff_y * (sx2 - sx1)) / r_cross_s
    u = (diff_x * dy - diff_y * dx) / r_cross_s

    # Allow slight overshoot for endpoint hits (u in [-epsilon, 1+epsilon])
    if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return (rx + t * dx, ry + t * dy)
    return None

def _compute_unit_vectors(coords: np.ndarray) -> np.ndarray:
    """
    Computes unit vectors for all lines in one pass.
    Input: (N, 4) array [x1, y1, x2, y2]
    Output: (N, 2) array [dx, dy] normalized
    """
    dx = coords[:, 2] - coords[:, 0]
    dy = coords[:, 3] - coords[:, 1]
    lengths = np.hypot(dx, dy)
    # Avoid division by zero
    lengths[lengths < 1e-9] = 1.0 
    return np.stack((dx / lengths, dy / lengths), axis=1)

# ----------------------------
# 3. Core Logic
# ----------------------------

def _extend_undershoots_bulk(geoms: List[LineString], max_dist: float) -> List[LineString]:
    if not geoms: return []
    
    # 1. Setup Data
    coords = get_coordinates(geoms).reshape(-1, 4)
    tree = STRtree(geoms)
    
    p1_pts = [ShapelyPoint(c[0], c[1]) for c in coords]
    p2_pts = [ShapelyPoint(c[2], c[3]) for c in coords]
    
    # 2. Bulk Query
    p1_indices = tree.query(p1_pts, predicate="dwithin", distance=max_dist)
    p2_indices = tree.query(p2_pts, predicate="dwithin", distance=max_dist)

    updates = {} # Map row_idx -> (dist, new_point, is_p1)

    # 3. Process Hits
    # We iterate twice: once for Start points (P1), once for End points (P2)
    # definition: (indices, is_p1, ray_dir_sign)
    passes = [
        (p1_indices, True, -1), # P1: ray goes P2->P1 (backwards relative to line vector)
        (p2_indices, False, 1)  # P2: ray goes P1->P2
    ]

    for (q_idxs, t_idxs), is_p1, sign in passes:
        for q, t in zip(q_idxs, t_idxs):
            if q == t: continue
            
            # Origin
            src_x = coords[q][0] if is_p1 else coords[q][2]
            src_y = coords[q][1] if is_p1 else coords[q][3]
            
            # Direction: (dx, dy) of the line * sign
            line_dx = coords[q][2] - coords[q][0]
            line_dy = coords[q][3] - coords[q][1]
            
            target_seg = ((coords[t][0], coords[t][1]), (coords[t][2], coords[t][3]))
            
            hit = _intersect_ray_segment_optimized(
                (src_x, src_y), 
                (line_dx * sign, line_dy * sign), 
                target_seg
            )
            
            if hit:
                d = math.hypot(hit[0] - src_x, hit[1] - src_y)
                if d < max_dist:
                    # Update if closer
                    key = (q, is_p1)
                    if key not in updates or d < updates[key][0]:
                        updates[key] = (d, hit)

    # 4. Apply Updates
    new_geoms = []
    for i in range(len(geoms)):
        p1_new = updates.get((i, True), (None, (coords[i][0], coords[i][1])))[1]
        p2_new = updates.get((i, False), (None, (coords[i][2], coords[i][3])))[1]
        
        if (i, True) in updates or (i, False) in updates:
            new_geoms.append(LineString([p1_new, p2_new]))
        else:
            new_geoms.append(geoms[i])
            
    return new_geoms

def _bridge_gaps_bulk(geoms: List[LineString], bridge_dist: float) -> List[LineString]:
    if not geoms: return []
    
    coords = get_coordinates(geoms).reshape(-1, 4)
    
    # Pre-calculate unit vectors for all lines (N, 2)
    # Vector points P1 -> P2
    unit_vecs = _compute_unit_vectors(coords)
    
    # Build points for query
    # Flattened list: [L0_Start, L0_End, L1_Start, L1_End, ...]
    points = []
    for c in coords:
        points.append(ShapelyPoint(c[0], c[1]))
        points.append(ShapelyPoint(c[2], c[3]))

    tree = STRtree(points)
    pairs = tree.query(points, predicate="dwithin", distance=bridge_dist)
    
    new_bridges = []
    seen_pairs = set()

    # Iterate over nearby endpoint pairs
    for i, j in zip(pairs[0], pairs[1]):
        if i >= j: continue 
        
        line_i, pt_i_type = divmod(i, 2) # type 0=Start, 1=End
        line_j, pt_j_type = divmod(j, 2)
        
        if line_i == line_j: continue 
        
        pair_key = tuple(sorted((i, j)))
        if pair_key in seen_pairs: continue
        seen_pairs.add(pair_key)
        
        p1 = points[i]
        p2 = points[j]
        
        # Micro-optimization: Squared distance check before sqrt
        dist_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
        if dist_sq < 1e-6: continue 

        # --- Vector Logic ---
        # Get vectors pointing OUT from the endpoint
        # If endpoint is Start (0), Out vector is P1 - P2 = -Vec
        # If endpoint is End (1), Out vector is P2 - P1 = +Vec
        u1 = unit_vecs[line_i] * (-1 if pt_i_type == 0 else 1)
        u2 = unit_vecs[line_j] * (-1 if pt_j_type == 0 else 1)
        
        dot = u1[0]*u2[0] + u1[1]*u2[1]
        
        should_bridge = False
        
        # 1. Collinear/Facing (Opposite directions -> dot < -0.9)
        if dot < -DOT_PARALLEL:
            # Displacement vector
            disp_x, disp_y = p2.x - p1.x, p2.y - p1.y
            disp_len = math.sqrt(dist_sq)
            if disp_len > 1e-9:
                # Check if u1 points towards p2
                if (u1[0]*(disp_x/disp_len) + u1[1]*(disp_y/disp_len)) > DOT_PARALLEL:
                    should_bridge = True
        
        # 2. Perpendicular (Cross corner)
        elif abs(dot) < DOT_PERP:
            should_bridge = True
            
        # 3. Side-by-Side (Parallel -> dot > 0.9)
        elif dot > DOT_PARALLEL:
            # Bridge must be perpendicular to lines
            disp_x, disp_y = p2.x - p1.x, p2.y - p1.y
            disp_len = math.sqrt(dist_sq)
            if disp_len > 1e-9:
                # Dot of u1 and displacement should be ~0
                if abs(u1[0]*(disp_x/disp_len) + u1[1]*(disp_y/disp_len)) < DOT_PERP:
                    should_bridge = True

        if should_bridge:
            new_bridges.append(LineString([p1, p2]))
        
    return geoms + new_bridges

# ----------------------------
# 4. Processing Pipeline
# ----------------------------

def _postprocess_polygons(polys: List[Polygon], cfg: DetectionConfig) -> List[Polygon]:
    # 1. Area Filter
    polys = [p for p in polys if p.area >= cfg.min_area]
    if not polys: return []

    # 2. Deduplication
    uniq = {p.simplify(0.0).wkt: p for p in polys}
    polys = sorted(list(uniq.values()), key=lambda p: p.area)

    # 3. Optimized Containment (STRtree)
    tree = STRtree(polys)
    contains_indices = tree.query(polys, predicate="contains")
    
    removed = set()
    for inner_idx, outer_idx in zip(contains_indices[0], contains_indices[1]):
        if inner_idx == outer_idx: continue
        
        inner = polys[inner_idx]
        outer = polys[outer_idx]
        
        # Precise check
        if outer.buffer(cfg.contain_tol).contains(inner):
            ratio = outer.area / max(inner.area, 1e-9)
            if ratio < cfg.area_ratio_thresh:
                removed.add(outer_idx)

    polys = [p for i, p in enumerate(polys) if i not in removed]

    # 4. Compactness
    def compactness(p): 
        return (4.0 * math.pi * p.area) / (p.length**2) if p.length > 0 else 0
    polys = [p for p in polys if compactness(p) >= cfg.min_compactness]

    if not polys: return []

    # 5. Non-Destructive Merge (Exact Geometry)
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
    
    # Boundary Conversion
    # (Assuming io_json.Line object has .to_shapely() based on architecture discussion)
    # If not, fallback manual conversion:
    shapely_lines = []
    for ln in lines:
        if hasattr(ln, 'to_shapely'):
            shapely_lines.append(ln.to_shapely())
        else:
            shapely_lines.append(LineString([(ln.start.x, ln.start.y), (ln.end.x, ln.end.y)]))

    if not shapely_lines: return [], {"coverage_pct": 0.0, "missing_lines": []}

    # Pipeline
    shapely_lines = _extend_undershoots_bulk(shapely_lines, config.extension_dist)
    shapely_lines = _bridge_gaps_bulk(shapely_lines, config.bridge_dist)

    # Noding & Snapping
    processed_geom = unary_union(set_precision(MultiLineString(shapely_lines), grid_size=config.snap_tol))

    # Polygonize
    raw_polys = list(polygonize(processed_geom))

    # Post-Process
    clean_polys = _postprocess_polygons(raw_polys, config)

    # Stats
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