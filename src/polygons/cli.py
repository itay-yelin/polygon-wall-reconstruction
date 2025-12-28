import argparse
from .io_json import load_lines_from_json
from .detect import detect_polygons
from .io_out import write_polygons_json
from .visualize import save_visualization
from .visualize import save_all_polygons_only

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    lines = load_lines_from_json(args.input)
    polygons, stats = detect_polygons(lines)

    out_json = write_polygons_json(polygons, args.outdir)
    
    missing = stats.get("missing_lines", [])
    out_png = save_visualization(lines, polygons, args.outdir, missing_lines=missing)

    out_all = save_all_polygons_only(polygons, args.outdir)
    
    print(f"--- Stats ---")
    print(f"Lines: {len(lines)}")
    print(f"Polygons: {len(polygons)}")
    print(f"Coverage: {stats.get('coverage_pct', 0):.2f}%")
    print(f"Missing Segments: {len(missing)}")
    print(f"------------")
    print(f"Wrote: {out_json}")
    print(f"Image: {out_png}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())