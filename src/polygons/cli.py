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
    polygons, _stats = detect_polygons(lines)

    out_json = write_polygons_json(polygons, args.outdir)
    out_png = save_visualization(lines, polygons, args.outdir)

    out_all = save_all_polygons_only(polygons, args.outdir)
    print(f"all polygons image: {out_all}")
    print(f"lines: {len(lines)}")
    print(f"polygons: {len(polygons)}")
    print(f"wrote: {out_json}")
    print(f"image: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
