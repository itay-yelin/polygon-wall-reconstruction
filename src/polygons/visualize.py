import os
from typing import Sequence, Any, List, Tuple
import matplotlib.pyplot as plt
from .io_json import Line

def save_visualization(
    lines: Sequence[Line], 
    polygons: list[dict[str, Any]], 
    outdir: str,
    missing_lines: List[List[Tuple[float, float]]] = None
) -> str:
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, "polygons.png")
    out_svg = os.path.join(outdir, "polygons.svg")
    out_png_only = os.path.join(outdir, "polygons_only.png")

    xs = []
    ys = []
    for ln in lines:
        xs.extend([ln.start.x, ln.end.x])
        ys.extend([ln.start.y, ln.end.y])

    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        padx = (maxx - minx) * 0.02
        pady = (maxy - miny) * 0.02
    else:
        minx = miny = 0.0
        maxx = maxy = 1.0
        padx = pady = 0.0

    def draw(show_lines: bool, path: str):
        plt.figure(figsize=(12, 12))

        if show_lines:
            for ln in lines:
                plt.plot(
                    [ln.start.x, ln.end.x], [ln.start.y, ln.end.y],
                    color="gray", linewidth=0.6, alpha=0.35, label="_nolegend_",
                )
            plt.plot([], [], color="gray", linewidth=1.5, alpha=0.35, label="Input lines")

        drew_poly = False
        for poly in polygons:
            pts = poly.get("points")
            if not pts: continue
            xs2 = [p[0] for p in pts]
            ys2 = [p[1] for p in pts]
            if xs2[0] != xs2[-1] or ys2[0] != ys2[-1]:
                xs2.append(xs2[0])
                ys2.append(ys2[0])
            plt.plot(xs2, ys2, color="tab:blue", linewidth=2.2, alpha=0.95, label="Detected" if not drew_poly else "_nolegend_")
            plt.fill(xs2, ys2, color="tab:blue", alpha=0.12, label="_nolegend_")
            drew_poly = True

        drew_missing = False
        if show_lines and missing_lines:
            for m_seg in missing_lines:
                m_xs = [p[0] for p in m_seg]
                m_ys = [p[1] for p in m_seg]
                plt.plot(m_xs, m_ys, color="tab:red", linewidth=2.0, alpha=0.9, label="Missing" if not drew_missing else "_nolegend_")
                drew_missing = True

        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)
        ax.set_title("Detected polygons")
        handles, labels = ax.get_legend_handles_labels()
        if labels: ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    draw(show_lines=True, path=out_png)
    draw(show_lines=True, path=out_svg)
    draw(show_lines=False, path=out_png_only)

    return out_png

def save_all_polygons_only(polygons: list[dict[str, Any]], outdir: str, color: str = "tab:blue", linewidth: float = 3.0) -> str:
    import matplotlib.pyplot as plt
    import os
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "all_polygons.png")
    plt.figure(figsize=(12, 12))
    for poly in polygons:
        pts = poly.get("points")
        if not pts or len(pts) < 3: continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0])
            ys.append(ys[0])
        plt.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.95)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path