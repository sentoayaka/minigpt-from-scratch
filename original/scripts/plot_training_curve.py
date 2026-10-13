import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_series(path):
    series = defaultdict(list)
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            series[row["split"]].append((int(row["step"]), float(row["loss"])))
    return series


def points_to_path(points, x_min, x_max, y_min, y_max, width, height, pad):
    coords = []
    for step, loss in points:
        x = pad + (step - x_min) / max(1, x_max - x_min) * (width - 2 * pad)
        y = height - pad - (loss - y_min) / max(1e-12, y_max - y_min) * (height - 2 * pad)
        coords.append(f"{x:.2f},{y:.2f}")
    return "M " + " L ".join(coords)


def main():
    parser = argparse.ArgumentParser(description="Plot train/valid loss CSV as a small SVG.")
    parser.add_argument("--log_path", default="outputs/logs/tinystories_experiment.csv")
    parser.add_argument("--output_path", default="outputs/figures/tinystories_loss.svg")
    args = parser.parse_args()

    series = read_series(args.log_path)
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        raise ValueError(f"no rows found in {args.log_path}")

    x_min = min(step for step, _ in all_points)
    x_max = max(step for step, _ in all_points)
    y_min = min(loss for _, loss in all_points)
    y_max = max(loss for _, loss in all_points)

    width, height, pad = 760, 420, 56
    colors = {"train": "#2563eb", "valid": "#dc2626"}
    paths = []
    for split, points in series.items():
        if len(points) < 2:
            continue
        color = colors.get(split, "#111827")
        d = points_to_path(points, x_min, x_max, y_min, y_max, width, height, pad)
        paths.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3"/>')

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="32" font-family="Arial" font-size="22" font-weight="700">Training Curve</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>
  <text x="{width//2}" y="{height-14}" font-family="Arial" font-size="14" text-anchor="middle">step</text>
  <text x="16" y="{height//2}" font-family="Arial" font-size="14" transform="rotate(-90 16 {height//2})" text-anchor="middle">loss</text>
  <text x="{pad}" y="{height-pad+22}" font-family="Arial" font-size="12">{x_min}</text>
  <text x="{width-pad}" y="{height-pad+22}" font-family="Arial" font-size="12" text-anchor="end">{x_max}</text>
  <text x="{pad-8}" y="{height-pad}" font-family="Arial" font-size="12" text-anchor="end">{y_min:.3f}</text>
  <text x="{pad-8}" y="{pad+4}" font-family="Arial" font-size="12" text-anchor="end">{y_max:.3f}</text>
  {''.join(paths)}
  <circle cx="{width-170}" cy="28" r="5" fill="#2563eb"/><text x="{width-158}" y="33" font-family="Arial" font-size="13">train</text>
  <circle cx="{width-100}" cy="28" r="5" fill="#dc2626"/><text x="{width-88}" y="33" font-family="Arial" font-size="13">valid</text>
</svg>
"""
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
