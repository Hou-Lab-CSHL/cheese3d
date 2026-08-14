#!/usr/bin/env python
"""Render labelled previews of a DeepLabCut labeled-data folder, for picking by eye.

Draws each frame's keypoints over the image and writes the result to a new
directory, plus a contact sheet of every frame at thumbnail size and a CSV
index. The source folder is only ever read: previews go somewhere else, so
the labelled data stays exactly as it was.

Frames are numbered identically in the contact sheet, the per-frame files and
the index, so a number picked off the sheet identifies a frame everywhere.

    python scripts/label_preview.py <labeled-data/SESSION> [-o OUTPUT]

Points are coloured by view -- Cheese3D's bodypart names carry (left) and
(right) suffixes, and the two sides overlap heavily in a head-on frame, so
colouring by side is what makes an overlay readable. A frame's labelled-point
count is on both the contact sheet and the index: partially labelled frames
are the ones worth spotting before training on them.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402
from PIL import Image                    # noqa: E402

SIDE_COLOURS = {"left": "#ff5c8a", "right": "#3fc1c9", "": "#ffd166"}


def read_labels(folder: Path) -> pd.DataFrame:
    """Load the label table into a frame indexed by image filename.

    Two filenames appear in practice -- DeepLabCut's own CollectedData_*.csv
    and the annotation.csv that Cheese3D-converted sessions carry -- with the
    same scorer/bodyparts/coords header either way. The CSV is read rather
    than the .h5 beside it so this runs in any environment: the HDF5 copy
    needs pytables, the CSV needs nothing.
    """
    matches = (sorted(folder.glob("CollectedData_*.csv"))
               + sorted(folder.glob("annotation.csv")))
    if not matches:
        sys.exit(f"no CollectedData_*.csv or annotation.csv in {folder}")
    frame = pd.read_csv(matches[0], header=[0, 1, 2], index_col=[0, 1, 2])
    # Some exports repeat the index names as a fourth header line, which
    # pandas reads as an all-empty first row. Drop rows with no coordinates
    # rather than matching on that literal text.
    frame = frame.dropna(how="all")
    frame.index = [row[-1] for row in frame.index]      # keep the filename
    return frame


def side_of(bodypart: str) -> str:
    if "(left)" in bodypart:
        return "left"
    if "(right)" in bodypart:
        return "right"
    return ""


def points_of(row: pd.Series) -> list[tuple[str, float, float]]:
    """Every labelled (x, y) in one row, dropping unlabelled bodyparts."""
    found = []
    for bodypart in dict.fromkeys(name for _, name, _ in row.index):
        try:
            x = row.xs((bodypart, "x"), level=(1, 2)).iloc[0]
            y = row.xs((bodypart, "y"), level=(1, 2)).iloc[0]
        except KeyError:
            continue
        if pd.notna(x) and pd.notna(y):
            found.append((bodypart, float(x), float(y)))
    return found


def draw_frame(image_path: Path, points, index: int, out: Path,
               annotate: bool) -> None:
    image = Image.open(image_path)
    width, height = image.size
    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(image, cmap="gray")
    axes.set_axis_off()

    for bodypart, x, y in points:
        colour = SIDE_COLOURS[side_of(bodypart)]
        axes.plot(x, y, "o", markersize=7, markerfacecolor=colour,
                  markeredgecolor="black", markeredgewidth=0.8)
        if annotate:
            axes.annotate(bodypart.replace("(base)", "").replace("(left)", "")
                          .replace("(right)", "").strip(),
                          (x, y), textcoords="offset points", xytext=(6, 4),
                          fontsize=6, color="white",
                          path_effects=None)

    axes.text(6, 6, f"#{index:03d}  {image_path.name}  {len(points)} pts",
              va="top", fontsize=11, color="white",
              bbox=dict(facecolor="black", alpha=0.6, pad=3, edgecolor="none"))
    figure.savefig(out, dpi=dpi)
    plt.close(figure)


def contact_sheet(previews: list[Path], counts: list[int], out: Path,
                  columns: int = 6, thumb: int = 320) -> None:
    """One grid of every frame, for picking numbers off at a glance."""
    rows = math.ceil(len(previews) / columns)
    figure, axes = plt.subplots(rows, columns,
                               figsize=(columns * thumb / 100, rows * thumb / 100),
                               dpi=100)
    figure.patch.set_facecolor("#111111")
    for position, cell in enumerate(figure.axes):
        cell.set_axis_off()
        if position >= len(previews):
            continue
        cell.imshow(Image.open(previews[position]))
        # Red where a frame has fewer points than the fullest one: those are
        # the frames to look at twice before picking them.
        colour = "#ffffff" if counts[position] == max(counts) else "#ff5c5c"
        cell.set_title(f"#{position:03d}  {counts[position]} pts",
                       fontsize=9, color=colour, pad=3)
    figure.tight_layout()
    figure.savefig(out, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="a labeled-data/<session> directory")
    parser.add_argument("-o", "--output", default="",
                        help="where previews go (default: "
                             "~/cheese3d_label_previews/<session>)")
    parser.add_argument("--names", action="store_true",
                        help="write the bodypart name beside every point; "
                             "readable on a single frame, cluttered on 38")
    parser.add_argument("--columns", type=int, default=6,
                        help="contact sheet width, in frames (default 6)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"not a directory: {folder}")
    output = (Path(args.output) if args.output
              else Path.home() / "cheese3d_label_previews" / folder.name)
    output.mkdir(parents=True, exist_ok=True)

    labels = read_labels(folder)
    images = sorted(p for p in folder.glob("*.png"))
    print(f"{len(images)} images, {len(labels)} labelled rows -> {output}")

    previews, counts, rows = [], [], []
    for index, image_path in enumerate(images):
        if image_path.name not in labels.index:
            print(f"  #{index:03d} {image_path.name}: no label row, skipped")
            continue
        found = points_of(labels.loc[image_path.name])
        preview = output / f"{index:03d}_{image_path.stem}.png"
        draw_frame(image_path, found, index, preview, args.names)
        previews.append(preview)
        counts.append(len(found))
        rows.append((index, image_path.name, len(found), str(preview)))

    sheet = output / "contact_sheet.png"
    contact_sheet(previews, counts, sheet, columns=args.columns)

    index_file = output / "index.csv"
    with index_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["number", "image", "labelled_points", "preview"])
        writer.writerows(rows)

    complete = sum(1 for _, _, n, _ in rows if n == max(counts))
    print(f"\ncontact sheet: {sheet}")
    print(f"index:         {index_file}")
    print(f"{complete}/{len(rows)} frames carry the full {max(counts)} points; "
          f"the rest are partially labelled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
