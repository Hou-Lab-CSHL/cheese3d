#!/usr/bin/env python
"""Build a video from a list of labelled frames, plus a CSV of their labels.

Reads a file of ``session/image.png`` lines -- the same format
selected_images.txt uses -- and writes one video frame per line, in the order
listed. The CSV carries the labels for those frames, keyed by the frame number
in the video, so a frame paused at index 7 is row ``frame == 7``.

Coordinates are written in the video's pixel space. That is only the same as
the source labels while every frame shares one size, which is checked rather
than assumed: mixed sizes are refused instead of being silently letterboxed
into coordinates that no longer match the image.

    python scripts/selected_video.py <project_dir> [-o OUTPUT_DIR]

The project directory is the DLC project holding labeled-data/, and the frame
list defaults to selected_images.txt inside it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd


def read_labels(folder: Path) -> pd.DataFrame:
    """Load a session's label table, whichever of the two layouts it uses."""
    matches = (sorted(folder.glob("CollectedData_*.csv"))
               + sorted(folder.glob("annotation.csv")))
    if not matches:
        return pd.DataFrame()
    frame = pd.read_csv(matches[0], header=[0, 1, 2], index_col=[0, 1, 2])
    frame = frame.dropna(how="all")          # drop a repeated index-name row
    frame.index = [row[-1] for row in frame.index]
    return frame


def bodyparts_of(frame: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(name for _, name, _ in frame.columns))


def coordinates(row: pd.Series, parts: List[str]) -> Dict[str, Tuple[float, float]]:
    found = {}
    for part in parts:
        try:
            x = row.xs((part, "x"), level=(1, 2)).iloc[0]
            y = row.xs((part, "y"), level=(1, 2)).iloc[0]
        except KeyError:
            continue
        if pd.notna(x) and pd.notna(y):
            found[part] = (float(x), float(y))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", help="DLC project directory containing labeled-data/")
    parser.add_argument("-l", "--list", default="",
                        help="frame list (default: selected_images.txt in the project)")
    parser.add_argument("-o", "--output", default="",
                        help="where the video and CSV go (default: the project)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="video frame rate (default 1)")
    parser.add_argument("--name", default="selected_frames",
                        help="basename for the .mp4 and .csv (default selected_frames)")
    parser.add_argument("--annotate", action="store_true",
                        help="also draw the keypoints onto the video")
    parser.add_argument("--exclude", default="ref(head-post)",
                        help="comma-separated bodyparts to leave out of the CSV. "
                             "Defaults to the head-post reference: it is a rig "
                             "fiducial rather than anatomy, and scoring a model "
                             "on it measures hardware placement, not pose. Pass "
                             "an empty string to keep everything")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    listing = Path(args.list) if args.list else project / "selected_images.txt"
    output = Path(args.output) if args.output else project
    if not listing.is_file():
        sys.exit(f"frame list not found: {listing}")
    output.mkdir(parents=True, exist_ok=True)

    wanted = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if not wanted:
        sys.exit(f"{listing} lists no frames")

    # One label table per session, read once even though sessions repeat.
    tables: Dict[str, pd.DataFrame] = {}
    rows, sizes = [], set()
    for entry in wanted:
        session, _, name = entry.partition("/")
        path = project / "labeled-data" / session / name
        if not path.is_file():
            sys.exit(f"frame listed but not present: {path}")
        image = cv2.imread(str(path))
        if image is None:
            sys.exit(f"could not read {path}")
        sizes.add((image.shape[1], image.shape[0]))
        if session not in tables:
            tables[session] = read_labels(project / "labeled-data" / session)
        rows.append((entry, session, name, path, image))

    if len(sizes) != 1:
        sys.exit(f"frames differ in size ({sorted(sizes)}); writing them to one "
                 f"video would rescale some, and the label coordinates would no "
                 f"longer match the image")
    width, height = sizes.pop()

    # mp4v is what this OpenCV build ships with; avc1 needs a system codec that
    # is not always present.
    writer = cv2.VideoWriter(str(output / f"{args.name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (width, height))
    if not writer.isOpened():
        sys.exit("could not open the video writer")

    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    every_part: List[str] = []
    for table in tables.values():
        for part in bodyparts_of(table):
            if part not in every_part and part not in excluded:
                every_part.append(part)
    if excluded:
        print(f"excluded: {', '.join(sorted(excluded))}")

    records = []
    for number, (entry, session, name, path, image) in enumerate(rows):
        table = tables[session]
        found = ({} if table.empty or name not in table.index
                 else coordinates(table.loc[name], every_part))
        if args.annotate:
            for x, y in found.values():
                cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)
                cv2.circle(image, (int(round(x)), int(round(y))), 4, (0, 0, 0), 1)
        writer.write(image)

        record = {"frame": number,
                  "time_s": round(number / args.fps, 3),
                  "session": session,
                  "image": name,
                  "path": str(path),
                  "labelled_points": len(found)}
        for part in every_part:
            x, y = found.get(part, ("", ""))
            record[f"{part}_x"] = x
            record[f"{part}_y"] = y
        records.append(record)
    writer.release()

    csv_path = output / f"{args.name}.csv"
    with csv_path.open("w", newline="") as stream:
        fields = ["frame", "time_s", "session", "image", "path", "labelled_points"]
        fields += [f"{p}_{axis}" for p in every_part for axis in ("x", "y")]
        writer_csv = csv.DictWriter(stream, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(records)

    counts = [r["labelled_points"] for r in records]
    print(f"{len(records)} frames at {args.fps:g} fps  ({width}x{height})")
    print(f"video: {output / f'{args.name}.mp4'}")
    print(f"csv:   {csv_path}")
    print(f"bodypart columns: {len(every_part)}")
    print(f"labelled points per frame: min {min(counts)}, max {max(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
