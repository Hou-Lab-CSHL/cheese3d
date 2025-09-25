import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

def keypoints_by_group(keypoints):
    kp_by_group = {}
    for kp in keypoints:
        for group in kp["groups"]:
            if group in kp_by_group:
                kp_by_group[group].append(kp["label"])
            else:
                kp_by_group[group] = [kp["label"]]

    return kp_by_group

def load_keypoints_and_skeleton(config_path):
    """Load bodyparts and skeleton edges from a YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    keypoints = config.get("keypoints", [])
    kp_skeletons = keypoints_by_group(keypoints)
    skeleton_edges = []
    for loop in kp_skeletons.values():
        skeleton_edges.extend(list(zip(loop, [*loop[1:], loop[0]])))
    keypoints = [kp["label"] for kp in keypoints]

    return keypoints, []


def create_empty_annotations(image_paths: List[str | Path],
                             yaml_path: str | Path,
                             keypoints: List[str]):
    """Create an empty annotations file with all (x, y) fields set to NaN.

    Parameters
    ----------
    image_paths : list of str or Path
        List of image paths.
    yaml_path : str
        Path to write the empty YAML file.
    keypoints : list of str
        List of keypoints from config.
    """
    annotations = {kp: {os.path.basename(img): [[None, None]]
                        for img in image_paths}
                   for kp in keypoints}
    with open(yaml_path, "w") as f:
        yaml.safe_dump(annotations, f)

def read_annotations(yaml_path: str | Path):
    with open(yaml_path, "r") as f:
        annotations = yaml.safe_load(f)
    rows = []
    for kp, imgs in annotations.items():
        for img, pt in imgs.items():
            x, y = (np.nan if p is None else p for p in pt[0])
            rows.append({"filename": img, "keypoint": kp, "x": x, "y": y})
    df = pd.DataFrame(rows)

    return df

def write_annotations(df: pd.DataFrame, yaml_path: str | Path):
    annotations = {}
    for _, row in df.iterrows():
        kp = str(row["keypoint"])
        filename = str(row["filename"])
        x = None if row["x"] is None else row["x"]
        y = None if row["y"] is None else row["y"]
        if kp in annotations:
            if filename in annotations[kp]:
                logging.warning(f"Encountered multiple rows with {kp=} and {filename=}, skipping...")
                continue
            else:
                annotations[kp][filename] = [[x, y]]
        else:
            annotations[kp] = {filename: [[x, y]]}

def find_keypoint_conflicts(df: pd.DataFrame, config_keypoints: List[str]) -> List[str]:
    """Find body parts that exist in df but not in config."""
    annotated_parts = df["keypoint"].unique()

    return sorted(set(annotated_parts).symmetric_difference(config_keypoints))


def ensure_images_in_yaml(image_files: List[str],
                          yaml_path: str | Path,
                          keypoints: List[str]) -> None:
    """
    Guarantee that every image in `image_files` exists in the annotations file.
    Any missing (filename, bodypart) rows are appended and filled with NaN.
    The annotations are rewritten in-place.
    """
    # load existing annotations
    with open(yaml_path, "r") as f:
        annotations = yaml.safe_load(f)
    # build in new entries as needed
    n_files_added = 0
    for kp in keypoints:
        if kp in annotations:
            for img in image_files:
                if img not in annotations[kp]:
                    annotations[kp][img] = [[None, None]]
                    n_files_added += 1
    # overwrite existing file
    with open(yaml_path, "w") as f:
        yaml.safe_dump(annotations, f)
    print(f"▶︎ Added {n_files_added} new image(s) to {os.path.basename(yaml_path)}")
