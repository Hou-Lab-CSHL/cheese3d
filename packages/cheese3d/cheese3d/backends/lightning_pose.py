from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich import print as rprint

from cheese3d.backends.core import Pose2dBackend, register_pose_backend
from cheese3d.config import KeypointConfig
from cheese3d.utils import BoundingBox

def read_lightning_pose_predictions(csv_path: str | Path,
                                    scorer: Optional[str] = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    keep_cols = []
    for col in df.columns:
        top, bodypart, coord = col
        if str(top).lower() == "set":
            continue
        if pd.isna(bodypart) or pd.isna(coord):
            continue
        if str(coord) not in {"x", "y", "likelihood"}:
            continue
        keep_cols.append(col)
    df = df.loc[:, keep_cols]
    if scorer is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coord) for _, bodypart, coord in df.columns],
            names=["scorer", "bodyparts", "coords"]
        )
    else:
        df.columns = pd.MultiIndex.from_tuples(list(df.columns),
                                               names=["scorer", "bodyparts", "coords"])
    return df.infer_objects()

def dlc_df_to_h5(df: pd.DataFrame,
                 h5_path: str | Path,
                 scorer: Optional[str] = None) -> Path:
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in df.columns if col[2] in {"x", "y", "likelihood"}]
    df = df.loc[:, keep_cols].copy()
    if scorer is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coord) for _, bodypart, coord in df.columns],
            names=["scorer", "bodyparts", "coords"]
        )
    if h5_path.exists():
        return h5_path
    df.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")
    return h5_path

def lightning_pose_csv_to_dlc_h5(csv_path: str | Path,
                                 h5_path: str | Path,
                                 scorer: Optional[str] = None) -> Path:
    df = read_lightning_pose_predictions(csv_path, scorer=scorer)
    return dlc_df_to_h5(df, h5_path, scorer=scorer)

class LightningPoseBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 crops: Optional[List[BoundingBox]] = None,
                 scorer: Optional[str] = None):
        from lightning_pose.api import Model

        super().__init__()
        self.name = name
        self.root_dir = Path(root_dir).absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.scorer = scorer
        self.model = Model.from_dir(self.project_path)

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      *args,
                      **kwargs):
        raise NotImplementedError("Importing existing Lightning Pose models is not implemented.")

    @property
    def project_path(self):
        return self.root_dir

    def import_c3d_labels(self, videos: Dict[str, Path]):
        raise NotImplementedError("Use Lightning Pose tooling to manage labeled frames.")

    def export_c3d_labels(self, videos: Dict[str, Path]):
        raise NotImplementedError("Use Lightning Pose tooling to manage labeled frames.")

    def extract_frames(self, videos: Optional[List[Path]] = None):
        raise NotImplementedError("Use Lightning Pose tooling to extract frames.")

    def train(self, gpu, iterate_dataset: bool = True):
        raise NotImplementedError("Use Lightning Pose tooling to train models.")

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None) -> bool:
        from lightning_pose.utils.predictions import predict_video

        resolved_videos = {view: Path(video).resolve() for view, video in videos.items()}
        video_list = list(resolved_videos.values())
        output_dir.mkdir(parents=True, exist_ok=True)
        rprint(f"[bold green]Tracking 2D pose with Lightning Pose:[/bold green] "
               f"{output_dir.parent.name}")
        missing_videos = [video for video in video_list
                          if not (output_dir / f"{video.stem}.h5").exists()]
        if len(missing_videos) == 0:
            return True

        existing = [self.model.video_preds_dir() / f"{video.stem}.csv"
                    for video in missing_videos]
        if not all(path.exists() for path in existing):
            if self.model.config.is_multi_view():
                view_names = list(self.model.config.cfg.data.view_names)
                videos_by_view = {}
                for view_name in view_names:
                    matches = [video for video in missing_videos if f"_{view_name}_" in video.stem]
                    if len(matches) != 1:
                        raise ValueError(f"Expected one video for view {view_name}, found {matches}")
                    videos_by_view[view_name] = matches[0]
                ordered_videos = [videos_by_view[view_name] for view_name in view_names]
                output_files = [self.model.video_preds_dir() / f"{video.stem}.csv"
                                for video in ordered_videos]
                existing_outputs = [path for path in output_files if path.exists()]
                if len(existing_outputs) > 0:
                    raise RuntimeError("Lightning Pose multiview tracking would overwrite existing "
                                       f"prediction files: {existing_outputs}")
                self.model._load()
                predict_video(video_file=[str(video) for video in ordered_videos],
                              model=self.model,
                              output_pred_file=[str(path) for path in output_files])
            else:
                for video in missing_videos:
                    prediction_csv = self.model.video_preds_dir() / f"{video.stem}.csv"
                    if prediction_csv.exists():
                        continue
                    self.model.predict_on_video_file(video,
                                                     compute_metrics=False,
                                                     generate_labeled_video=False)
        for video in missing_videos:
            csv_path = self.model.video_preds_dir() / f"{video.stem}.csv"
            h5_path = output_dir / f"{video.stem}.h5"
            lightning_pose_csv_to_dlc_h5(csv_path, h5_path, scorer=self.scorer)
            rprint(f"Wrote {h5_path}")

        return True

register_pose_backend("lightning_pose", LightningPoseBackend)
