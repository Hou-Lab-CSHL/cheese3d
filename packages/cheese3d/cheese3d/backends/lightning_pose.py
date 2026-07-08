import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from cheese3d.backends.core import Pose2dBackend, register_pose_backend
from cheese3d.config import KeypointConfig
from cheese3d.utils import BoundingBox

def read_lp_preds(csv_path: str | Path, scorer: Optional[str] = None) -> pd.DataFrame:
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

def dlc_df_to_h5(df: pd.DataFrame, h5_path: str | Path, scorer: Optional[str] = None) -> Path:
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in df.columns if col[2] in {"x", "y", "likelihood"}]
    df = df.loc[:, keep_cols].copy()
    if scorer is not None:
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coord) for _, bodypart, coord in df.columns],
            names=["scorer", "bodyparts", "coords"]
        )
    if not h5_path.exists():
        df.to_hdf(h5_path, key="df_with_missing", format="table", mode="w")

    return h5_path

def lp_csv_to_dlc_h5(csv_path: str | Path, h5_path: str | Path, scorer: Optional[str] = None) -> Path:
    return dlc_df_to_h5(read_lp_preds(csv_path, scorer=scorer), h5_path, scorer=scorer)

class LightningPoseBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 crops: Optional[List[BoundingBox]] = None,
                 scorer: Optional[str] = None,
                 **cfg_options):
        from lightning_pose.api import Model

        super().__init__()
        self.name = name
        self.root_dir = Path(root_dir).absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.scorer = scorer
        self._update_config(**cfg_options)
        self.model = Model.from_dir(self.project_path)

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      *args,
                      **kwargs):
        raise NotImplementedError("Importing existing Lightning Pose models is not implemented.")

    def _update_config(self, **cfg_options):
        from lightning_pose.api import ModelConfig
        from omegaconf import OmegaConf

        cfg = ModelConfig.from_yaml_file(self.project_path / "config.yaml")
        cfg.cfg.merge_with(cfg_options)
        OmegaConf.save(cfg.cfg, self.project_path / "config.yaml")

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

        output_dir.mkdir(parents=True, exist_ok=True)
        # outputs are <session> / pose-2d by Ch3DProject orchestration
        print(output_dir.parent.name)
        # resolve paths to absolute
        videos = {view: Path(video).resolve() for view, video in videos.items()}
        # list of videos that we need to generate
        # exit early if there are none
        missing_videos = [video for video in videos.values()
                          if not (output_dir / f"{video.stem}.h5").exists()]
        if len(missing_videos) == 0:
            return True
        # list of videos that lp model has already created internal preds for
        existing = [self.model.video_preds_dir() / f"{video.stem}.csv"
                    for video in missing_videos]
        if not all(path.exists() for path in existing):
            # NOTE: currently the config is saying multiview
            #       but lenny says this is a SVT
            # if self.model.config.is_multi_view():
            #     view_names = list(self.model.config.cfg.data.view_names)
            #     if set(view_names) != set(videos.keys()):
            #         raise RuntimeError("Session view names do not match view names for trained LP model!"
            #                            f" Expected {set(view_names)}, got {set(videos.keys())}")
            #     # get input/output file path ordering matching view order
            #     ordered_videos = [videos[view_name] for view_name in view_names]
            #     output_files = [self.model.video_preds_dir() / f"{video.stem}.csv"
            #                     for video in ordered_videos]
            #     existing_outputs = [path for path in output_files if path.exists()]
            #     if len(existing_outputs) > 0:
            #         raise RuntimeError("Lightning Pose multiview tracking would overwrite existing "
            #                            f"prediction files: {existing_outputs}")
            #     # automated collection utility does not match view names for C3D format correctly
            #     # do this manually (otherwise could use model.predict_on_video_file_multiview)
            #     self.model._load()
            #     predict_video(video_file=[str(video) for video in ordered_videos],
            #                   model=self.model,
            #                   output_pred_file=[str(path) for path in output_files])
            # else:
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
            lp_csv_to_dlc_h5(csv_path, h5_path, scorer=self.scorer)
            print(f"Wrote {h5_path}")

        return True

register_pose_backend("lightning_pose", LightningPoseBackend)
