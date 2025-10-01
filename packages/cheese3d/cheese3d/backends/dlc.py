import os
import shutil
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from omegaconf import OmegaConf

from cheese3d.backends.core import Pose2dBackend
from cheese3d.config import KeypointConfig
from cheese3d.utils import maybe, reglob, BoundingBox, VideoFrames, dlc_folder_to_components

class DLCBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 experimenter: str = "default",
                 date: Optional[str] = None,
                 crops: Optional[List[BoundingBox]] = None,
                 frames_per_video: int = 5):
        super().__init__()
        self.name = name
        self.root_dir = root_dir
        self.experimenter = experimenter
        self.date = maybe(date, datetime.now().strftime("%Y-%m-%d"))
        self.videos = videos
        self.crops = maybe(crops, [[None, None, None, None] for _ in videos])
        self.keypoints = keypoints
        self.frames_per_video = frames_per_video

        # Check if project already exists, if not create it
        if not self.project_path.exists():
            self.create()
            self.overwrite_config()
            self.fix_video_symlinks()
        else:
            self.overwrite_config()
            self.fix_video_symlinks()

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      videos: List[Path],
                      keypoints: List[KeypointConfig],
                      crops: Optional[List[BoundingBox]]):
        # copy project over
        def _ignore(src, names):
            if Path(src).name == "videos":
                return names
            else:
                return []
        shutil.copytree(project_path, root_dir / project_path.name,
                        ignore=_ignore,
                        dirs_exist_ok=True,
                        symlinks=True,
                        ignore_dangling_symlinks=True)
        # create links for video files
        for video in videos:
            abspath = video.resolve()
            relpath = os.path.relpath(abspath, root_dir / project_path.name / "videos")
            os.symlink(relpath, root_dir / project_path.name / "videos" / video.name)
        # create dlc project
        name, experimenter, date = dlc_folder_to_components(project_path)
        return cls(name=name,
                   root_dir=root_dir,
                   videos=videos,
                   keypoints=keypoints,
                   experimenter=experimenter,
                   date=date,
                   crops=crops)

    @property
    def dlc_name(self):
        return f"{self.name}-{self.experimenter}-{self.date}"

    @property
    def project_path(self):
        return self.root_dir / self.dlc_name

    @property
    def config_path(self):
        return self.project_path / "config.yaml"

    def create(self):
        import deeplabcut as dlc
        dlc.create_new_project(
            project=self.name,
            experimenter=self.experimenter,
            working_directory=self.root_dir,
            videos=self.videos,
            copy_videos=False
        )

    def overwrite_config(self):
        # load dlc config file
        dlc_config = OmegaConf.load(self.config_path)
        # overwrite videos
        videos = {}
        for (video, crop) in zip(self.videos, self.crops):
            if crop is None or any(c is None for c in crop):
                height, width = VideoFrames.get_dims(video)
                crop = (maybe(crop[0], 0),
                        maybe(crop[2], int(width)),
                        maybe(crop[1], 0),
                        maybe(crop[3], int(height)))
            else:
                crop = (crop[0], crop[2], crop[1], crop[3])
            video_path = self.project_path / "videos" / video.name
            videos[str(video_path.absolute())] = {
                "crop": ", ".join(map(str, crop))
            }
        dlc_config.video_sets = videos
        # overwrite bodyparts
        dlc_config.bodyparts = [kp.label for kp in self.keypoints]
        # overwrite number of frames to pick
        dlc_config.numframes2pick = self.frames_per_video
        # dump updated dlc config to disk
        OmegaConf.save(dlc_config, self.config_path, resolve=True)

    def fix_video_symlinks(self):
        videos = [Path(p) for p in reglob(r".*", str(self.project_path / "videos"))]
        project_videos = [p.name for p in self.videos]
        for video in videos:
            if video.name in project_videos:
                abspath = video.resolve()
                relpath = os.path.relpath(abspath, self.project_path / "videos")
                os.remove(video)
                os.symlink(relpath, video)

    def import_c3d_labels(self, videos: Dict[str, Path]):
        for name, path in videos.items():
            label_folder = self.project_path / "labeled-data" / name
            if label_folder.exists():
                images = reglob(r".*\.png", str(path))
                for image in images:
                    src_image = Path(image)
                    dst_image = label_folder / src_image.name
                    if dst_image.exists():
                        os.remove(dst_image)
                    relpath = os.path.relpath(src_image, label_folder)
                    os.symlink(relpath, dst_image)
                annotations_yaml = path / "annotations.yaml"
                hdf = label_folder / f"CollectedData_{self.experimenter}.h5"
                csv = label_folder / f"CollectedData_{self.experimenter}.csv"
                if annotations_yaml.exists():
                    with open(annotations_yaml, "r") as f:
                        annotations = yaml.safe_load(f)
                    data_dict = {}
                    for kp, files in annotations.items():
                        for file, coords in files.items():
                            # create index for this row
                            idx = ('labeled-data', str(label_folder.name), file)
                            # get x and y coordinates (could be null/None)
                            x_coord = coords[0][0]
                            y_coord = coords[0][1]
                            # create column keys for x and y
                            x_col = (self.experimenter, kp, 'x')
                            y_col = (self.experimenter, kp, 'y')
                            # store in data dictionary
                            if idx not in data_dict:
                                data_dict[idx] = {}
                            data_dict[idx][x_col] = x_coord
                            data_dict[idx][y_col] = y_coord
                    # convert to dataframe
                    index = pd.MultiIndex.from_tuples(list(data_dict.keys()))
                    columns = pd.MultiIndex.from_tuples(
                        [(self.experimenter, bp, coord)
                         for bp in annotations.keys()
                         for coord in ['x', 'y']],
                        names=["scorer", "bodyparts", "coords"]
                    )
                    # create empty df with the right structure
                    df = pd.DataFrame(index=index, columns=columns)
                    # fill in the values
                    for idx, values in data_dict.items():
                        for col, val in values.items():
                            df.loc[idx, col] = val
                    # convert to float (this will convert None/null to NaN)
                    df = df.astype(float)
                    # write dataframe to disk
                    if hdf.exists():
                        os.remove(hdf)
                    df.to_hdf(hdf, key="df", mode="w")
                    if csv.exists():
                        os.remove(csv)
                    df.to_csv(csv)

    def export_c3d_labels(self, videos: Dict[str, Path]):
        for name, path in videos.items():
            label_folder = self.project_path / "labeled-data" / name
            if label_folder.exists():
                images = reglob(r".*\.png", str(label_folder))
                for image in images:
                    image = Path(image)
                    if image.resolve() != (path / image.name).resolve():
                        shutil.copy2(image, path)
                    relpath = os.path.relpath(path / image.name, label_folder)
                    os.remove(image)
                    os.symlink(relpath, image)
                hdf = label_folder / f"CollectedData_{self.experimenter}.h5"
                if hdf.exists():
                    df = pd.read_hdf(hdf)
                    annotations = {}
                    for kp in self.keypoints:
                        annotations[kp.label] = {}
                        coords_df = df[self.experimenter][kp.label] # type: ignore
                        for i in range(len(coords_df)):
                            pt = coords_df.iloc[i]
                            x, y = pt["x"], pt["y"]
                            x = None if pd.isna(x) else float(x)
                            y = None if pd.isna(y) else float(y)
                            file = pt.name[2]
                            annotations[kp.label][file] = [[x, y]]
                    with open(path / "annotations.yaml", "w") as f:
                        yaml.safe_dump(annotations, f)

    def extract_frames(self, videos: Optional[List[Path]] = None):
        import deeplabcut as dlc
        project_videos = maybe(videos, [p.name for p in self.videos])
        videos_list = [p for p in reglob(r".*", str(self.project_path / "videos"))
                         if Path(p).name in project_videos]
        dlc.extract_frames(config=self.config_path,
                           userfeedback=False,
                           crop=True,
                           videos_list=videos_list)

    def train(self, gpu):
        import deeplabcut as dlc
        training_datasets = reglob("iteration-[0-9]+", path=str(self.project_path / "training-datasets"))
        if len(training_datasets) > 0:
            dlc.merge_datasets(config=self.config_path)
        dlc.create_training_dataset(config=self.config_path,
                                    userfeedback=False,
                                    net_type="resnet_50",
                                    augmenter_type="imgaug")
        dlc.train_network(config=self.config_path,
                          gputouse=gpu)
        dlc.evaluate_network(config=self.config_path,
                             gputouse=gpu,
                             per_keypoint_evaluation=True)
