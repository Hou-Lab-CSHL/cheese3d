import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from omegaconf import OmegaConf

from cheese3d.backends.core import Pose2dBackend
from cheese3d.config import KeypointConfig
from cheese3d.utils import maybe, reglob, BoundingBox, VideoFrames

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
            if src == project_path:
                return ["videos"]
            else:
                return []
        shutil.copytree(project_path, root_dir / project_path.name,
                        ignore=_ignore,
                        dirs_exist_ok=True,
                        symlinks=True)
        # create dlc project
        name, experimenter, date = project_path.name.split("-", 2)
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
            videos[str(video.absolute())] = {
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

    def export_c3d_labels(self, videos: Dict[str, Path]):
        for name, path in videos.items():
            label_folder = self.project_path / "labeled-data" / name
            if label_folder.exists():
                images = reglob(r".*\.png", str(label_folder))
                for image in images:
                    image = Path(image)
                    shutil.copy2(image, path)
                    relpath = os.path.relpath(path / image.name, label_folder)
                    os.remove(image)
                    os.symlink(relpath, image)
                # hdf = label_folder / f"CollectedData_{self.experimenter}.h5"
                # if hdf.exists():

    def extract_frames(self):
        import deeplabcut as dlc
        project_videos = [p.name for p in self.videos]
        videos = [p for p in reglob(r".*", str(self.project_path / "videos"))
                    if Path(p).name in project_videos]
        dlc.extract_frames(config=self.config_path,
                           userfeedback=False,
                           crop=True,
                           videos_list=videos)

    def train(self, gpu):
        import deeplabcut as dlc
        dlc.train_network(config=self.config_path,
                          gputouse=gpu)
        dlc.evaluate_network(config=self.config_path,
                             gputouse=gpu,
                             per_keypoint_evaluation=True)
