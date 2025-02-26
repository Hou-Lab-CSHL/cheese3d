import os
import toml
import re
import logging
import shutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence
from pathlib import Path

from cheese3d.recording import load_sessions
from cheese3d.utils import maybe
from cheese3d.regex import RECORDING_SPLIT_FULL_REGEX

@dataclass
class AniposeProject:
    name: str
    root_dir: str
    dlc_model: str
    bodyparts: Dict[str, Sequence[str]]
    bodypart_ignore: Sequence[str]
    axes: Sequence[Sequence[str]]
    ref_point: str
    filter_type: Optional[str]
    videos: Dict[str, Dict[str, Sequence[str]]]

    def project_dir(self):
        """Return the path to the Anipose project."""
        return os.sep.join([self.root_dir, f"{self.name}"])

    def config_path(self):
        """Return the path to the Anipose configuration file."""
        return os.sep.join([self.project_dir(), "config.toml"])

    @property
    def anipose_config(self):
        from anipose.anipose import load_config
        return load_config(self.config_path())

    def build_labeling(self):
        labeling = []
        for cluster in self.bodyparts.values():
            if len(cluster) > 2:
                labeling.append([*cluster, cluster[0]])
            else:
                labeling.append(cluster)

        return labeling

    def setup_project(self):
        # create project directory
        os.makedirs(self.project_dir(), exist_ok=True)
        # create and write the configuration
        config = {
            "project": self.name,

            # Working dir is 4 levels below mouse-fe-analysis
            "model_folder": os.path.relpath(self.dlc_model, self.project_dir()),
            "nesting": 1,
            "pipeline": {
                "videos-raw": "videos-raw",
            },
            "labeling": {
                "scheme": self.build_labeling(),
                "ignore": self.bodypart_ignore
            },
            "filter": {
                "enabled": (self.filter_type is not None),
                "type": maybe(self.filter_type, "medfilt"),
                "medfilt": 13, # length of median filter
                "offset_threshold": 5, # offset from median filter to count as jump
                "score_threshold": 0.8, # score below which to count as bad
                "spline": False, # interpolate using linearly instead of cubic spline
            },
            "calibration": {
                "board_type": "charuco",
                "board_size": [7, 7],
                "board_marker_bits": 4,
                "board_marker_dict_number": 50,
                "board_marker_length": 4.5, # mm
                "board_square_side_length": 6 # mm
            },
            "triangulation": {
                "triangulate": True,
                "cam_regex": "_([LRCTB]{1,2})(?=_|$)",
                "manually_verify": False,
                "axes": self.axes,
                "reference_point": self.ref_point,
                "optim": True,
                "score_threshold": 0.9,
                "scale_smooth": 0.0,
            }
        }
        with open(self.config_path(), "w") as f:
            toml.dump(config, f)
        # symlink the videos
        for group, files in self.videos.items():
            video_dir = os.sep.join([self.project_dir(), group, "videos-raw"])
            cal_dir = os.sep.join([self.project_dir(), group, "calibration"])
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(cal_dir, exist_ok=True)
            for file in files["raw"]:
                relfile = os.path.relpath(os.path.realpath(file), start=video_dir)
                dst = os.sep.join([video_dir, os.path.basename(file)])
                if not os.path.exists(dst):
                    if os.path.lexists(dst): # file exists but symlink broken
                        os.remove(dst)
                    os.symlink(relfile, dst)
            for file in files["calibration"]:
                relfile = os.path.relpath(os.path.realpath(file), start=video_dir)
                dst = os.sep.join([cal_dir, os.path.basename(file)])
                if not os.path.exists(dst):
                    if os.path.lexists(dst): # file exists but symlink broken
                        os.remove(dst)
                    os.symlink(relfile, dst)

    def analyze(self):
        from anipose.pose_videos import pose_videos_all
        print('Analyzing videos...')
        pose_videos_all(self.anipose_config)

    def filter(self):
        from anipose.filter_pose import filter_pose_all
        print('Filtering tracked points...')
        if self.anipose_config['filter']['enabled']:
            filter_pose_all(self.anipose_config)

    def filter_3d(self):
        from anipose.filter_3d import filter_pose_3d_all
        print('Filtering 3D points...')
        if self.anipose_config['filter3d']['enabled']:
            filter_pose_3d_all(self.anipose_config)

    def calibrate(self):
        from anipose.calibrate import calibrate_all
        print('Calibrating...')
        calibrate_all(self.anipose_config)

    def triangulate(self):
        from anipose.triangulate import triangulate_all
        print('Triangulating points...')
        triangulate_all(self.anipose_config)

    def reproject_3d(self):
        from anipose.project_2d import project_2d_all
        print("Reprojecting 3D to 2D...")
        project_2d_all(self.anipose_config)

    def label_2d(self):
        from anipose.label_videos import label_videos_all, label_videos_filtered_all
        print('Labeling videos in 2D...')
        if self.anipose_config['filter']['enabled']:
            label_videos_filtered_all(self.anipose_config)
        else:
            label_videos_all(self.anipose_config)

    def label_3d(self):
        from anipose.label_videos_3d import label_videos_3d_all
        print('Labeling videos in 3D...')
        label_videos_3d_all(self.anipose_config)

    def label_combined(self):
        from anipose.label_combined import label_combined_all
        print('Labeling combined videos...')
        label_combined_all(self.anipose_config)

    def compare_viz(self):
        from anipose.label_videos import label_videos_all, label_videos_filtered_all
        print('Labeling videos in 2D...')
        label_videos_all(self.anipose_config)

        if self.anipose_config['filter']['enabled']:
            print('Labeling filtered videos in 2D...')
            label_videos_filtered_all(self.anipose_config)

        from anipose.label_videos_proj import label_proj_all
        print('Projecting 3D points back to 2D...')
        label_proj_all(self.anipose_config)

        from anipose.label_filter_compare import label_filter_compare_all
        print('Labeling videos to compare 2D views')
        label_filter_compare_all(self.anipose_config)

    def run_data(self):
        self.analyze()
        self.filter()
        self.calibrate()
        self.triangulate()
        self.filter_3d()

    def run_viz(self):
        self.label_2d()
        self.compare_viz()
        self.label_3d()
        self.label_combined()

    def run_all(self):
        self.run_data()
        self.run_viz()

    def clean_data(self):
        data_dirs = Path(self.project_dir()).glob('**/pose-?d*')
        for d in data_dirs:
            print(f"Deleting: {d}")
            shutil.rmtree(d)

    def clean_viz(self):
        data_dirs = Path(self.project_dir()).glob('**/videos-compare')
        for d in data_dirs:
            print(f"Deleting: {d}")
            shutil.rmtree(d)
        data_dirs = Path(self.project_dir()).glob('**/videos-2d-proj')
        for d in data_dirs:
            print(f"Deleting: {d}")
            shutil.rmtree(d)
        data_dirs = Path(self.project_dir()).glob('**/videos-labeled*')
        for d in data_dirs:
            print(f"Deleting: {d}")
            shutil.rmtree(d)

    def clean(self):
        self.clean_data()
        self.clean_viz()

@dataclass
class AniposeConfig:
    """
    An Anipose configuration to load a `fepipeline.anipose.AniposeProject`.

    Arguments:
    - `name`: the name of the Anipose project
    - `dlc_model`: the path to the DLC project used for this Anipose project
    - `sessions`: a sequence of recording sessions as dicts of form
        `{"name": [...]}` containing the name of a session
        (optionally, the keys "config", "condition", and "run" can be used to
        specify loading a subset of a session)
    - `default_config`: the default config under `configs/dataset` to use
        for loading sessions
    - `axes`: the Anipose axes config (see Anipose configuration docs)
    - `ref_point`: the Anipose reference point config (see Anipose configuration docs)
    """
    name: str
    dlc_model: str
    sessions: List[Dict[str, str]]
    default_config: str
    axes: List[List[str]]
    ref_point: str
    bodypart_ignore: Optional[List[str]]
    filter_type: Optional[str]

    def load_sessions(self):
        """Load the session configurations."""
        return load_sessions(self.sessions, self.default_config)

    def instantiate(self, root_dir):
        """Instantiate a `fepipeline.anipose.AniposeProject` from the configuration.
        The project is stored under `root_dir`."""
        videos_dict = {}
        # convert session strings into config objects
        session_cfgs = self.load_sessions()
        bodyparts = session_cfgs[0].bodyparts

        for session in session_cfgs:
            if session.bodyparts != bodyparts:
                logging.warn(f"Bodyparts for {session.name} does not match "
                             f"bodyparts for initial session:\n{bodyparts}")

            videos = []
            for video in session.videos.as_list():
                if isinstance(video.path, list):
                    videos.extend(video.path)
                else:
                    videos.append(video.path)
            cals = []
            for cal in session.calibration.as_list():
                if isinstance(cal.path, list):
                    cals.extend(cal.path)
                else:
                    cals.append(cal.path)

            for video in videos:
                # get group type
                pattern = re.compile(RECORDING_SPLIT_FULL_REGEX)
                pattern_match = pattern.search(video)
                try:
                    group = (f"{pattern_match.group(1)}_"
                             f"{pattern_match.group(2)}_"
                             f"{pattern_match.group(3)}_"
                             f"{pattern_match.group(4)}_"
                             f"{pattern_match.group(5)}_"
                             f"{pattern_match.group(7)}")
                except:
                    print(f"'{video}' group failed.")
                    continue

                if group not in videos_dict:
                    videos_dict[group] = {"raw": [], "calibration": cals}
                videos_dict[group]["raw"].append(video)

        # for group, values in videos_dict.items():
        #     print(f"Group: {group}")
        #     for key in ['raw', 'calibration']:
        #         num_items = len(values[key])
        #         print(f"    {key}: {num_items} items")
        return AniposeProject(name=self.name,
                              root_dir=root_dir,
                              dlc_model=self.dlc_model,
                              bodyparts=bodyparts,
                              bodypart_ignore=maybe(self.bodypart_ignore, []),
                              axes=self.axes,
                              ref_point=self.ref_point,
                              filter_type=self.filter_type,
                              videos=videos_dict)
