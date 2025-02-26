import os
import re
import yaml
import subprocess
import logging
import hydra
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
from omegaconf import OmegaConf
from collections import defaultdict
from datetime import datetime
from glob import glob
from tqdm.auto import tqdm
from pathlib import Path

from cheese3d.recording import load_sessions
from cheese3d.utils import unzip

def flatten_bodyparts(bodyparts):
    """Flatten a nested dictionary of bodyparts into a flat dictionary."""
    flat_bodyparts = []
    for bodypart in bodyparts.values():
        if isinstance(bodypart, dict):
            bodypart = flatten_bodyparts(bodypart)
        flat_bodyparts = flat_bodyparts + bodypart

    return list(dict.fromkeys(flat_bodyparts))

def training_set_iteration(project_dir):
    training_sets = glob(os.sep.join([project_dir, "/dlc-models/iteration-*"]))
    if not training_sets:
        return -1
    return sorted(training_sets)[-1][-1]


def select_snapshot(project_dir, snapshot = "latest"):
    """
    Select the DLC training snapshot for a given project.

    Arguments:
    - `project_dir`: the path to the DLC project directory
    - `snapshot`: the snapshot (training checkpoint) number to select
        (defaults to "latest" to select the last saved snapshot)
    """
    snapshots = glob(os.sep.join([project_dir,
                                  "/dlc-models/iteration-*/*/train/snapshot-*.index"]))
    if not snapshots:
        return None
    paths, names = unzip([os.path.split(s) for s in snapshots])
    snapshot_ids = [int(os.path.splitext(f)[0].split("-")[1]) for f in names]
    if snapshot == "latest":
        selected_snapshot = max(snapshot_ids)
    else:
        snapshot = int(snapshot)
        if snapshot in snapshot_ids:
            selected_snapshot = snapshot
        else:
            RuntimeError(f"Could not find snapshot #{snapshot} in "
                         f"project at {project_dir}")

    i = snapshot_ids.index(selected_snapshot)

    return selected_snapshot, os.sep.join([paths[i], names[i]])

def video_to_landmark_path(video_path, network, augmentation):
    return f"{os.path.splitext(video_path)[0]}_landmarks_{network}_{augmentation}.h5"



@dataclass
class DLCProject:
    """
    A DLC project for facial expression analysis.

    Arguments:
    - `name`: the name of the DLC project
    - `date`: the DLC project date
    ` `experimenter`: the DLC project experimenter
    - `root_dir`: the absolute parent directory that should hold the DLC project directory
    - `bodyparts`: a nested dictionary of bodyparts for labeling
    - `videos`: a list of absolute paths to the videos for training
    - `crops`: a list of crops applied to each video in `videos`
    - `augmentation`: the augmentation style applied to the training data
        (defaults to "imgaug"; see https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#f-create-training-dataset-s)
    - `model`: the pre-trained model to use
        (defaults to "resnet_50"; see https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html#g-train-the-network)
    - `snapshot`: the snapshot to use for analyzing videos _after training_
    """
    name: str
    date: str
    experimenter: str
    root_dir: str
    bodyparts: Dict[str, Sequence[str]]
    # Dictionary of camera view to list of bodyparts we expect in that view
    cam_bodyparts: Dict[str, Sequence[str]]
    videos: Sequence[str]
    crops: Sequence[str]
    augmentation: str = "imgaug"
    model: str = "resnet_50"
    snapshot: str = "latest"
    cam_regex: str = "_([LRCTB]{1,2})[_,\.]"
    # Landmark confidence threshold to use when extracting outlier frames
    confidence_threshold: float = 0.5
    # If True, will save annotated images when extracting frames
    savelabeled: bool = False
    # Learning rate for refinement training
    refine_lr: float = 0.001
    # Number of steps to continue training when refining
    refine_steps: int = 100000
    # Number of frames to label for a new model
    frames2pick_new: int = 20
    # Number of frames to label when refining labels
    frames2pick_refine: int = 5

    def project_dir(self):
        """Return the path to the DLC project."""
        return os.sep.join([self.root_dir,
                            f"{self.name}-{self.experimenter}-{self.date}"])

    def config_path(self):
        """Return the path to the DLC configuration file."""
        return os.sep.join([self.project_dir(), "config.yaml"])

    def video_to_view(self, video_path):
        match = re.search(self.cam_regex, video_path)
        if match:
            return match[1]
        return None

    def setup_project(self):
        """Setup the DLC project directory if it does not yet exist
        and populate the configuration file with the default settings."""
        import deeplabcut as dlc

        if not os.path.exists(self.config_path()):
            dlc.create_new_project(self.name,
                                   self.experimenter,
                                   self.videos,
                                   working_directory=self.root_dir,
                                   copy_videos=False,
                                   multianimal=False)
            self.populate_config()
            self.fix_symlinks()

    def fix_symlinks(self):
        video_dir = os.sep.join([self.project_dir(), "videos"])
        for video in self.videos:
            video_name = os.path.basename(video)
            relpath = os.path.relpath(os.path.realpath(video), start=video_dir)
            dlc_video = os.sep.join([video_dir, video_name])
            if os.path.exists(dlc_video):
                os.remove(dlc_video)
            os.symlink(relpath, dlc_video)

    def delete_empty_labeled_data(self):
        """
        When extract frames does not find any outliers, it still
        creates an empty directory. This breaks the merge dataset step,
        which checks that every directory contains `CollectedData` labels

        We can fix it by simply deleting any empty directories in labeled-data
        """
        labeled_data = Path(self.project_dir()) / 'labeled-data'
        for folder in labeled_data.iterdir():
            if not folder.is_dir():
                continue
            children = [p for p in folder.iterdir()]
            if not children:
                # Safe operation, will only delete empty directories
                folder.rmdir()

    def set_numframes2pick(self, numframes2pick):
        """
        DLC uses the config value `numframes2pick` for two purposes:
        1 - Choosing how many frames to label when creating a new model
        2 - Choosing how many frames to label when refining labels with an existing model

        We want to be able to use a lower number of frames when refining.
        This function will overwrite the config file with the updated value
        """
        from deeplabcut.pose_estimation_tensorflow.config import load_config

        cfg = load_config(self.config_path())
        cfg['numframes2pick'] = numframes2pick
        with open(self.config_path(), 'w') as file:
            yaml.dump(cfg, file)

    def populate_config(self):
        """Populate the DLC configuration file with the default settings.
        This overrides the default bodyparts and crops for each video."""
        # load DLC config.yaml
        dlc_config = OmegaConf.load(self.config_path())

        # populate the DLC with crops from dataset config
        dlc_videos = dlc_config.video_sets

        # Default to arbitrary crop for videos not currently in DLC config
        default_crop = next(iter(dlc_videos.values()))
        dlc_videos_by_filename = defaultdict(lambda: default_crop,
                                             {Path(key).name: value for key, value in dlc_videos.items()})
        new_dlc_videos = {}
        for (video, crop) in zip(self.videos, self.crops):
            prev_crop = dlc_videos_by_filename[Path(video).name].crop.split(", ")
            new_crop = (oldc if newc is None else str(newc)
                        for oldc, newc in zip(prev_crop, crop))
            new_dlc_videos[video] = {"crop": ", ".join(new_crop)}

        dlc_config.video_sets = new_dlc_videos
        # override bodyparts
        dlc_config.bodyparts = flatten_bodyparts(self.bodyparts)
        # write the new DLC config to disk
        OmegaConf.save(config=dlc_config, f=self.config_path())
        # get user to confirm config file
        print(f"DLC config file ({self.config_path()}) has be pre-populated "
              "with default values.\nPlease open this file and verify it.")

    def extract_frames(self, videos, disable_active_learning = False):
        """Extract the frames for labeling."""
        import deeplabcut as dlc

        # If a trained model exists, use it to extract outlier frames
        snapshot = select_snapshot(self.project_dir())
        if not disable_active_learning and snapshot:
            print("Will use previous DLC model to extract outlier frames for new videos")
            print(f"Snapshot: {snapshot}")

            self.set_numframes2pick(self.frames2pick_refine)
            videos_by_view = [(self.video_to_view(video), video) for video in videos if video]
            for view, video in tqdm(videos_by_view):
                if view is None:
                    print(f"Could not extract view for {video=}")
                    continue
                bp = self.cam_bodyparts[view]
                dlc.analyze_videos(self.config_path(), video)
                dlc.extract_outlier_frames(
                    self.config_path(),
                    video,
                    automatic=True,
                    outlieralgorithm='uncertain',
                    comparisonbodyparts=bp,
                    p_bound=self.confidence_threshold,
                    savelabeled=self.savelabeled,
                )
            self.delete_empty_labeled_data()

        else:
            print("No DLC model, running naive extract frames")
            self.set_numframes2pick(self.frames2pick_new)
            dlc.extract_frames(self.config_path(),
                               mode='automatic',
                               algo='kmeans',
                               userfeedback=False,
                               crop=True)

    def label_frames(self):
        """Open the DLC labeling tool to label frames."""
        import deeplabcut as dlc

        label_dir = os.sep.join([self.project_dir(), "labeled-data"])
        views = [os.sep.join([label_dir, v])
                 for v in os.listdir(label_dir) if not v.startswith(".")]
        if len(views) == 0:
            dlc.label_frames()
        elif len(glob(f"{views[0]}/CollectedData_*.h5")) == 0:
            dlc.label_frames([views[0], self.config_path()])
        else:
            dlc.label_frames(views[0])

    def check_labels(self):
        """Run the built-in DLC label checking utility."""
        import deeplabcut as dlc
        dlc.check_labels(self.config_path())

    def build_dataset(self):
        """Build the training dataset for the project."""
        import deeplabcut as dlc

        result = dlc.create_training_dataset(self.config_path(),
                                             userfeedback=False,
                                             net_type=self.model,
                                             augmenter_type=self.augmentation)
        if result is None:
            raise RuntimeError("Could not create training dataset!")

    def merge_dataset(self):
        """ Merge dataset (after refining labels) """
        import deeplabcut as dlc

        prev_iteration = training_set_iteration(self.project_dir())
        dlc.merge_datasets(self.config_path())
        self.build_dataset()
        new_iteration = training_set_iteration(self.project_dir())
        if prev_iteration == new_iteration:
            exit("Failed to update training set. Check that all folders in labeled-data contain the CollectedData files (from labeling)")

    def init_weights(self):
        """ Initialize weights to previous checkpoint """
        from deeplabcut.pose_estimation_tensorflow.config import load_config

        steps, snapshot = select_snapshot(self.project_dir())
        path = Path(snapshot)
        while 'models' not in path.name and path.name != '':
            path = path.parent
        latest_iter = sorted(path.iterdir(), reverse=True)[0]
        cfg_path = next(latest_iter.glob('*/train/pose_cfg.yaml'))
        cfg = load_config(cfg_path)
        cfg['init_weights'] = Path(snapshot).with_suffix('').as_posix()
        # Max iterations is set in multi_step and is the number of additional iterations (not the total)
        cfg['multi_step'] = [[self.refine_lr, self.refine_steps]]
        cfg['save_iters'] = min(self.refine_steps, 50000)
        with open(cfg_path, 'w') as file:
            yaml.dump(cfg, file)

    def train(self, gpu = 0, evaluate = True):
        """
        Train the DLC model.

        Arguments:
        - `gpu`: the GPU number to use (defaults to 0)
        - `evaluate`: run evaluation after training (defaults to `True`)
        """
        import deeplabcut as dlc

        dlc.train_network(self.config_path(), gputouse=gpu)
        if evaluate:
            dlc.evaluate_network(self.config_path(), gputouse=gpu)

    def analyze(self, videos, landmarks,
                labeledvideos = False, filteroutput = False):
        """
        Analyze a new set of videos using the trained DLC model.

        Arguments:
        - `videos`: a list of paths to video files for analysis
        - `landmarks`: a list of paths for each video in `videos` declaring
            where the extracted landmark data should be stored
        - `labeledvideos`: use DLC to create videos annotated with the extracted
            landmarks (defaults to `False`)
        - `filteroutput`: use DLC to filter the extract landmarks for saving them
            (defaults to `False; _not recommended_)
        """
        import deeplabcut as dlc

        # check that project exists
        config_path = self.config_path()
        if not os.path.exists(config_path):
            raise RuntimeError(f"Cannot find DLC config {config_path}! "
                               "Did you run setup_project() first?")
        # analyze videos
        dlc.analyze_videos(config_path, videos)
        if filteroutput:
            dlc.filterpredictions(config_path, videos)
        if labeledvideos:
            dlc.create_labeled_video(config_path, videos)
        # rename video output
        cfg_date = datetime.strptime(self.date, "%Y-%m-%d").strftime("%b%-d")
        snapshot, _ = select_snapshot(self.project_dir(), self.snapshot)
        for video, landmark in zip(videos, landmarks):
            name = os.path.splitext(video)[0]
            if self.model == "resnet_50":
                model = "resnet50"
            else:
                model = self.model
            srcname = f"{name}DLC_{model}_{self.name}{cfg_date}shuffle1_{snapshot}"
            dstname = os.path.splitext(landmark)[0]
            os.rename(f"{srcname}_meta.pickle", f"{dstname}_meta.pickle")
            if filteroutput:
                os.rename(f"{srcname}.h5", f"{dstname}_unfiltered.h5")
                os.rename(f"{srcname}_filtered.h5", landmark)
                os.remove(f"{srcname}_filtered.csv")
            else:
                os.rename(f"{srcname}.h5", landmark)
            if labeledvideos:
                os.rename(f"{srcname}_labeled.mp4", f"{dstname}_labeled.mp4")

def load_dlc_config(session, default_config):
    """
    Load the DLC config for `session`.

    Arguments:
    - `session`: a dictionary of the form {"name": [...]} containing the
        name of a session (optionally, the keys "config", "condition", and "run"
        can be used to override the defaults for these values)
    """
    session_str = '{' + ','.join(f"{k}: {v}" for (k,v) in session.items()) + '}'
    overrides = [f"dlc.name={session['name']}",
                 f'dlc.sessions=[{session_str}]',
                 f'dlc.default_config={default_config}']
    common_cfg = hydra.compose(config_name="common",
                               overrides=["+dataset={}", "+dlc={}"],
                               return_hydra_config=True)
    dlc_cfg = hydra.compose(config_name="dlc/common", overrides=overrides)
    dlc_cfg = OmegaConf.merge(common_cfg, dlc_cfg)

    return OmegaConf.to_object(dlc_cfg.dlc)

@dataclass
class DLCConfig:
    """
    A DLC configuration to load a `fepipeline.dlc.DLCProject`.

    Arguments:
    - `name`: the name of the DLC project
    - `date`: the DLC project date
    - `experimenter`: the DLC project experimenter
    - `sessions`: a sequence of recording sessions as dicts of form
        `{"name": [...]}` containing the name of a session
        (optionally, the keys "config", "condition", and "run" can be used to
        specify loading a subset of a session)
    - `default_config`: the default config under `configs/dataset` to use
        for loading sessions
    - `augmentation`: the DLC project dataset augmentation to use
    - `network`: the DLC project pre-trained network to use
    - `snapshot`: the DLC project snapshot to use.
    """
    name: str
    date: str
    experimenter: str
    sessions: List[Dict[str, Any]]
    default_config: str
    augmentation: str = "imgaug"
    network: str = "resnet_50"
    snapshot: str = "latest"

    def load_sessions(self):
        """Load the session configurations."""
        return load_sessions(self.sessions, self.default_config)

    def merge_labels(self, dst_project: DLCProject, root_dir):
        """Merge the labels for each session's single DLC project into
        this DLC project's path."""

        dst_cfg_path = dst_project.config_path()
        labels_to_sync = []
        for session in self.sessions:
            single_dlc_cfg = load_dlc_config(session, self.default_config)
            single_dlc_project = single_dlc_cfg.instantiate(root_dir)
            src_cfg_path = single_dlc_project.config_path()
            if src_cfg_path != dst_cfg_path:
                src_labels = os.sep.join([single_dlc_project.project_dir(),
                                          "labeled-data"])
                labels_to_sync.append(src_labels)

        if len(labels_to_sync) > 0:
            ret = subprocess.call(["rsync", "-r", "-h", "--progress", "--ignore-existing"] +
                                  labels_to_sync + [dst_project.project_dir()])
            if ret:
                raise RuntimeError(f"Failed to copy files (return code = {ret})")

    def instantiate(self, root_dir):
        """Instantiate a `fepipeline.dlc.DLCProject` from the configuration.
        The project is stored under `root_dir`."""
        videos = []
        crops = []
        # convert session strings into config objects
        session_cfgs = self.load_sessions()
        bodyparts = session_cfgs[0].bodyparts
        for session in session_cfgs:
            for video in session.videos.as_list():
                if isinstance(video.path, list):
                    videos.extend(video.path)
                    crops.extend([video.crop] * len(video.path))
                else:
                    videos.append(video.path)
                    crops.append(video.crop)

            if session.bodyparts != bodyparts:
                logging.warn(f"Bodyparts for {session.name} does not match "
                             f"bodyparts for initial session:\n{bodyparts}")

        return DLCProject(name=self.name,
                          date=self.date,
                          experimenter=self.experimenter,
                          root_dir=root_dir,
                          bodyparts=bodyparts,
                          videos=videos,
                          crops=crops,
                          augmentation=self.augmentation,
                          model=self.network,
                          snapshot=self.snapshot,
                          cam_bodyparts=session_cfgs[0].cam_bodyparts,
                          frames2pick_new=2)
