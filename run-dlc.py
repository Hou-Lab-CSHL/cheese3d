import logging
import hydra
import napari

# We need to import deeplabcut before napari.settings
# to prevent an error that will result in the message:
# "DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)"
import deeplabcut
from napari.settings import get_settings

from omegaconf import OmegaConf, DictConfig

from cheese3d.dlc import video_to_landmark_path
from cheese3d.configutils import setup_hydra
from cheese3d.recording import load_sessions
from cheese3d.utils import flatten

def session_to_str(session):
    return "_".join([session["name"], *(session[k] for k in ("mouse", "condition", "run") if k in session)])

def session_to_video(sessions, target, default_cfg):
    target = session_to_str(target)
    matches = [session for session in sessions if session_to_str(session) == target]
    if len(matches) > 1:
        logging.warn(f"Multiple matching sessions found for refinement session {target} so all will be used. ({matches=})")
    elif len(matches) == 0:
        logging.warn(f"No matching sessions found for refinement session {target} so will skip. ({matches=})")
    videos = []
    for session in load_sessions(matches, default_cfg):
        for v in session.videos.as_list():
            if isinstance(v.path, str):
                videos.append(v.path)
            else:
                videos.extend(v.path)

    return videos

@hydra.main(config_path="./configs", config_name="dlc-main", version_base=None)
def main(cfg: DictConfig):
    # get the DLC config
    dlc_cfg = OmegaConf.to_object(cfg.dlc)
    # create the DLC project
    dlc_project = dlc_cfg.instantiate(root_dir=cfg.paths.dlc.projects)

    # set up the DLC project if necessary
    if cfg.setupproject:
        dlc_project.setup_project()

    # fix symlinks if necessary
    if cfg.replace_symlinks:
        dlc_project.fix_symlinks()

    # repopulate the config
    if cfg.write_dlc_config:
        dlc_project.populate_config()

    # extract frames
    if cfg.extractframes:
        extract_videos = flatten([session_to_video(cfg.dlc.sessions, session, cfg.dlc.default_config)
                                  for session in cfg.refinement_sessions])
        dlc_project.extract_frames(extract_videos, cfg.disable_active_learning)

    # Merge previously created labels from multiple sessions
    if cfg.mergelabels:
        dlc_cfg.merge_labels(dlc_project, root_dir=cfg.paths.dlc.projects)

    # label frames
    if cfg.labelframes:
        dlc_project.label_frames()
        napari.run()

    # visualize labels
    if cfg.checklabels:
        dlc_project.check_labels()

    # build training data
    if cfg.builddataset:
        dlc_project.build_dataset()

    # merge datasets (increment iteration after refining labels)
    if cfg.mergedataset:
        dlc_project.merge_dataset()

    # Resume training from latest checkpoint
    if cfg.initweights:
        dlc_project.init_weights()

    # train the network
    if cfg.trainnetwork:
        dlc_project.train(gpu=cfg.gpu, evaluate=cfg.evaluatenetwork)

    # analyze new data if requested
    if cfg.analyzedataset and (cfg.get("dataset", None) is not None):
        session_cfg = OmegaConf.to_object(cfg.dataset)
        videos = [v.path for v in session_cfg.videos.as_list()]
        landmarks = [video_to_landmark_path(video,
                                            dlc_project.model,
                                            dlc_project.augmentation)
                     for video in videos]
        dlc_project.analyze(videos, landmarks,
                            labeledvideos=cfg.labeledvideos,
                            filteroutput=cfg.filteroutput)
    elif cfg.analyzedataset:
        logging.warn("analyzedataset=true in the config, "
                     "but dataset is not set in config. "
                     "Specify a dataset using +dataset=...")

if __name__ == "__main__":
    # this will prevent napari from launching the label viewer
    # until we want to launch it
    # this will allow the script to pause, get labels, then continue
    get_settings().application.ipy_interactive = False
    setup_hydra()
    main()
