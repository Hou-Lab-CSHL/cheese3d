import os
import re
import subprocess
import logging
from datetime import datetime
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from cheese3d.utils import reglob, maybe
from cheese3d.regex import (RECORDING_MOUSE_REGEX,
                            RECORDING_COND_REGEX,
                            RECORDING_RUN_REGEX,
                            RECORDING_DATE_REGEX,
                            RECORDING_EXP_REGEX,
                            RECORDING_END_REGEX)
from cheese3d.recording import SessionConfig
from cheese3d.dlc import DLCConfig
from cheese3d.anipose import AniposeConfig

def default_dlc_date(proj_path, name, experimenter, current_date):
    projects = reglob(f"{os.sep}{name}-{experimenter}-*", proj_path)
    if len(projects) == 0:
        return current_date
    else:
        dates = ["-".join(os.path.basename(p).split("-")[-3:])
                 for p in projects]
        if len(dates) == 1:
            return dates[0]
        else:
            dates_str = "\n  - ".join(dates)
            raise RuntimeError(
                "Cannot auto-resolve config variable for `date`."
                "\n\nConfig variable `date` was not set, "
                "and multiple matching DLC projects were found.\n"
                "Explicitly set `date=YYYY-MM-DD` to one of the following "
                "(or choose a new date to create a new project):\n"
                f"  - {dates_str}"
            )

def _sort_files_by_date(files):
    if len(files) == 0:
        return None

    files = sorted((re.match(f"(^.*){RECORDING_END_REGEX}.(avi|csv|xdat.json)$", f).groups()
                    for f in files),
                   key=lambda t: datetime.strptime(t[1], "%H-%M-%S"),
                   reverse=True)
    if all(f[0] == files[0][0] for f in files):
        return [f"{f[0]}_{f[1]}.{f[2]}" for f in files]
    else:
        return None

def find_video(repo_path, dataset, view, mouse, condition, run,
               concat = False, data_dir = "fe-data"):
    """
    Find the matching video file(s) under the directory
    `{repo_path}/{data_dir}/{dataset}`. The format of the filename should be:
    ```
    {YYYYMMdd}_{B{9-0} or test or all}_{exp}_{cond}_{run}_{view}_{HH}-{mm}-{SS}.avi
    ```
    For example, `20230324_B3_tastant_plain_000_L_15-06-21.avi` is matched as:
    - `{YYYYMMdd} = 20230324`
    - `{B{9-0} or test or all}` = B3`
    - `{exp} = tastant`
    - `{cond} = plain`
    - `{view} = L`
    - `{run} = 000`
    - `{HH} = 15`
    - `{mm} = 06`
    - `{SS} = 21`

    If multiple videos are matched and `concat` is true,
    then all videos will be concatenated into a single video.
    The concatenated video will have the name:
    ```
    {dataset}_concat_{view}.avi
    ```
    where `dataset` is the argument passed into this function.
    When `concat` is false, multiple matched videos will be returned as a list.

    Arguments:
    - `repo_path`: the path to the repo root directory
    - `dataset`: the name of the dataset directory
    - `view`: the camera view identifier
    - `mouse`: the substring indicating the mouse (must match the format above,
        can be `None` to match any mouse)
    - `condition`: the substring indicating the condition (variant) of the video
        (can be `None` to match any condition)
    - `run`: a unique substring (three digits in 0-9) separating repeated conditions
        (can be `None` to match any run)
    - `concat`: set to true to concatenate multiple videos
    - `data_dir`: the directory under `repo_path` containing all the datasets
    """
    dataset_path = os.sep.join([repo_path, data_dir, dataset])
    mice = maybe(mouse, RECORDING_MOUSE_REGEX)
    cond = maybe(condition, RECORDING_COND_REGEX)
    run_id = maybe(run, RECORDING_RUN_REGEX)
    video_regex = "_".join([RECORDING_DATE_REGEX,
                            mice,
                            RECORDING_EXP_REGEX,
                            cond,
                            run_id,
                            view,
                            f"{RECORDING_END_REGEX}.avi"])
    video = reglob(video_regex, dataset_path)
    sorted_videos = _sort_files_by_date(video)

    if len(video) > 1 and sorted_videos is not None:
        logging.warn("Found multiple videos for "
                     f"{dataset=} {view=} {mouse=} {condition=} {run=} "
                      "with different times. Will return most recent video.")
        return sorted_videos[0]

    elif len(video) > 1 and concat:
        logging.warn("Found multiple videos for "
                     f"{dataset=} {view=} {mouse=} {condition=} {run=}. "
                      "Will concatenate all matching videos.")
        out_file = os.sep.join([dataset_path, f"{dataset}_concat_{view}.avi"])
        file_list = os.sep.join([dataset_path, f"{dataset}_file_list.txt"])
        # do not overwrite an existing file
        if os.path.exists(out_file):
            return out_file
        # concat videos
        with open(file_list, "w") as f:
            for file in video:
                f.write(f"file {file}\n")
        ret = subprocess.call([
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list,
            '-c', 'copy',
            out_file
        ])
        if ret:
            raise RuntimeError(f"Failed to concatenate recordings (ret code = {ret})")
        else:
            os.remove(file_list)

            return out_file

    elif len(video) > 1:
        logging.warn("Found multiple videos for "
                     f"{dataset=} {view=} {mouse=} {condition=} {run=}. "
                      "Will return all matching videos.")
        return video

    elif len(video) == 0:
        logging.warn(
            f"Found no videos for dataset in {dataset_path} "
            f"({mouse=}, {condition=}, {run=}, {video_regex=}). "
            "Will default to empty string."
        )

        return ""

    return video[0]

def find_csv(repo_path, dataset, mouse, condition, run, data_dir = "fe-data"):
    """
    Find the matching CSV metadata file under the directory
    `{repo_path}/{data_dir}/{dataset}`. The format of the filename should be:
    ```
    {YYYYMMdd}_{B{9-0} or test or all}_{exp}_{cond}_{run}_{HH}-{mm}-{SS}.csv
    ```
    For example, `20230324_B3_tastant_plain_000_15-06-21.csv` is matched as:
    - `{YYYYMMdd} = 20230324`
    - `{B{9-0} or test or all}` = B3`
    - `{exp} = tastant`
    - `{cond} = plain`
    - `{run} = 000`
    - `{HH} = 15`
    - `{mm} = 06`
    - `{SS} = 21`

    If multiple CSVs are found, the first matching value is used.

    Arguments:
    - `repo_path`: the path to the repo root directory
    - `dataset`: the name of the dataset directory
    - `mouse`: the substring indicating the mouse (must match the format above,
        can be `None` to match any mouse)
    - `condition`: the substring indicating the condition (variant) of the video
        (can be `None` to match any condition)
    - `run`: a unique substring (three digits in 0-9) separating repeated conditions
        (can be `None` to match any run)
    - `data_dir`: the directory under `repo_path` containing all the datasets
    """
    dataset_path = os.sep.join([repo_path, data_dir, dataset])
    mice = maybe(mouse, RECORDING_MOUSE_REGEX)
    cond = maybe(condition, RECORDING_COND_REGEX)
    run_id = maybe(run, RECORDING_RUN_REGEX)
    csv = reglob("_".join([RECORDING_DATE_REGEX,
                           mice,
                           RECORDING_EXP_REGEX,
                           cond,
                           run_id,
                           f"{RECORDING_END_REGEX}.csv"]), dataset_path)
    sorted_csvs = _sort_files_by_date(csv)

    if len(csv) != 1 and sorted_csvs is not None:
        logging.warn(
            f"Found multiple CSVs for dataset in {dataset_path} "
            "with different timestamps. Will return the latest CSV."
        )
        csv = sorted_csvs
    elif len(csv) != 1:
        csv_list = "\n- ".join(csv)
        logging.warn(
            f"Found multiple (or no) CSVs for dataset in {dataset_path}. "
            "The following CSVs were matched for identifier = "
            f"(mouse = {mouse}, cond = {condition}, run = {run}):\n"
            f"- {csv_list}\n"
            f"Will default to first value (or empty string)."
        )
        # append an empty string in case there was no match
        csv.append("")

    return csv[0]

def find_allego(repo_path, dataset, mouse, run, data_dir = "ephys-data"):
    """
    Find the matching Open Ephys files under the directory
    `{repo_path}/{data_dir}/{dataset}`. The format of the filename should be:
    ```
    {YYYYMMdd}_{B{9-0} or test or all}_{exp}_{cond}_{run}_{view}_{HH}-{mm}-{SS}.avi
    ```
    For example, `20230324_B3_tastant_plain_000_L_15-06-21.avi` is matched as:
    - `{YYYYMMdd} = 20230324`
    - `{B{9-0} or test or all}` = B3`
    - `{exp} = tastant`
    - `{cond} = plain`
    - `{view} = L`
    - `{run} = 000`
    - `{HH} = 15`
    - `{mm} = 06`
    - `{SS} = 21`

    Multiple matched recordings will be returned as a list.

    Arguments:
    - `repo_path`: the path to the repo root directory
    - `dataset`: the name of the dataset directory
    - `view`: the camera view identifier
    - `mouse`: the substring indicating the mouse (must match the format above,
        can be `None` to match any mouse)
    - `condition`: the substring indicating the condition (variant) of the video
        (can be `None` to match any condition)
    - `run`: a unique substring (three digits in 0-9) separating repeated conditions
        (can be `None` to match any run)
    - `data_dir`: the directory under `repo_path` containing all the datasets
    """
    dataset_path = os.sep.join([repo_path, data_dir, dataset])
    mice = maybe(mouse, RECORDING_MOUSE_REGEX)
    if run is None:
        run_id = r"(\d+)"
    else:
        run_id = str(int(run))
    date = r"_uid\d{4}-" + RECORDING_END_REGEX
    json_regex = "_".join([mice, run_id, f"{date}.xdat.json"])
    json = reglob(json_regex, dataset_path)
    sorted_json = _sort_files_by_date(json)

    if len(json) > 1 and sorted_json is not None:
        logging.warn("Found multiple .xdat.json files for "
                     f"{dataset=} {mouse=} {run=} "
                      "with different times. Will return most recent file.")
        return sorted_json[0]

    elif len(json) > 1:
        logging.warn("Found multiple .xdat.json files for "
                     f"{dataset=} {mouse=} {run=}. "
                      "Will return all matching files.")
        return json

    elif len(json) == 0:
        logging.warn(
            f"Found no .xdat.json files for dataset in {dataset_path} "
            f"({mouse=} {run=}). "
            "Will default to empty string."
        )

        return ""

    return json[0]

def find_openephys(repo_path, dataset, mouse, run, data_dir = "ephys-data"):
    """
    Find the matching Open Ephys files under the directory
    `{repo_path}/{data_dir}/{dataset}`. The format of the filename should be:
    ```
    {YYYYMMdd}_{B{9-0} or test or all}_{exp}_{cond}_{run}_{view}_{HH}-{mm}-{SS}.avi
    ```
    For example, `20230324_B3_tastant_plain_000_L_15-06-21.avi` is matched as:
    - `{YYYYMMdd} = 20230324`
    - `{B{9-0} or test or all}` = B3`
    - `{exp} = tastant`
    - `{cond} = plain`
    - `{view} = L`
    - `{run} = 000`
    - `{HH} = 15`
    - `{mm} = 06`
    - `{SS} = 21`

    Multiple matched recordings will be returned as a list.

    Arguments:
    - `repo_path`: the path to the repo root directory
    - `dataset`: the name of the dataset directory
    - `view`: the camera view identifier
    - `mouse`: the substring indicating the mouse (must match the format above,
        can be `None` to match any mouse)
    - `condition`: the substring indicating the condition (variant) of the video
        (can be `None` to match any condition)
    - `run`: a unique substring (three digits in 0-9) separating repeated conditions
        (can be `None` to match any run)
    - `data_dir`: the directory under `repo_path` containing all the datasets
    """
    dataset_path = os.sep.join([repo_path, data_dir, dataset])
    mice = maybe(mouse, RECORDING_MOUSE_REGEX)
    # if run is None:
    #     run_id = r"(\d+)"
    # else:
    #     run_id = str(int(run))
    end_regex = r"([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}-[0-9]{2}-[0-9]{2})_(\d{3})$"
    oe_folder_regex = "_".join([mice, end_regex])
    oe_folder = reglob(oe_folder_regex, dataset_path)
    # sorted_oe_folder = _sort_files_by_date(oe_folder)

    # if len(oe_folder) > 1 and sorted_oe_folder is not None:
    #     logging.warn("Found multiple OE folders for "
    #                  f"{dataset=} {mouse=} {run=} "
    #                   "with different times. Will return most recent file.")
    #     return sorted_oe_folder[0]

    if len(oe_folder) > 1:
        logging.warn("Found multiple OE folders for "
                     f"{dataset=} {mouse=} {run=}. "
                      "Will return all matching files.")
        return oe_folder

    elif len(oe_folder) == 0:
        logging.warn(
            f"Found no OE folders for dataset in {dataset_path} "
            f"({mouse=} {run=}). "
            "Will default to empty string."
        )

        return ""

    return oe_folder[0]

def find_dsi(repo_path, dataset, mouse, run, data_dir = "ephys-data"):
    """
    Find the matching Open Ephys files under the directory
    `{repo_path}/{data_dir}/{dataset}`. The format of the filename should be:
    ```
    {YYYYMMdd}_{B{9-0} or test or all}_{exp}_{cond}_{run}_{view}_{HH}-{mm}-{SS}.avi
    ```
    For example, `20230324_B3_tastant_plain_000_L_15-06-21.avi` is matched as:
    - `{YYYYMMdd} = 20230324`
    - `{B{9-0} or test or all}` = B3`
    - `{exp} = tastant`
    - `{cond} = plain`
    - `{view} = L`
    - `{run} = 000`
    - `{HH} = 15`
    - `{mm} = 06`
    - `{SS} = 21`

    Multiple matched recordings will be returned as a list.

    Arguments:
    - `repo_path`: the path to the repo root directory
    - `dataset`: the name of the dataset directory
    - `view`: the camera view identifier
    - `mouse`: the substring indicating the mouse (must match the format above,
        can be `None` to match any mouse)
    - `condition`: the substring indicating the condition (variant) of the video
        (can be `None` to match any condition)
    - `run`: a unique substring (three digits in 0-9) separating repeated conditions
        (can be `None` to match any run)
    - `data_dir`: the directory under `repo_path` containing all the datasets
    """
    dataset_path = os.sep.join([repo_path, data_dir, dataset])
    mice = maybe(mouse, RECORDING_MOUSE_REGEX)
    # if run is None:
    #     run_id = r"(\d+)"
    # else:
    #     run_id = str(int(run))
    eeg_txt_regex = "_".join([RECORDING_DATE_REGEX, mice, RECORDING_COND_REGEX, "eeg.txt"])
    eeg_txt = reglob(eeg_txt_regex, dataset_path)
    # sorted_oe_folder = _sort_files_by_date(oe_folder)

    # if len(eeg_txt) > 1 and sorted_eeg_txt is not None:
    #     logging.warn("Found multiple OE folders for "
    #                  f"{dataset=} {mouse=} {run=} "
    #                   "with different times. Will return most recent file.")
    #     return sorted_eeg_txt[0]

    if len(eeg_txt) > 1:
        logging.warn("Found multiple DSI .txt files for "
                     f"{dataset=} {mouse=} {run=}. "
                      "Will return all matching files.")
        return eeg_txt

    elif len(eeg_txt) == 0:
        logging.warn(
            f"Found no DSI .txt files for dataset in {dataset_path} "
            f"({mouse=} {run=}). "
            "Will default to empty string."
        )

        return ""

    return eeg_txt[0]

def setup_hydra():
    """Call this before loading any Hydra config.
    This registers some global Hydra utilities of our lab's config."""
    OmegaConf.register_new_resolver("find_video", find_video, replace=True)
    OmegaConf.register_new_resolver("find_csv", find_csv, replace=True)
    OmegaConf.register_new_resolver("find_allego", find_allego, replace=True)
    OmegaConf.register_new_resolver("find_openephys", find_openephys, replace=True)
    OmegaConf.register_new_resolver("find_dsi", find_dsi, replace=True)
    OmegaConf.register_new_resolver("default_dlc_date", default_dlc_date, replace=True)

    cs = ConfigStore()
    cs.store(group="labutils", name="sessionconfig", node=SessionConfig)
    cs.store(group="labutils", name="dlcconfig", node=DLCConfig)
    cs.store(group="labutils", name="aniposeconfig", node=AniposeConfig)
