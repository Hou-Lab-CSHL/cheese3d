# this file generally mirrors video generation in anipose
# with some custom patches to support what we want in cheese3d
# it also decouples the visualization from anipose allowing
# us to iterate a bit quicker
import os
import re
import cv2
import queue
import threading
import skvideo.io
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import colors
from pathlib import Path
from seaborn import color_palette
from tqdm import trange

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)

    return params

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)
    params = get_video_params_cap(cap)
    cap.release()

    return params

def get_nframes(vidname):
    try:
        metadata = skvideo.io.ffprobe(vidname)
        length = int(metadata['video']['@nb_frames'])

        return length
    except KeyError:
        return 0

def tiny_cmap(center, delta, n, cmap="colorblind"):
    if isinstance(cmap, str):
        _cmap = color_palette(cmap, as_cmap=True)
    else:
        _cmap = cmap
    if isinstance(_cmap, list):
        _cmap = colors.LinearSegmentedColormap.from_list("tiny_cmap", _cmap)

    return list(_cmap(np.linspace(center - delta, center + delta, n), bytes=True))

def color_scheme(scheme, cmap="colorblind", spread=0.02):
    bp_to_color = {}
    cluster_centers = np.linspace(0, 1, len(scheme))
    for idx, cluster in enumerate(scheme):
        cluster_size = len(cluster)
        cluster_cmap = tiny_cmap(cluster_centers[idx], spread, cluster_size, cmap)
        for bp in cluster:
            bp_to_color[bp] = [int(c) for c in cluster_cmap.pop()]
    return bp_to_color

def connect(img, points, bps, bodyparts, col=(0, 255, 0, 255)):
    try:
        ixs = [bodyparts.index(bp) for bp in bps]
    except ValueError:
        return
    for a, b in zip(ixs, ixs[1:]):
        if np.any(np.isnan(points[[a, b]])):
            continue
        pa = tuple(np.int32(points[a])) # type: ignore
        pb = tuple(np.int32(points[b])) # type: ignore
        cv2.line(img, tuple(pa), tuple(pb), col, 4)

def connect_all(img, points, scheme, bodyparts, bp_to_color):
    for bps in scheme:
        col = bp_to_color[bps[0]]
        connect(img, points, bps, bodyparts, col)

def label_frame(img, points, scheme, bodyparts, bp_to_color):
    connect_all(img, points, scheme, bodyparts, bp_to_color)
    for lnum, (x, y) in enumerate(points):
        if np.isnan(x) or np.isnan(y):
            continue
        x = np.clip(x, 1, img.shape[1] - 1)
        y = np.clip(y, 1, img.shape[0] - 1)
        x = int(round(x))
        y = int(round(y))
        col = bp_to_color[bodyparts[lnum]]
        cv2.circle(img, (x, y), 7, col[:3], -1)

    return img

def visualize_labels(scheme, bodyparts, points, scores, vid_fname, outname,
                     colormap="colorblind"):
    bp_to_color = color_scheme(scheme, cmap=colormap)
    cap = cv2.VideoCapture(vid_fname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        '-framerate': str(fps),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '28',
        '-pix_fmt': 'yuv420p',
        '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    })
    last = points.shape[2]
    scores = scores.copy()
    scores[np.isnan(scores)] = 0
    scores[np.isnan(points[:, 0, :])] = 0
    good = scores > 0.1
    points = points.copy()
    points[:, 0, :][~good] = np.nan
    points[:, 1, :][~good] = np.nan
    for ix in trange(last, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_points = points[:, :, ix]
        img = label_frame(img, frame_points, scheme, bodyparts, bp_to_color)
        writer.writeFrame(img)
    cap.release()
    writer.close()

def _find_matching_video(h5_basename, videos_raw_dir):
    h5_base = Path(h5_basename)
    raw_dir = Path(videos_raw_dir)
    candidates = []
    for video_file in sorted(raw_dir.iterdir()):
        if not video_file.is_file():
            continue
        # DLC appends its scorer/model identifier directly after the source
        # video stem. LP outputs normally retain the exact source stem.
        if (h5_base.stem == video_file.stem or
            h5_base.stem.startswith(video_file.stem + "DLC")):
            candidates.append(video_file)
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        choices = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Pose output {h5_base.name} matches multiple raw videos: {choices}"
        )
    return None

def _read_pose_2d(h5_path, bodyparts_hint=None):
    dlabs = pd.read_hdf(h5_path)
    if len(dlabs.columns.levels) > 2:
        scorer = dlabs.columns.levels[0][0]
        dlabs = dlabs.loc[:, scorer]
    if bodyparts_hint is not None:
        bodyparts = bodyparts_hint
    else:
        bodyparts = list(dlabs.columns.levels[0])
    points = np.array([(dlabs[bp]['x'], dlabs[bp]['y']) for bp in bodyparts])
    scores = np.array([dlabs[bp]['likelihood'] for bp in bodyparts])

    return bodyparts, points, scores

def generate_videos_2d(scheme, bodyparts, videos_raw_dir, pose_dir, out_dir):
    pose_dir = Path(pose_dir)
    out_dir = Path(out_dir)
    labels_fnames = sorted(pose_dir.glob('*.h5'), key=lambda p: p.name)
    if len(labels_fnames) == 0:
        print(f"No pose HDF5 files found in {pose_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for fname in labels_fnames:
        basename = fname.stem
        vid_path = _find_matching_video(basename, videos_raw_dir)
        if vid_path is None:
            print(f"Skipping {fname.name}: no matching raw video found in {videos_raw_dir}")
            continue
        # Name labeled videos after the raw videos. The comparison-video stage
        # uses those names, and pose filenames may contain a DLC scorer suffix.
        out_fname = str(out_dir / (Path(vid_path).stem + '.mp4'))
        if (os.path.exists(out_fname) and
            abs(get_nframes(out_fname) - get_nframes(vid_path)) < 100):
            completed += 1
            continue
        _, points, scores = _read_pose_2d(str(fname), bodyparts_hint=bodyparts)
        print(out_fname)
        visualize_labels(scheme, bodyparts, points, scores, vid_path, out_fname)
        if os.path.exists(out_fname):
            completed += 1

    return completed

def write_frame_thread(writer, q):
    while True:
        frame = q.get(block=True)
        if frame is None:
            return
        writer.writeFrame(frame)

def read_frames(caps):
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            return False, None
        frames.append(frame)
    return True, frames

def get_plotting_params(caps, n_rows=4):
    params = [get_video_params_cap(c) for c in caps]
    height = max([p['height'] for p in params])
    widths = [round(p['width'] * height / p['height']) for p in params]

    width_total = sum(widths)
    height_total = height * n_rows

    nframes = min([p['nframes'] for p in params])
    fps = np.mean([p['fps'] for p in params])

    return {
        'height': height,
        'widths': widths,
        'nframes': nframes,
        'fps': fps,
        'width_total': width_total,
        'height_total': height_total,
    }

def draw_data(start_img, frames_raw, frames_2d, frames_2d_filt, frames_2d_proj, pp):
    height = pp['height']
    widths = pp['widths']
    has_filt = frames_2d_filt is not None
    frames_raw_resized = [cv2.resize(f, (w, height))
                         for f, w in zip(frames_raw, widths)]
    frames_2d_resized = [cv2.resize(f, (w, height))
                         for f, w in zip(frames_2d, widths)]
    frames_2d_proj_resized = [cv2.resize(f, (w, height))
                              for f, w in zip(frames_2d_proj, widths)]
    imout = np.copy(start_img)
    imout[0:height] = np.hstack(frames_raw_resized)
    imout[height:height*2] = np.hstack(frames_2d_resized)
    if has_filt:
        frames_2d_filt_resized = [cv2.resize(f, (w, height))
                                  for f, w in zip(frames_2d_filt, widths)]
        imout[height*2:height*3] = np.hstack(frames_2d_filt_resized)
        imout[height*3:height*4] = np.hstack(frames_2d_proj_resized)
    else:
        imout[height*2:height*3] = np.hstack(frames_2d_proj_resized)

    return imout

def visualize_compare(fnames_raw, fnames_2d, fnames_2d_filt, fnames_2d_proj, out_fname):
    has_filt = fnames_2d_filt is not None
    n_rows = 4 if has_filt else 3
    caps_raw = [cv2.VideoCapture(v) for v in fnames_raw]
    caps_2d = [cv2.VideoCapture(v) for v in fnames_2d]
    if has_filt:
        caps_2d_filt = [cv2.VideoCapture(v) for v in fnames_2d_filt]
    caps_2d_proj = [cv2.VideoCapture(v) for v in fnames_2d_proj]
    pp = get_plotting_params(caps_2d, n_rows=n_rows)
    nframes = pp['nframes']
    fps = pp['fps']
    start_img = np.zeros((pp['height_total'], pp['width_total'], 3), dtype='uint8')
    writer = skvideo.io.FFmpegWriter(out_fname, inputdict={
        '-framerate': str(round(fps, ndigits=2)),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '28',
        '-pix_fmt': 'yuv420p',
        '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    })
    q = queue.Queue(maxsize=50)
    thread = threading.Thread(target=write_frame_thread, args=(writer, q))
    thread.start()
    for _ in trange(nframes, ncols=70):
        ret0, frames_raw = read_frames(caps_raw)
        ret1, frames_2d = read_frames(caps_2d)
        if has_filt:
            ret2, frames_2d_filt = read_frames(caps_2d_filt)
        else:
            ret2 = True
            frames_2d_filt = None
        ret3, frames_2d_proj = read_frames(caps_2d_proj)
        if not all([ret0, ret1, ret2, ret3]):
            break
        imout = draw_data(start_img, frames_raw, frames_2d, frames_2d_filt, frames_2d_proj, pp)
        imout = cv2.cvtColor(imout, cv2.COLOR_BGR2RGB)
        q.put(imout)
    for cap in caps_raw:
        cap.release()
    for cap in caps_2d:
        cap.release()
    if has_filt:
        for cap in caps_2d_filt:
            cap.release()
    for cap in caps_2d_proj:
        cap.release()
    q.put(None)
    thread.join()
    writer.close()

def generate_compare_video(videos_raw_dir,
                           videos_labeled_dir,
                           videos_labeled_filt_dir,
                           videos_2d_proj_dir,
                           out_dir, cam_regex):
    videos_raw_dir = Path(videos_raw_dir)
    videos_labeled_dir = Path(videos_labeled_dir)
    videos_labeled_filt_dir = Path(videos_labeled_filt_dir)
    videos_2d_proj_dir = Path(videos_2d_proj_dir)
    out_dir = Path(out_dir)
    vid_fnames = list(videos_raw_dir.glob("*"))
    if len(vid_fnames) == 0:
        return 0
    fnames_raw = defaultdict(list)
    for vid in vid_fnames:
        basename = vid.stem
        vidname = re.sub(cam_regex, '', basename).strip()
        fnames_raw[vidname].append(str(vid))

    out_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for vidname in sorted(fnames_raw.keys()):
        out_fname = str(out_dir / (vidname + '.mp4'))
        vids_raw = sorted(fnames_raw[vidname])
        vid_fname = vids_raw[0]
        if (os.path.exists(out_fname) and
            abs(get_nframes(out_fname) - get_nframes(vid_fname)) < 100):
            completed += 1
            continue
        vids_2d = [str(videos_labeled_dir / (Path(f).stem + '.mp4'))
                   for f in vids_raw]
        vids_2d_filtered = [str(videos_labeled_filt_dir / (Path(f).stem + '.mp4'))
                            for f in vids_raw]
        vids_2d_projected = [str(videos_2d_proj_dir / (Path(f).stem + '.mp4'))
                             for f in vids_raw]
        if not all([os.path.exists(f) for f in vids_2d]):
            print(out_fname, 'missing labeled 2d videos')
            continue
        if not all([os.path.exists(f) for f in vids_2d_projected]):
            print(out_fname, 'missing projected 2d videos')
            continue
        if not all([os.path.exists(f) for f in vids_2d_filtered]):
            vids_2d_filtered = None
        print(out_fname)
        visualize_compare(vids_raw, vids_2d, vids_2d_filtered, vids_2d_projected, out_fname)
        if os.path.exists(out_fname):
            completed += 1

    return completed
