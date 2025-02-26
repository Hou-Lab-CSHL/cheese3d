import os
import subprocess
from glob import glob
from omegaconf import DictConfig

from cheese3d.utils import processed_video_name

# ffmpeg -i -i input.mp4 -vf "curves=all='0/0 0.4/1 1/1'" -ss 00:36.00 -t 6.6 -c:v libvpx -b:v 3200k -an -sn -copyts -threads 4 ground.webm
# ffmpeg -i <video>
#        -vf "curves=all='0/0 0.4/1 1/1'"      (https://hhsprings.bitbucket.io/docs/programming/examples/ffmpeg/manipulating_video_colors/curves.html)
#            "eq=brightness=0.06:saturation=2" (https://ffmpeg.org/ffmpeg-filters.html#eq)
#        -c:v libvpx      (-c:v == video codec)
#        -b:v 3200k       (-b:v == video bitrate)
#        -copyts          (copy input timestampts w/o processing)
#        -pix_fmt gray8   (extract grayscale frames)

def ffmpeg_create_frames(movie):
    img_path = os.path.splitext(movie)[0]
    ret = subprocess.call([
        'ffmpeg',
        # '-r', '10',
        '-i', movie,
        # '-r', '1/1',
        '-pix_fmt', 'gray8',
        f'{img_path} (%d).png',
    ])

    if ret:
        os.remove(img_path + "*.png")
        raise RuntimeError(f"Failed to extract frames using ffmpeg. (code = {ret})")
    else:
        return img_path + "*.png"

def extract_frames(videos):
    frame_paths = []
    for video in videos:
        path = os.path.splitext(video)[0]
        frames = glob(path + "*.png")
        if len(frames) == 0:
            frame_paths.append(ffmpeg_create_frames(video))
        else:
            frame_paths.append(path + "*.png")

    return frame_paths

def ffmpeg_eq_filter(contrast = 1, brightness = 0, saturation = 1):
    assert -1000 < contrast < 1000, (
        f"contrast must be in [-1000.0, 1000.0] (got {contrast})")
    assert -1 < brightness < 1, (
        f"brightness must be in [-1.0, 1.0] (got {brightness})")
    assert -1000 < saturation < 1000, (
        f"saturation must be in [0.0, 3.0] (got {saturation})")

    return f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation}'

def ffmpeg_filter_video(movie, filter_str, nthreads = 4):
    fname, ext = os.path.splitext(movie)
    outfile = f"{fname}_processed{ext}"

    ret = subprocess.call([
        "ffmpeg",
        "-i", movie,
        "-vf", filter_str,
        # "-c:v", "libx264",
        "-b:v", "14000k",
        "-copyts",
        "-threads", str(nthreads),
        outfile
    ])

    if ret:
        os.remove(outfile)
        raise RuntimeError(f"Failed to filter video using ffmpeg. (code = {ret})")
    else:
        return outfile

def filter_videos(videos, filters, **kwargs):
    assert len(videos) == len(filters), (
        "# of videos must == # of filters. Set filter to 'null' in config for no filter.")

    video_paths = []
    for video, f in zip(videos, filters):
        processed_video = processed_video_name(video, f)
        if os.path.exists(processed_video):
            video_paths.append(processed_video)
        else:
            if isinstance(f, DictConfig):
                fstr = ffmpeg_eq_filter(**f)
            else:
                fstr = f

            video_paths.append(ffmpeg_filter_video(video, fstr, **kwargs))

    return video_paths
