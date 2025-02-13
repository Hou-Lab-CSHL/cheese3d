import pandas as pd
import os
import re
import numpy as np
import h5py
from glob import glob

def read_2d_data(data_dir, landmark_names, landmarks_masks):
    """Load 2D landmark data from anipose for multiple views

    ### Arguments:
    - `data_dir`: anipose directory for the video to load
    - `landmark_names`: a list of names for each landmark
    - `landmark_masks`: a dictionary mapping view names to boolean
      mask for which landmarks are visible in each view

    ### Returns:
    -  TODO
    """
    files = glob(os.sep.join([data_dir, "pose-2d-filtered", "*.h5"]))
    landmarks = []
    masks = []
    for file in files:
        data = h5py.File(file)["df_with_missing/table"]
        data = np.stack([data_i[1] for data_i in data])
        landmarks.append(np.reshape(data, (-1, len(landmark_names), 3)))
        view = re.match(r".*_(TL|TR|L|R|BC|TC)_.*\.h5", file).group(1)
        masks.append(landmarks_masks[view])

    landmarks = np.stack(landmarks, axis=1)
    masks = np.stack(masks)

    return landmarks[:, masks, :]

def read_3d_data(data_dir, filter_func=None, filter_kwargs=None):
    """Load 3D landmark data from Anipose.

    ### Arguments
    - `data_dir`: the Anipose directory for the video to load
    - `filer_func`: optional, a callable filter function to apply to the 3D data
    - `filter_kwargs`: optional, kwargs to pass to the filter_func

    ### Returns
    A dictionary of 3D data where each key is a landmark name
    and the associated value is an array of shape `(time, 3)`
    """
    files = glob(os.sep.join([data_dir, "pose-3d", "*.csv"]))
    data = pd.read_csv(files[0])
    cols = data.head() # name of all the columns
    landmark_names = np.unique([s.split('_')[0]
                                for s in cols if s.endswith(('_x', '_y', '_z'))])
    landmarks = {landmark: np.stack([data[f"{landmark}_x"],
                                     data[f"{landmark}_y"],
                                     data[f"{landmark}_z"]], axis=-1)
                 for landmark in landmark_names}
    if filter_func is not None:
        filter_kwargs = filter_kwargs or {}
        landmarks = {lm: filter_func(vals, **filter_kwargs) for lm, vals in landmarks.items()}
    return landmarks
