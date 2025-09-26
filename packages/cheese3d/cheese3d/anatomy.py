import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

def compute_euclidean_dist(a, b):
    """Compute the Euclidean distance between two 3D points.

    ### Arguments
    - `a`, `b`: the 3D vectors as arrays of shape `(batch..., 3)`

    ### Returns
    The distance as an array of shape `(batch...,)`.
    """
    return np.linalg.norm(a - b, axis=-1)

def project_vector(a, b):
    """Project a 3D vector `a` onto another 3D vector `b`.

    ### Arguments
    - `a`, `b`: the 3D vectors as arrays of shape `(batch..., 3)`

    ### Returns
    An array of shape `(batch..., 3)` for the projected vector(s).
    """
    bnorm = np.sum(b * b, axis=-1, keepdims=True)
    scale = np.where(bnorm > 0, np.sum(a * b, axis=-1, keepdims=True) / bnorm, 0)

    return scale * b

def compute_normal(a, b, c):
    """Given three points, `a`, `b`, and `c`, compute the vector
    normal to the plane defined by the points.

    ### Arguments
    - `a`, `b`, `c`: the 3D points as arrays of shape `(batch..., 3)`

    ### Returns
    An array of shape `(batch..., 3)` for the normal vectors(s).
    """
    return np.cross(a - c, b - c, axis=-1)

def compute_angle(a, b):
    """Compute the angle between two vectors.

    ### Arguments
    - `a`, `b`: the N-D vectors as arrays of shape `(batch..., N)`

    ### Returns
    An array angles of shape `(batch...,)`.
    """
    scale = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    angle = np.arccos(np.sum(a * b, axis=-1) / scale)

    return angle * 180 / np.pi

def compute_triangle_area(a, b, c):
    """Given the 3D coordinates of three points, compute the area of
    the enclosed triangle.

    ### Arguments
    - `a`,`b`,`c`,`d`: a 3D coordinate for each point of the triangle as
        an array of shape `(batch..., 3)`

    ### Returns
    The area of the triangle as an array of shape `(batch...,)`.
    """
    # get two sides of triangle
    left_edge = a - c
    right_edge = b - c
    # compute cross product formula for area of triangle in 3d
    cross_prod = np.cross(left_edge, right_edge)
    area = np.linalg.norm(cross_prod, axis=-1) / 2

    return area

def compute_tetrahedron_volume(a, b, c, d):
    """Given the 3D coordinates of four points, compute the volume of
    the enclosed tetrahedron.

    ### Arguments
    - `a`,`b`,`c`,`d`: a 3D coordinate for each point of the volume as
        an array of shape `(batch..., 3)`

    ### Returns
    The volume of the tetrahedron as an array of shape `(batch...,)`.

    #### Reference
    https://en.wikipedia.org/wiki/Tetrahedron#Volume
    """
    ad = a - d
    bd = b - d
    cd = c - d

    return np.abs(np.sum(ad * np.cross(bd, cd), axis=-1)) / 6

def compute_hull_volume(*points):
    """Given the 3D coordinates of any number of points, compute the volume of
    the convex hull enclosing all points. Note that, by definition, not every
    point must lie on the surface of the hull.

    ### Arguments
    - `*points`: a variable number of 3D points with shape `(batch..., 3)`

    ### Returns
    The volume of the hull as an array of shape `(batch...,)`.
    """
    *batch_size, _ = points[0].shape
    assert all(p.shape == (*batch_size, 3) for p in points), (
        "All points must have the same shape `(batch..., 3)`"
    )

    # flatten batch dimensions
    points = [np.reshape(p, (-1, 3)) for p in points]
    # stack together points in sets and build convex hull
    hulls = [ConvexHull(np.stack(ps)) for ps in zip(*points)]
    # get volume and reshape into batch dimensions
    volume = np.reshape(np.stack([h.volume for h in hulls]), batch_size)

    return volume

def compute_ellipse_area(major_pts, minor_pts):
    """Given the major and minor points of an ellipse, compute the area.

    ### Arguments
    - `major_pts`: a 2-tuple of the major axis 3D coordinates each with shape
        `(batch..., 3)`
    - `minor_pts`: a 2-tuple of the minor axis 3D coordinates each with shape
        `(batch..., 3)`

    ### Returns
    The area of the ellipse as an array of shape `(batch...,)`.
    """
    major_axis = np.linalg.norm(major_pts[1] - major_pts[0], axis=-1)
    midpoint = (major_pts[0] + major_pts[1]) / 2
    minor_axis = (np.linalg.norm(minor_pts[1] - midpoint, axis=-1) +
                  np.linalg.norm(minor_pts[0] - midpoint, axis=-1))

    return np.pi / 4 * major_axis * minor_axis

def compute_mouth_sidelen(upperlip, lowerlip):
    """Compute the side length of the mouth opening as the distance from
    the upperlip to lowerlip.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `lowerlip`: an array of lowerlip positions with shape `(batch..., 3)`
    - `upperlip`: an array of upperlip positions with shape `(batch..., 3)`

    ### Returns
    The side length of the mouth as an array of shape `(batch...)`.
    """
    return compute_euclidean_dist(upperlip, lowerlip)

def compute_mouth_area(lowerlip, upperlip_left, upperlip_right):
    """Compute the area of mouth opening given a set of 3D landmarks.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `lowerlip`: an array of lowerlip positions with shape `(batch..., 3)`
    - `upperlip_left`: an array of left upperlip positions with shape
        `(batch..., 3)`
    - `upperlip_right`: an array of right upperlip positions with shape
        `(batch..., 3)`

    ### Returns
    An array of size `(batch...,)` where each element is the mouth area.
    """
    return compute_triangle_area(upperlip_left, upperlip_right, lowerlip)

def compute_eye_height(eye_top, eye_bottom):
    """Compute the height of the eye opening given the top and bottom of the
    3D eye coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `eye_top`: an array of top of the eye positions with shape `(batch..., 3)`
    - `eye_bottom`: an array of bottom of the eye positions with shape
        `(batch..., 3)`

    ### Returns
    An array of size `(batch...,)` where each element is the eye opening height.
    """
    return compute_euclidean_dist(eye_top, eye_bottom)

def compute_eye_width(eye_front, eye_back):
    """Compute the width of the eye opening given the front and back of the
    3D eye coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `eye_front`: an array of front of the eye positions with shape
        `(batch..., 3)`
    - `eye_back`: an array of back of the eye positions with shape
        `(batch..., 3)`

    ### Returns
    An array of size `(batch...,)` where each element is the eye opening width.
    """
    return compute_euclidean_dist(eye_front, eye_back)

def compute_eye_orbital_tightness(eye_front, eye_back, eye_top, eye_bottom):
    """Compute the orbital tightness (area) of the eye opening given the
    front, back, top, and bottom 3D eye coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `eye_front`: an array of front of the eye positions with shape
        `(batch..., 3)`
    - `eye_back`: an array of back of the eye positions with shape
        `(batch..., 3)`
    - `eye_top`: an array of top of the eye positions with shape `(batch..., 3)`
    - `eye_bottom`: an array of bottom of the eye positions with shape
        `(batch..., 3)`

    ### Returns
    An array of size `(batch...,)` where each element is the orbital tightness.
    """
    # define the major and minor axis of the ellipse
    major_axis = (eye_front, eye_back)
    minor_axis = (eye_top, eye_bottom)
    # compute orbital ellipse area
    area = compute_ellipse_area(major_axis, minor_axis)

    return area

def compute_proptosis(eye_front, eye_back, eye_top, eye_bottom, *,
                      reference = ("bottom", "top", "back"), target = "back"):
    def select_pt(name):
        if name == "front":
            return eye_front
        elif name == "back":
            return eye_back
        elif name == "top":
            return eye_top
        elif name == "bottom":
            return eye_bottom
        else:
            raise ValueError(f"{name} is not a valid eye point"
                             "(use front, back, top, or bottom).")

    normal = compute_normal(*(select_pt(pt) for pt in reference))
    proj = project_vector(select_pt(target), normal)

    return np.sum(proj * proj, axis=-1)

def compute_nose_height(nose_top, nose_bottom):
    """Compute the height of the nose as the distance from nose top to nose bottom.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `nose_top`: an array of top of the nose positions with shape `(batch..., 3)`
    - `nose_bottom`: an array of bottom of the nose positions with shape
        `(batch..., 3)`

    ### Returns
    The height of the nose as an array of shape `(batch...)`.
    """
    return compute_euclidean_dist(nose_top, nose_bottom)

def compute_nose_angle(nose_tip, nose_top, nose_bottom):
    """Compute the angle of the nose tip.

    ### Arguments
    - `nose_tip`: an array of nose tip positions with shape `(batch..., 3)`
    - `nose_top`: an array of top of the nose positions with shape `(batch..., 3)`
    - `nose_bottom`: an array of bottom of the nose positions with shape
        `(batch..., 3)`

    ### Returns
    The angle at the nose tip as an array of shape `(batch...)`.
    """
    return compute_angle(nose_top - nose_tip, nose_bottom - nose_tip)

def compute_nose_bulge_volume(eye_left, eye_right, nose_top, pad_left, pad_right):
    """Compute the nose bulge volume based on the front of the eye, top of the
    nose, and top of the pad as 3D coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `eye_left`: an array of front of the left eye positions with shape `(batch..., 3)`
    - `eye_right`: an array of front of the right eye positions with shape `(batch..., 3)`
    - `nose_top`: an array of top of the nose positions with shape `(batch..., 3)`
    - `pad_left`: an array of left pad top positions with shape `(batch..., 3)`
    - `pad_right`: an array of right pad top positions with shape `(batch..., 3)`

    ### Returns
    The volume of the nose bulge as an array of shape `(batch...)`.
    """
    # define the bridge point between the eyes "on" the snout
    bridge_point = (eye_left + eye_right) / 2
    # compute the tetrahedron volume
    volume = compute_tetrahedron_volume(bridge_point, nose_top, pad_left, pad_right)

    return volume

def compute_pad_angle(pad_center, pad_top, pad_side):
    """Compute the angle of the whisker pad segments.

    ### Arguments
    - `pad_center`: an array of pad center positions with shape `(batch..., 3)`
    - `pad_top`: an array of pad top positions with shape `(batch..., 3)`
    - `pad_side`: an array of pad side positions with shape `(batch..., 3)`

    ### Returns
    The angle at the whisker pad as an array of shape `(batch...)`.
    """
    return compute_angle(pad_top - pad_center, pad_side - pad_center)

def compute_cheek_bulge_volume(nose_bottom,
                               pad_top_left,
                               pad_top_right,
                               pad_side_left,
                               pad_side_right):
    """Compute the cheek bulge volume using the convex hull of nose bottom,
    pad top (left + right), and pad side (left + right) 3D coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `nose_bottom`: an array of bottom of the nose positions with shape `(batch..., 3)`
    - `pad_top_left`: an array of left pad top positions with shape `(batch..., 3)`
    - `pad_top_right`: an array of right pad top positions with shape `(batch..., 3)`
    - `pad_side_left`: an array of left pad side positions with shape `(batch..., 3)`
    - `pad_side_right`: an array of right pad side positions with shape `(batch..., 3)`

    ### Returns
    The cheek bulge volume as an array of shape `(batch...)`.
    """
    # compute the hull volume
    volume = compute_hull_volume(nose_bottom,
                                 pad_top_left,
                                 pad_top_right,
                                 pad_side_left,
                                 pad_side_right)

    return volume

def compute_ear_height(ear_tip, ear_base):
    """Compute the height of the ear given the tip and base of the
    3D ear coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `ear_tip`: an array of ear tip positions with shape `(batch..., 3)`
    - `ear_base`: an array of ear base positions with shape `(batch..., 3)`

    ### Returns
    The height (distance between tip and base) of the ear as an array of shape `(batch...)`.
    """
    return compute_euclidean_dist(ear_tip, ear_base)

def compute_ear_width(ear_top, ear_bottom):
    """Compute the width of the ear given the top and bottom of the
    3D ear coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `ear_top`: an array of ear top positions with shape `(batch..., 3)`
    - `ear_bottom`: an array of ear bottom positions with shape `(batch..., 3)`

    ### Returns
    The width (distance between top and bottom) of the ear as an array of shape `(batch...)`.
    """
    return compute_euclidean_dist(ear_top, ear_bottom)

def compute_ear_angle(ear_tip, ear_base, pad_center):
    """Compute the angle between ear tip to ear base vs. ear base to pad center
    segments after projecting the segments onto the sagittal plane.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `ear_tip`: an array of ear tip positions with shape `(batch..., 3)`
    - `ear_base`: an array of ear base positions with shape `(batch..., 3)`
    - `pad_center`: an array of pad center positions with shape `(batch..., 3)`

    ### Returns
    The angle at the ear base as an array of shape `(batch...)`.
    """
    # define segments
    a = ear_tip - ear_base
    b = pad_center - ear_base

    # projecting onto the sagittal plane is equivalent to taking the YZ coords only
    # return compute_angle(a[..., 1:], b[..., 1:])
    # Not taking the projection
    return compute_angle(a[..., 0:], b[..., 0:])

def compute_ear_area(ear_base, ear_tip, ear_top, ear_bottom):
    """Compute the ear area (like a 'footprint') given the
    front, back, top, and bottom 3D ear coordinates.
    The XYZ coordinates must be in the final axis of each landmark.

    ### Arguments
    - `ear_base`: an array of front of the ear positions with shape
        `(batch..., 3)`
    - `ear_tip`: an array of back of the ear positions with shape
        `(batch..., 3)`
    - `ear_top`: an array of top of the ear positions with shape `(batch..., 3)`
    - `ear_bottom`: an array of bottom of the ear positions with shape
        `(batch..., 3)`

    ### Returns
    An array of size `(batch...,)` where each element is the orbital tightness.
    """
    # define the major and minor axis of the ellipse
    major_axis = (ear_base, ear_tip)
    minor_axis = (ear_top, ear_bottom)
    # compute orbital ellipse area
    area = compute_ellipse_area(major_axis, minor_axis)

    return area


def compute_anatomical_measurements(landmarks, exclude = None):
    structure = {
        "eye": [
            "eye-height-left-mm",
            "eye-height-right-mm",
            "eye-width-left-mm",
            "eye-width-right-mm",
            "eye-area-left-mm2",
            "eye-area-right-mm2"
        ],
        "ear": [
            "ear-height-left-mm",
            "ear-height-right-mm",
            "ear-width-left-mm",
            "ear-width-right-mm",
            "ear-angle-left-deg",
            "ear-angle-right-deg",
            "ear-area-left-mm2",
            "ear-area-right-mm2"
        ],
        "mouth": ["mouth-area-mm2"],
        "nose": ["nose-bulge-volume-mm3"],
        "cheek": ["cheek-bulge-volume-mm3"]
    }
    exclude = [] if exclude is None else exclude

    measurements = {}
    for group, names in structure.items():
        measurements[group] = {}
        for name in names:
            if name in exclude:
                continue

            if name == "eye-height-left-mm":
                measurements[group][name] = compute_eye_height(
                    landmarks["eye(top)(left)"],
                    landmarks["eye(bottom)(left)"]
                )
            elif name == "eye-height-right-mm":
                measurements[group][name] = compute_eye_height(
                    landmarks["eye(top)(right)"],
                    landmarks["eye(bottom)(right)"]
                )
            elif name == "eye-width-left-mm":
                measurements[group][name] = compute_eye_width(
                    landmarks["eye(front)(left)"],
                    landmarks["eye(back)(left)"]
                )
            elif name == "eye-width-right-mm":
                measurements[group][name] = compute_eye_width(
                    landmarks["eye(front)(right)"],
                    landmarks["eye(back)(right)"]
                )
            elif name == "eye-area-left-mm2":
                measurements[group][name] = compute_eye_orbital_tightness(
                    landmarks["eye(front)(left)"],
                    landmarks["eye(back)(left)"],
                    landmarks["eye(top)(left)"],
                    landmarks["eye(bottom)(left)"]
                )
            elif name == "eye-area-right-mm2":
                measurements[group][name] = compute_eye_orbital_tightness(
                    landmarks["eye(front)(right)"],
                    landmarks["eye(back)(right)"],
                    landmarks["eye(top)(right)"],
                    landmarks["eye(bottom)(right)"]
                )
            elif name == "ear-height-left-mm":
                measurements[group][name] = compute_ear_height(
                    landmarks["ear(base)(left)"],
                    landmarks["ear(tip)(left)"]
                )
            elif name == "ear-height-right-mm":
                measurements[group][name] = compute_ear_height(
                    landmarks["ear(base)(right)"],
                    landmarks["ear(tip)(right)"]
                )
            elif name == "ear-width-left-mm":
                measurements[group][name] = compute_ear_width(
                    landmarks["ear(top)(left)"],
                    landmarks["ear(bottom)(left)"]
                )
            elif name == "ear-width-right-mm":
                measurements[group][name] = compute_ear_width(
                    landmarks["ear(top)(right)"],
                    landmarks["ear(bottom)(right)"]
                )
            elif name == "ear-angle-left-deg":
                measurements[group][name] = compute_ear_angle(
                    landmarks["ear(tip)(left)"],
                    landmarks["ear(base)(left)"],
                    landmarks["pad(center)"]
                )
            elif name == "ear-angle-right-deg":
                measurements[group][name] = compute_ear_angle(
                    landmarks["ear(tip)(right)"],
                    landmarks["ear(base)(right)"],
                    landmarks["pad(center)"]
                )
            elif name == "ear-area-left-mm2":
                measurements[group][name] = compute_ear_area(
                    landmarks["ear(base)(left)"],
                    landmarks["ear(tip)(left)"],
                    landmarks["ear(top)(left)"],
                    landmarks["ear(bottom)(left)"]
                )
            elif name == "ear-area-right-mm2":
                measurements[group][name] = compute_ear_area(
                    landmarks["ear(base)(right)"],
                    landmarks["ear(tip)(right)"],
                    landmarks["ear(top)(right)"],
                    landmarks["ear(bottom)(right)"]
                )
            elif name == "mouth-area-mm2":
                measurements[group][name] = compute_mouth_area(
                    landmarks["lowerlip"],
                    landmarks["upperlip(left)"],
                    landmarks["upperlip(right)"]
                )
            elif name == "nose-bulge-volume-mm3":
                measurements[group][name] = compute_nose_bulge_volume(
                    landmarks["eye(front)(left)"],
                    landmarks["eye(front)(right)"],
                    landmarks["nose(top)"],
                    landmarks["pad(top)(left)"],
                    landmarks["pad(top)(right)"]
                )
            elif name == "cheek-bulge-volume-mm3":
                measurements[group][name] = compute_cheek_bulge_volume(
                    landmarks["nose(bottom)"],
                    landmarks["pad(top)(left)"],
                    landmarks["pad(top)(right)"],
                    landmarks["pad(side)(left)"],
                    landmarks["pad(side)(right)"]
                )

    return measurements

def compute_measurements_df(coord_data,
                            key_columns = ("mouse", "source", "condition"),
                            exclude = None):
    """Given a dictionary of 3D coordinates from Anipose, create a Pandas
    dataframe with the associated anatomical measurements.

    ### Arguments:
    - `coord_data`: a dictionary with keys of as a tuple (e.g. `(B6, "rig2", "awake")`)
    and values as 3D coordinates produced  by `fepipeline.features.landmarks.read_3d_data`.

    ### Returns
    A dataframe with the computed anatomical measurements.
    """
    data = []
    for data_keys, coords in coord_data.items():
        measurements = compute_anatomical_measurements(coords, exclude)
        for region, rmeasures in measurements.items():
            for name, measure in rmeasures.items():
                if "height" in name or "width" in name:
                    measure_units = "mm"
                    measure_type = "distance"
                elif "angle" in name:
                    measure_units = "deg"
                    measure_type = "angle"
                elif "area" in name:
                    measure_units = "mm^2"
                    measure_type = "area"
                elif "volume" in name:
                    measure_units = "mm^3"
                    measure_type = "volume"
                else:
                    measure_units = None
                    measure_type = "unknown"
                data.append([*data_keys,
                             region, name, measure_units, measure_type,
                             np.mean(measure), np.std(measure), len(measure), measure])

    df = pd.DataFrame(data, columns=[*key_columns,
                                     "measurement_group",
                                     "measurement_name",
                                     "measurement_units",
                                     "measurement_type",
                                     "measurement_value",
                                     "std",
                                     "count",
                                     "timeseries"])
    df.sort_values("mouse", inplace=True)

    return df
