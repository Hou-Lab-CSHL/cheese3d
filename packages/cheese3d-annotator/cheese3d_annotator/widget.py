import os
import numpy as np
import pandas as pd
from typing import List
from magicgui.widgets import Container, FileEdit, ComboBox, Label, CheckBox
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QMessageBox, QSizePolicy
from qtpy.QtGui import QFont
from napari import Viewer
from glob import glob
from skimage.io import imread
from skimage.color import rgb2gray
from vispy.color import get_colormap
from pathlib import Path
from cheese3d_annotator.data import (load_keypoints_and_skeleton,
                                     read_annotations,
                                     write_annotations,
                                     create_empty_annotations,
                                     find_keypoint_conflicts,
                                     ensure_images_in_yaml)

class FrameAnnotatorWidget(Container):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.layers.clear()
        # self.viewer.window.remove_dock_widget("all") # type: ignore
        self.viewer.grid.enabled = False

        # Main UI elements
        self.root_folder = FileEdit(label="Root Folder", mode="d") # type: ignore
        self.config_path = FileEdit(label="Config File", mode="r") # type: ignore
        self.keypoint_dropdown = ComboBox(label="Keypoint", choices=[])
        self.keypoint_dropdown.enabled = False
        # List of recording folders
        self.folder_list = QListWidget()
        self.folder_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.folder_list.setFixedHeight(394)
        font = QFont()
        font.setPointSize(12)
        self.folder_list.setFont(font)
        self.folder_list.setStyleSheet("QListWidget::item { height: 32px; }")
        # Options UI elements
        self.auto_next_frame = CheckBox(label="Auto next frame", value=False)
        self.auto_next_keypoint = CheckBox(label="Auto next keypoint", value=False)
        self.auto_nav_container = Container(widgets=[
            self.auto_next_frame,
            self.auto_next_keypoint
        ], layout="horizontal")
        # Help instructions
        self.help_label = Label(value="""
            <b>Instructions:</b><br>
            - Select the <b>Root Folder</b> containing recording subfolders.<br>
            - Load a config file with keypoint definition.<br>
            - Click a folder to view and annotate.<br>
            - <b>Left-click</b> to add a point.<br>
            - <b>Right-click</b> to delete a point.<br>
            - Use <b>W/S</b> to change body part.<br>
            - Use <b>A/D</b> to change frame.
            """)

        # Layout
        self.extend([
            self.root_folder,
            self.config_path,
            self.keypoint_dropdown,
            self.auto_nav_container
        ])
        self.native.layout().addWidget(self.folder_list)
        self.native.layout().addWidget(self.help_label.native)

        # Signals
        self.root_folder.changed.connect(self.check_ready)
        self.config_path.changed.connect(self.check_ready)
        self.folder_list.itemClicked.connect(self.load_subfolder)

        # Data
        self.df = pd.DataFrame(columns=["filename", "keypoint", "x", "y"])
        self.image_filenames = []  # list of image names in current folder
        self.keypoints = []
        self.keypoint_colors = {}
        self.annotation_path = None

        # Layers
        self.image_layer = self.viewer.add_image(np.zeros((1, 1, 1), dtype=np.float32),
                                                 name="image_placeholder")
        self.points_layer = self.viewer.add_points(np.empty((0, 3)),
                                                   name="annotations",
                                                   ndim=3,
                                                   size=6,
                                                   face_color="yellow",
                                                   symbol="o",
                                                   properties={"keypoint": []})
        self.cursor_layer = self.viewer.add_points(np.empty((0, 3)),
                                                   name="cursor_label",
                                                   ndim=3,
                                                   size=8,
                                                   face_color="transparent",
                                                   border_color="transparent",
                                                   text={"string": "{keypoint}",
                                                         "color": "white",
                                                         "size": 8,
                                                         "anchor": "upper_left"},
                                                   properties={"keypoint": []})
        self.skeleton_layer = self.viewer.add_shapes([np.empty((2, 3))],
                                                     ndim=3,
                                                     shape_type="line",
                                                     name="skeleton",
                                                     edge_color="white",
                                                     edge_width=2,
                                                     face_color="white")

        # Setup
        self.auto_next_frame.changed.connect(
            lambda val: self.auto_next_keypoint.set_value(False) if val else None
        )
        self.auto_next_keypoint.changed.connect(
            lambda val: self.auto_next_frame.set_value(False) if val else None
        )
        self.viewer.dims.events.current_step.connect(self._on_frame_change)
        self._bind_callbacks()
        self._bind_keys()

    def check_ready(self):
        if str(self.root_folder.value) != '.' and str(self.config_path.value) != '.':
            self.load_config()
            self.refresh_folders()

    def set_file_dialogs(self, img_folder: str | Path, config_file: str | Path):
        self.root_folder.value = Path(img_folder)
        self.config_path.value = Path(config_file)

    def load_config(self):
        self.keypoints, self.skeleton_edges = load_keypoints_and_skeleton(self.config_path.value)

        self.keypoint_dropdown.choices = self.keypoints
        if len(self.keypoints) > 0:
            self.keypoint_dropdown.set_choice(self.keypoints[0])
        self.keypoint_dropdown.enabled = True
        self.keypoint_dropdown.show()

        cmap = get_colormap("husl")
        colors = cmap.map(np.linspace(0, 1, len(self.keypoints)))
        self.keypoint_colors = {kp: colors[i] for i, kp in enumerate(self.keypoints)}

    def refresh_folders(self):
        self.folder_list.clear()
        base = self.root_folder.value
        if (base is None) or isinstance(base, tuple) or not base.is_dir():
            return
        subfolders = sorted([f.name for f in base.iterdir() if f.is_dir()])
        for folder in subfolders:
            self.folder_list.addItem(folder)

    def load_subfolder(self, item: QListWidgetItem):
        folder_name = item.text()
        base = self.root_folder.value
        if (base is None) or isinstance(base, tuple) or not base.is_dir():
            return
        folder_path = base / folder_name
        # Load images
        image_paths = sorted(glob(os.path.join(folder_path, "*.png")))
        if not image_paths:
            QMessageBox.warning(None, "No images", f"No PNGs found in {folder_name}")
            return

        stack = np.stack([rgb2gray(imread(p)) if imread(p).ndim == 3 else imread(p)
                          for p in image_paths], axis=0)
        self.image_filenames = [os.path.basename(p) for p in image_paths]
        self.image_layer.data = stack # type: ignore
        self.image_layer.name = folder_name # type: ignore
        self.viewer.reset_view()
        self.annotation_path = folder_path / "annotations.yaml"

        # ─── 1 ▸ make sure the YAML know about every *.png* FIRST ───────
        if self.annotation_path.exists():
            ensure_images_in_yaml(self.image_filenames,
                                  self.annotation_path,
                                  self.keypoints)
            # ─── 2 ▸ *now* load the tidy annotations into memory ─────────────────
            self.df = read_annotations(self.annotation_path)
        else:
            create_empty_annotations(image_paths, # type: ignore
                                     self.annotation_path,
                                     self.keypoints)
            self.df = read_annotations(self.annotation_path)
            print(f"Created new annotation sheet: {self.annotation_path}")

        conflicts = find_keypoint_conflicts(self.df, self.keypoints)
        if conflicts:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("keypoint definitions conflict between config and annotation.")
            msg.setInformativeText("Use config.yaml or annotation.yaml as reference?"
                                   " (using config as reference will overwrite annotation.yaml)")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)
            msg.button(QMessageBox.Yes).setText("Use Config")
            msg.button(QMessageBox.No).setText("Use Annotation")
            choice = msg.exec_()

            if choice == QMessageBox.Yes: # use config keypoints
                self.df = self._resolve_conflicts(self.df, self.keypoints)
                create_empty_annotations(image_paths, # type: ignore
                                         self.annotation_path,
                                         self.keypoints)
                write_annotations(self.df, self.annotation_path)
            else:
                self.viewer.close()
                print("Using annotation as-is (conflicts ignored).")

        self.refresh_points()
        self.viewer.layers.selection.active = self.image_layer

    def refresh_points(self):
        frame_idx = self.viewer.dims.current_step[0]
        if frame_idx >= len(self.image_filenames):
            return

        filename = self.image_filenames[frame_idx]
        frame_df = self.df[self.df["filename"] == filename]

        if frame_df.empty:
            self.points_layer.data = np.empty((0, 3))
            self.points_layer.properties = {"keypoint": []}
            self.points_layer.face_color = "yellow"
            self.skeleton_layer.data = [np.random.rand(2, 3)]
            return

        # (T, y, x) format: T is frame index for z-stack visualization
        coords = np.stack([np.full(len(frame_df), frame_idx), frame_df["y"], frame_df["x"]], axis=1)
        properties = {"keypoint": frame_df["keypoint"].values}
        colors = [self.keypoint_colors[bp] for bp in properties["keypoint"]]

        self.points_layer.data = coords
        self.points_layer.properties = properties
        self.points_layer.face_color = colors

        # Skeleton lines
        point_map = {row["keypoint"]: np.array([frame_idx, row["y"], row["x"]])
                     for _, row in frame_df.iterrows()}

        if len(point_map) < 2:
            self.skeleton_layer.data = [np.random.rand(2, 3)]
            return

        lines = []
        for p1, p2 in self.skeleton_edges:
            if p1 in point_map and p2 in point_map:
                line = np.stack([point_map[p1], point_map[p2]])  # shape: (2, 3)
                lines.append(line)

        if lines:
            self.skeleton_layer.data = lines
        else:
            self.skeleton_layer.data = [np.random.rand(2, 3)]

    def _bind_callbacks(self):
        self.image_layer.mouse_drag_callbacks.clear()
        self.image_layer.mouse_double_click_callbacks.clear()
        self.image_layer.mouse_drag_callbacks.append(self._on_click)
        self.image_layer.mouse_drag_callbacks.append(self._on_click_drag)

    def _on_click(self, layer, event):
        if event.type != "mouse_press" or event.button != 2:
            return  # Only handle right-clicks

        frame_idx = self.viewer.dims.current_step[0]
        filename = self.image_filenames[frame_idx]
        pos = event.position[1:]

        mask = self.df["filename"] == filename
        dist_sq = (self.df["y"] - pos[0])**2 + (self.df["x"] - pos[1])**2
        candidates = self.df[mask & (dist_sq < 36)]

        if not candidates.empty:
            deleted_index = candidates.index[0]
            deleted_keypoint = self.df.loc[deleted_index, "keypoint"]

            # Set x and y to NaN instead of dropping the row
            self.df.loc[deleted_index, ["x", "y"]] = np.nan

            self.keypoint_dropdown.value = deleted_keypoint
            self.update_cursor_label()

            write_annotations(self.df, self.annotation_path) # type: ignore
            self.refresh_points()


    def _on_click_drag(self, layer, event):
        if event.button != 1:
            return  # Only handle left-clicks

        frame_idx = self.viewer.dims.current_step[0]
        filename = self.image_filenames[frame_idx]
        keypoint = self.keypoint_dropdown.value
        start_pos = np.array(event.position[1:])

        yield  # Start interaction

        max_dist = 0
        while event.type == 'mouse_move':
            current_pos = np.array(event.position[1:])
            dist = np.linalg.norm(current_pos - start_pos)
            max_dist = max(max_dist, dist)
            yield

        if max_dist < 5:  # Treat as click, not drag
            # Remove previous annotation for same keypoint on this frame
            self.df.loc[(self.df["filename"] == filename) &
                        (self.df["keypoint"] == keypoint),
                        ["x", "y"]] = start_pos[1], start_pos[0]

            write_annotations(self.df, self.annotation_path)
            self.refresh_points()

            if self.auto_next_keypoint.value:
                idx = self.keypoints.index(keypoint)
                self.keypoint_dropdown.value = self.keypoints[(idx + 1) % len(self.keypoints)]
                self.update_cursor_label()

            if self.auto_next_frame.value:
                step = list(self.viewer.dims.current_step)
                step[0] = min(step[0] + 1, len(self.image_filenames) - 1)
                self.viewer.dims.current_step = tuple(step)
                self.refresh_points()

    def _bind_keys(self):
        @self.viewer.bind_key("d", overwrite=True)
        def _next_frame(viewer):
            step = list(viewer.dims.current_step)
            num_frames = self.image_layer.data.shape[0]
            step[0] = min(step[0] + 1, num_frames - 1)
            viewer.dims.current_step = tuple(step)
            self.viewer.layers.selection.active = self.image_layer

        @self.viewer.bind_key("a", overwrite=True)
        def _prev_frame(viewer):
            step = list(viewer.dims.current_step)
            num_frames = self.image_layer.data.shape[0]
            step[0] = max(step[0] - 1, 0)
            viewer.dims.current_step = tuple(step)
            self.viewer.layers.selection.active = self.image_layer

        @self.viewer.bind_key("w", overwrite=True)
        def _next_part(viewer):
            idx = self.keypoints.index(self.keypoint_dropdown.value)
            self.keypoint_dropdown.value = self.keypoints[(idx + 1) % len(self.keypoints)]
            self.update_cursor_label()

        @self.viewer.bind_key("s", overwrite=True)
        def _prev_part(viewer):
            idx = self.keypoints.index(self.keypoint_dropdown.value)
            self.keypoint_dropdown.value = self.keypoints[(idx - 1) % len(self.keypoints)]
            self.update_cursor_label()

        @self.viewer.mouse_move_callbacks.append
        def update_cursor_text(viewer, event):
            if not self.image_layer.visible:
                self.cursor_layer.data = np.empty((0, 3))
                return

            frame = viewer.dims.current_step[0]
            pos = event.position
            self.cursor_layer.data = np.array([[frame, pos[1], pos[2]]])
            self.cursor_layer.properties = {'keypoint': [self.keypoint_dropdown.value]}

            # Set text color to match the body part color
            color = self.keypoint_colors.get(self.keypoint_dropdown.value, [1, 1, 1, 1])
            self.cursor_layer.text = {
                'string': '{keypoint}',
                'color': color,
                'size': 8,
                'anchor': 'upper_left',
                'translation': [0, 15, 15]
            }

    def update_cursor_label(self):
        current = self.keypoint_dropdown.value
        self.cursor_layer.properties = {'keypoint': [current]}
        color = self.keypoint_colors.get(current, [1, 1, 1, 1])
        self.cursor_layer.text = {
            'string': '{keypoint}',
            'color': color,
            'size': 8,
            'anchor': 'upper_left',
            'translation': [0, 15, 15]
        }

    def _on_frame_change(self, event=None):
        self.refresh_points()


    def _resolve_conflicts(self, df: pd.DataFrame, valid_keypoints: List[str]) -> pd.DataFrame:
        """
        Keep only valid keypoints from config, and for each filename,
        ensure all config keypoints are present (fill missing with NaN).
        """
        filenames = df["filename"].unique()

        # Drop keypoints not in config
        df = df[df["keypoint"].isin(valid_keypoints)] # type: ignore

        # Collect (filename, keypoint) already in the annotation
        existing_keys = set(zip(df["filename"], df["keypoint"]))

        # Fill missing (filename, keypoint) with NaNs
        missing_rows = []
        for fname in filenames:
            for bp in valid_keypoints:
                if (fname, bp) not in existing_keys:
                    missing_rows.append({
                        "filename": fname,
                        "keypoint": bp,
                        "x": np.nan,
                        "y": np.nan
                    })

        if missing_rows:
            df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

        return df
