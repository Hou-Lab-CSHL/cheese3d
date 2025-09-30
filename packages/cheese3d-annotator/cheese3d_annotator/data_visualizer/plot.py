#!/usr/bin/env python3
"""
plot.py — Cheese3D features plotter (Qt/PyQtGraph), with column toggles and frame cursor

Usage (standalone):
    python plot.py /path/to/cheese3d_features.csv
    # optional: start with just a subset of columns checked
    python plot.py /path/to/cheese3d_features.csv --cols yaw pitch roll
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qtpy import QtCore, QtWidgets
import pyqtgraph as pg

# ─────────────────────────────────────────────── Feature colors (RGBA 0–1 → 0–255)
_FEATURE_COLORS = {
    'mouth-area':           (197, 84, 63, 255),
    'cheek-bulge-volume':   (22, 156, 104, 255),
    'ear-width-left':       (198, 98, 8, 255),
    'ear-width-right':      (198, 98, 8, 255),
    'eye-height-left':      (204, 118, 179, 255),
    'eye-height-right':     (204, 118, 179, 255),
    'eye-area-left':        (202, 144, 102, 255),
    'eye-area-right':       (202, 144, 102, 255),
    'eye-width-left':       (249, 174, 223, 255),
    'eye-width-right':      (249, 174, 223, 255),
    'ear-height-left':      (149, 148, 149, 255),
    'ear-height-right':     (149, 148, 149, 255),
    'ear-area-left':        (234, 224, 53, 255),
    'ear-area-right':       (234, 224, 53, 255),
    'nose-bulge-volume':    (86, 180, 233, 255),
    'ear-angle-left':       (146, 114, 37, 255),
    'ear-angle-right':      (146, 114, 37, 255),
}

# ─────────────────────────────────────────────── PyQtGraph compatibility patch
try:
    from pyqtgraph.widgets.PlotWidget import PlotWidget as _PGPlotWidget
    if not hasattr(_PGPlotWidget, "autoRangeEnabled"):
        def _autoRangeEnabled(self):  # noqa: ANN001
            vb = self.getViewBox()
            if hasattr(vb, "autoRangeEnabled"):
                return vb.autoRangeEnabled()
            return (False, False)
        _PGPlotWidget.autoRangeEnabled = _autoRangeEnabled
except Exception:
    pass

_NUMERIC_DTYPE_KINDS = set("fc")
_EXCLUDE_COLS = {"Unnamed: 0", "index", "Index"}


def _is_numeric_series(s: pd.Series) -> bool:
    if s.dtype.kind in _NUMERIC_DTYPE_KINDS or np.issubdtype(s.dtype, np.integer):
        return True
    try:
        pd.to_numeric(s, errors="raise")
        return True
    except Exception:
        return False


class FeaturesPlotWidget(QtWidgets.QWidget):
    currentFrameChanged = QtCore.Signal(int)

    def __init__(self, csv_path: Path | str, cols: Optional[Sequence[str]] = None,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("FeaturesPlotWidget")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._csv_path = Path(csv_path)
        if not self._csv_path.exists():
            raise FileNotFoundError(self._csv_path)

        self.df = pd.read_csv(self._csv_path)
        self._x_name, self.x = self._choose_x(self.df)
        self.numeric_cols = self._list_numeric_columns(self.df, exclude=self._x_name)
        if len(self.x) != len(self.df):
            self._x_name, self.x = "index", np.arange(len(self.df), dtype=float)

        self._initial_cols = set(map(str, cols)) if cols else None

        self._build_ui()
        self._populate_columns()
        self._attach_plot()
        self._replot()

    def _choose_x(self, df: pd.DataFrame) -> Tuple[str, np.ndarray]:
        for name in df.columns:
            lname = str(name).strip().lower()
            if lname in {c.lower() for c in _EXCLUDE_COLS}:
                continue
            if lname in ("frame", "frames", "time", "t", "sec", "seconds"):
                return name, pd.to_numeric(df[name], errors="coerce").to_numpy()
        return "index", np.arange(len(df), dtype=float)

    def _list_numeric_columns(self, df: pd.DataFrame, exclude: Optional[str]) -> list[str]:
        return [
            str(c) for c in df.columns
            if c != exclude and c not in _EXCLUDE_COLS and _is_numeric_series(df[c])
        ]

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel(str(self._csv_path))
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.path_label, 1)
        self.xaxis_label = QtWidgets.QLabel(f"X: {self._x_name}")
        top.addWidget(self.xaxis_label, 0, QtCore.Qt.AlignRight)
        root.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(split, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter columns (substring or regex)")
        left_layout.addWidget(self.filter_edit)

        self.col_list = QtWidgets.QListWidget()
        self.col_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.col_list.setAlternatingRowColors(False)
        left_layout.addWidget(self.col_list, 1)

        btns = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("All")
        self.btn_none = QtWidgets.QPushButton("None")
        btns.addWidget(self.btn_all)
        btns.addWidget(self.btn_none)
        left_layout.addLayout(btns)

        frm_box = QtWidgets.QGroupBox("Navigation")
        frm_lay = QtWidgets.QVBoxLayout(frm_box)
        nav_row = QtWidgets.QHBoxLayout()
        nav_row.addWidget(QtWidgets.QLabel("Frame:"))
        self.spn_frame = QtWidgets.QSpinBox()
        self.spn_frame.setRange(0, max(0, len(self.x) - 1))
        nav_row.addWidget(self.spn_frame, 1)
        frm_lay.addLayout(nav_row)
        self.sld_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_frame.setRange(0, max(0, len(self.x) - 1))
        frm_lay.addWidget(self.sld_frame)
        left_layout.addWidget(frm_box)
        left_layout.addStretch(1)

        split.addWidget(left)

        self.plot = pg.PlotWidget(background="k")
        split.addWidget(self.plot)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        self._pi: pg.PlotItem = self.plot.getPlotItem()
        self._vb: pg.ViewBox = self._pi.getViewBox()

        self._pi.addLegend()
        self._pi.showGrid(x=True, y=True, alpha=0.2)
        self._vb.setMouseEnabled(x=True, y=False)
        self._pi.setMenuEnabled(False)

        self.vline = pg.InfiniteLine(angle=90, movable=True,
                                     pen=pg.mkPen((200, 200, 255), width=1))
        self._pi.addItem(self.vline, ignoreBounds=True)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.btn_all.clicked.connect(lambda: self._set_all_checked(True))
        self.btn_none.clicked.connect(lambda: self._set_all_checked(False))
        self.col_list.itemChanged.connect(self._replot)

        self.spn_frame.valueChanged.connect(self._on_frame_spin)
        self.sld_frame.valueChanged.connect(self._on_frame_slider)
        self.vline.sigPositionChanged.connect(self._on_vline_moved)
        self._pi.scene().sigMouseClicked.connect(self._on_plot_clicked)

    def _populate_columns(self):
        self.col_list.blockSignals(True)
        self.col_list.clear()
        for name in self.numeric_cols:
            it = QtWidgets.QListWidgetItem(name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            checked = (
                (self._initial_cols is None) or (name in self._initial_cols)
            )
            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.col_list.addItem(it)
        self.col_list.blockSignals(False)

    def _iter_checked_columns(self) -> Iterable[str]:
        for i in range(self.col_list.count()):
            it = self.col_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                yield it.text()

    def _apply_filter(self):
        pattern = self.filter_edit.text().strip()
        for i in range(self.col_list.count()):
            it = self.col_list.item(i)
            name = it.text()
            show = True
            if pattern:
                try:
                    show = QtCore.QRegularExpression(pattern).match(name).hasMatch()
                except Exception:
                    show = pattern.lower() in name.lower()
            it.setHidden(not show)

    def _attach_plot(self):
        pg.setConfigOptions(antialias=False)

    def _replot(self):
        self._pi.clear()
        self._pi.addLegend()
        self._pi.showGrid(x=True, y=True, alpha=0.2)
        self._pi.addItem(self.vline, ignoreBounds=True)

        cols = list(self._iter_checked_columns())
        if not cols:
            self._pi.setTitle("No columns selected")
            return

        self._pi.setTitle(f"X: {self._x_name}   |   {len(cols)} signal(s)")
        x = self.x.astype(float)

        for idx, c in enumerate(cols):
            y = pd.to_numeric(self.df[c], errors="coerce").to_numpy(dtype=float)
            color = _FEATURE_COLORS.get("-".join(c.split("-")[:-1]), pg.intColor(idx))
            item = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2), name=c)
            if hasattr(item, "setClipToView"):
                item.setClipToView(True)
            if hasattr(item, "setDownsampling"):
                try:
                    item.setDownsampling(auto=True, method='peak')
                except TypeError:
                    try:
                        item.setDownsampling(auto=True)
                    except Exception:
                        pass
            elif hasattr(item, "setAutoDownsample"):
                try:
                    item.setAutoDownsample(True)
                except Exception:
                    pass
            self._pi.addItem(item)

        if hasattr(self._pi, "enableAutoRange"):
            try:
                self._pi.enableAutoRange(y=True)
            except Exception:
                pass
        self._vb.setMouseEnabled(x=True, y=False)
        self._set_vline_x_from_index(int(self.spn_frame.value()))

    def _index_to_x(self, idx: int) -> float:
        idx = int(np.clip(idx, 0, max(0, len(self.x) - 1)))
        xv = self.x[idx]
        if not np.isfinite(xv):
            finite = np.where(np.isfinite(self.x))[0]
            if finite.size == 0:
                return float(idx)
            near = finite[np.argmin(np.abs(finite - idx))]
            xv = self.x[near]
        return float(xv)

    def _x_to_index(self, xv: float) -> int:
        x = self.x
        i = int(np.searchsorted(x, xv))
        if i <= 0:
            return 0
        if i >= len(x):
            return len(x) - 1
        return i if abs(x[i] - xv) < abs(xv - x[i - 1]) else i - 1

    def _set_vline_x_from_index(self, idx: int):
        xv = self._index_to_x(idx)
        try:
            self.vline.blockSignals(True)
            self.vline.setPos(xv)
        finally:
            self.vline.blockSignals(False)

    def _on_frame_spin(self, idx: int):
        self.sld_frame.blockSignals(True)
        self.sld_frame.setValue(idx)
        self.sld_frame.blockSignals(False)
        self._set_vline_x_from_index(idx)
        self.currentFrameChanged.emit(idx)

    def _on_frame_slider(self, idx: int):
        self.spn_frame.blockSignals(True)
        self.spn_frame.setValue(idx)
        self.spn_frame.blockSignals(False)
        self._set_vline_x_from_index(idx)
        self.currentFrameChanged.emit(idx)

    def _on_vline_moved(self):
        xv = float(self.vline.value())
        idx = self._x_to_index(xv)
        self.spn_frame.blockSignals(True)
        self.sld_frame.blockSignals(True)
        try:
            self.spn_frame.setValue(idx)
            self.sld_frame.setValue(idx)
        finally:
            self.spn_frame.blockSignals(False)
            self.sld_frame.blockSignals(False)
        self.currentFrameChanged.emit(idx)

    def _on_plot_clicked(self, evt):
        if evt.button() != QtCore.Qt.LeftButton:
            return
        if not self._vb.sceneBoundingRect().contains(evt.scenePos()):
            return
        mouse_point = self._vb.mapSceneToView(evt.scenePos())
        idx = self._x_to_index(float(mouse_point.x()))
        self.goto_frame(idx, center=False)

    def goto_frame(self, idx: int, center: bool = False):
        idx = int(np.clip(idx, 0, max(0, len(self.x) - 1)))
        self.spn_frame.setValue(idx)
        if center:
            xv = self._index_to_x(idx)
            xr = self._vb.viewRange()[0]
            halfw = 0.5 * (xr[1] - xr[0])
            self._vb.setXRange(xv - halfw, xv + halfw, padding=0.0)

    def current_frame(self) -> int:
        return int(self.spn_frame.value())

    def keyPressEvent(self, e):
        key = e.key()
        mod = e.modifiers()
        step = 10 if (mod & QtCore.Qt.ShiftModifier) else 1
        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self.goto_frame(self.current_frame() + step)
            e.accept(); return
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self.goto_frame(self.current_frame() - step)
            e.accept(); return
        if key == QtCore.Qt.Key_Home:
            self.goto_frame(0, center=True); e.accept(); return
        if key == QtCore.Qt.Key_End:
            self.goto_frame(len(self.x) - 1, center=True); e.accept(); return
        super().keyPressEvent(e)

    def _set_all_checked(self, checked: bool):
        self.col_list.blockSignals(True)
        for i in range(self.col_list.count()):
            it = self.col_list.item(i)
            if not it.isHidden():
                it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self.col_list.blockSignals(False)
        self._replot()


def _main(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description="Plot Cheese3D features CSV.")
    ap.add_argument("csv", type=str, help="Path to cheese3d_features.csv")
    ap.add_argument("--cols", nargs="*", default=None,
                    help="Optional subset of column names initially checked (default: all numeric).")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FeaturesPlotWidget(args.csv, cols=args.cols)
    w.resize(1200, 700)
    w.setWindowTitle("Cheese3D Features")
    w.show()
    app.exec_()


if __name__ == "__main__":
    _main()
