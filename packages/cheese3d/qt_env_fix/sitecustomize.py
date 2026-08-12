"""Auto-imported by Python at interpreter startup (see the site module docs) --
placed on PYTHONPATH by pixi.toml's [feature.sleap.activation.env] so it
applies to every entry point (`sleap`, `sleap-label`, `sleap-track`, ...),
not just ones Cheese3D wraps explicitly.

Two environment-level fixes for SLEAP's GUI:

1. Qt binding selection. SLEAP 1.6.1's GUI is written for PySide6 -- its
   frame-loader thread (sleap/gui/widgets/video_worker.py) imports PySide6
   directly rather than going through qtpy -- but qtpy prefers PyQt5
   whenever it's importable, and this environment ends up with both
   installed. Running the GUI on PyQt5 produced a stream of PyQt5-only
   type-strictness crashes (QGraphicsScene.addRect with QRect, QRect with
   float args, QColor(..., a=...), a changedPlot signal emitted with None,
   QGraphicsView.mapToScene with floats -- each confirmed by direct
   reproduction) and, fatally, a frame loader emitting PySide6 QImages into
   a PyQt5 view whose setImage does a strict `type(image) is QImage` check
   against the PyQt5 class, so video frames never displayed at all.
   Forcing qtpy to the binding SLEAP was developed against fixes the whole
   class of problems at once. pixi.toml's [feature.sleap.activation.env]
   also sets QT_API=pyside6; the setdefault here covers invocations that
   get this file on PYTHONPATH without going through pixi activation.

2. cv2's Qt plugin path override. Importing cv2 (a SLEAP dependency) sets
   QT_QPA_PLATFORM_PLUGIN_PATH to cv2's own bundled Qt plugins, built
   against a different Qt ABI than the GUI's own Qt. Qt5-based bindings
   abort the whole process (SIGABRT) when they find the incompatible
   plugin on that overridden path; PySide6 was observed to tolerate it and
   fall back correctly (both here and for DLC's PySide6 GUI), but clearing
   the override costs nothing and protects every launch path. cv2 only
   sets it the *first* time it's imported in a process, so importing it
   here -- before any SLEAP entry point's own code runs -- and immediately
   clearing the variable protects every later `import cv2` too. This
   mirrors Cheese3D's own fix for launching Napari (see project.py's
   _clear_opencv_qt_platform_plugin_override) and cheese3d.sleap_label.
"""
import logging
import os

os.environ.setdefault("QT_API", "pyside6")

# sleap.gui.app transitively imports numexpr, whose own __init__ logs
# "NumExpr detected N cores..." at INFO level on first import, landing on
# stdout for every SLEAP entry point invocation (including headless/CLI
# ones) unless silenced.
logging.getLogger("numexpr").setLevel(logging.WARNING)

try:
    import cv2  # noqa: F401
except ImportError:
    pass
else:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)
