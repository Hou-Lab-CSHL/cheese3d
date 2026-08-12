"""Launch SLEAP's own labeling GUI (sleap-label) without the cv2/Qt crash.

Importing cv2 (a SLEAP dependency, pulled in during its own startup) sets
QT_QPA_PLATFORM_PLUGIN_PATH to cv2's bundled Qt plugins, which are built
against a different Qt ABI than the PySide/PyQt build sleap-label's own
window uses. When sleap-label's Qt application later tries to load its
"xcb" platform plugin, it finds cv2's incompatible one on that overridden
path instead of falling back to its own -- "found... but could not be
loaded" -- and Qt aborts the whole process (SIGABRT) instead of recovering.

cv2 only sets this the *first* time it's imported in a process (its own
package __init__ side effect), so importing it here first and immediately
clearing the variable protects sleap-label's own later `import cv2` too:
Python reuses the already-imported module from sys.modules without
re-running its poisoning side effect. This mirrors Cheese3D's own fix for
launching Napari (see project.py's _clear_opencv_qt_platform_plugin_override).
"""
import os
import sys


def main() -> int:
    import cv2  # noqa: F401  (import side effect is the point; see module docstring)

    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)

    from sleap.gui.app import main as sleap_label_main

    return sleap_label_main()


if __name__ == "__main__":
    sys.exit(main())
