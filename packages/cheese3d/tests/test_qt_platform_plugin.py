from cheese3d.project import _clear_opencv_qt_platform_plugin_override


def test_removes_opencv_qt_platform_plugin_override():
    """cv2's bundled, ABI-mismatched plugin path must not survive into Napari's Qt app."""
    environ = {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/data/disk2/home/tony/cheese3d/.pixi/envs/lp/lib/python3.11/site-packages/cv2/qt/plugins",
        "QT_PLUGIN_PATH": "/data/disk2/home/tony/cheese3d/.pixi/envs/lp/lib/python3.11/site-packages/cv2/qt",
    }

    _clear_opencv_qt_platform_plugin_override(environ)

    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in environ
    assert "QT_PLUGIN_PATH" not in environ


def test_noop_when_neither_variable_is_set():
    """An environment cv2 never touched (or that already cleared it) stays unaffected."""
    environ = {"SOME_OTHER_VAR": "unrelated"}

    _clear_opencv_qt_platform_plugin_override(environ)

    assert environ == {"SOME_OTHER_VAR": "unrelated"}
