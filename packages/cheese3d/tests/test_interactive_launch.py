from pathlib import Path

import pandas as pd
import pytest

from cheese3d import interactive
from cheese3d.backends.core import partition_videos_by_gpu
from cheese3d.interactive import (
    _build_lp_dlc_augmentation_config,
    _open_sized_pty,
    _TeeOutput,
    _TrainingProgressOutput,
    CreateWizardLoading,
    DialogBox,
    MainScreen,
    TextualStdout,
)


def test_training_completion_forces_a_repaint_not_a_bare_dialog():
    """A long worker-thread turn can leave a Textual Serve client's page stale
    even though Python-side state is already correct, appearing completely
    frozen with no way to recover except restarting the whole GUI. track()/
    triangulate() already force a repaint before showing their completion
    dialog; train_model() and generate_videos() formerly pushed a bare
    DialogBox directly instead, missing that same fix -- confirmed as the
    actual cause of a real reported freeze during video generation, which
    process-level dumps showed was NOT a Python deadlock (every thread was
    idle) but a stale, never-repainted page.
    """
    import asyncio

    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Button, TabbedContent, TabPane

    class _MinimalScreen(interactive.Screen):
        def compose(self) -> ComposeResult:
            with TabbedContent(id="all_tabs"):
                with TabPane("training"):
                    with Horizontal(id="training_buttons"):
                        yield Button("Train", id="train")
                    yield Button("Stop", id="stop_train")

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield _MinimalScreen()

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            refresh_calls = []
            screen._refresh_after_inference = lambda *a, **k: refresh_calls.append(True)

            MainScreen._complete_training_ui(screen, "Model training completed!")
            await pilot.pause()

            assert refresh_calls, "completion must force a repaint, not just push a dialog"
            dialog = app.screen
            assert isinstance(dialog, DialogBox)
            assert dialog.button_text == "Done — refresh GUI"

    asyncio.run(run())


def test_tracking_completion_forces_a_repaint_not_a_bare_dialog():
    """track()/triangulate() route through _complete_inference_ui, which must
    keep forcing the same repaint _complete_training_ui/_complete_visualization_ui
    do -- this locks that in so a future edit can't silently regress the one
    completion path (inference/tracking) that was already correct.
    """
    import asyncio

    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import Button, TabbedContent, TabPane

    class _MinimalScreen(interactive.Screen):
        def compose(self) -> ComposeResult:
            with TabbedContent(id="all_tabs"):
                with TabPane("pose"):
                    with Horizontal(id="pose_buttons"):
                        yield Button("Track", id="track")

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield _MinimalScreen()

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            refresh_calls = []
            screen._refresh_after_inference = lambda *a, **k: refresh_calls.append(True)
            screen._enable_pose_done = lambda: None

            MainScreen._complete_inference_ui(screen, "2D pose tracking completed!")
            await pilot.pause()

            assert refresh_calls, "completion must force a repaint, not just push a dialog"
            dialog = app.screen
            assert isinstance(dialog, DialogBox)
            assert dialog.button_text == "Done — refresh GUI"

    asyncio.run(run())


def test_open_sized_pty_sets_a_nonzero_window_size():
    """os.openpty() alone leaves the window size at 0x0, which silently
    disables Lightning/tqdm's live progress-bar rendering (confirmed by
    direct reproduction: a raw os.openpty()-backed PTY produced zero visible
    training-progress output, while the same run with a real window size
    produced the expected per-step tqdm bar). A plain non-tty pipe redirect
    does NOT hit this failure mode, which is why it went unnoticed.
    """
    import fcntl
    import os
    import struct
    import termios

    master_fd, slave_fd = _open_sized_pty(rows=24, columns=120)
    try:
        raw = fcntl.ioctl(slave_fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        rows, columns, _, _ = struct.unpack("HHHH", raw)
        assert (rows, columns) == (24, 120)
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_training_log_collapses_carriage_return_progress_into_one_line():
    """The training log must show/update a live progress line, not drop it.

    _TrainingProgressOutput forwards one full record per write with the \r
    separator trailing it (e.g. "Epoch 0: 50%|...|\r"), never embedded mid
    string. TextualStdout.write() used text.split("\r")[-1] to find the
    current frame, which is always "" for a trailing \r -- every subprocess
    training update was silently dropped, so the GUI's training log never
    showed anything for DLC/LP/SLEAP's PTY-subprocess training path (verified
    by direct reproduction: 0 lines written before this fix).
    """
    import asyncio

    from textual.app import App, ComposeResult

    class _FakeTerminal:
        def write(self, text):
            pass

        def flush(self):
            pass

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield TextualStdout(id="training_log")

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            log = app.query_one("#training_log")
            tee = _TeeOutput(log, _FakeTerminal())
            progress = _TrainingProgressOutput(tee)

            sample = (
                "Epoch 0:   4%|... 1/25 [00:07<02:54, 0.14it/s]\r"
                "Epoch 0:   8%|... 2/25 [00:11<02:10, 0.18it/s]\r"
                "Epoch 0:  12%|... 3/25 [00:15<01:50, 0.20it/s]\r"
            )

            def feed_all():
                for character in sample:
                    progress.feed(character)
                progress.flush()

            await asyncio.to_thread(feed_all)
            await pilot.pause()

            assert len(log.lines) == 1
            assert "12%" in str(log.lines[0])
            assert "4%" not in str(log.lines[0])

    asyncio.run(run())


def test_importing_dlc_project_into_lightning_pose_does_not_need_deeplabcut(
    monkeypatch, tmp_path
):
    """Choosing 'import' + a non-dlc backend must seed that backend, not open DLC.

    Previously the GUI's import path always called Ch3DProject.from_path with
    model_import=<path>, which unconditionally builds a DLCBackend (and thus
    imports deeplabcut) regardless of which backend the user picked in the
    model_backend selector -- crashing with ModuleNotFoundError in any Pixi
    environment (e.g. lp) that doesn't have deeplabcut installed.
    """
    pytest.importorskip("lightning_pose")

    dlc_project = tmp_path / "source" / "testmodel-tester-2026-01-01"
    label_dir = dlc_project / "labeled-data" / "cam_L"
    label_dir.mkdir(parents=True)
    image_name = "cam_L.png"
    (label_dir / image_name).write_bytes(b"image")
    columns = pd.MultiIndex.from_tuples([
        ("tester", bodypart, coord)
        for bodypart in ["nose"] for coord in ("x", "y")
    ])
    index = pd.MultiIndex.from_tuples([("labeled-data", "cam_L", image_name)])
    pd.DataFrame([[1.0, 2.0]], index=index, columns=columns).to_hdf(
        label_dir / "CollectedData_tester.h5", key="df"
    )

    project_root = tmp_path / "projects"
    project_root.mkdir()
    screen = CreateWizardLoading.__new__(CreateWizardLoading)
    screen.project_config = {
        "name": "proj", "root": str(project_root),
        "video_root": "videos", "fps": 100,
        "video_regex": {"_path_": r".*_{{type}}_{{view}}.*\.avi",
                        "type": r"[^_]+", "view": r"TL|TR|L|R|TC|BC"},
    }
    screen.ephys_config = None
    screen.model_config = {"path": str(dlc_project), "backend_type": "lightning_pose"}

    screen.create_config()

    from omegaconf import OmegaConf
    saved = OmegaConf.load(project_root / "proj" / "config.yaml")
    assert saved.model.backend_type == "lightning_pose"
    assert saved.model.backend_options.source_project_path == str(dlc_project)
    assert saved.model.backend_options.source_format == "dlc"
    csv_path = (project_root / "proj" / "model" / saved.model.name /
               "backend" / "data" / "CollectedData.csv")
    assert csv_path.is_file()


def test_create_config_uses_selected_root_not_process_cwd(tmp_path, monkeypatch):
    """A new project must be created under the picked location, not the cwd.

    Previously the location was hardcoded to ".", so a new project always
    landed wherever the served Textual process happened to be launched from,
    with no way to choose a different directory from the GUI.
    """
    (tmp_path / "somewhere-else").mkdir()
    monkeypatch.chdir(tmp_path / "somewhere-else")
    project_root = tmp_path / "chosen-location"
    project_root.mkdir()

    screen = CreateWizardLoading.__new__(CreateWizardLoading)
    screen.project_config = {
        "name": "proj", "root": str(project_root),
        "video_root": "videos", "fps": 100,
        "video_regex": {"_path_": r".*_{{type}}_{{view}}.*\.avi",
                        "type": r"[^_]+", "view": r"TL|TR|L|R|TC|BC"},
    }
    screen.ephys_config = None
    screen.model_config = {}

    screen.create_config()

    assert (project_root / "proj" / "config.yaml").is_file()
    assert not (tmp_path / "somewhere-else" / "proj").exists()


def test_lp_gui_exposes_all_supported_backbone_families():
    """The selector must include CNN, animal-pretrained, and DINO choices."""
    choices = set(interactive.LIGHTNING_POSE_BACKBONES)

    assert {"resnet18", "resnet50_animal_ap10k", "efficientnet_b2",
            "vits_dinov2", "vitb_imagenet"} <= choices
    assert len(choices) == len(interactive.LIGHTNING_POSE_BACKBONES)
    assert "vitb_sam" not in choices


def test_training_input_label_is_not_a_focus_dependent_border_title():
    """Training setting names must remain visible when their input is unfocused."""
    field = interactive.TrainingInput("Batch size", "batch_size", "8", "integer")

    assert field.field_label == "Batch size"
    assert field.input_id == "batch_size"
    assert field.input_value == "8"
    assert field.input_type == "integer"
    assert "training_field_label" in interactive.TrainingInput.compose.__doc__ or \
        "label" in interactive.TrainingInput.compose.__doc__.lower()


def test_camera_videos_are_distributed_round_robin_across_gpus(tmp_path):
    """Six camera videos should be balanced across two inference GPU workers."""
    videos = [tmp_path / f"camera_{index}.mp4" for index in range(6)]

    assignments = partition_videos_by_gpu(videos, ["0", "1"])

    assert assignments == [("0", videos[::2]), ("1", videos[1::2])]


def test_lp_dlc_augmentation_gui_defaults_match_lightning_pose_preset():
    """Editable GUI defaults must reproduce LP 2.2's built-in DLC augmentation."""
    values = {
        "rotation_probability": 0.4, "rotation_degrees": 25,
        "motion_blur_probability": 0.5, "motion_blur_kernel": 5,
        "motion_blur_angle": 90, "dropout_probability": 0.5,
        "dropout_pixel_probability": 0.02, "dropout_size_percent": 0.3,
        "dropout_per_channel_probability": 0.5, "salt_probability": 0.5,
        "pepper_probability": 0.5, "salt_pepper_pixel_probability": 0.01,
        "salt_pepper_size_min": 0.05, "salt_pepper_size_max": 0.1,
        "elastic_probability": 0.5, "elastic_alpha_min": 0,
        "elastic_alpha_max": 10, "elastic_sigma": 5,
        "histogram_probability": 0.1, "clahe_probability": 0.1,
        "emboss_probability": 0.1, "emboss_alpha_max": 0.5,
        "emboss_strength_min": 0.5, "emboss_strength_max": 1.5,
        "crop_probability": 0.4, "crop_percent": 0.15,
    }

    config = _build_lp_dlc_augmentation_config(values)

    assert config["Affine"] == {"p": 0.4, "kwargs": {"rotate": [-25.0, 25.0]}}
    assert config["MotionBlur"]["kwargs"]["k"] == 5
    assert config["CoarseDropout"]["kwargs"]["p"] == 0.02
    assert config["ElasticTransformation"]["kwargs"]["alpha"] == [0.0, 10.0]
    assert config["CropAndPad"]["kwargs"]["percent"] == [-0.15, 0.15]


def test_training_command_relaunches_selected_project_in_same_environment(tmp_path):
    """GUI training must use the active Pixi Python for either configured backend."""
    project = tmp_path / "demo2"

    assert interactive._training_command(project, gpu=2) == [
        interactive.sys.executable, "-m", "cheese3d", "--path",
        str(tmp_path), "train", "demo2", "--gpu", "2",
    ]


def test_training_command_serializes_multi_gpu_backend_settings(tmp_path):
    """The GUI must pass two GPUs and edited augmentation values to the CLI."""
    command = interactive._training_command(
        tmp_path / "demo2", gpu="0,1", settings={"epochs": 25, "imgaug": "dlc"}
    )

    assert command[-4:] == [
        "--gpu", "0,1", "--training-settings",
        '{"epochs": 25, "imgaug": "dlc"}',
    ]


def test_training_interrupt_targets_process_group_on_posix(monkeypatch):
    """Early stopping must reach backend/data-loader children, not only the CLI."""
    calls = []
    process = type("Process", (), {"pid": 4321})()
    monkeypatch.setattr(interactive.os, "name", "posix")
    monkeypatch.setattr(interactive.os, "killpg",
                        lambda pid, sig: calls.append((pid, sig)))

    interactive._interrupt_training_process(process)

    assert calls == [(4321, interactive.signal.SIGINT)]


def test_directory_picker_starts_at_home_not_repository(monkeypatch, tmp_path):
    """The web UI must expose sibling projects instead of starting in the repo."""
    monkeypatch.setattr(interactive.Path, "home", classmethod(lambda cls: tmp_path))

    picker = interactive._directory_picker("Choose a project")

    assert picker._location == tmp_path
    assert picker._title == "Choose a project"


def test_directory_picker_has_global_parent_shortcuts():
    """Backspace must remain available even when the list has lost focus."""
    bindings = {
        binding.key: binding
        for binding in interactive._Cheese3DDirectoryPicker.BINDINGS
    }

    assert bindings["backspace"].action == "parent_directory"
    assert bindings["backspace"].priority is True
    assert bindings["alt+left"].action == "parent_directory"


def test_directory_picker_includes_parent_button(monkeypatch, tmp_path):
    """Mouse users must have a visible control for returning to the parent."""
    monkeypatch.setattr(interactive.Path, "home", classmethod(lambda cls: tmp_path))
    picker = interactive._directory_picker()

    input_widgets = list(picker._input_bar())

    assert input_widgets[-1].id == "parent_directory"
    assert "Parent" in str(input_widgets[-1].label)


def test_fast_directory_navigation_avoids_per_file_stat_calls(tmp_path):
    """Listing a directory full of files must not stat() every single one.

    The upstream DirectoryNavigation calls Path.is_dir() (an uncached stat
    syscall) on every entry just to filter files out, even though pickers
    here only ever display directories -- this is the actual lag the folder
    browser complaint was about. Mounts both the original and the patched
    navigation on the same directory (5 subdirs, 500 files) and compares real
    stat() counts, proving the fix is not just "still produces the right
    directories" but genuinely does far less filesystem work to get there.
    """
    import asyncio

    from textual.app import App, ComposeResult
    from textual_fspicker.parts.directory_navigation import (
        DirectoryNavigation as _UpstreamDirectoryNavigation,
    )

    from cheese3d.interactive import _FastDirectoryNavigation

    for i in range(5):
        (tmp_path / f"dir{i}").mkdir()
    for i in range(500):
        (tmp_path / f"file{i}.txt").touch()

    stat_calls = {"count": 0}
    original_stat = Path.stat

    def counting_stat(self, *a, **k):
        stat_calls["count"] += 1
        return original_stat(self, *a, **k)

    async def mount_and_load(navigation_cls):
        class _TestApp(App):
            def compose(self) -> ComposeResult:
                # Matches SelectDirectory.on_mount(), which is what a real
                # directory picker sets; bare DirectoryNavigation defaults to
                # showing files too.
                navigation = navigation_cls(tmp_path)
                navigation.show_files = False
                yield navigation

        app = _TestApp()
        async with app.run_test() as pilot:
            navigation = app.query_one(navigation_cls)
            await pilot.pause()
            for _ in range(50):
                if navigation._entries:
                    break
                await asyncio.sleep(0.05)
            return sorted(entry.location.name for entry in navigation._entries)

    import unittest.mock
    with unittest.mock.patch.object(Path, "stat", counting_stat):
        stat_calls["count"] = 0
        slow_entries = asyncio.run(mount_and_load(_UpstreamDirectoryNavigation))
        slow_stat_calls = stat_calls["count"]

        stat_calls["count"] = 0
        fast_entries = asyncio.run(mount_and_load(_FastDirectoryNavigation))
        fast_stat_calls = stat_calls["count"]

    expected = sorted(f"dir{i}" for i in range(5))
    assert slow_entries == expected
    assert fast_entries == expected
    # The original implementation stats every one of the 500 files just to
    # discard them; scandir's cached type avoids that for the common case.
    assert slow_stat_calls >= 500
    assert fast_stat_calls < slow_stat_calls / 5


class ImmediateTimer:
    def __init__(self, _delay, callback, args):
        self.callback = callback
        self.args = args
        self.daemon = False

    def start(self):
        self.callback(*self.args)


def test_web_ui_uses_terminal_child_and_opens_browser(monkeypatch):
    calls = {}

    class FakeServer:
        def __init__(self, command, **kwargs):
            calls["server"] = (command, kwargs)

        def serve(self):
            calls["served"] = True

    monkeypatch.setattr(interactive, "Server", FakeServer)
    monkeypatch.setattr(interactive, "Timer", ImmediateTimer)
    monkeypatch.setattr(interactive.webbrowser, "open",
                        lambda url: calls.setdefault("browser", url) is not None)

    interactive.run_interative()

    assert calls["server"] == (
        f"{interactive.sys.executable} -m cheese3d --path "
        f"{interactive.Path.home()} interactive --terminal",
        {"host": "localhost", "port": 8000, "title": "Cheese3D"},
    )
    assert calls["browser"] == "http://localhost:8000"
    assert calls["served"] is True


def test_terminal_ui_remains_available(monkeypatch):
    calls = {}

    class FakeApp:
        def __init__(self, start_directory=None):
            calls["start_directory"] = start_directory

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(interactive, "Cheese3dApp", FakeApp)

    interactive.run_interative(web_mode=False)

    assert calls["ran"] is True
    assert calls["start_directory"] == interactive.Path.home()


def test_web_ui_propagates_explicit_project_root_to_terminal_child(monkeypatch, tmp_path):
    """The served child must not lose --path and reopen at an unrelated directory."""
    calls = {}

    class FakeServer:
        def __init__(self, command, **_kwargs):
            calls["command"] = command

        def serve(self):
            pass

    monkeypatch.setattr(interactive, "Server", FakeServer)
    interactive.run_interative(
        start_directory=tmp_path, open_browser=False,
    )

    assert calls["command"] == (
        f"{interactive.sys.executable} -m cheese3d --path {tmp_path} "
        "interactive --terminal"
    )


def test_persist_visualization_threshold_on_config_missing_visualization_section(tmp_path):
    """Projects saved before the 'visualization' config field existed must not crash."""
    from omegaconf import OmegaConf

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"name": "demo1"}), config_path)

    interactive._persist_visualization_threshold(tmp_path, 0.42)

    saved = OmegaConf.load(config_path)
    assert saved.visualization.keypoint_probability_threshold == 0.42


def test_project_wizard_mounts_without_error(tmp_path):
    """ProjectWizard must actually mount through Textual's real lifecycle.

    Previously self.location was created in on_mount() but referenced by
    compose(), which Textual always calls first -- every earlier test here
    only exercised widget logic via __new__, bypassing compose()/on_mount()
    entirely, so none of them caught this AttributeError before it shipped.
    """
    import asyncio

    from textual.app import App, ComposeResult

    from cheese3d.interactive import ProjectWizard

    class _TestApp(App):
        start_directory = tmp_path

        def compose(self) -> ComposeResult:
            yield ProjectWizard()

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            wizard = app.query_one(ProjectWizard)
            assert wizard.location.value == str(tmp_path)

    asyncio.run(run())


def test_start_menu_loads_typed_path_without_requiring_browse(monkeypatch, tmp_path):
    """Typing a project path directly must work, not just browsing to it.

    Also covers the same compose()/on_mount() ordering hazard as
    ProjectWizard: load_path is referenced in compose() and only its default
    value is set in on_mount(). StartMenu is a Screen, so it must be pushed
    via push_screen (matching real usage in Cheese3dApp.on_mount), not
    composed as a plain child widget.
    """
    import asyncio

    from textual.app import App
    from textual.screen import Screen

    from cheese3d.interactive import StartMenu

    project_path = tmp_path / "existing-project"
    project_path.mkdir()
    (project_path / "config.yaml").touch()

    opened = []

    def _fake_main_screen(path):
        opened.append(path)
        return Screen()

    monkeypatch.setattr(interactive, "MainScreen", _fake_main_screen)

    class _TestApp(App):
        start_directory = tmp_path

        def on_mount(self) -> None:
            self.push_screen(StartMenu())

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            menu = app.screen
            assert isinstance(menu, StartMenu)
            assert menu.load_path.value == str(tmp_path)
            menu.load_path.value = str(project_path)
            menu.load_project()
            await pilot.pause()

    asyncio.run(run())
    assert opened == [project_path]


def test_start_menu_shows_friendly_error_for_invalid_typed_path(tmp_path):
    """A typed path with no config.yaml must show a message, not crash."""
    import asyncio

    from textual.app import App

    from cheese3d.interactive import DialogBox, StartMenu

    class _TestApp(App):
        start_directory = tmp_path

        def on_mount(self) -> None:
            self.push_screen(StartMenu())

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            menu = app.screen
            assert isinstance(menu, StartMenu)
            menu.load_path.value = str(tmp_path / "not-a-project")
            menu.load_project()
            await pilot.pause()
            assert isinstance(app.screen, DialogBox)

    asyncio.run(run())


def test_model_wizard_locks_backend_to_the_installed_pose_package(monkeypatch):
    """The backend selector must match this environment and not be editable.

    Previously it was a free-form dropdown that always defaulted to "dlc"
    regardless of which pose package is actually installed, and doubled as
    the "what format is the imported source" choice. Picking "dlc" to mean
    the latter while running lp/lp-cu13/sleap actually tried to build a
    DLCBackend, crashing with ModuleNotFoundError: No module named
    'deeplabcut'. The backend is now locked to the active environment; a
    separate model_source_format selector answers the source-format question.
    """
    import asyncio

    from textual.app import App, ComposeResult
    from textual.widgets import Select

    from cheese3d import interactive
    from cheese3d.interactive import ModelWizard

    monkeypatch.setattr(interactive, "active_pose_backend", lambda: "lightning_pose")

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield ModelWizard()

    async def run():
        app = _TestApp()
        async with app.run_test():
            wizard = app.query_one(ModelWizard)
            backend_select = wizard.query_one("#model_backend", Select)
            assert backend_select.value == "lightning_pose"
            assert backend_select.disabled is True

    asyncio.run(run())


def test_create_wizard_loading_shows_dialog_instead_of_crashing_on_failure(tmp_path):
    """A failed project creation must show a message, not crash the whole app.

    Previously create_config()'s exceptions (e.g. a mismatched pose backend
    needing a package this Pixi environment doesn't have) were unhandled,
    crashing the entire Textual app instead of returning to the start menu.
    """
    import asyncio

    from textual.app import App

    from cheese3d.interactive import CreateWizardLoading

    class _TestApp(App):
        def on_mount(self) -> None:
            self.push_screen(CreateWizardLoading(
                {"name": "proj", "root": "/nonexistent/-definitely-not-here-",
                 "video_root": "videos", "fps": 100,
                 "video_regex": {"_path_": r".*_{{type}}_{{view}}.*\.avi",
                                 "type": r"[^_]+", "view": r"TL|TR|L|R|TC|BC"}},
                None, {},
            ))

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CreateWizardLoading)
            assert screen._failed is True

    asyncio.run(run())


def test_model_wizard_get_config_reflects_chosen_source_format_after_import_selected(
    monkeypatch, tmp_path
):
    """Selecting a source format after choosing 'import' must be preserved,
    independent of the (locked) active backend.

    Mounted and driven exactly as a user would: select "import" (which no
    longer force-opens a directory picker -- the path can be typed directly),
    type a path, then change the source-format selector afterward. Earlier
    unit tests only synthesized model_config by hand and never exercised this
    real widget interaction sequence, which is exactly the class of gap that
    let the previous ProjectWizard mount-order bug ship unnoticed.
    """
    import asyncio

    from textual.app import App, ComposeResult
    from textual.widgets import Select

    from cheese3d import interactive
    from cheese3d.interactive import LabeledInput, ModelWizard

    monkeypatch.setattr(interactive, "active_pose_backend", lambda: "lightning_pose")

    sleap_path = tmp_path / "some-sleap-project"
    sleap_path.mkdir()

    class _TestApp(App):
        def compose(self) -> ComposeResult:
            yield ModelWizard()

    async def run():
        app = _TestApp()
        async with app.run_test() as pilot:
            wizard = app.query_one(ModelWizard)
            wizard.query_one("#model_mode", Select).value = "import"
            await pilot.pause()
            assert not wizard.name_or_path.disabled, \
                "the path field must stay typeable, not force browsing only"
            assert not wizard.query_one("#model_source_format", Select).disabled, \
                "the source-format selector must become editable in import mode"
            wizard.query_one("#name_or_path", LabeledInput).value = str(sleap_path)
            wizard.query_one("#model_source_format", Select).value = "sleap"
            await pilot.pause()

            config = wizard.get_config()
            assert config["backend_type"] == "lightning_pose"
            assert config["source_format"] == "sleap"
            assert config["path"] == str(sleap_path)

    asyncio.run(run())


def test_pipeline_output_mirrors_gui_and_terminal(monkeypatch):
    gui = Buffer()
    terminal = Buffer()
    monkeypatch.setattr(interactive.os, "name", "not-posix")
    monkeypatch.setattr(interactive.sys, "__stderr__", terminal)

    with interactive._pipeline_output(gui):
        print("pipeline progress")

    assert "pipeline progress" in gui.value
    assert "pipeline progress" in terminal.value


class Buffer:
    def __init__(self):
        self.value = ""

    def write(self, text):
        self.value += text
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False
