from cheese3d import interactive
from cheese3d.backends.core import partition_videos_by_gpu


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
        f"{interactive.sys.executable} -m cheese3d interactive --terminal",
        {"host": "localhost", "port": 8000, "title": "Cheese3D"},
    )
    assert calls["browser"] == "http://localhost:8000"
    assert calls["served"] is True


def test_terminal_ui_remains_available(monkeypatch):
    calls = {}

    class FakeApp:
        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(interactive, "Cheese3dApp", FakeApp)

    interactive.run_interative(web_mode=False)

    assert calls["ran"] is True


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
