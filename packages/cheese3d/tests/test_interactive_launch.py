from cheese3d import interactive


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
        "cheese3d interactive --terminal",
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
