from omegaconf import OmegaConf
from typing import Callable, Optional, List, Tuple
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from threading import Timer
import os
import json
import signal
import shlex
import subprocess
import sys
import traceback
import webbrowser
from textual import work, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual_serve.server import Server
from textual.reactive import reactive
from textual.containers import (Horizontal,
                                HorizontalGroup,
                                HorizontalScroll,
                                Vertical,
                                VerticalGroup,
                                VerticalScroll,
                                CenterMiddle)
from textual.widgets import (Checkbox,
                             Footer,
                             Header,
                             Button,
                             Input,
                             Static,
                             Select,
                             Collapsible,
                             LoadingIndicator,
                             TabbedContent,
                             TabPane,
                             RichLog,
                             SelectionList,
                             OptionList)
from textual_fspicker import SelectDirectory
from textual_fspicker.parts import DirectoryNavigation

from cheese3d.config import _DEFAULT_VIDEO_REGEX
from cheese3d.utils import maybe, reglob
from cheese3d.project import Ch3DProject, RecordingKey
from cheese3d.config import ProjectConfig, ModelConfig
from cheese3d.backends.dlc import DLC3_PYTORCH_MODELS


def _training_command(project_path: str | Path, gpu: str = "0",
                      settings: Optional[dict] = None) -> List[str]:
    """Build the same-backend Cheese3D CLI command used by GUI training."""
    project_path = Path(project_path)
    command = [sys.executable, "-m", "cheese3d", "--path",
               str(project_path.parent), "train", project_path.name, "--gpu", str(gpu)]
    if settings:
        command.extend(["--training-settings", json.dumps(settings)])
    return command


def _interrupt_training_process(process: subprocess.Popen) -> None:
    """Interrupt a trainer and all of its data-loader/backend descendants."""
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.send_signal(signal.SIGINT)


class _Cheese3DDirectoryPicker(SelectDirectory):
    """Directory picker whose parent shortcut works regardless of focus.

    ``textual-fspicker`` already handles Backspace inside its directory list,
    but that binding becomes inactive after a button or another control receives
    focus. Priority screen bindings make Backspace and Alt+Left reliable across
    the entire dialog.
    """

    BINDINGS = [
        Binding("backspace", "parent_directory", "Parent", priority=True),
        Binding("alt+left", "parent_directory", "Parent", priority=True),
    ]

    def _input_bar(self) -> ComposeResult:
        """Add a physical parent button beside the current-directory display."""
        # Keep the dependency's existing current-directory display rather than
        # replacing it; the extra control is appended to the same input bar.
        yield from super()._input_bar()
        yield Button("↑ Parent", id="parent_directory", variant="primary")

    def action_parent_directory(self) -> None:
        """Move to the current directory's parent without leaving the picker."""
        navigation = self.query_one(DirectoryNavigation)
        if not navigation.is_root:
            navigation.location = navigation.location.parent
        # Restore list focus so arrows and Enter work immediately after going up.
        navigation.focus()

    @on(Button.Pressed, "#parent_directory")
    def press_parent_directory(self, event: Button.Pressed) -> None:
        """Handle the visible Parent button using the keyboard action's logic."""
        event.stop()
        self.action_parent_directory()


def _directory_picker(title: str = "Select directory") -> SelectDirectory:
    """Create an unrestricted picker starting outside the repository checkout.

    The Textual web server inherits the repository as its working directory, so
    ``SelectDirectory()`` used to open inside Cheese3D on every invocation. The
    user's home directory is a more useful starting point and the picker's ``..``
    entry still permits navigation all the way to the filesystem root.
    """
    # Former behavior is preserved here for context; it made the repository the
    # initial location whenever Cheese3D was launched from its source checkout.
    # return SelectDirectory()
    return _Cheese3DDirectoryPicker(location=Path.home(), title=title)

_REGEX_HELP_MSG = """
A utility to help with building named grouped
[link='https://www.regular-expressions.info/quickstart.html']regex strings[/link].

Either:
- Enter a proper regex string under the 'full string' field
- Enter a pseudo-regex string with named placeholders like 'my_{{string}}_here.avi'
    - Use the '+' and '-' buttons to define how to match each placeholder
        (e.g. set 'field name' to 'string' and 'field value' to '[0-9]+' to match digits)
- Hints:
    - '[0-9]' matches any digit
    - '[a-z]' / '[A-Z]' matches all lower / upper case letters
    - '.' matches any character
    - Put '*' after a match to match 0 or more instances (e.g. '.*')
    - Put '+' after a match to match 1 or more instances (e.g. '[0-9]+')
    - Put '{n}' after a match to match exactly n instances (e.g. '[a-z,A-Z]{3}')
"""

_MAIN_HELP_MSG = """
Use the tabs to navigate through the Cheese3D pipeline (typically from left to right).
Project is live-loaded from disk whenever you switch tabs (so you can edit your config file).

Tab info:
- [bold]"summary":[/bold] an overview of your project including detected videos (and ephys)
- [bold]"select sessions":[/bold] select video/ephys recordings to include in project
- [bold]"model":[/bold] model-related actions like labeling frames and training
- [bold]"pose tracking":[/bold] analysis-related actions like camera
    calibration, keypoint tracking, and triangulation
- [bold]"visualization":[/bold] generate quality-control videos and
    launch data visualizer tool
"""

_EXTRACTING_POPUP = """
[bold]Extracting frames (may take a few seconds to load) ...
close napari to dismiss.[/bold]
"""

_LABELING_POPUP = """
[bold]Labeling frames (may take a few seconds to load) ...
close napari to dismiss.[/bold]
"""

_VISUALIZATION_POPUP = """
[bold]Visualizing results (may take a few seconds to load) ...
close napari to dismiss.[/bold]
"""

class RichConsole(RichLog):
    def print(self, *args, **kwargs):
        self.write(*args, **kwargs)

class TextualStdout(RichLog):
    """Custom stdout-like object that writes to a RichLog widget and handles progress bars."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_line = ""
        self.last_was_progress = False

    def write(self, text: str) -> int:
        # Handle carriage return (progress bar updates)
        if "\r" in text and not text.endswith("\n"):
            # Extract the progress line (last part after \r)
            progress_text = text.split("\r")[-1].strip()
            if progress_text:
                if self.last_was_progress:
                    # Update last line by clearing and rewriting
                    self.app.call_from_thread(self._update_progress_line, progress_text)
                else:
                    # First progress line
                    self.app.call_from_thread(super().write, progress_text)
                self.last_was_progress = True
        elif text.strip():
            # Regular line - write normally
            self.app.call_from_thread(super().write, text.rstrip())
            self.last_was_progress = False

        return len(text)

    def _update_progress_line(self, new_text: str):
        """Update the last line for progress bars."""
        # remove last line
        if self.lines:
            self.lines.pop()
        # clear cache of old line
        y = len(self.lines)
        scroll_x, _ = self.scroll_offset
        width = self.scrollable_content_region.width
        key = (y + self._start_line, scroll_x, width, self._widest_line_width)
        self._line_cache.discard(key)
        # write new line
        super().write(new_text)
        # refresh just this line
        self.refresh_line(y)

    def flush(self):
        pass  # RichLog handles its own flushing

    def close(self):
        pass # no need to "close" this output stream


class _TeeOutput:
    """Write pipeline output to both a GUI log and the launching terminal."""

    def __init__(self, gui_log, terminal):
        self.gui_log = gui_log
        self.terminal = terminal

    def write(self, text: str) -> int:
        self.gui_log.write(text)
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


@contextmanager
def _pipeline_output(gui_log):
    """Mirror redirected worker output to the original terminal when possible."""
    terminal = None
    try:
        if os.name == "posix":
            # Textual Serve captures the child process's stdout/stderr. The
            # controlling TTY still refers to the terminal that launched it.
            terminal = open("/dev/tty", "w", buffering=1)
        else:
            terminal = sys.__stderr__
    except OSError:
        terminal = sys.__stderr__

    tee = _TeeOutput(gui_log, terminal)
    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            yield
    finally:
        if terminal not in (sys.__stdout__, sys.__stderr__):
            terminal.close()

class LabeledInput(Input):

    label = reactive("")

    def __init__(self, label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def watch_label(self, label):
        self.border_title = label


class TrainingInput(VerticalGroup):
    """Training input with a permanently visible label above the field."""

    def __init__(self, label: str, input_id: str, value: str,
                 input_type: str = "text"):
        super().__init__(classes="training_field")
        self.field_label = label
        self.input_id = input_id
        self.input_value = value
        self.input_type = input_type

    def compose(self) -> ComposeResult:
        """Render the label independently of the input's focus state."""
        yield Static(self.field_label, classes="training_field_label")
        yield Input(value=self.input_value, id=self.input_id, type=self.input_type)


class KeyValuePair(VerticalGroup):

    pair = reactive(("", ""))

    def __init__(self, name: Optional[str] = None, value: Optional[str] = None):
        super().__init__()
        self.name_input = LabeledInput(label="field name", value=name, placeholder="Name of field")
        self.value_input = LabeledInput(label="field value", value=value, placeholder="Value of field")

    def compose(self) -> ComposeResult:
        yield self.name_input
        yield self.value_input

    def compute_pair(self):
        return self.name_input.value, self.value_input.value

class RegexInput(VerticalScroll):

    disable_remove = reactive(False)

    class Ready(Message):
        def __init__(self, ready: bool):
            super().__init__()
            self.ready = ready

    def __init__(self, label: str,
                 regex: Optional[str],
                 required: Optional[List[str]],
                 **fields):
        super().__init__()
        self.label = label
        self.init_regex = regex
        self.init_fields = fields
        self.required = maybe(required, [])

    def on_mount(self) -> None:
        self.border_title = self.label
        for name in self.required:
            self.fields.mount(KeyValuePair(name, self.init_fields[name]))
        for name, value in self.init_fields.items():
            if name in self.required:
                continue
            self.fields.mount(KeyValuePair(name, value))
        if len(self.fields.children) <= len(self.required):
            self.disable_remove = True

    def compose(self) -> ComposeResult:
        with Collapsible(classes="helpmenu", title="help", collapsed=True):
            yield Static(_REGEX_HELP_MSG)
        with Collapsible(classes="helpmenu", title="expand controls", collapsed=True):
            yield LabeledInput(label="full string", id="path", value=self.init_regex, placeholder="Full regex string")
            self.fields = HorizontalScroll()
            with Horizontal():
                with Vertical(id="buttons"):
                    yield Button("+", id="add_field", variant="success")
                    yield Button("-", id="remove_field", variant="error")
                yield self.fields

    def watch_disable_remove(self, disable: bool):
        button = self.query_one("#remove_field")
        button.disabled = disable

    @on(LabeledInput.Changed)
    def check_ready(self):
        def _check_kv(k, v):
            return (maybe(k, "") != "") and (maybe(v, "") != "")
        ready = all(_check_kv(k, v) for k, v in self.get_regex().items())
        self.post_message(RegexInput.Ready(ready))

    @on(Button.Pressed, "#add_field")
    def add_field(self):
        nchildren = len(self.fields.children)
        self.fields.mount(KeyValuePair())
        self.disable_remove = ((nchildren + 1) <= len(self.required))

    @on(Button.Pressed, "#remove_field")
    def remove_field(self):
        nchildren = len(self.fields.children)
        if nchildren > 1:
            self.fields.children[-1].remove()
            nchildren -= 1
        self.disable_remove = (nchildren <= len(self.required))

    def get_regex(self):
        return {
            "_path_": self.query_one("#path").value,
            **dict(kv.pair for kv in self.fields.children)
        }

class DialogBox(ModalScreen):
    def __init__(self, message: str = "Completed step", button_text: str = "Continue", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message
        self.button_text = button_text

    @on(Button.Pressed, "#continue")
    def close(self):
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static(f"[bold]{self.message}[/bold]", id="msg"))
            yield Horizontal(Button(self.button_text, id="continue", variant="success"))

class ChoiceBox(ModalScreen):
    def __init__(self, message: str, button_text: Tuple[str, str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message
        self.button_text = button_text

    @on(Button.Pressed, "#choice_a")
    def close_a(self):
        self.dismiss(0)

    @on(Button.Pressed, "#choice_b")
    def close_b(self):
        self.dismiss(1)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static(f"[bold]{self.message}[/bold]", id="msg"))
            with Horizontal():
                yield Horizontal(Button(self.button_text[0], id="choice_a", variant="primary"))
                yield Horizontal(Button(self.button_text[1], id="choice_b", variant="primary"))

class SelectionBox(ModalScreen):
    def __init__(self, message: str, options: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message
        self.options = options

    @on(OptionList.OptionSelected, "#recording_list")
    def close(self, msg: OptionList.OptionMessage):
        self.dismiss(msg.option.prompt)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static(f"[bold]{self.message}[/bold]", id="msg"))
            yield Horizontal(OptionList(*self.options, id="recording_list"))

class BlockScreen(ModalScreen):
    def __init__(self, launch_callback: Callable, message: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.launch_callback = launch_callback
        self.message = message

    def on_show(self) -> None:
        self.launch_callback()
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static(self.message))

class ProjectWizard(VerticalGroup):

    labels_ready = reactive(False)
    regex_ready = reactive(True)

    class Ready(Message):
        def __init__(self, ready: bool):
            super().__init__()
            self.ready = ready

    def on_mount(self) -> None:
        self.border_title = "project info"

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="project name", id="project_name", placeholder="Project name")
        yield LabeledInput(label="video dir", id="video_dir", value="videos", placeholder="Video recordings sub-directory")
        yield LabeledInput(label="fps", id="fps", value="100", type="integer", placeholder="Frames per second")
        yield RegexInput(label="video regex",
                         regex=_DEFAULT_VIDEO_REGEX["_path_"],
                         required=["type", "view"],
                         **{k: v for k, v in _DEFAULT_VIDEO_REGEX.items() if k != "_path_"})

    @on(RegexInput.Ready)
    def set_regex_ready(self, msg: RegexInput.Ready) -> None:
        self.regex_ready = msg.ready

    @on(LabeledInput.Changed, "#project_name, #video_dir, #fps")
    def set_labels_ready(self) -> None:
        name = maybe(self.query_one("#project_name").value, "")
        video_dir = maybe(self.query_one("#video_dir").value, "")
        fps = maybe(self.query_one("#fps").value, "")
        self.labels_ready = (name != "") and (video_dir != "") and (fps != "")

    def check_ready(self):
        self.post_message(ProjectWizard.Ready(self.regex_ready and self.labels_ready))

    def watch_labels_ready(self):
        self.check_ready()

    def watch_regex_ready(self):
        self.check_ready()

    def get_config(self):
        return {
            "name": self.query_one("#project_name").value,
            "video_root": self.query_one("#video_dir").value,
            "fps": int(self.query_one("#fps").value),
            "video_regex": self.query_one("RegexInput").get_regex()
        }

class AllegoParams(HorizontalGroup):

    class Ready(Message):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

    def on_mount(self) -> None:
        self.border_title = "allego sync parameters"

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="sync channel", id="sync_channel", value="32", type="integer")
        yield LabeledInput(label="sync threshold",
                           id="sync_threshold",
                           value="0.2",
                           type="number",
                           placeholder="Voltage threshold for sync pulse")
        yield LabeledInput(label="sample rate", id="sample_rate", value="30000", type="number")

    @on(LabeledInput.Changed)
    def check_ready(self):
        ready = all(maybe(input.value, "") != "" for input in self.children)
        self.post_message(AllegoParams.Ready(ready))

    def get_config(self):
        return {
            "sync_channel": int(self.query_one("#sync_channel").value),
            "sync_threshold": float(self.query_one("#sync_threshold").value),
            "sample_rate": int(self.query_one("#sample_rate").value)
        }

class OpenEphysParams(HorizontalGroup):

    class Ready(Message):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

    def on_mount(self) -> None:
        self.border_title = "open ephys sync parameters"

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="sync channel", id="sync_channel", value="32", type="integer")
        yield LabeledInput(label="sync threshold",
                           id="sync_threshold",
                           value="0.2",
                           type="number",
                           placeholder="Voltage threshold for sync pulse")
        yield LabeledInput(label="sample rate", id="sample_rate", value="30000", type="number")

    @on(LabeledInput.Changed)
    def check_ready(self):
        ready = all(maybe(input.value, "") != "" for input in self.children)
        self.post_message(OpenEphysParams.Ready(ready))

    def get_config(self):
        return {
            "sync_channel": int(self.query_one("#sync_channel").value),
            "sync_threshold": float(self.query_one("#sync_threshold").value),
            "sample_rate": int(self.query_one("#sample_rate").value)
        }

class DSIParams(HorizontalGroup):

    class Ready(Message):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

    def on_mount(self) -> None:
        self.border_title = "dsi sync parameters"

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="sync threshold",
                           id="sync_threshold",
                           value="0.2",
                           type="number",
                           placeholder="Voltage threshold for sync pulse")
        yield LabeledInput(label="sample rate", id="sample_rate", value="1000", type="number")

    @on(LabeledInput.Changed)
    def check_ready(self):
        ready = all(maybe(input.value, "") != "" for input in self.children)
        self.post_message(DSIParams.Ready(ready))

    def get_config(self):
        return {
            "sync_threshold": float(self.query_one("#sync_threshold").value),
            "sample_rate": int(self.query_one("#sample_rate").value)
        }

class EphysWizard(VerticalGroup):

    ephys_type = reactive(None, recompose=True)
    ephys_params_ready = reactive(False)

    class Ready(Message):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

    def __init__(self):
        super().__init__()
        self.allego_params = AllegoParams(classes="ephys_params")
        self.oe_params = OpenEphysParams(classes="ephys_params")
        self.dsi_params = DSIParams(classes="ephys_params")

    def on_mount(self) -> None:
        self.border_title = "ephys info"

    def compose(self) -> ComposeResult:
        enabled = (self.ephys_type is not None)
        with HorizontalGroup():
            yield Checkbox("Enable ephys?", id="enable_ephys", value=enabled)
            yield LabeledInput(label="ephys dir",
                               id="ephys_dir",
                               value="ephys",
                               placeholder="Ephys recordings sub-directory",
                               disabled=(not enabled))
            yield LabeledInput(label="ephys regex",
                               id="ephys_regex",
                               value=r".*\.xdat\.json",
                               placeholder="Regex for identifying ephys source files",
                               disabled=(not enabled))
            yield Select.from_values(("allego", "openephys", "dsi"),
                                     id="ephys_type",
                                     value=(self.ephys_type if enabled else Select.BLANK),
                                     prompt="Ephys source type", disabled=(not enabled))
        if self.ephys_type == "allego":
            yield self.allego_params
        elif self.ephys_type == "openephys":
            yield self.oe_params
        elif self.ephys_type == "dsi":
            yield self.dsi_params

    @on(Checkbox.Changed, "#enable_ephys")
    def enable_ephys(self, event: Checkbox.Changed) -> None:
        if event.checkbox.value:
            for child in event.checkbox.parent.children[1:]: # type: ignore
                child.disabled = False
        else:
            for child in event.checkbox.parent.children[1:]: # type: ignore
                child.disabled = True
        self.check_ready()

    @on(Select.Changed, "#ephys_type")
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.value == Select.BLANK:
            self.ephys_type = None
            self.ephys_params_ready = False
        else:
            self.ephys_type = event.select.value

    @on(AllegoParams.Ready)
    def set_allego_params_ready(self, msg: AllegoParams.Ready):
        if self.ephys_type == "allego":
            self.ephys_params_ready = msg.ready

    @on(OpenEphysParams.Ready)
    def set_oe_params_ready(self, msg: OpenEphysParams.Ready):
        if self.ephys_type == "openephys":
            self.ephys_params_ready = msg.ready

    @on(DSIParams.Ready)
    def set_dsi_params_ready(self, msg: DSIParams.Ready):
        if self.ephys_type == "dsi":
            self.ephys_params_ready = msg.ready

    def watch_ephys_params_ready(self):
        self.check_ready()

    def check_ready(self):
        if self.query_one("#enable_ephys").value:
            ready = all(maybe(input.value, "") != "" for input in self.query_children("LabeledInput"))
            self.post_message(EphysWizard.Ready(ready and self.ephys_params_ready))
        else:
            self.post_message(EphysWizard.Ready(True))

    def get_config(self):
        if not self.query_one("#enable_ephys").value:
            return None

        if self.ephys_type == "allego":
            params = self.allego_params.get_config()
        elif self.ephys_type == "openephys":
            params = self.oe_params.get_config()
        elif self.ephys_type == "dsi":
            params = self.dsi_params.get_config()

        return {
            "ephys_root": self.query_one("#ephys_dir").value,
            "ephys_regex": self.query_one("#ephys_regex").value,
            "ephys_param": {
                "type": self.ephys_type,
                **params
            }
        }

class ModelWizard(Horizontal):

    class Ready(Message):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

    def on_mount(self) -> None:
        self.border_title = "model info"

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="model dir", value="model", placeholder="Model and label sub-directory")
        yield Select.from_values(("create", "import"), allow_blank=False, value="create")
        self.name_or_path = LabeledInput(label="model name", id="name_or_path", placeholder="Name of your model")
        yield self.name_or_path
        self.choose_path = Button("Choose path", disabled=True)
        yield self.choose_path

    @on(Button.Pressed)
    @work
    async def select_directory(self, event: Button.Pressed) -> None:
        # Imported models may live anywhere accessible to the Cheese3D process.
        model_path = await self.app.push_screen_wait(
            _directory_picker("Select model directory")
        )
        if model_path is None:
            model_path = ""
        else:
            model_path = str(model_path.absolute())
        self.name_or_path.value = model_path

    @on(Select.Changed)
    @work
    async def select_mode(self, event: Select.Changed) -> None:
        if event.select.value == "create":
            self.name_or_path.label = "model name"
            self.name_or_path.placeholder = "Name of your model"
            self.name_or_path.value = ""
            self.name_or_path.disabled = False
            self.choose_path.disabled = True
        elif event.select.value == "import":
            # Use the same unrestricted browser when import mode is selected.
            model_path = await self.app.push_screen_wait(
                _directory_picker("Select model directory")
            )
            if model_path is None:
                model_path = ""
            else:
                model_path = str(model_path.absolute())
            self.name_or_path.label = "model path"
            self.name_or_path.placeholder = "Click 'Choose path' to fill in model path"
            self.name_or_path.value = model_path
            self.name_or_path.disabled = True
            self.choose_path.disabled = False

    @on(LabeledInput.Changed)
    def check_ready(self):
        ready = all(maybe(input.value, "") != "" for input in self.query_children("LabeledInput"))
        self.post_message(ModelWizard.Ready(ready))

    def get_config(self):
        if self.query_one("Select").value == "create":
            return {
                "name": self.name_or_path.value
            }
        else:
            return {
                "path": self.name_or_path.value
            }

class StartMenu(Screen):
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Horizontal():
            yield Button("Create new project", id="create_project", variant="primary")
            yield Button("Load existing project", id="load_project", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#create_project")
    @work
    async def create_project(self):
        project_path = await self.app.push_screen_wait(CreateWizard())
        if project_path is not None:
            self.app.push_screen(MainScreen(project_path))

    @on(Button.Pressed, "#load_project")
    @work
    async def load_project(self):
        # Start at the user's home rather than the web server's repository CWD.
        project_path = await self.app.push_screen_wait(
            _directory_picker("Select Cheese3D project")
        )
        if project_path is not None:
            self.app.push_screen(MainScreen(project_path))

class CreateWizardLoading(ModalScreen):
    def __init__(self, project_config, ephys_config, model_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_config = project_config
        self.ephys_config = ephys_config
        self.model_config = model_config

    def create_config(self) -> None:
        config = self.project_config
        ephys_config = self.ephys_config
        model_config = self.model_config
        # create the project
        Ch3DProject.initialize(config["name"], root=".")
        # read in config to overwrite
        yaml_config = ProjectConfig.load(Path(".") / config["name"] / "config.yaml")
        # overwrite project parameters
        yaml_config.video_root = config["video_root"]
        yaml_config.fps = config["fps"]
        yaml_config.video_regex = config["video_regex"]
        # overwrite ephys
        if ephys_config is not None:
            yaml_config.ephys_root = ephys_config["ephys_root"]
            yaml_config.ephys_regex = ephys_config["ephys_regex"]
            yaml_config.ephys_param = ephys_config["ephys_param"]
        # overwrite model
        if "name" in model_config:
            yaml_config.model = ModelConfig(model_config["name"])
        # write yaml
        with Path(".") / config["name"] / "config.yaml" as f:
            OmegaConf.save(yaml_config, f)
        # import model if needed
        if "path" in model_config:
            project = Ch3DProject.from_path(Path(".") / config["name"],
                                            model_import=model_config["path"])
            project._export_labels()
            yaml_config.model.name = project.model.name
            yaml_config.model.backend_options = {
                "experimenter": project.model.experimenter,
                "date": project.model.date
            }
            with Path(".") / config["name"] / "config.yaml" as f:
                OmegaConf.save(yaml_config, f)

    def on_show(self) -> None:
        self.create_config()
        # close screen
        msg = self.query_one("#msg")
        msg.update(f"[bold]Created new project at: {Path('.') / self.project_config['name']}[/bold]")
        self.query_one("#loading").remove()
        self.query_one("#modal").mount(Horizontal(Button("Done", id="done", variant="success")))

    @on(Button.Pressed, "#done")
    def close(self):
        self.dismiss(Path(".") / self.project_config["name"])

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static("[bold]Creating config file...[/bold]", id="msg"))
            yield Horizontal(LoadingIndicator(), id="loading")

class CreateWizard(Screen):

    project_ready = reactive(False)
    ephys_ready = reactive(True)
    model_ready = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Button("Continue", id="accept_config", variant="success", disabled=True)
            yield ProjectWizard()
            yield ModelWizard()
            yield EphysWizard()
        yield Footer()

    @on(Button.Pressed, "#accept_config")
    @work
    async def accept_config(self):
        config = self.query_one("ProjectWizard").get_config()
        ephys_config = self.query_one("EphysWizard").get_config()
        model_config = self.query_one("ModelWizard").get_config()
        project_path = await self.app.push_screen_wait(
            CreateWizardLoading(config, ephys_config, model_config)
        )
        self.dismiss(project_path)

    @on(ProjectWizard.Ready)
    def set_project_ready(self, msg: ProjectWizard.Ready):
        self.project_ready = msg.ready

    @on(EphysWizard.Ready)
    def set_ephys_ready(self, msg: EphysWizard.Ready):
        self.ephys_ready = msg.ready

    @on(ModelWizard.Ready)
    def set_model_ready(self, msg: ModelWizard.Ready):
        self.model_ready = msg.ready

    def check_ready(self):
        if self.project_ready and self.ephys_ready and self.model_ready:
            self.query_one("#accept_config").disabled = False
        else:
            self.query_one("#accept_config").disabled = True

    def watch_project_ready(self):
        self.check_ready()

    def watch_ephys_ready(self):
        self.check_ready()

    def watch_model_ready(self):
        self.check_ready()

class MainScreen(Screen):
    class ExtractionSelectionComplete(Message):
        def __init__(self, selection):
            super().__init__()
            self.selection = selection

    def __init__(self, project_path: str | Path):
        self.project = Ch3DProject.from_path(project_path)
        # The GUI launches training in a process, rather than its own worker
        # thread, so both DLC and LP can be interrupted without killing the UI.
        self._training_process: Optional[subprocess.Popen] = None
        self._training_stop_requested = False
        self._checkpoint_records: Dict[str, dict] = {}
        super().__init__()

    def on_mount(self) -> None:
        self._refresh_summary()

    def _check_in_sessions(self, session: str):
        k = RecordingKey(session, "")
        return any(k2.matches(k) for k2 in self.project.sessions.keys())

    def _refresh_summary(self):
        summary_log = self.query_one("#summary_log")
        summary_log.clear()
        self.project.summarize(summary_log)

    def _refresh_recording_list(self):
        config = ProjectConfig.load(self.project.path / "config.yaml")
        video_root = str(self.project.path / config.video_root) # type: ignore
        self.query_one("#recording_path").update(
            f"[bold]Available sessions under:[/bold] {video_root}"
        )
        sessions = reglob(".*", path=video_root)
        sessions = [Path(p) for p in sessions]
        in_project = [self._check_in_sessions(p.name) for p in sessions]
        select_list = self.query_one("#select_sessions")
        select_list.clear_options()
        select_list.add_options([(path.name, path, select)
                                 for path, select in zip(sessions, in_project)])

    def _refresh_checkpoint_list(self) -> None:
        """Populate pose checkpoint choices and show their validation metrics."""
        selector = self.query_one("#pose_checkpoint", Select)
        records = self.project.model.list_checkpoints() if self.project.model else []
        self._checkpoint_records = {item["path"]: item for item in records}
        options = []
        for item in records:
            epoch = item.get("epoch")
            label = f"epoch {epoch}: {Path(item['path']).name}" if epoch is not None \
                else Path(item["path"]).name
            if "shuffle" in item:
                label = (
                    f"{item.get('train_fraction_percent', '?')}% train | "
                    f"shuffle {item['shuffle']} | {label}"
                )
            options.append((label, item["path"]))
        selector.set_options(options)
        if records:
            # Default visibly to the newest available checkpoint; the change
            # handler then makes that exact choice active in the backend.
            selector.value = records[-1]["path"]
        else:
            self.query_one("#checkpoint_metrics", Static).update(
                "No trained checkpoints were found for this backend."
            )

    def _read_tracking_settings(self) -> dict:
        """Validate pose-inference device IDs and batch size from the GUI."""
        gpu_ids = self.query_one("#tracking_gpus", Input).value.strip()
        if gpu_ids and any(not item.strip().isdigit() for item in gpu_ids.split(",")):
            raise ValueError("Tracking GPU IDs must be comma-separated integers")
        batch_size = int(self.query_one("#tracking_batch_size", Input).value)
        if batch_size <= 0:
            raise ValueError("Tracking batch size must be positive")
        settings = {"gpu_ids": gpu_ids, "batch_size": batch_size}
        if not self._uses_lightning_pose():
            shuffle = int(self.query_one("#dlc_tracking_shuffle", Input).value)
            if shuffle <= 0:
                raise ValueError("DLC tracking shuffle must be positive")
            settings["shuffle"] = shuffle
        return settings

    def _enable_model_done(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#model_buttons").children:
            button.disabled = False

    def _disable_model_in_progress(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = True
        for button in self.query_one("#model_buttons").children:
            button.disabled = True

    def _set_training_controls(self, training: bool) -> None:
        """Expose only the stop control while a DLC or LP trainer is active."""
        # Former model-wide disabling also disabled the enclosing tab, making an
        # early-stop button impossible to reach while training was in progress.
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#training_buttons").children:
            button.disabled = training
        self.query_one("#stop_train", Button).disabled = not training

    def _uses_lightning_pose(self) -> bool:
        """Select backend-specific controls without importing the other backend."""
        return self.project.model is not None and \
            self.project.model.__class__.__name__ == "LightningPoseBackend"

    def _read_training_settings(self) -> tuple[str, dict]:
        """Validate and collect common plus backend-specific training controls."""
        def integer(widget_id: str) -> int:
            return int(self.query_one(widget_id, Input).value)

        def number(widget_id: str) -> float:
            return float(self.query_one(widget_id, Input).value)

        gpu_ids = self.query_one("#training_gpus", Input).value.strip()
        if not gpu_ids or any(not item.strip().isdigit() for item in gpu_ids.split(",")):
            raise ValueError("GPU IDs must be comma-separated integers, for example 0,1")
        settings = {
            "epochs": integer("#training_epochs"),
            "batch_size": integer("#training_batch_size"),
            "learning_rate": number("#training_learning_rate"),
            "save_every_n_epochs": integer("#training_save_epochs"),
            "validate_every_n_epochs": integer("#training_val_epochs"),
        }
        if settings["epochs"] <= 0 or settings["batch_size"] <= 0 \
                or settings["learning_rate"] <= 0 \
                or settings["save_every_n_epochs"] <= 0 \
                or settings["validate_every_n_epochs"] <= 0:
            raise ValueError(
                "Epochs, batch size, learning rate, save interval, and validation "
                "interval must be positive"
            )
        if self._uses_lightning_pose():
            settings.update({
                "imgaug": str(self.query_one("#lp_imgaug", Select).value),
                "horizontal_flip": self.query_one("#lp_hflip", Checkbox).value,
                "train_prob": number("#lp_train_prob"),
                "val_prob": number("#lp_val_prob"),
                "unfreezing_epoch": integer("#lp_unfreezing_epoch"),
                "early_stopping": self.query_one("#lp_early_stopping", Checkbox).value,
                "early_stop_patience": integer("#lp_early_stop_patience"),
            })
            if settings["train_prob"] < 0 or settings["val_prob"] < 0 \
                    or settings["train_prob"] + settings["val_prob"] > 1:
                raise ValueError("Lightning Pose train + validation fractions must be <= 1")
        else:
            settings.update({
                "network_architecture": str(
                    self.query_one("#dlc_network_architecture", Select).value
                ),
                "train_fraction_percent": number("#dlc_train_fraction_percent"),
                "training_shuffle": integer("#dlc_training_shuffle"),
                "max_snapshots_to_keep": integer("#dlc_max_snapshots_to_keep"),
                "rotation": number("#dlc_rotation"),
                "scale_min": number("#dlc_scale_min"),
                "scale_max": number("#dlc_scale_max"),
                "crop_width": integer("#dlc_crop_width"),
                "crop_height": integer("#dlc_crop_height"),
                "motion_blur": self.query_one("#dlc_motion_blur", Checkbox).value,
                "gaussian_noise": number("#dlc_gaussian_noise"),
            })
            if settings["scale_min"] <= 0 or settings["scale_min"] > settings["scale_max"]:
                raise ValueError("DLC minimum scale must be positive and <= maximum scale")
            if not 1 <= settings["train_fraction_percent"] <= 99:
                raise ValueError("DLC training percentage must be between 1 and 99")
            if settings["training_shuffle"] <= 0:
                raise ValueError("DLC training shuffle must be positive")
            if settings["max_snapshots_to_keep"] <= 0:
                raise ValueError("DLC maximum snapshots to keep must be positive")
            if settings["crop_width"] <= 0 or settings["crop_height"] <= 0:
                raise ValueError("DLC crop width and height must be positive")
        return gpu_ids, settings

    def _enable_pose_done(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#pose_buttons").children:
            button.disabled = False

    def _disable_pose_in_progress(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = True
        for button in self.query_one("#pose_buttons").children:
            button.disabled = True

    def _enable_visualize_done(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#visualize_buttons").children:
            button.disabled = False

    def _disable_visualize_in_progress(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = True
        for button in self.query_one("#visualize_buttons").children:
            button.disabled = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Collapsible(classes="helpmenu", title="help", collapsed=True):
                yield Static(_MAIN_HELP_MSG, id="main_help")
            with TabbedContent(initial="summary", id="all_tabs"):
                with TabPane(title="summary", id="summary"):
                    yield RichConsole(id="summary_log")
                with TabPane(title="select sessions", id="sessions"):
                    with Vertical(id="sessions_list"):
                        yield Static("", id="recording_path")
                        yield SelectionList[str](id="select_sessions")
                with TabPane(title="model", id="model"):
                    with Vertical():
                        with CenterMiddle(classes="buttons_group"):
                            with HorizontalGroup(id="model_buttons"):
                                yield Button("Extract frames", id="extract")
                                yield Button("Label frames", id="label")
                        yield TextualStdout(id="model_log")
                with TabPane(title="training", id="training"):
                    with VerticalScroll():
                        yield Static("Common training settings")
                        yield TrainingInput("GPU IDs (comma-separated)", "training_gpus", "0")
                        default_epochs = "300" if self._uses_lightning_pose() else "200"
                        default_batch = "16" if self._uses_lightning_pose() else "8"
                        default_lr = "0.001" if self._uses_lightning_pose() else "0.0005"
                        yield TrainingInput("Epochs", "training_epochs", default_epochs,
                                            "integer")
                        yield TrainingInput("Batch size", "training_batch_size", default_batch,
                                            "integer")
                        yield TrainingInput("Learning rate", "training_learning_rate", default_lr,
                                            "number")
                        default_save = "25" if self._uses_lightning_pose() else "25"
                        default_val = "5" if self._uses_lightning_pose() else "10"
                        yield TrainingInput("Save every N epochs", "training_save_epochs",
                                            default_save, "integer")
                        yield TrainingInput("Validate every N epochs", "training_val_epochs",
                                            default_val, "integer")
                        yield Static("Data augmentation and backend settings")
                        if self._uses_lightning_pose():
                            yield Static("Augmentation preset", classes="training_field_label")
                            yield Select([("DLC-style", "dlc"), ("Default", "default")],
                                         value="dlc", id="lp_imgaug")
                            yield Checkbox("Random horizontal flip", id="lp_hflip")
                            yield TrainingInput("Training fraction", "lp_train_prob", "0.95", "number")
                            yield TrainingInput("Validation fraction", "lp_val_prob", "0.05", "number")
                            yield TrainingInput("Backbone unfreezing epoch", "lp_unfreezing_epoch",
                                                "20", "integer")
                            yield Checkbox("Early stopping", id="lp_early_stopping")
                            yield TrainingInput("Early-stop patience", "lp_early_stop_patience",
                                                "3", "integer")
                        else:
                            yield Static("DLC3 network architecture",
                                         classes="training_field_label")
                            # The selector is independent of checkpoint selection:
                            # this chooses the network built for a new training run.
                            yield Select.from_values(
                                DLC3_PYTORCH_MODELS, value="resnet_50",
                                allow_blank=False, id="dlc_network_architecture"
                            )
                            yield TrainingInput("Training split (%)",
                                                "dlc_train_fraction_percent",
                                                "95", "number")
                            yield TrainingInput("Training shuffle",
                                                "dlc_training_shuffle", "1", "integer")
                            yield Static(
                                "The remaining images form the test split (95% train = 5% test).",
                                classes="training_field_note"
                            )
                            yield TrainingInput("Maximum snapshots to keep",
                                                "dlc_max_snapshots_to_keep",
                                                "5", "integer")
                            yield TrainingInput("Rotation range (degrees)", "dlc_rotation", "30", "number")
                            yield TrainingInput("Minimum scale", "dlc_scale_min", "0.5", "number")
                            yield TrainingInput("Maximum scale", "dlc_scale_max", "1.25", "number")
                            yield TrainingInput("Crop width", "dlc_crop_width", "448", "integer")
                            yield TrainingInput("Crop height", "dlc_crop_height", "448", "integer")
                            yield Checkbox("Motion blur", value=True, id="dlc_motion_blur")
                            yield TrainingInput("Gaussian noise standard deviation",
                                                "dlc_gaussian_noise", "12.75", "number")
                        with HorizontalGroup(id="training_buttons"):
                            yield Button("Train network", id="train", variant="success")
                            yield Button("Stop training", id="stop_train",
                                         variant="error", disabled=True)
                        yield TextualStdout(id="training_log")
                with TabPane(title="pose tracking", id="pose"):
                    with Vertical():
                        yield Static("Tracking checkpoint", classes="training_field_label")
                        yield Select([], id="pose_checkpoint", prompt="Select checkpoint")
                        yield Static("Select a checkpoint to view validation metrics.",
                                     id="checkpoint_metrics")
                        yield Static("Inference settings", classes="training_field_label")
                        yield TrainingInput("GPU IDs (videos split across GPUs)",
                                            "tracking_gpus", "0")
                        tracking_batch = "32" if self._uses_lightning_pose() else "8"
                        yield TrainingInput("Inference batch size", "tracking_batch_size",
                                            tracking_batch, "integer")
                        if not self._uses_lightning_pose():
                            yield TrainingInput("DLC training shuffle",
                                                "dlc_tracking_shuffle", "1", "integer")
                        yield Static(
                            "With multiple IDs (for example 0,1), independent camera videos "
                            "are assigned round-robin to one inference process per GPU.",
                            id="tracking_gpu_note",
                        )
                        with CenterMiddle(classes="buttons_group"):
                            with HorizontalGroup(id="pose_buttons"):
                                yield Button("Calibrate", id="calibrate")
                                yield Button("Track", id="track")
                                yield Button("Triangulate", id="triangulate")
                        yield TextualStdout(id="pose_log")
                with TabPane(title="visualization", id="visualization"):
                    with Vertical():
                        with CenterMiddle(classes="buttons_group"):
                            with HorizontalGroup(id="visualize_buttons"):
                                yield Button("Visualize (interactive)", id="visualize")
                                yield Button("Generate videos", id="generate_videos")
                        default_video_workers = str(max(1, (os.cpu_count() or 1) - 2))
                        yield TrainingInput("Video generation CPU core budget",
                                            "video_generation_workers",
                                            default_video_workers, "integer")
                        yield Static(
                            "Default uses all detected CPU cores except two. Camera processes "
                            "share the remaining budget through multithreaded FFmpeg encoders.",
                            id="video_worker_note",
                        )
                        yield TextualStdout(id="visualize_log")
        yield Footer()

    @on(TabbedContent.TabActivated)
    def update_tabs(self, msg: TabbedContent.TabActivated):
        self.project = Ch3DProject.from_path(self.project.path)
        if msg.pane.id == "summary":
            self._refresh_summary()
        elif msg.pane.id == "sessions":
            self._refresh_recording_list()
        elif msg.pane.id == "pose":
            self._refresh_checkpoint_list()

    @on(Select.Changed, "#pose_checkpoint")
    def select_pose_checkpoint(self, msg: Select.Changed) -> None:
        """Activate the selected weights and display associated validation metrics."""
        path = str(msg.value)
        record = self._checkpoint_records.get(path)
        if record is None or self.project.model is None:
            return
        self.project.model.select_checkpoint(path)
        if "shuffle" in record and not self._uses_lightning_pose():
            # Selecting weights also selects their DLC model folder. The field
            # remains editable for advanced/manual configuration recovery.
            self.query_one("#dlc_tracking_shuffle", Input).value = str(record["shuffle"])
        metrics = record.get("metrics", {})
        metric_text = " | ".join(
            f"{name.split('/')[-1]}: {float(value):.5g}"
            for name, value in sorted(metrics.items())
        ) or "No validation metrics stored in this checkpoint."
        self.query_one("#checkpoint_metrics", Static).update(
            f"Epoch: {record.get('epoch', 'unknown')} | "
            f"Shuffle: {record.get('shuffle', 'n/a')} | {metric_text}"
        )

    @on(SelectionList.SelectedChanged, "#select_sessions")
    def update_selected_sessions(self, msg: SelectionList.SelectedChanged):
        config = ProjectConfig.load(self.project.path / "config.yaml")
        current = set(recording["name"] for recording in config.sessions) # type: ignore
        selections = [selection.name for selection in msg.selection_list.selected]
        new_sessions = []
        for recording in config.sessions: # type: ignore
            name = recording.get("name", "")
            if name in selections:
                new_sessions.append(recording)
        for selection in selections:
            if selection not in current:
                new_sessions.append({"name": selection}) # type: ignore
        config.sessions = new_sessions # type: ignore
        OmegaConf.save(config, self.project.path / "config.yaml")

    @work(thread=True)
    def _automatic_extraction(self):
        log = self.query_one("#model_log")
        log.clear() # type: ignore
        with _pipeline_output(log):
            self.project.extract_frames()

    @on(Button.Pressed, "#extract")
    @work
    async def extract_frames(self):
        self._disable_model_in_progress()
        choice = await self.app.push_screen_wait(
            ChoiceBox(message="How do you want to extract frames?",
                      button_text=("Automatic", "Manual"))
        )
        if choice == 0:
            await self._automatic_extraction().wait()
        else:
            options = {recording.name: recording
                       for recording in self.project.sessions.keys()}
            selection_name = await self.app.push_screen_wait(
                SelectionBox(message="Select a recording to extract",
                             options=list(options.keys()))
            )
            await self.app.push_screen_wait(
                BlockScreen(lambda: self.project.extract_frames([options[selection_name]], manual=True),
                            message=_EXTRACTING_POPUP)
            )
        self._enable_model_done()
        self.app.push_screen(DialogBox("Frame extraction completed!"))

    @on(Button.Pressed, "#label")
    @work
    async def label_frames(self):
        await self.app.push_screen_wait(
            BlockScreen(lambda: self.project.label_frames(),
                        message=_LABELING_POPUP)
        )

    @on(Button.Pressed, "#train")
    @work(thread=True)
    def train_model(self):
        """Run training in an interruptible child process for either backend."""
        self._training_stop_requested = False
        self._set_training_controls(True)
        log = self.query_one("#training_log")
        log.clear() # type: ignore
        project_path = self.project.path
        try:
            gpu_ids, settings = self._read_training_settings()
        except ValueError as error:
            self._set_training_controls(False)
            self.app.call_from_thread(self.app.push_screen, DialogBox(str(error)))
            return
        command = _training_command(project_path, gpu=gpu_ids, settings=settings)
        terminal = None
        try:
            try:
                # Textual Serve owns stdout, while /dev/tty is the terminal that
                # launched Cheese3D and must continue showing training progress.
                terminal = open("/dev/tty", "w", buffering=1) if os.name == "posix" \
                    else sys.__stderr__
            except OSError:
                terminal = sys.__stderr__
            output = _TeeOutput(log, terminal)
            output.write(f"Starting training subprocess: {shlex.join(command)}\n")
            child_environment = os.environ.copy()
            child_environment["PYTHONUNBUFFERED"] = "1"
            # A new process group lets Stop training reach DataLoader and backend
            # descendants as well as the direct Cheese3D CLI child.
            self._training_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=child_environment,
                start_new_session=(os.name == "posix")
            )
            # Former inherited output disappeared into Textual Serve. Forward
            # every unbuffered line to both the GUI model log and launching TTY.
            if self._training_process.stdout is not None:
                for line in iter(self._training_process.stdout.readline, ""):
                    output.write(line)
                self._training_process.stdout.close()
            return_code = self._training_process.wait()
            stopped = self._training_stop_requested
        except Exception:
            traceback.print_exc(file=sys.__stderr__)
            return_code = -1
            stopped = self._training_stop_requested
        finally:
            self._training_process = None
            self._set_training_controls(False)
            if terminal not in (None, sys.__stdout__, sys.__stderr__):
                terminal.close()

        if stopped:
            message = "Model training stopped early. Existing checkpoints were kept."
        elif return_code == 0:
            message = "Model training completed!"
        else:
            message = f"Model training failed (exit code {return_code}); see terminal output."
        self.app.call_from_thread(self.app.push_screen, DialogBox(message))

    @on(Button.Pressed, "#stop_train")
    def stop_training(self) -> None:
        """Request graceful early stopping of the active DLC or LP subprocess."""
        process = self._training_process
        if process is None or process.poll() is not None:
            return
        self._training_stop_requested = True
        # The training reader thread will forward the backend's interrupt output
        # to the GUI and terminal after this immediate on-screen notification.
        stop_message = "Stopping training early (sending SIGINT)..."
        model_log = self.query_one("#training_log", TextualStdout)
        # Bypass TextualStdout's worker-thread adapter because button handlers
        # already execute on Textual's UI thread.
        RichLog.write(model_log, stop_message)
        try:
            with open("/dev/tty", "w", buffering=1) as terminal:
                terminal.write(f"{stop_message}\n")
        except OSError:
            print(stop_message, file=sys.__stderr__, flush=True)
        # Preserve the former uninterrupted training path as the default; this
        # branch runs only when the user explicitly presses the new stop button.
        _interrupt_training_process(process)
        self.query_one("#stop_train", Button).disabled = True

    @on(Button.Pressed, "#calibrate")
    @work(thread=True)
    def calibrate(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with _pipeline_output(log):
            self.project.calibrate()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("Camera calibration completed!"))

    @on(Button.Pressed, "#track")
    @work(thread=True)
    def track(self):
        """Run inference off-thread while all widget changes stay on Textual's UI thread."""
        # Formerly the worker thread directly disabled and re-enabled widgets.
        # Textual widget mutation is not thread-safe and could leave the served
        # page permanently frozen after inference had already completed.
        self.app.call_from_thread(self._disable_pose_in_progress)
        log = self.app.call_from_thread(self.query_one, "#pose_log")
        self.app.call_from_thread(log.clear)  # type: ignore
        try:
            settings = self.app.call_from_thread(self._read_tracking_settings)
            with _pipeline_output(log):
                self.project.track(tracking_settings=settings)
        except Exception as error:
            # The former narrow exception handler skipped cleanup for CUDA,
            # subprocess, and filesystem failures, leaving every pose control disabled.
            with _pipeline_output(log):
                traceback.print_exc()
            message = f"2D pose tracking failed: {error}"
        else:
            message = "2D pose tracking completed!"
        finally:
            # Always restore the page, including after worker or progress-monitor failure.
            self.app.call_from_thread(self._enable_pose_done)
        self.app.call_from_thread(self.app.push_screen, DialogBox(message))

    @on(Button.Pressed, "#triangulate")
    @work(thread=True)
    def triangulate(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with _pipeline_output(log):
            self.project.triangulate()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("3D triangulation completed!"))

    @on(Button.Pressed, "#generate_videos")
    @work(thread=True)
    def generate_videos(self):
        """Render QC videos off-thread while mutating widgets on the UI thread."""
        self.app.call_from_thread(self._disable_visualize_in_progress)
        log = self.app.call_from_thread(self.query_one, "#visualize_log")
        self.app.call_from_thread(log.clear)  # type: ignore
        try:
            with _pipeline_output(log):
                print("Starting video generation ...")
                max_workers = int(self.app.call_from_thread(
                    lambda: self.query_one("#video_generation_workers", Input).value
                ))
                if max_workers <= 0:
                    raise ValueError("Video generation CPU core budget must be positive")
                completed = self.project.generate_videos(max_workers=max_workers)
        except Exception as exc:
            with _pipeline_output(log):
                traceback.print_exc()
            self.app.call_from_thread(
                self.app.push_screen,
                DialogBox(f"Video generation failed: {exc}")
            )
        else:
            self.app.call_from_thread(
                self.app.push_screen,
                DialogBox(f"Video generation completed: {completed} output(s) available.")
            )
        finally:
            # The former worker-thread widget call could leave the web client
            # disabled even after every labeled video had been written.
            self.app.call_from_thread(self._enable_visualize_done)

    @on(Button.Pressed, "#visualize")
    @work
    async def visualize(self):
        self._disable_visualize_in_progress()
        options = {recording.name: recording
                   for recording in self.project.sessions.keys()}
        selection = await self.app.push_screen_wait(
            SelectionBox(message="Select a recording to visualize",
                            options=list(options.keys()))
        )
        await self.app.push_screen_wait(
            BlockScreen(lambda: self.project.visualize(options[selection]),
                        message=_VISUALIZATION_POPUP)
        )
        self._enable_visualize_done()
        self.app.push_screen(DialogBox("Visualization completed!"))

class Cheese3dApp(App):
    """Interactive Cheese3D TUI via Textual."""

    BINDINGS = [
        ("q", "quit", "Quit the GUI"),
        ("d", "toggle_dark", "Toggle dark mode")
    ]

    CSS_PATH = "interactive_styles/app.css"

    def on_mount(self) -> None:
        self.title = "Cheese3D Interative GUI"
        self.sub_title = "Use mouse or keyboard to navigate"
        self.push_screen(StartMenu())


def _open_web_ui(url: str) -> None:
    """Open the served Textual application in the user's default browser."""
    if not webbrowser.open(url):
        print(f"Could not open a browser automatically. Open {url} manually.")


def run_interative(web_mode = True, open_browser = True):
    if web_mode:
        url = "http://localhost:8000"
        # The served child must use terminal mode; otherwise the default web
        # mode would recursively start another Textual server.
        # Reuse this process's Python so the served child cannot fall back to a
        # different PATH entry (for example Pixi without Lightning Pose).
        child_command = shlex.join([
            sys.executable, "-m", "cheese3d", "interactive", "--terminal"
        ])
        # Former behavior is retained for diagnosis; PATH could resolve it to
        # `.pixi/envs/default/bin/cheese3d` even after launching from Conda:
        # child_command = "cheese3d interactive --terminal"
        server = Server(child_command,
                        host="localhost",
                        port=8000,
                        title="Cheese3D")
        print(f"Cheese3D GUI: {url}")
        print("Server output will remain visible here. Press Ctrl+C to stop it.")
        if open_browser:
            browser_timer = Timer(1.0, _open_web_ui, args=(url,))
            browser_timer.daemon = True
            browser_timer.start()
        server.serve()
    else:
        app = Cheese3dApp()
        app.run()

if __name__ == "__main__":
    run_interative()
