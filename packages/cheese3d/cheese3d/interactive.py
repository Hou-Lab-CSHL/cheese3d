from omegaconf import OmegaConf
from typing import Callable, Optional, List, Tuple
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from threading import Timer
import errno
import faulthandler
import fcntl
import os
import json
import re
import signal
import shlex
import struct
import subprocess
import sys
import termios
import traceback
import webbrowser
from textual import work, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual_serve.server import Server
from textual.reactive import reactive
from textual.containers import (Center,
                                Horizontal,
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
from textual.worker import get_current_worker
from textual_fspicker import SelectDirectory
from textual_fspicker.parts import DirectoryNavigation
from textual_fspicker.parts.directory_navigation import DirectoryEntry

from cheese3d.config import _DEFAULT_VIDEO_REGEX
from cheese3d.utils import maybe, reglob, dlc_folder_to_components
from cheese3d.project import Ch3DProject, RecordingKey
from cheese3d.config import ProjectConfig, ModelConfig
from cheese3d.backends.dlc import DLC3_PYTORCH_MODELS
from cheese3d.backends.core import active_pose_backend

# Keep this dependency-free list aligned with Lightning Pose 2.2 so the GUI can
# open in DLC-only environments without importing torch or Lightning Pose.
LIGHTNING_POSE_BACKBONES = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet50_animal_apose", "resnet50_animal_ap10k",
    "resnet50_human_jhmdb", "resnet50_human_res_rle",
    "resnet50_human_top_res", "resnet50_human_hand",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vits_dino", "vits_dinov2", "vits_dinov3",
    "vitb_dino", "vitb_dinov2", "vitb_dinov3", "vitb_imagenet", "vitb_sam",
)
# Mirror SLEAP-NN 0.1.0 presets without importing its PyTorch stack in the DLC UI.
SLEAP_BACKBONES = (
    "unet", "unet_medium_rf", "unet_large_rf",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "swint_tiny", "swint_small", "swint_base",
)


def _training_command(project_path: str | Path, gpu: str = "0",
                      settings: Optional[dict] = None) -> List[str]:
    """Build the same-backend Cheese3D CLI command used by GUI training."""
    project_path = Path(project_path)
    command = [sys.executable, "-m", "cheese3d", "--path",
               str(project_path.parent), "train", project_path.name, "--gpu", str(gpu)]
    if settings:
        command.extend(["--training-settings", json.dumps(settings)])
    return command


def _persist_visualization_threshold(project_path: str | Path,
                                     probability_threshold: float) -> None:
    """Save the GUI's keypoint-probability threshold into the project config."""
    config_path = Path(project_path) / "config.yaml"
    config = OmegaConf.load(config_path)
    # Projects saved before 'visualization' existed as a ProjectConfig field have
    # no 'visualization:' section on disk; a plain attribute assignment on this
    # schema-less loaded config raises ConfigAttributeError instead of creating
    # the missing section, so update with merge=True to auto-create it.
    OmegaConf.update(config, "visualization.keypoint_probability_threshold",
                     probability_threshold, merge=True)
    OmegaConf.save(config, config_path)


def _open_sized_pty(rows: int = 24, columns: int = 120) -> Tuple[int, int]:
    """Open a PTY pair with a real (non-zero) window size.

    os.openpty() leaves the window size at 0x0. Lightning's TQDMProgressBar
    (and rich-based bars) silently disable their live progress rendering when
    the terminal size can't be determined -- unlike an ordinary non-tty pipe,
    which falls back to printing full progress records instead of nothing at
    all. A real window size restores the same live epoch bar a directly
    launched training run would show.
    """
    master_fd, slave_fd = os.openpty()
    winsize = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    return master_fd, slave_fd


def _interrupt_training_process(process: subprocess.Popen) -> None:
    """Interrupt a trainer and all of its data-loader/backend descendants."""
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.send_signal(signal.SIGINT)


def _build_lp_dlc_augmentation_config(values: dict) -> dict:
    """Build Lightning Pose's DLC-style imgaug dictionary from GUI values."""
    probability_names = [name for name in values if name.endswith("_probability")]
    for name in probability_names:
        if not 0.0 <= float(values[name]) <= 1.0:
            raise ValueError(f"{name.replace('_', ' ')} must be between 0 and 1")
    if float(values["rotation_degrees"]) < 0:
        raise ValueError("Rotation range must be non-negative")
    if int(values["motion_blur_kernel"]) <= 0:
        raise ValueError("Motion-blur kernel must be positive")
    if float(values["salt_pepper_size_min"]) > float(values["salt_pepper_size_max"]):
        raise ValueError("Salt/pepper minimum size must be <= maximum size")
    if float(values["elastic_alpha_min"]) > float(values["elastic_alpha_max"]):
        raise ValueError("Elastic alpha minimum must be <= maximum")
    if float(values["emboss_strength_min"]) > float(values["emboss_strength_max"]):
        raise ValueError("Emboss strength minimum must be <= maximum")

    rotation = float(values["rotation_degrees"])
    blur_angle = float(values["motion_blur_angle"])
    crop = float(values["crop_percent"])
    # This dictionary mirrors Lightning Pose 2.2's `dlc` preset while making
    # every transform probability and strength editable in Cheese3D.
    return {
        "Affine": {
            "p": float(values["rotation_probability"]),
            "kwargs": {"rotate": [-rotation, rotation]},
        },
        "MotionBlur": {
            "p": float(values["motion_blur_probability"]),
            "kwargs": {
                "k": int(values["motion_blur_kernel"]),
                "angle": [-blur_angle, blur_angle],
            },
        },
        "CoarseDropout": {
            "p": float(values["dropout_probability"]),
            "kwargs": {
                "p": float(values["dropout_pixel_probability"]),
                "size_percent": float(values["dropout_size_percent"]),
                "per_channel": float(values["dropout_per_channel_probability"]),
            },
        },
        "CoarseSalt": {
            "p": float(values["salt_probability"]),
            "kwargs": {
                "p": float(values["salt_pepper_pixel_probability"]),
                "size_percent": [float(values["salt_pepper_size_min"]),
                                 float(values["salt_pepper_size_max"])],
            },
        },
        "CoarsePepper": {
            "p": float(values["pepper_probability"]),
            "kwargs": {
                "p": float(values["salt_pepper_pixel_probability"]),
                "size_percent": [float(values["salt_pepper_size_min"]),
                                 float(values["salt_pepper_size_max"])],
            },
        },
        "ElasticTransformation": {
            "p": float(values["elastic_probability"]),
            "kwargs": {
                "alpha": [float(values["elastic_alpha_min"]),
                          float(values["elastic_alpha_max"])],
                "sigma": float(values["elastic_sigma"]),
            },
        },
        "AllChannelsHistogramEqualization": {
            "p": float(values["histogram_probability"]), "kwargs": {},
        },
        "AllChannelsCLAHE": {
            "p": float(values["clahe_probability"]), "kwargs": {},
        },
        "Emboss": {
            "p": float(values["emboss_probability"]),
            "kwargs": {
                "alpha": [0.0, float(values["emboss_alpha_max"])],
                "strength": [float(values["emboss_strength_min"]),
                             float(values["emboss_strength_max"])],
            },
        },
        "CropAndPad": {
            "p": float(values["crop_probability"]),
            "kwargs": {"percent": [-crop, crop], "keep_size": False},
        },
    }


class _FastDirectoryNavigation(DirectoryNavigation):
    """DirectoryNavigation that avoids a redundant stat() per listed entry.

    Upstream's ``_load`` calls the library's safe ``Path.is_dir()`` wrapper
    (a fresh, uncached stat syscall) on every entry just to filter files out,
    even though Cheese3D's pickers only ever display directories. Navigating
    into any folder with many files (labeled-data image folders routinely
    hold thousands) pays one uncached stat per file just to discard it --
    this is the actual source of picker lag, not Textual's rendering.
    ``os.scandir`` reuses the directory-entry type the OS's own readdir
    syscall already returned, avoiding that extra stat in the common case.
    """

    @work(exclusive=True, thread=True)
    def _load(self) -> None:
        self._entries = []
        worker = get_current_worker()
        styles = self._styles
        try:
            with os.scandir(self._location) as scan:
                for entry in scan:
                    try:
                        keep = entry.is_dir() or (entry.is_file() and self.show_files)
                    except OSError:
                        keep = False
                    if keep:
                        self._entries.append(
                            DirectoryEntry(self._location / entry.name, styles)
                        )
                    if worker.is_cancelled:
                        return
        except PermissionError:
            self.post_message(self.PermissionError(self, self._location))
        self.app.call_from_thread(self._repopulate_display)


def _install_fast_directory_navigation() -> None:
    """Swap the picker's DirectoryNavigation for the faster subclass above.

    textual_fspicker's FileSystemPickerScreen.compose() instantiates
    DirectoryNavigation by name from its own module's namespace, so this
    replaces that module-level reference rather than editing any installed
    package file; every SelectDirectory-based picker created afterward
    (including Cheese3D's own _Cheese3DDirectoryPicker subclass) picks it up.
    """
    import textual_fspicker.base_dialog as _base_dialog
    _base_dialog.DirectoryNavigation = _FastDirectoryNavigation


_install_fast_directory_navigation()


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


def _directory_picker(title: str = "Select directory",
                      location: Optional[str | Path] = None) -> SelectDirectory:
    """Create an unrestricted picker starting outside the repository checkout.

    The Textual web server inherits the repository as its working directory, so
    ``SelectDirectory()`` used to open inside Cheese3D on every invocation. The
    user's home directory is a more useful starting point and the picker's ``..``
    entry still permits navigation all the way to the filesystem root.
    """
    # Former behavior is preserved here for context; it made the repository the
    # initial location whenever Cheese3D was launched from its source checkout.
    # return SelectDirectory()
    start = Path.home() if location is None else Path(location).expanduser().resolve()
    # A stale/moved CLI path should recover to home rather than making the
    # picker library silently substitute the web server's repository CWD.
    if not start.is_dir():
        start = Path.home()
    return _Cheese3DDirectoryPicker(location=start, title=title)

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
            # Extract the current frame: whatever follows the last \r, or --
            # when the \r is the trailing character, as _TrainingProgressOutput
            # always sends one complete record per write -- whatever precedes
            # it. The old text.split("\r")[-1] alone always evaluated to ""
            # for a trailing \r, silently dropping every subprocess-relayed
            # progress update instead of showing/updating it.
            before, _, after = text.rpartition("\r")
            progress_text = (after or before).strip()
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


class _TrainingProgressOutput:
    """Stream metric progress while suppressing routine distributed-rank noise."""

    _rank_noise = re.compile(
        r"(?:\[?rank\s*\d+\]?|global_rank|local_rank|member:\s*\d+/\d+|"
        r"initializing distributed|distributed backend)", re.IGNORECASE
    )
    _important = re.compile(
        r"(?:traceback|error|exception|warning|epoch|loss|validation|val[_ /]|"
        r"train[_ /]|lr[= :]|gpu|%\||it/s)", re.IGNORECASE
    )
    _nccl_shutdown_noise = re.compile(
        r"(?:failed to check the .should dump. flag on tcpstore|"
        r"tcpstore.*sendbytes failed.*broken pipe)", re.IGNORECASE
    )

    def __init__(self, destination: _TeeOutput):
        self.destination = destination
        self.buffer = ""
        self.suppress_native_stack = False

    def _emit(self, separator: str) -> None:
        """Forward one complete log/progress record unless it is rank boilerplate."""
        record, self.buffer = self.buffer, ""
        if not record:
            return
        # PyTorch can emit a native stack after rank zero has already closed the
        # TCPStore during an otherwise successful DDP shutdown. Suppress only
        # that exact teardown warning and its immediately following C++ frames;
        # genuine Python, CUDA, and NCCL training failures remain visible.
        if self._nccl_shutdown_noise.search(record):
            self.suppress_native_stack = True
            return
        if self.suppress_native_stack and re.match(
                r"^(?:\x1b\[[0-9;]*m)*\s*(?:frame #\d+:|"
                r"exception raised from sendbytes\b|\s*$)",
                record, re.IGNORECASE):
            return
        self.suppress_native_stack = False
        if self._rank_noise.search(record) and not self._important.search(record):
            return
        self.destination.write(record + separator)

    def feed(self, text: str) -> None:
        """Split both newline logs and tqdm carriage-return updates immediately."""
        for character in text:
            if character == "\n":
                self._emit("\n")
            elif character == "\r":
                self._emit("\r")
            else:
                self.buffer += character

    def flush(self) -> None:
        """Forward the final unterminated subprocess record."""
        self._emit("\n")


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
        # self.app (and thus the launch --path) is only reliably resolvable
        # once this widget is mounted, but compose() -- which runs first --
        # already needs self.location to exist to yield it; only the default
        # value is set here, not the widget itself.
        self.location.value = str(getattr(self.app, "start_directory", Path.home()))

    def compose(self) -> ComposeResult:
        yield LabeledInput(label="project name", id="project_name", placeholder="Project name")
        # Former behavior always created the project under the served process's
        # working directory, which the GUI never surfaced or let users change.
        # Default to the launch --path so existing behavior is unsurprising,
        # while "Choose path" still lets a different location be selected.
        # Typing the path directly is also supported -- "Choose path" is a
        # convenience for browsing, not the only way to set a location.
        self.location = LabeledInput(label="location", id="project_location",
                                     value=str(Path.home()),
                                     placeholder="Directory to create the project under")
        yield self.location
        yield Button("Choose path", id="choose_project_location")
        yield LabeledInput(label="video dir", id="video_dir", value="videos", placeholder="Video recordings sub-directory")
        yield LabeledInput(label="fps", id="fps", value="100", type="integer", placeholder="Frames per second")
        yield RegexInput(label="video regex",
                         regex=_DEFAULT_VIDEO_REGEX["_path_"],
                         required=["type", "view"],
                         **{k: v for k, v in _DEFAULT_VIDEO_REGEX.items() if k != "_path_"})

    @on(Button.Pressed, "#choose_project_location")
    @work
    async def select_location(self, event: Button.Pressed) -> None:
        location = await self.app.push_screen_wait(
            _directory_picker("Select where to create the project", location=self.location.value)
        )
        if location is not None:
            self.location.value = str(location.absolute())
            self.set_labels_ready()

    @on(RegexInput.Ready)
    def set_regex_ready(self, msg: RegexInput.Ready) -> None:
        self.regex_ready = msg.ready

    @on(LabeledInput.Changed, "#project_name, #project_location, #video_dir, #fps")
    def set_labels_ready(self) -> None:
        name = maybe(self.query_one("#project_name").value, "")
        location = maybe(self.location.value, "")
        video_dir = maybe(self.query_one("#video_dir").value, "")
        fps = maybe(self.query_one("#fps").value, "")
        self.labels_ready = (name != "") and (location != "") and (video_dir != "") and (fps != "")

    def check_ready(self):
        self.post_message(ProjectWizard.Ready(self.regex_ready and self.labels_ready))

    def watch_labels_ready(self):
        self.check_ready()

    def watch_regex_ready(self):
        self.check_ready()

    def get_config(self):
        return {
            "name": self.query_one("#project_name").value,
            "root": self.location.value,
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
        yield Select.from_values(("create", "import"), allow_blank=False,
                                 value="create", id="model_mode")
        # The active backend must match the Pixi environment used to launch
        # Cheese3D, so it is locked to whatever pose package is actually
        # installed here rather than being a free-form choice. Former
        # behavior let this dropdown pick both "which backend to build" and
        # "what format the imported source is", so choosing "dlc" to mean
        # the latter while running lp/lp-cu13/sleap actually tried to build a
        # DLCBackend, crashing with ModuleNotFoundError: No module named
        # 'deeplabcut'. The two concerns are now independent: this selector
        # is purely informational, and model_source_format (below) answers
        # "what format is the source project" only in import mode.
        active_backend = active_pose_backend() or "dlc"
        yield Select.from_values((active_backend,), allow_blank=False,
                                 value=active_backend, id="model_backend",
                                 disabled=True)
        self.source_format = Select.from_values(
            ("dlc", "lightning_pose", "sleap"), allow_blank=False,
            value="dlc", id="model_source_format", disabled=True,
        )
        yield self.source_format
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
    def select_mode(self, event: Select.Changed) -> None:
        if event.select.id != "model_mode":
            return
        if event.select.value == "create":
            self.name_or_path.label = "model name"
            self.name_or_path.placeholder = "Name of your model"
            self.name_or_path.value = ""
            self.name_or_path.disabled = False
            self.choose_path.disabled = True
            self.source_format.disabled = True
        elif event.select.value == "import":
            # Former behavior force-opened the directory picker the instant
            # this mode was selected, before the user could type anything --
            # typing the path directly is also supported now, so switching
            # modes just enables the field; "Choose path" remains available
            # for browsing.
            self.name_or_path.label = "model path"
            self.name_or_path.placeholder = ("Path to an existing DLC/Lightning Pose/SLEAP "
                                             "project, or click 'Choose path'")
            self.name_or_path.disabled = False
            self.choose_path.disabled = False
            self.source_format.disabled = False

    @on(LabeledInput.Changed)
    def check_ready(self):
        ready = all(maybe(input.value, "") != "" for input in self.query_children("LabeledInput"))
        self.post_message(ModelWizard.Ready(ready))

    def get_config(self):
        backend_type = str(self.query_one("#model_backend", Select).value)
        if self.query_one("#model_mode", Select).value == "create":
            return {
                "name": self.name_or_path.value,
                "backend_type": backend_type,
            }
        else:
            return {
                "path": self.name_or_path.value,
                "backend_type": backend_type,
                # Independent of backend_type: which framework the source
                # project at `path` is actually formatted as. Any active
                # backend can import a DLC, Lightning Pose, or SLEAP source.
                "source_format": str(self.query_one("#model_source_format", Select).value),
            }

class StartMenu(Screen):
    def on_mount(self) -> None:
        # self.app (and thus the launch --path) is only reliably resolvable
        # once this widget is mounted, but compose() -- which runs first --
        # already needs load_path to exist to yield it; see ProjectWizard for
        # the same ordering constraint.
        self.load_path.value = str(getattr(self.app, "start_directory", Path.home()))

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Vertical():
            # A Vertical's align:center shares one left edge across all direct
            # children rather than centering each independently, so each row
            # is wrapped in its own Center to keep both blocks on the same
            # true center line regardless of their individual widths.
            with Center():
                yield Button("Create new project", id="create_project", variant="primary")
            with Center():
                with Horizontal():
                    # Typing the path directly is also supported -- "Browse" is
                    # a convenience, not the only way to open an existing project.
                    self.load_path = LabeledInput(
                        label="existing project path", id="load_project_path",
                        value=str(Path.home()),
                        placeholder="Path to an existing Cheese3D project",
                    )
                    yield self.load_path
                    yield Button("Browse", id="browse_load_path")
                    yield Button("Load existing project", id="load_project", variant="primary")
        yield Footer()

    @on(Button.Pressed, "#create_project")
    @work
    async def create_project(self):
        project_path = await self.app.push_screen_wait(CreateWizard())
        if project_path is not None:
            self.app.push_screen(MainScreen(project_path))

    @on(Button.Pressed, "#browse_load_path")
    @work
    async def browse_load_path(self, event: Button.Pressed) -> None:
        # Use the directory propagated through the web-server child rather than
        # whichever CWD Textual Serve happened to inherit for this connection.
        project_path = await self.app.push_screen_wait(
            _directory_picker("Select Cheese3D project", location=self.load_path.value)
        )
        if project_path is not None:
            self.load_path.value = str(project_path.absolute())

    @on(Button.Pressed, "#load_project")
    def load_project(self):
        project_path = Path(self.load_path.value)
        if (project_path / "config.yaml").is_file():
            self.app.push_screen(MainScreen(project_path))
        else:
            self.app.push_screen(DialogBox(
                f"'{project_path}' does not look like a Cheese3D project "
                "(no config.yaml found there)."
            ))

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
        # The GUI's project-location picker replaces the former hardcoded "."
        # (the served process's working directory), which was never surfaced
        # or selectable, so a new project always landed wherever the server
        # happened to be launched from.
        root = Path(config["root"])
        project_path = root / config["name"]
        # create the project
        Ch3DProject.initialize(config["name"], root=root)
        # read in config to overwrite
        yaml_config = ProjectConfig.load(project_path / "config.yaml")
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
            yaml_config.model = ModelConfig(
                model_config["name"],
                backend_type=model_config.get("backend_type", "dlc"),
            )
        # write yaml
        with project_path / "config.yaml" as f:
            OmegaConf.save(yaml_config, f)
        # import model if needed
        if "path" in model_config:
            backend_type = model_config.get("backend_type", "dlc")
            source_format = model_config.get("source_format", "dlc")
            if backend_type == "dlc" and source_format == "dlc":
                # Both source and target are DLC: open the project directly,
                # nothing to seed.
                project = Ch3DProject.from_path(project_path,
                                                model_import=model_config["path"])
                project._export_labels()
                yaml_config.model.name = project.model.name
                yaml_config.model.backend_options = {
                    "experimenter": project.model.experimenter,
                    "date": project.model.date
                }
            else:
                # Any other combination -- a non-DLC target, or a DLC target
                # seeded from a non-DLC source -- seeds the active backend's
                # native labels from the generic record readers/writers
                # instead of forcing a DLCBackend construction, which would
                # require deeplabcut even when the active Pixi environment
                # only has lightning_pose/sleap.
                if source_format == "dlc":
                    name, *_ = dlc_folder_to_components(model_config["path"])
                else:
                    # Lightning Pose/SLEAP source projects have no DLC-style
                    # "name-experimenter-date" folder naming to parse.
                    name = Path(model_config["path"]).name
                yaml_config.model = ModelConfig(
                    name, backend_type=backend_type,
                    backend_options={
                        "source_project_path": model_config["path"],
                        "source_format": source_format,
                    },
                )
                # build_model_backend reads backend_type/backend_options back
                # from disk, so the seeded config must be saved before this.
                with project_path / "config.yaml" as f:
                    OmegaConf.save(yaml_config, f)
                project = Ch3DProject.from_path(project_path)
                project._export_labels()
                return
            with project_path / "config.yaml" as f:
                OmegaConf.save(yaml_config, f)

    def on_show(self) -> None:
        # Former behavior let any failure here (e.g. a mismatched pose backend
        # needing a package this Pixi environment doesn't have) propagate as
        # an unhandled exception, crashing the whole app instead of returning
        # to the start menu with an explanation.
        self._failed = False
        try:
            self.create_config()
        except Exception as error:
            self._failed = True
            hint = ""
            for package, backend, env in (
                ("deeplabcut", "dlc", "dlc"),
                ("lightning_pose", "lightning_pose", "lp"),
                ("sleap", "sleap", "sleap"),
            ):
                if package in str(error):
                    hint = (
                        f"\n\nThe '{backend}' backend needs the matching Pixi "
                        f"environment ('pixi run -e {env} ...'). Either "
                        "relaunch Cheese3D there, or pick a different model "
                        "backend above."
                    )
                    break
            msg = self.query_one("#msg")
            msg.update(f"[bold red]Failed to create project:[/bold red] {error}{hint}")
            self.query_one("#loading").remove()
            self.query_one("#modal").mount(Horizontal(Button("Back", id="done", variant="error")))
            return
        # close screen
        project_path = Path(self.project_config["root"]) / self.project_config["name"]
        msg = self.query_one("#msg")
        msg.update(f"[bold]Created new project at: {project_path}[/bold]")
        self.query_one("#loading").remove()
        self.query_one("#modal").mount(Horizontal(Button("Done", id="done", variant="success")))

    @on(Button.Pressed, "#done")
    def close(self):
        if self._failed:
            self.dismiss(None)
        else:
            self.dismiss(Path(self.project_config["root"]) / self.project_config["name"])

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
        if self._uses_sleap():
            peak_threshold = float(self.query_one("#sleap_peak_threshold", Input).value)
            if not 0 <= peak_threshold <= 1:
                raise ValueError("SLEAP peak threshold must be between 0 and 1")
            settings["peak_threshold"] = peak_threshold
        if not self._uses_lightning_pose() and not self._uses_sleap():
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

    def _uses_sleap(self) -> bool:
        """Select SLEAP controls without importing SLEAP in other environments."""
        return self.project.model is not None and \
            self.project.model.__class__.__name__ == "SLEAPBackend"

    def _lightning_pose_backbone(self) -> str:
        """Return the project's configured LP backbone for the GUI default."""
        # Reading YAML directly avoids importing LP in a DLC-only environment
        # and preserves a backbone selected during an earlier training run.
        config_path = Path(self.project.model.project_path) / "config.yaml"
        try:
            configured = OmegaConf.select(OmegaConf.load(config_path), "model.backbone")
            if configured in LIGHTNING_POSE_BACKBONES:
                return str(configured)
        except (OSError, ValueError, TypeError):
            pass
        return "resnet50_animal_ap10k"

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
            augmentation_preset = str(self.query_one("#lp_imgaug", Select).value)
            if augmentation_preset == "dlc":
                augmentation_values = {
                    "rotation_probability": number("#lp_aug_rotation_probability"),
                    "rotation_degrees": number("#lp_aug_rotation_degrees"),
                    "motion_blur_probability": number("#lp_aug_motion_blur_probability"),
                    "motion_blur_kernel": integer("#lp_aug_motion_blur_kernel"),
                    "motion_blur_angle": number("#lp_aug_motion_blur_angle"),
                    "dropout_probability": number("#lp_aug_dropout_probability"),
                    "dropout_pixel_probability": number("#lp_aug_dropout_pixel_probability"),
                    "dropout_size_percent": number("#lp_aug_dropout_size_percent"),
                    "dropout_per_channel_probability": number(
                        "#lp_aug_dropout_per_channel_probability"
                    ),
                    "salt_probability": number("#lp_aug_salt_probability"),
                    "pepper_probability": number("#lp_aug_pepper_probability"),
                    "salt_pepper_pixel_probability": number(
                        "#lp_aug_salt_pepper_pixel_probability"
                    ),
                    "salt_pepper_size_min": number("#lp_aug_salt_pepper_size_min"),
                    "salt_pepper_size_max": number("#lp_aug_salt_pepper_size_max"),
                    "elastic_probability": number("#lp_aug_elastic_probability"),
                    "elastic_alpha_min": number("#lp_aug_elastic_alpha_min"),
                    "elastic_alpha_max": number("#lp_aug_elastic_alpha_max"),
                    "elastic_sigma": number("#lp_aug_elastic_sigma"),
                    "histogram_probability": number("#lp_aug_histogram_probability"),
                    "clahe_probability": number("#lp_aug_clahe_probability"),
                    "emboss_probability": number("#lp_aug_emboss_probability"),
                    "emboss_alpha_max": number("#lp_aug_emboss_alpha_max"),
                    "emboss_strength_min": number("#lp_aug_emboss_strength_min"),
                    "emboss_strength_max": number("#lp_aug_emboss_strength_max"),
                    "crop_probability": number("#lp_aug_crop_probability"),
                    "crop_percent": number("#lp_aug_crop_percent"),
                }
                augmentation = _build_lp_dlc_augmentation_config(augmentation_values)
            else:
                augmentation = augmentation_preset
            settings.update({
                "backbone": str(self.query_one("#lp_backbone", Select).value),
                "imgaug": augmentation,
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
        elif self._uses_sleap():
            settings.update({
                "backbone": str(self.query_one("#sleap_backbone", Select).value),
                "validation_fraction_percent": number("#sleap_validation_percent"),
                "val_batch_size": integer("#sleap_val_batch_size"),
                "optimizer": str(self.query_one("#sleap_optimizer", Select).value),
                "min_steps_per_epoch": integer("#sleap_min_steps_per_epoch"),
                "steps_per_epoch": integer("#sleap_steps_per_epoch"),
                "save_top_k": integer("#sleap_save_top_k"),
                "save_last": self.query_one("#sleap_save_last", Checkbox).value,
                "early_stopping": self.query_one("#sleap_early_stopping", Checkbox).value,
                "early_stop_patience": integer("#sleap_early_stop_patience"),
                "use_augmentation": self.query_one("#sleap_augmentation", Checkbox).value,
                "rotation_min": number("#sleap_rotation_min"),
                "rotation_max": number("#sleap_rotation_max"),
                "scale_min": number("#sleap_scale_min"),
                "scale_max": number("#sleap_scale_max"),
                "translate": number("#sleap_translate"),
            })
            if not 1 <= settings["validation_fraction_percent"] <= 50:
                raise ValueError("SLEAP validation percentage must be between 1 and 50")
            if settings["val_batch_size"] <= 0 or settings["min_steps_per_epoch"] <= 0 \
                    or settings["steps_per_epoch"] < 0 or settings["save_top_k"] <= 0:
                raise ValueError("SLEAP batch/step/checkpoint values are invalid")
            if settings["rotation_min"] > settings["rotation_max"]:
                raise ValueError("SLEAP minimum rotation must be <= maximum rotation")
            if settings["scale_min"] <= 0 or settings["scale_min"] > settings["scale_max"]:
                raise ValueError("SLEAP minimum scale must be positive and <= maximum scale")
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

    def _refresh_after_inference(self, _result=None) -> None:
        """Force a complete browser repaint after closing the inference modal."""
        # Textual Serve clients can retain stale disabled widget styles after a
        # long worker turn even though Python state is already enabled. Refresh
        # the tab container, screen, and application after the modal is removed.
        tabs = self.query_one("#all_tabs")
        tabs.refresh(layout=True, repaint=True)
        self.refresh(layout=True, repaint=True)
        self.app.refresh(layout=True, repaint=True)

    def _complete_inference_ui(self, message: str) -> None:
        """Atomically restore controls and present the inference completion UI."""
        self._enable_pose_done()
        self._refresh_after_inference()
        # The callback runs after dismissal, when the underlying page is visible
        # again, and fixes the stale web-client state that a browser reload did not.
        self.app.push_screen(
            DialogBox(message, button_text="Done — refresh GUI"),
            callback=self._refresh_after_inference,
        )

    def _complete_visualization_ui(self, message: str) -> None:
        """Restore and repaint the browser after a visualize-tab worker finishes.

        Shared by both Napari closing and generate_videos() completing: a long
        worker-thread turn can leave a Textual Serve client's page stale even
        though Python-side state (including a freshly pushed dialog) is
        already correct, so the same forced multi-level repaint is needed
        regardless of which visualize-tab operation just finished.
        """
        self._enable_visualize_done()
        self._refresh_after_inference()
        self.app.push_screen(
            DialogBox(message, button_text="Done — refresh GUI"),
            callback=self._refresh_after_inference,
        )

    def _complete_training_ui(self, message: str) -> None:
        """Repaint and present the completion dialog after training finishes.

        train_model() already resets its own button state (_set_training_controls)
        before this runs, since the stop button needs to stay reachable earlier
        in the same finally block; this only needs the same forced repaint the
        other long-running operations require -- training is typically the
        longest-running operation of all, and formerly pushed its completion
        dialog directly without it, leaving Textual Serve clients on a stale
        page with no way to recover except restarting the whole GUI.
        """
        self._refresh_after_inference()
        self.app.push_screen(
            DialogBox(message, button_text="Done — refresh GUI"),
            callback=self._refresh_after_inference,
        )

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
                        default_epochs = "300" if self._uses_lightning_pose() else \
                            ("100" if self._uses_sleap() else "200")
                        default_batch = "16" if self._uses_lightning_pose() else \
                            ("4" if self._uses_sleap() else "8")
                        default_lr = "0.001" if self._uses_lightning_pose() else \
                            ("0.0001" if self._uses_sleap() else "0.0005")
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
                            yield Static("Model backbone", classes="training_field_label")
                            # Changing this selection creates a different model
                            # architecture with its own compatible checkpoints.
                            yield Select(
                                [(name, name) for name in LIGHTNING_POSE_BACKBONES],
                                value=self._lightning_pose_backbone(),
                                id="lp_backbone",
                            )
                            yield Static(
                                "ViT/DINO backbones automatically use square 512×512 input.",
                                classes="training_field_label",
                            )
                            yield Static("Augmentation preset", classes="training_field_label")
                            yield Select([("DLC-style (customizable)", "dlc"),
                                          ("Default (resize only)", "default")],
                                         value="dlc", id="lp_imgaug")
                            yield Checkbox("Random horizontal flip", id="lp_hflip")
                            yield Static(
                                "DLC-style augmentation controls (ignored by resize-only Default)",
                                classes="training_field_label",
                            )
                            yield TrainingInput("Rotation probability", "lp_aug_rotation_probability",
                                                "0.4", "number")
                            yield TrainingInput("Rotation range (degrees)", "lp_aug_rotation_degrees",
                                                "25", "number")
                            yield TrainingInput("Motion blur probability",
                                                "lp_aug_motion_blur_probability", "0.5", "number")
                            yield TrainingInput("Motion blur kernel (pixels)",
                                                "lp_aug_motion_blur_kernel", "5", "integer")
                            yield TrainingInput("Motion blur angle range (degrees)",
                                                "lp_aug_motion_blur_angle", "90", "number")
                            yield TrainingInput("Coarse dropout probability",
                                                "lp_aug_dropout_probability", "0.5", "number")
                            yield TrainingInput("Dropout pixel probability",
                                                "lp_aug_dropout_pixel_probability", "0.02", "number")
                            yield TrainingInput("Dropout block size fraction",
                                                "lp_aug_dropout_size_percent", "0.3", "number")
                            yield TrainingInput("Dropout per-channel probability",
                                                "lp_aug_dropout_per_channel_probability",
                                                "0.5", "number")
                            yield TrainingInput("Coarse salt probability",
                                                "lp_aug_salt_probability", "0.5", "number")
                            yield TrainingInput("Coarse pepper probability",
                                                "lp_aug_pepper_probability", "0.5", "number")
                            yield TrainingInput("Salt/pepper pixel probability",
                                                "lp_aug_salt_pepper_pixel_probability",
                                                "0.01", "number")
                            yield TrainingInput("Salt/pepper minimum block fraction",
                                                "lp_aug_salt_pepper_size_min", "0.05", "number")
                            yield TrainingInput("Salt/pepper maximum block fraction",
                                                "lp_aug_salt_pepper_size_max", "0.1", "number")
                            yield TrainingInput("Elastic transform probability",
                                                "lp_aug_elastic_probability", "0.5", "number")
                            yield TrainingInput("Elastic alpha minimum",
                                                "lp_aug_elastic_alpha_min", "0", "number")
                            yield TrainingInput("Elastic alpha maximum",
                                                "lp_aug_elastic_alpha_max", "10", "number")
                            yield TrainingInput("Elastic sigma", "lp_aug_elastic_sigma",
                                                "5", "number")
                            yield TrainingInput("Histogram equalization probability",
                                                "lp_aug_histogram_probability", "0.1", "number")
                            yield TrainingInput("CLAHE probability", "lp_aug_clahe_probability",
                                                "0.1", "number")
                            yield TrainingInput("Emboss probability", "lp_aug_emboss_probability",
                                                "0.1", "number")
                            yield TrainingInput("Emboss maximum alpha",
                                                "lp_aug_emboss_alpha_max", "0.5", "number")
                            yield TrainingInput("Emboss minimum strength",
                                                "lp_aug_emboss_strength_min", "0.5", "number")
                            yield TrainingInput("Emboss maximum strength",
                                                "lp_aug_emboss_strength_max", "1.5", "number")
                            yield TrainingInput("Crop/pad probability", "lp_aug_crop_probability",
                                                "0.4", "number")
                            yield TrainingInput("Crop/pad maximum fraction", "lp_aug_crop_percent",
                                                "0.15", "number")
                            yield TrainingInput("Training fraction", "lp_train_prob", "0.95", "number")
                            yield TrainingInput("Validation fraction", "lp_val_prob", "0.05", "number")
                            yield TrainingInput("Backbone unfreezing epoch", "lp_unfreezing_epoch",
                                                "20", "integer")
                            yield Checkbox("Early stopping", id="lp_early_stopping")
                            yield TrainingInput("Early-stop patience", "lp_early_stop_patience",
                                                "3", "integer")
                        elif self._uses_sleap():
                            yield Static("Single-instance model backbone",
                                         classes="training_field_label")
                            yield Select.from_values(
                                SLEAP_BACKBONES, value="unet_medium_rf",
                                allow_blank=False, id="sleap_backbone",
                            )
                            yield TrainingInput("Validation split (%)",
                                                "sleap_validation_percent", "10", "number")
                            yield TrainingInput("Validation batch size",
                                                "sleap_val_batch_size", "4", "integer")
                            yield Static("Optimizer", classes="training_field_label")
                            yield Select.from_values(
                                ("Adam", "AdamW"), value="Adam", allow_blank=False,
                                id="sleap_optimizer",
                            )
                            yield TrainingInput("Minimum steps per epoch",
                                                "sleap_min_steps_per_epoch", "200", "integer")
                            yield TrainingInput("Exact steps per epoch (0 = automatic)",
                                                "sleap_steps_per_epoch", "0", "integer")
                            yield TrainingInput("Keep best N checkpoints",
                                                "sleap_save_top_k", "3", "integer")
                            yield Checkbox("Also save last checkpoint", value=True,
                                           id="sleap_save_last")
                            yield Checkbox("Early stopping", value=True,
                                           id="sleap_early_stopping")
                            yield TrainingInput("Early-stop patience",
                                                "sleap_early_stop_patience", "10", "integer")
                            yield Checkbox("Use augmentation", value=True,
                                           id="sleap_augmentation")
                            yield TrainingInput("Minimum rotation (degrees)",
                                                "sleap_rotation_min", "-15", "number")
                            yield TrainingInput("Maximum rotation (degrees)",
                                                "sleap_rotation_max", "15", "number")
                            yield TrainingInput("Minimum scale", "sleap_scale_min",
                                                "0.9", "number")
                            yield TrainingInput("Maximum scale", "sleap_scale_max",
                                                "1.1", "number")
                            yield TrainingInput("Maximum translation fraction",
                                                "sleap_translate", "0.0", "number")
                            yield Static(
                                "SLEAP epochs use at least the configured minimum steps; "
                                "batch size is per GPU under DDP.",
                                classes="training_field_note",
                            )
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
                        if self._uses_sleap():
                            yield TrainingInput("SLEAP peak confidence threshold",
                                                "sleap_peak_threshold", "0.2", "number")
                        if not self._uses_lightning_pose() and not self._uses_sleap():
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
                        # Half the machine's cores/threads by default -- using
                        # nearly all of them (formerly -2) starved the GUI's own
                        # event-loop/websocket thread badly enough during heavy
                        # multi-camera FFmpeg rendering that the browser's
                        # connection silently dropped and never recovered.
                        default_video_workers = str(max(1, (os.cpu_count() or 1) // 2))
                        yield TrainingInput("Video generation CPU core budget",
                                            "video_generation_workers",
                                            default_video_workers, "integer")
                        yield TrainingInput(
                            "Minimum keypoint probability (p)",
                            "video_keypoint_probability_threshold",
                            str(self.project.visualization.keypoint_probability_threshold),
                            "number",
                        )
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
        if "shuffle" in record and not self._uses_lightning_pose() \
                and not self._uses_sleap():
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
            progress_output = _TrainingProgressOutput(output)
            output.write(f"Starting training subprocess: {shlex.join(command)}\n")
            child_environment = os.environ.copy()
            child_environment["PYTHONUNBUFFERED"] = "1"
            if self._uses_lightning_pose():
                # Lightning Pose's DDP ranks currently leave Trainer.fit at
                # slightly different times. PyTorch's optional NCCL heartbeat
                # can therefore contact rank zero's already-closed TCPStore and
                # print a native Broken-pipe stack after successful training.
                # Disabling only this watchdog keeps NCCL/DDP computation and
                # Cheese3D's process-group Stop button fully operational.
                child_environment.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
            # A pseudo-terminal is intentional on POSIX: Lightning disables or
            # buffers its TQDM epoch bar when stdout is an ordinary subprocess
            # pipe.  The PTY makes the trainer behave exactly as it does when
            # launched directly from a terminal while Cheese3D can still mirror
            # every update into both the GUI log and the launching terminal.
            if os.name == "posix":
                # A real window size is required too: os.openpty() alone still
                # leaves the progress bar invisible (see _open_sized_pty).
                child_environment.setdefault("COLUMNS", "120")
                child_environment.setdefault("LINES", "24")
                master_fd, slave_fd = _open_sized_pty()
                try:
                    self._training_process = subprocess.Popen(
                        command, stdout=slave_fd, stderr=slave_fd,
                        env=child_environment, start_new_session=True,
                        close_fds=True,
                    )
                finally:
                    # Only the child owns the slave; retaining it here prevents
                    # EOF on the master after all trainer ranks have exited.
                    os.close(slave_fd)
                try:
                    with os.fdopen(master_fd, "r", encoding="utf-8",
                                   errors="replace", buffering=1) as stream:
                        # Character streaming preserves carriage-return TQDM
                        # refreshes instead of withholding them until an epoch ends.
                        while True:
                            try:
                                character = stream.read(1)
                            except OSError as error:
                                # Linux PTY masters report EIO, rather than an
                                # empty read, after the final slave is closed.
                                if error.errno == errno.EIO:
                                    break
                                raise
                            if not character:
                                break
                            progress_output.feed(character)
                finally:
                    progress_output.flush()
            else:
                # Retain the pipe fallback for platforms without POSIX PTYs.
                self._training_process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=child_environment,
                    start_new_session=False,
                )
                if self._training_process.stdout is not None:
                    for character in iter(
                            lambda: self._training_process.stdout.read(1), ""):
                        progress_output.feed(character)
                    progress_output.flush()
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
        self.app.call_from_thread(self._complete_training_ui, message)

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
        """Run calibration off-thread while all widget changes stay on Textual's UI thread."""
        self.app.call_from_thread(self._disable_pose_in_progress)
        log = self.app.call_from_thread(self.query_one, "#pose_log")
        self.app.call_from_thread(log.clear)  # type: ignore
        try:
            with _pipeline_output(log):
                self.project.calibrate()
        except Exception as error:
            with _pipeline_output(log):
                traceback.print_exc()
            message = f"Camera calibration failed: {error}"
        else:
            message = "Camera calibration completed!"
        finally:
            self.app.call_from_thread(self._complete_inference_ui, message)

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
            # One UI-thread transaction restores state, repaints the served page,
            # and opens a completion popup. Separate calls formerly allowed the
            # browser to remain visually disabled after inference returned.
            self.app.call_from_thread(self._complete_inference_ui, message)

    @on(Button.Pressed, "#triangulate")
    @work(thread=True)
    def triangulate(self):
        """Run triangulation off-thread while all widget changes stay on Textual's UI thread."""
        self.app.call_from_thread(self._disable_pose_in_progress)
        log = self.app.call_from_thread(self.query_one, "#pose_log")
        self.app.call_from_thread(log.clear)  # type: ignore
        try:
            with _pipeline_output(log):
                self.project.triangulate()
        except Exception as error:
            with _pipeline_output(log):
                traceback.print_exc()
            message = f"3D triangulation failed: {error}"
        else:
            message = "3D triangulation completed!"
        finally:
            self.app.call_from_thread(self._complete_inference_ui, message)

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
                probability_threshold = float(self.app.call_from_thread(
                    lambda: self.query_one(
                        "#video_keypoint_probability_threshold", Input
                    ).value
                ))
                if not 0.0 <= probability_threshold <= 1.0:
                    raise ValueError("Minimum keypoint probability must be between 0 and 1")
                # Persist the GUI choice so CLI generation and future sessions
                # use the same cutoff instead of reverting to the old default.
                self.project.visualization.keypoint_probability_threshold = probability_threshold
                _persist_visualization_threshold(self.project.path, probability_threshold)
                completed = self.project.generate_videos(
                    max_workers=max_workers,
                    probability_threshold=probability_threshold,
                )
        except Exception as exc:
            with _pipeline_output(log):
                traceback.print_exc()
            message = f"Video generation failed: {exc}"
        else:
            message = f"Video generation completed: {completed} output(s) available."
        finally:
            # This formerly pushed its completion dialog directly, without the
            # same forced multi-level repaint track()/triangulate() already
            # needed via _complete_visualization_ui: Textual Serve clients can
            # retain a stale, un-repainted page after a long worker-thread
            # turn even though Python state (including the pushed dialog) is
            # already correct -- appearing completely frozen with no way to
            # recover except restarting the whole GUI.
            self.app.call_from_thread(self._complete_visualization_ui, message)

    @on(Button.Pressed, "#visualize")
    @work
    async def visualize(self):
        """Open Napari and force the served GUI to recover after it closes."""
        self._disable_visualize_in_progress()
        try:
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
        except Exception as error:
            traceback.print_exc(file=sys.__stderr__)
            message = f"Visualization failed: {error}"
        else:
            message = "Visualization completed!"
        finally:
            # The completion transaction runs only after Napari's event loop has
            # returned, ensuring the browser receives fresh enabled widget state.
            self._complete_visualization_ui(message)

class Cheese3dApp(App):
    """Interactive Cheese3D TUI via Textual."""

    BINDINGS = [
        ("q", "quit", "Quit the GUI"),
        ("d", "toggle_dark", "Toggle dark mode")
    ]

    CSS_PATH = "interactive_styles/app.css"

    def __init__(self, start_directory: Optional[str | Path] = None, *args, **kwargs):
        """Create the GUI with one deterministic project-picker start directory."""
        self.start_directory = Path(
            Path.home() if start_directory is None else start_directory
        ).expanduser().resolve()
        if not self.start_directory.is_dir():
            self.start_directory = Path.home()
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.title = "Cheese3D Interative GUI"
        self.sub_title = "Use mouse or keyboard to navigate"
        self.push_screen(StartMenu())


def _open_web_ui(url: str) -> None:
    """Open the served Textual application in the user's default browser."""
    if not webbrowser.open(url):
        print(f"Could not open a browser automatically. Open {url} manually.")


def _register_freeze_dump_handler() -> Path:
    """Let `kill -USR1 <pid>` dump every thread's stack, no ptrace required.

    py-spy/gdb need ptrace, which the default Yama LSM policy
    (ptrace_scope=1) restricts to a process's own direct parent -- useless
    for attaching to an already-running Cheese3D process from a fresh shell
    without root. Sending a signal to your own process needs no special
    permission, and faulthandler.dump_traceback(all_threads=True) shows
    exactly what py-spy would: every thread's current Python stack,
    including the asyncio event loop and any @work(thread=True) worker.
    """
    log_path = Path.home() / ".cheese3d" / "freeze_dump.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    faulthandler.register(signal.SIGUSR1, file=log_file, all_threads=True, chain=False)
    return log_path


def run_interative(web_mode=True, open_browser=True,
                   start_directory: Optional[str | Path] = None):
    """Launch Cheese3D while preserving the project root across web processes."""
    freeze_dump_path = _register_freeze_dump_handler()
    print(f"If the GUI ever freezes: kill -USR1 <pid> dumps every thread's "
          f"stack to {freeze_dump_path}")
    start_directory = Path(
        Path.home() if start_directory is None else start_directory
    ).expanduser().resolve()
    if not start_directory.is_dir():
        start_directory = Path.home()
    if web_mode:
        url = "http://localhost:8000"
        # The served child must use terminal mode; otherwise the default web
        # mode would recursively start another Textual server.
        # Reuse this process's Python so the served child cannot fall back to a
        # different PATH entry (for example Pixi without Lightning Pose).
        child_command = shlex.join([
            sys.executable, "-m", "cheese3d", "--path", str(start_directory),
            "interactive", "--terminal"
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
        app = Cheese3dApp(start_directory=start_directory)
        app.run()

if __name__ == "__main__":
    run_interative()
