from omegaconf import OmegaConf
from typing import Optional, List
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from textual import work, on
from textual.app import App, ComposeResult
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
                             SelectionList)
from textual_fspicker import SelectDirectory

from cheese3d.config import _DEFAULT_VIDEO_REGEX
from cheese3d.utils import maybe, reglob
from cheese3d.project import Ch3DProject, RecordingKey
from cheese3d.config import ProjectConfig, ModelConfig

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
- [bold]"select recordings":[/bold] select video recordings to include in project
- [bold]"model":[/bold] model-related actions like labeling frames and training
- [bold]"pose estimation":[/bold] analysis-related actions like camera
    calibration, keypoint tracking, and triangulation
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

class LabeledInput(Input):

    label = reactive("")

    def __init__(self, label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    def watch_label(self, label):
        self.border_title = label

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
        self.fields.mount(KeyValuePair())

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
            "recording_root": self.query_one("#video_dir").value,
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
        model_path = await self.app.push_screen_wait(SelectDirectory())
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
            model_path = await self.app.push_screen_wait(SelectDirectory())
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
        project_path = await self.app.push_screen_wait(SelectDirectory())
        if project_path is not None:
            self.app.push_screen(MainScreen(project_path))

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
            yield EphysWizard()
            yield ModelWizard()
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

class LabelFramesScreen(ModalScreen):
    def __init__(self, project: Ch3DProject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project

    def on_show(self) -> None:
        self.project.label_frames()
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Horizontal(Static("[bold]Labeling frames ... close napari to dismiss.[/bold]"))

class MainScreen(Screen):
    def __init__(self, project_path: str | Path):
        self.project = Ch3DProject.from_path(project_path)
        super().__init__()

    def on_mount(self) -> None:
        self._refresh_summary()

    def _check_in_sessions(self, session: str):
        k = RecordingKey(session, "")
        return any(k2.matches(k) for k2 in self.project.recordings.keys())

    def _refresh_summary(self):
        summary_log = self.query_one("#summary_log")
        summary_log.clear()
        self.project.summarize(summary_log)

    def _refresh_recording_list(self):
        config = ProjectConfig.load(self.project.path / "config.yaml")
        recording_root = str(self.project.path / config.recording_root) # type: ignore
        self.query_one("#recording_path").update(
            f"[bold]Available recordings under:[/bold] {recording_root}"
        )
        recordings = reglob(".*", path=recording_root)
        recordings = [Path(p) for p in recordings]
        in_project = [self._check_in_sessions(p.name) for p in recordings]
        select_list = self.query_one("#select_recordings")
        select_list.clear_options()
        select_list.add_options([(path.name, path, select)
                                 for path, select in zip(recordings, in_project)])

    def _enable_model_done(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#model_buttons").children:
            button.disabled = False

    def _disable_model_in_progress(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = True
        for button in self.query_one("#model_buttons").children:
            button.disabled = True

    def _enable_pose_done(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = False
        for button in self.query_one("#pose_buttons").children:
            button.disabled = False

    def _disable_pose_in_progress(self):
        self.query_one("#all_tabs").query_one("ContentTabs").disabled = True
        for button in self.query_one("#pose_buttons").children:
            button.disabled = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Collapsible(classes="helpmenu", title="help", collapsed=True):
                yield Static(_MAIN_HELP_MSG, id="main_help")
            with TabbedContent(initial="summary", id="all_tabs"):
                with TabPane(title="summary", id="summary"):
                    yield RichConsole(id="summary_log")
                with TabPane(title="select recordings", id="recordings"):
                    with Vertical(id="recordings_list"):
                        yield Static("", id="recording_path")
                        yield SelectionList[str](id="select_recordings")
                with TabPane(title="model", id="model"):
                    with Vertical():
                        with CenterMiddle(classes="buttons_group"):
                            with HorizontalGroup(id="model_buttons"):
                                yield Button("Extract frames", id="extract")
                                yield Button("Label frames", id="label")
                                yield Button("Train network", id="train")
                        yield TextualStdout(id="model_log")
                with TabPane(title="pose estimation", id="pose"):
                    with Vertical():
                        with CenterMiddle(classes="buttons_group"):
                            with HorizontalGroup(id="pose_buttons"):
                                yield Button("Calibrate", id="calibrate")
                                yield Button("Track", id="track")
                                yield Button("Triangulate", id="triangulate")
                                yield Button("Visualize", id="visualize")
                        yield TextualStdout(id="pose_log")
        yield Footer()

    @on(TabbedContent.TabActivated)
    def update_tabs(self, msg: TabbedContent.TabActivated):
        self.project = Ch3DProject.from_path(self.project.path)
        if msg.pane.id == "summary":
            self._refresh_summary()
        elif msg.pane.id == "recordings":
            self._refresh_recording_list()

    @on(SelectionList.SelectedChanged, "#select_recordings")
    def update_selected_recordings(self, msg: SelectionList.SelectedChanged):
        config = ProjectConfig.load(self.project.path / "config.yaml")
        current = set(recording["name"] for recording in config.recordings) # type: ignore
        selections = [selection.name for selection in msg.selection_list.selected]
        new_recordings = []
        for recording in config.recordings: # type: ignore
            name = recording.get("name", "")
            if name in selections:
                new_recordings.append(recording)
        for selection in selections:
            if selection not in current:
                new_recordings.append({"name": selection}) # type: ignore
        config.recordings = new_recordings # type: ignore
        OmegaConf.save(config, self.project.path / "config.yaml")

    @on(Button.Pressed, "#extract")
    @work(thread=True)
    def extract_frames(self):
        self._disable_model_in_progress()
        log = self.query_one("#model_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.extract_frames()
        self._enable_model_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("Frame extraction completed!"))

    @on(Button.Pressed, "#label")
    def label_frames(self):
        self.app.push_screen(LabelFramesScreen(self.project))

    @on(Button.Pressed, "#train")
    @work(thread=True)
    def train_model(self):
        self._disable_model_in_progress()
        log = self.query_one("#model_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.train(0)
        self._enable_model_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("Model training completed!"))

    @on(Button.Pressed, "#calibrate")
    @work(thread=True)
    def calibrate(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.calibrate()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("Camera calibration completed!"))

    @on(Button.Pressed, "#track")
    @work(thread=True)
    def track(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.track()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("2D pose tracking completed!"))

    @on(Button.Pressed, "#triangulate")
    @work(thread=True)
    def triangulate(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.triangulate()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("3D triangulation completed!"))

    @on(Button.Pressed, "#visualize")
    @work(thread=True)
    def visualize(self):
        self._disable_pose_in_progress()
        log = self.query_one("#pose_log")
        log.clear() # type: ignore
        with redirect_stdout(log), redirect_stderr(log): # type: ignore
            self.project.visualize()
        self._enable_pose_done()
        self.app.call_from_thread(self.app.push_screen, DialogBox("Visualization completed!"))

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


def run_interative(web_mode = False):
    if web_mode:
        server = Server("cheese3d interactive")
        server.serve()
    else:
        app = Cheese3dApp()
        app.run()

if __name__ == "__main__":
    run_interative()
