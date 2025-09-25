import os
import typer
import rich
import questionary
from pathlib import Path
from typing import Annotated, Optional, List

from cheese3d.interactive import run_interative
from cheese3d.project import Ch3DProject
from cheese3d.utils import maybe

cli = typer.Typer(no_args_is_help=True)

def _build_project(path, name, configs, overrides, **kwargs):
    full_path = Path(path) / name
    config_dir = Path(path) / configs
    overrides = maybe(overrides, [])

    return Ch3DProject.from_path(full_path, config_dir,
                                 overrides=overrides, **kwargs)

@cli.command()
def setup(name: str, path = os.getcwd()):
    """Setup a new Cheese3D project called NAME under --path."""
    Ch3DProject.initialize(name=name, root=Path(path))
    rich.print(f"Successfully initialized {name} :tada:")

@cli.command(name="import")
def import_model(
    model: Annotated[str, typer.Argument(help="Path to existing model")],
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    model_type: Annotated[str, typer.Option(
        help="Type of project to import (only 'dlc' is valid for now)."
    )] = "dlc",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    """Import an existing pose model project into NAME."""
    project = _build_project(path, name, configs, config_overrides, model_import=model)
    project._export_labels()
    rich.print(f"Done importing {model.split(os.sep)[-1]} :white_check_mark:")

@cli.command()
def summarize(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    """Summarize a Cheese3D project based on its configuration file."""
    project = _build_project(path, name, configs, config_overrides)
    project.summarize()

@cli.command()
def sync(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    """Synchronize the video (and possibly ephys) files in a Cheese3D project."""
    project = _build_project(path, name, configs, config_overrides)
    project.synchronize()
    rich.print("Synchronization completed! :white_check_mark:")

@cli.command()
def extract(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None,
    manual: Annotated[bool, typer.Option(help="Set to manual frame picking GUI")] = False,
):
    """Extract frames from video data."""
    project = _build_project(path, name, configs, config_overrides)
    if manual:
        choices = [questionary.Choice(title=f"session: {k.session}, name: {k.name}",
                                      value=k)
                   for k in project.recordings.keys()]
        choices.append(questionary.Choice(title="exit (q)", value="exit", shortcut_key="q"))
        while True:
            chosen = questionary.select("Which recording would you like to extract frames for?",
                                        choices=choices).ask()
            if (chosen == "exit") or (chosen is None):
                break
            else:
                project.extract_frames(recordings=[chosen], manual=True)
    else:
        project.extract_frames(manual=False)
    rich.print("Frames extracted! :white_check_mark:")

@cli.command()
def label(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    """Label extracted frames for model training."""
    project = _build_project(path, name, configs, config_overrides)
    project.label_frames()
    rich.print("Labeling complete! :white_check_mark:")

@cli.command()
def train(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    gpu: Annotated[int, typer.Option(help="GPU ID(s) to use")] = 0,
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    """Train 2d pose model."""
    project = _build_project(path, name, configs, config_overrides)
    project.train(gpu)
    rich.print("Training complete :spaceship:")

@cli.command()
def calibrate(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    project = _build_project(path, name, configs, config_overrides)
    rich.print("Calibrating ...")
    project.calibrate()
    rich.print("Calibration complete :white_check_mark:")

@cli.command()
def track(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    project = _build_project(path, name, configs, config_overrides)
    rich.print("Tracking 2D pose ...")
    project.track()
    rich.print("Pose estimation complete :white_check_mark:")

@cli.command()
def triangulate(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    project = _build_project(path, name, configs, config_overrides)
    rich.print("Triangulating ...")
    project.triangulate()
    rich.print("Triangulation complete :white_check_mark:")

@cli.command()
def analyze(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    project = _build_project(path, name, configs, config_overrides)
    project.calibrate()
    project.track()
    project.triangulate()

@cli.command()
def visualize(
    name: Annotated[str, typer.Argument(help="Name of project")] = ".",
    path: Annotated[str, typer.Option(help="Path to project directory")] = os.getcwd(),
    configs: Annotated[str, typer.Option(
        help="Path to additional configs (relative to project)"
    )] = "configs",
    config_overrides: Annotated[Optional[List[str]], typer.Argument(
        help="Config overrides passed to Hydra (https://hydra.cc/docs/intro/)"
    )] = None
):
    project = _build_project(path, name, configs, config_overrides)
    project.visualize()

@cli.command()
def interactive(
    web: Annotated[bool, typer.Option(help="Set to enable web mode UI")] = False,
):
    run_interative(web_mode=web)
