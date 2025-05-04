import os
import typer
import rich
from pathlib import Path
from typing import Annotated, Optional, List

from cheese3d.project import Ch3DProject
from cheese3d.utils import maybe

cli = typer.Typer(no_args_is_help=True)

@cli.command()
def setup(name: str, path = os.getcwd()):
    """Setup a new Cheese3D project called NAME under --path."""
    Ch3DProject.initialize(name=name, root=Path(path))
    rich.print(f"Successfully initialized {name} :tada:")

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
    full_path = Path(path) / name
    config_dir = Path(path) / configs
    overrides = maybe(config_overrides, [])
    project = Ch3DProject.from_path(full_path, config_dir, overrides=overrides)
    project.summarize()
