import os
import typer
import logging
import rich
from shutil import rmtree
from pathlib import Path

from cheese3d.project import Ch3DProject

cli = typer.Typer(no_args_is_help=True)

@cli.command()
def setup(name: str, path = os.getcwd()):
    """Setup a new Cheese3D project called NAME under --path."""
    project = Ch3DProject(name=name, root=Path(path))
    project.initialize()
    rich.print(f"Successfully initialized {name} :tada:")

@cli.command()
def clean(name: str, path = os.getcwd()):
    """Completely delete a Cheese3D project called NAME under --path.

    Note that this is the same as deleting the folder manually.
    """
    project = Ch3DProject(name=name, root=Path(path))
    rmtree(project.path, ignore_errors=True)

if __name__ == "__main__":
    cli()
