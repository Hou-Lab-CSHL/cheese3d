import os
import re
from glob import glob

def unzip(iter):
    return tuple(list(x) for x in zip(*iter))

def maybe(this, that):
    return that if this is None else this

def reglob(pattern, path = None, recursive = False):
    """
    `glob` a filesystem using regex patterns.

    Arguments:
    - `pattern`: a regex pattern compatible with Python's `re`
    - `path`: the root under which to search
    - `recursive`: set to true to search the path recursively
    """
    path = os.getcwd() if path is None else path
    files = glob(os.sep.join([path, "**"]), recursive=recursive)
    regex = re.compile(pattern)

    return sorted([f for f in files if regex.search(f) is not None])
