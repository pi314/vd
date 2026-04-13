import os
from pathlib import Path

from .internal_utils import exporter
export, __all__ = exporter()


@export
def cwd(path=None):
    try:
        if path is None:
            here = Path.cwd()
            return here

        os.chdir(path)
        return Path.cwd()

    except:
        return False


dir_stack = []

@export
class pushd:
    def __init__(self, path=None):
        here = cwd()
        if not here:
            self.succ = False

        else:
            self.succ = cwd(path)
            if self.succ:
                dir_stack.append(here)

    def __bool__(self):
        return self.succ

    def __enter__(self):
        return self.succ

    def __exit__(self, exc_type, exc_value, traceback):
        popd()


@export
def popd(all=False):
    if not dir_stack:
        return False

    try:
        if all:
            os.chdir(dir_stack[0])
            dir_stack.clear()

        else:
            os.chdir(dir_stack.pop())

        return True

    except:
        pass

    return False


@export
def dirs(clear=False):
    if clear:
        dir_stack.clear()
    return list(dir_stack) + [cwd()]


@export
def home():
    return Path.home()


@export
def shrinkuser(path):
    path = str(path)
    trailing_slash = '/' if path.endswith('/') else ''

    import os.path
    path = path.rstrip('/') + '/'
    homepath = os.path.expanduser('~').rstrip('/') + '/'
    if path.startswith(homepath):
        ret = os.path.join('~', path[len(homepath):])
    else:
        ret = path
    return ret.rstrip('/') + trailing_slash
