import os
import shutil

from pathlib import Path


_dirs = []
def pushd(d):
    if not _dirs:
        _dirs.append(os.getcwd())

    _dirs.append(d)
    os.chdir(d)


def popd():
    _dirs.pop()
    os.chdir(_dirs[-1])


class EmptyFile:
    def __call__(self, path):
        Path(path).absolute().parent.mkdir(exist_ok=True)
        with open(path, 'w'):
            pass


class SymlinkTo:
    def __init__(self, dest):
        self.dest = dest

    def __call__(self, link):
        os.symlink(self.dest, link)


class Dir:
    def __init__(self, content=None):
        self.content = content

    def __call__(self, path):
        Path(path).mkdir()

        if self.content:
            for child_path, ctor in self.content.items():
                ctor(path / child_path)


class FIFO:
    def __call__(self, path):
        os.mkfifo(path)


class TestChamber:
    def __init__(self, path='test_chamber'):
        self.path = Path(path).absolute()

    def create(self, content):
        if self.path.exists():
            shutil.rmtree(self.path)

        self.path.mkdir()

        for path, ctor in content.items():
            ctor(self.path / path)

    def destroy(self):
        shutil.rmtree(self.path)
