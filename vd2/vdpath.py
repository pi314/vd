import os
import glob

from os.path import expanduser, join
from pathlib import Path

from .utils import *


class VDGlob:
    def __init__(self, text):
        self.txt = expanduser(text)

    def __repr__(self):
        return f'VDGlob({self.text})'

    @property
    def text(self):
        return shrinkuser(self.txt)

    def glob(self):
        ret = glob.glob(self.txt, recursive=True)
        return natsorted(ret)


class VDPath:
    def __init__(self, text):
        self.txt = text
        self.path = Path(text.rstrip('|/')).expanduser()

    def __repr__(self):
        return f'VDPath({self.text})'

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, VDPath):
            return self.path == other.path
        return self.path == Path(other).expanduser()

    @property
    def text(self):
        if not self.txt:
            return '.'

        ret = self.txt.rstrip('|/')

        # Add postfix to display text
        if self.isdir:
            ret += '/'
        elif self.isfifo:
            ret += '|'

        return shrinkuser(ret)

    @property
    def inode(self):
        if self.exists:
            return self.path.stat(follow_symlinks=False).st_ino

    @property
    def realpath(self):
        if self.islink:
            return str(self.path.parent.resolve() / self.path.name)
        return str(self.path.resolve())

    @property
    def exists(self):
        return self.path.exists() or self.islink

    @property
    def isdir(self):
        return self.path.is_dir() and not self.islink

    @property
    def isfile(self):
        return self.path.is_file() and not self.islink

    @property
    def isfifo(self):
        return self.path.is_fifo() and not self.islink

    @property
    def isexecutable(self):
        return os.access(self.path, os.X_OK)

    @property
    def islink(self):
        return self.path.is_symlink()

    def listdir(self, include_hidden):
        if not self.exists:
            return [self.txt]

        if not self.isdir:
            return ['.'] if not self.txt else [self.text]

        ret = []

        children = natsorted(p.name for p in self.path.iterdir())
        for child in children:
            if child.startswith('.') and not include_hidden:
                continue

            ret.append(child if not self.txt
                    else join(self.text, child)
                    )

        if not ret:
            ret = ['.'] if not self.txt else [self.text]

        return ret


class VDLink:
    def __init__(self, lnk_text, ref_text=None):
        self.lnk_text = lnk_text
        self.lnk = VDPath(self.lnk_text)

        self.ref_text = ref_text or os.readlink(self.lnk.path)
        self.ref = VDPath(self.ref_text)

    def __repr__(self):
        return f'VDLink({self.lnk} -> {self.ref})'

    def __hash__(self):
        return hash(self.lnk)

    def __eq__(self, other):
        if isinstance(other, VDPath):
            return self.lnk == other
        if isinstance(other, VDLink):
            return (self.lnk, self.ref) == (other.lnk, other.ref)
        return self.lnk == other

    @property
    def text(self):
        return self.lnk.text + ' -> ' + self.ref.text

    def __getattr__(self, attr):
        return getattr(self.lnk, attr)
