import glob

from pathlib import Path

from .utils import *
from .actions import *
from .errors import *


class VDGlob:
    def __init__(self, text):
        self.txt = os.path.expanduser(text)

    def __repr__(self):
        return f'VDGlob({self.text})'

    @property
    def text(self):
        return shrinkuser(self.txt)

    def glob(self):
        ret = glob.glob(self.txt, recursive=True)
        return fsorted(ret)


class VDPath:
    def __init__(self, text):
        self.txt = text
        self.path = Path(text).expanduser()

    def __repr__(self):
        return f'VDPath({self.text})'

    def __eq__(self, other):
        return type(self) == type(other) and self.text == other.text

    @property
    def text(self):
        if not self.txt:
            return '.'

        ret = self.txt.rstrip('/')

        # Add postfix to display text
        if self.isdir and not self.txt.endswith('*'):
            ret += '/'

        return shrinkuser(ret)

    def __eq__(self, other):
        return self.text == other.text

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
            return []

        if not self.isdir:
            return ['.'] if not self.txt else [self.text]

        ret = []

        children = fsorted(p.name for p in self.path.iterdir())
        for child in children:
            if child.startswith('.') and not include_hidden:
                continue

            ret.append(child if not self.txt
                    else os.path.join(self.text, child)
                    )

        if not ret:
            ret = ['.'] if not self.txt else [self.text]

        return ret


class TrackingItem:
    def __init__(self, iii, text, mark=None):
        # III = Inventory Item Index
        self.iii = iii

        if not mark or not isinstance(mark, str) or mark not in '#*+@':
            self.mark = '.'
        else:
            self.mark = mark

        self.path = VDPath(text)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.iii == other.iii and
                self.mark == other.mark and
                self.path == other.path)

    def __repr__(self):
        return f'{self.mark.ljust(1)} {self.iii} [{self.path}]'

    def __getattr__(self, attr):
        if hasattr(self.path, attr):
            return getattr(self.path, attr)

    @property
    def type(self):
        if not self.exists:
            return 9
        if self.isdir:
            return 1
        if self.isfile and self.isexecutable:
            return 2
        if self.islink:
            return 3
        if self.isfifo:
            return 4
        return 0


class Inventory:
    def __init__(self):
        self.content = []

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index):
        return self.content[index]

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return False

        return self.content == other.content

    def append(self, text, iii=None, mark=None):
        if text is None:
            if (self.content or [None])[-1] is not None:
                self.content.append(None)

        elif isinstance(text, TrackingItem):
            self.content.append(text)

        elif iii is not None:
            self.content.append(TrackingItem(int(iii, 10), text, mark=mark))

        elif '*' in text.replace('[*]', '_'):
            self.content.append(VDGlob(text))

        else:
            self.content.append(VDPath(text.replace('[*]', '*')))

    def freeze(self):
        while self.content and self.content[0] is None:
            self.content.pop(0)
        while self.content and self.content[-1] is None:
            self.content.pop(-1)

        path_iii_mapping = {}

        offset = 10 ** (len(str(len(self.content))))
        iii = 1
        for item in self.content:
            if isinstance(item, TrackingItem) and item.iii is None:
                if item.text in path_iii_mapping:
                    item.iii = path_iii_mapping[item.text]
                else:
                    item.iii = (offset + iii) * 10 + item.type
                    path_iii_mapping[item.text] = item.iii
                    iii += 1


class ItemChange:
    def __init__(self, src):
        self.src = src
        self.dst = []

    def append(self, dst):
        self.dst.append(dst)

    @property
    def changed(self):
        return [self.src] != self.dst
