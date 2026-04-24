from pathlib import Path

from .utils import *
from .errors import *
from .vdpath import *


class TrackingItem:
    def __init__(self, iii, text, mark=None):
        # III = Inventory Item Index
        self.iii = iii

        if not mark or not isinstance(mark, str) or mark not in '#*+@':
            self.mark = '.'
        else:
            self.mark = mark

        if isinstance(text, (VDPath, VDGlob, VDLink)):
            self.path = text
        else:
            self.path = VDPath(text)
            if self.path.islink:
                self.path = VDLink(text)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.iii == other.iii and
                self.mark == other.mark and
                self.path == other.path)

    def __repr__(self):
        return f'{self.mark.ljust(1)} {self.iii} [{self.path}]'

    def __getattr__(self, attr):
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
            if self.path.path.exists():
                return 3
            else:
                return 4
        if self.isfifo:
            return 5
        return 0


class Inventory:
    def __init__(self):
        self.content = []

    def __len__(self):
        return len(self.content)

    def __bool__(self):
        return bool(self.content)

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

        elif isinstance(text, (VDPath, VDGlob, VDLink)):
            self.content.append(text)

        else:
            vdpath = VDPath(text)
            if vdpath.islink:
                self.content.append(VDLink(text))
            else:
                self.content.append(vdpath)

    def contains(self, path):
        if isinstance(path, VDPath):
            vdpath = path
        elif isinstance(path, VDLink):
            vdpath = path.lnk
        else:
            vdpath = VDPath(path)

        for item in self.content:
            if not isinstance(item, TrackingItem) or item.mark != '.':
                continue

            if item.path == vdpath:
                return True
        return False

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
