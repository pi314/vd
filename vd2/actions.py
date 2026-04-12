import shutil

from pathlib import Path

from . import logger
from .vdpath import *
from .inventory import *
from .paints import *


class TicketPool:
    def __init__(self):
        self.by_path = {}
        self.ticket_list = []

    @property
    def paths(self):
        return self.by_path.keys()

    def __bool__(self):
        return bool(self.ticket_list)

    def __iter__(self):
        return iter(self.ticket_list)

    def to_path(self, arg):
        if isinstance(arg, VDPath):
            return arg.path
        if isinstance(arg, TrackingItem):
            return self.to_path(arg.path)
        if isinstance(arg, Path):
            return arg
        return Path(arg)

    def reserve(self, path):
        self.by_path[self.to_path(path)] = {}

    def register(self, *args):
        tag_path_list = []
        action = None
        for arg in args:
            if isinstance(arg, tuple):
                tag_path_list.append(arg)
            else:
                action = arg

        ticket = Ticket(action)

        for tag, path in tag_path_list:
            path = self.to_path(path)

            if path not in self.by_path:
                self.by_path[path] = {}
            if tag not in self.by_path[path]:
                self.by_path[path][tag] = []

            self.by_path[path][tag].append(ticket)
            ticket.participants.append(path)

        self.ticket_list.append(ticket)


class Ticket:
    def __init__(self, action=None, *participants):
        self.action = action
        self.participants = list(participants)

    def __repr__(self):
        return f'({self.action}, {self.participants})'


class VirtualAction:
    def __init__(self, *targets):
        self.targets = targets

    @property
    def src(self):
        return self.targets[0]

    @property
    def dst(self):
        return self.targets[1]

    def __repr__(self):
        return '<{} {}>'.format(
                self.__class__.__name__,
                ', '.join('[{}]'.format(t) for t in self.targets))


class MetaAction(VirtualAction):
    pass


class FSAction(VirtualAction):
    pass


class TrackAction(MetaAction):
    pass


class NoAction(MetaAction):
    pass


class ResolveLinkAction(MetaAction):
    pass
    # os.readlink(target)


class UntrackAction(MetaAction):
    pass


class GlobAction(MetaAction):
    pass


class GlobAllAction(MetaAction):
    pass


def trim_empty_folder(path):
    try:
        cwd = Path.cwd().resolve()
        for probe in path.resolve().parents:
            # Delete .DS_Store if present
            try:
                (probe / '.DS_Store').unlink()
            except:
                pass

            if probe == cwd:
                # dont delete cwd
                return True

            if not probe.is_dir():
                # something weird happen
                return

            for child in probe.iterdir():
                # if probe/ is not empty, return
                return True

            # probe/ is empty, delete it
            logger.cmd(['rmdir', probe])
            probe.rmdir()
    except:
        return


class DeleteAction(FSAction):
    def preview(self):
        logger.info(red('Delete:') + red('[') + self.src + red(']'))

    def apply(self):
        path = Path(self.src)
        try:
            if not path.is_dir() or path.is_symlink():
                logger.cmd(['rm', path])
                path.unlink()
            else:
                logger.cmd(['rm', '-r', path])
                shutil.rmtree(path)
        except:
            return False

        return trim_empty_folder(path)


class RenameAction(FSAction):
    def preview(self):
        if self.src == self.dst:
            return
        logger.info(yellow('Rename:') + yellow('[') + self.src + yellow(']'))
        logger.info(yellow('└─────►') + yellow('[') + self.dst + yellow(']'))

    def apply(self):
        if self.src == self.dst:
            return
        src = Path(self.src)
        dst = Path(self.dst)
        try:
            if not dst.parent.exists():
                logger.cmd(['mkdir', '-p', dst.parent])
                dst.parent.mkdir(parents=True, exist_ok=True)

            logger.cmd(['mv', src, dst])
            src.rename(dst)

            return trim_empty_folder(src)

        except:
            return False


class CopyAction(FSAction):
    def preview(self):
        logger.info(yellow('Copy:') + yellow('[') + self.src + yellow(']'))
        logger.info(yellow('└───►') + yellow('[') + self.dst + yellow(']'))

    def apply(self):
        src = Path(self.src)
        dst = Path(self.dst)
        try:
            if not dst.parent.exists():
                logger.cmd(['mkdir', '-p', dst.parent])
                dst.parent.mkdir(parents=True, exist_ok=True)

            logger.cmd(['cp', src, dst])
            src.copy(dst, follow_symlinks=False)
        except:
            return False


class DominoRenameAction(FSAction):
    pass


class RotateRenameAction(FSAction):
    pass
