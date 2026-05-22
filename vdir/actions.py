import shutil
import subprocess

from pathlib import Path

from . import logger
from .utils import *
from .vdpath import *
from .inventory import *
from .fscmd import *


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

    def __getitem__(self, key):
        return self.by_path.get(self.to_path(key), {})

    def to_path(self, arg):
        if isinstance(arg, (VDPath, VDLink)):
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

            if tag == 'nop' and self.by_path[path][tag]:
                continue

            self.by_path[path][tag].append(ticket)
            ticket.participants.append(path)

        if not isinstance(action, NoAction):
            self.ticket_list.append(ticket)

    def deregister(self, ticket):
        self.ticket_list.remove(ticket)

    def replace(self, old, new):
        self.ticket_list.remove(old)
        if new not in self.ticket_list:
            self.ticket_list.append(new)

        for path in old.participants:
            for tag, ticket_list in self.by_path[path].items():
                if old not in ticket_list:
                    continue
                ticket_list.remove(old)
                ticket_list.append(new)


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

    def __getitem__(self, index):
        return self.targets[index]

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        return '<{} {}>'.format(
                self.__class__.__name__,
                '[' + ', '.join('{}'.format(repr(t)) for t in self.targets) + ']')


class InvAction(VirtualAction):
    pass


class FSAction(VirtualAction):
    pass


class TrackAction(InvAction):
    def preview(self):
        if isinstance(self.src, VDShCmd):
            what = cyan('$(') + self.src.txt + cyan(')')
        else:
            what = cyan('[') + self.src.txt + cyan(']')
        logger.info(cyan('Track:') + what)


class NoAction(InvAction):
    pass


class ResolveLinkAction(InvAction):
    def preview(self):
        logger.info(cyan('Resolve:') + cyan('[') + self.src.txt + cyan(']'))


class UntrackAction(InvAction):
    def preview(self):
        logger.info(cyan('Untrack:') + cyan('[') + self.src.txt + cyan(']'))


class GlobAction(InvAction):
    def preview(self):
        logger.info(cyan('Expand:') + cyan('[') + self.src.txt + cyan(']'))


class GlobAllAction(InvAction):
    def preview(self):
        logger.info(cyan('ExpandAll:') + cyan('[') + self.src.txt + cyan(']'))


class SortInventoryAction(InvAction):
    def preview(self):
        logger.info(cyan('Sort:') + cyan('[') + self.src.text + cyan(']'))


class DeleteAction(FSAction):
    def preview(self):
        logger.info(red('Delete:') + red('[') + self.src.txt + red(']'))

    def apply(self):
        try:
            return DeleteCommand(self.src)()
        except Exception as e:
            logger.error(e)
            return False


class CopyAction(FSAction):
    def preview(self):
        logger.info(lime('Copy:') + lime('[') + self.src.txt + lime(']'))
        logger.info(lime('└───►') + lime('[') + self.dst.txt + lime(']'))

    def apply(self):
        try:
            return CopyCommand(self.src, self.dst)()
        except Exception as e:
            logger.error(e)
            return False


class RenameAction(FSAction):
    def preview(self):
        if len(self) == 2:
            A, B = fancy_diff_strings(self.src.txt, self.dst.txt)
            logger.info(yellow('Rename:') + yellow('[') + A + yellow(']'))
            if B:
                logger.info(yellow('└─────►') + yellow('[') + B + yellow(']'))
        else:
            for idx, target in enumerate(self.targets):
                logger.info(yellow('Rename:' + ('┌─' if idx == 0 else '└►')) +
                            yellow('[') + target.txt + yellow(']'))

    def apply(self):
        try:
            for src, dst in list(zip(self.targets, self.targets[1:]))[::-1]:
                return MoveCommand(src, dst)()

        except Exception as e:
            logger.error(e)
            return False


class RotateRenameAction(RenameAction):
    def preview(self):
        if len(self) == 2:
            logger.info(yellow('Swap:┌►') + yellow('[') + self.src.txt + yellow(']'))
            logger.info(yellow('Swap:└►') + yellow('[') + self.dst.txt + yellow(']'))
        else:
            total_len = len(self.targets)
            for idx, target in enumerate(self.targets):
                if idx == 0:
                    arrow = '┌►┌─'
                elif idx == total_len - 1:
                    arrow = '└───'
                else:
                    arrow = '│ └►'
                logger.info(yellow('Rotate:' + arrow) + yellow('[') + target.txt + yellow(']'))

    def apply(self):
        try:
            for p in self.targets:
                if not p.exists:
                    logger.error(red('File does not exist:[') + p.txt + red(']'))
            if logger.has_error():
                return False

            tmpdst = gen_tmp_file_name(self.targets[-1])

            mv_list = []
            mv_list.append((self.targets[-1], tmpdst))
            for src, dst in list(zip(self.targets, self.targets[1:]))[::-1]:
                mv_list.append((src, dst))
            mv_list.append((tmpdst, self.targets[0]))

            for src, dst in mv_list:
                MoveCommand(src, dst)()

        except Exception as e:
            logger.error(e)
            return False


class RelinkAction(FSAction):
    def preview(self):
        logger.info(yellow('Relink:') + yellow('[') + self.src.txt + yellow(']'))

        if isinstance(self.src, VDLink):
            ref = self.src.ref
        else:
            ref = VDPath(os.readlink(self.src.path))
        color = yellow if ref.exists else red
        logger.info(yellow('├──x──►') + color('[') + ref.txt + color(']'))

        color = yellow if self.dst.exists else red
        logger.info(yellow('└─────►') + color('[') + self.dst.txt + color(']'))

    def apply(self):
        try:
            return RelinkCommand(self.src, self.dst)()
        except Exception as e:
            logger.error(e)
            return False


class CompressAction(FSAction):
    def __init__(self, *targets, keep=True):
        super().__init__(*targets)
        self.keep = keep
        self.cmd = None

    def preview(self):
        color = lime if self.keep else yellow
        logger.info(color('Compress:') + color('[') + self.src.txt + color(']'))
        logger.info(color('└───────►') + color('[') + self.dst.txt + color(']'))
        self.cmd = CompressCommand(self.src, self.dst, self.keep)

    def apply(self):
        try:
            return self.cmd()
        except Exception as e:
            logger.error(e)
            return False


class UncompressAction(FSAction):
    def __init__(self, *targets, keep=True):
        super().__init__(*targets)
        self.keep = keep
        self.cmd = None

    def preview(self):
        color = lime if self.keep else yellow
        logger.info(color('Extract:') + color('[') + self.src.txt + color(']'))
        logger.info(color('└──────►') + color('[') + self.dst.txt + color(']'))
        self.cmd = UncompressCommand(self.src, self.dst, keep=self.keep)

    def apply(self):
        try:
            return self.cmd()
        except Exception as e:
            logger.error(e)
            return False
