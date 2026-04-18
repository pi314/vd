import shutil

from pathlib import Path

from . import logger
from .utils import *
from .vdpath import *
from .inventory import *


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

        if tag != 'nop':
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


class CopyCommand:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.res = None

    def __call__(self):
        try:
            mkdir_p(self.dst)
            self.preview()
            self.src.copy(self.dst, follow_symlinks=False)
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False
            self.preview()

        return self.res

    def preview(self):
        logger.cmd(['cp', self.src, self.dst], res=self.res)


class MoveCommand:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.res = None

    def __call__(self):
        try:
            mkdir_p(self.dst)
            self.preview()
            self.src.rename(self.dst)
            rmdir_p(self.src)
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False
            self.preview()

        return self.res

    def preview(self):
        logger.cmd(['mv', self.src, self.dst], res=self.res)


class DeleteCommand:
    def __init__(self, src):
        self.src = src
        self.res = None

    def __call__(self):
        try:
            self.preview()
            if not self.src.is_dir() or self.src.is_symlink():
                self.src.unlink()
            else:
                shutil.rmtree(self.src)
            rmdir_p(self.src)
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False
            self.preview()

        return self.res

    def preview(self):
        if not self.src.is_dir() or self.src.is_symlink():
            cmd = ['rm', self.src]
        else:
            cmd = ['rm', '-r', self.src]
        logger.cmd(cmd, res=self.res)


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
                ', '.join('[{}]'.format(t) for t in self.targets))


class MetaAction(VirtualAction):
    pass


class FSAction(VirtualAction):
    pass


class TrackAction(MetaAction):
    def preview(self):
        logger.info(cyan('Track:') + cyan('[') + self.src + cyan(']'))


class NoAction(MetaAction):
    pass


class ResolveLinkAction(MetaAction):
    pass
    # os.readlink(target)


class UntrackAction(MetaAction):
    pass


class GlobAction(MetaAction):
    def preview(self):
        logger.info(cyan('Expand:') + cyan('[') + self.src + cyan(']'))


class GlobAllAction(MetaAction):
    def preview(self):
        logger.info(cyan('ExpandAll:') + cyan('[') + self.src + cyan(']'))


def mkdir_p(path):
    try:
        if not path.parent.exists():
            logger.cmd(['mkdir', '-p', path.parent])
            path.parent.mkdir(parents=True, exist_ok=True)
    except:
        return


def rmdir_p(path):
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
        try:
            return DeleteCommand(Path(self.src))()
        except Exception as e:
            logger.error(e)
            return False


class CopyAction(FSAction):
    def preview(self):
        logger.info(yellow('Copy:') + yellow('[') + self.src + yellow(']'))
        logger.info(yellow('└───►') + yellow('[') + self.dst + yellow(']'))

    def apply(self):
        try:
            return CopyCommand(Path(self.src), Path(self.dst))()
        except Exception as e:
            logger.error(e)
            return False


class RenameAction(FSAction):
    def preview(self):
        if len(self) == 2:
            A, B = fancy_diff_strings(self.src, self.dst)
            logger.info(yellow('Rename:') + yellow('[') + A + yellow(']'))
            if B:
                logger.info(yellow('└─────►') + yellow('[') + B + yellow(']'))
        else:
            for idx, target in enumerate(self.targets):
                logger.info(yellow('Rename:' + ('┌─' if idx == 0 else '└►')) +
                            yellow('[') + target + yellow(']'))

    def apply(self):
        try:
            for src, dst in list(zip(self.targets, self.targets[1:]))[::-1]:
                return MoveCommand(Path(src), Path(dst))()

        except Exception as e:
            logger.error(e)
            return False


class RotateRenameAction(RenameAction):
    def preview(self):
        if len(self) == 2:
            logger.info(yellow('Swap:┌►') + yellow('[') + self.src + yellow(']'))
            logger.info(yellow('Swap:└►') + yellow('[') + self.dst + yellow(']'))
        else:
            total_len = len(self.targets)
            for idx, target in enumerate(self.targets):
                if idx == 0:
                    arrow = '┌►┌─'
                elif idx == total_len - 1:
                    arrow = '└───'
                else:
                    arrow = '│ └►'

                logger.info(yellow('Rotate:' + arrow) + yellow('[') + target + yellow(']'))

    def apply(self):
        try:
            for p in self.targets:
                if not Path(p).exists():
                    logger.error(red('File does not exist:[') + p + red(']'))
            if logger.has_error():
                return False

            tmpdst = gen_tmp_file_name(self.targets[-1])

            mv_list = []
            mv_list.append((self.targets[-1], tmpdst))
            for src, dst in list(zip(self.targets, self.targets[1:]))[::-1]:
                mv_list.append((src, dst))
            mv_list.append((tmpdst, self.targets[0]))

            for src, dst in mv_list:
                src = Path(src)
                dst = Path(dst)
                if not dst.parent.exists():
                    logger.cmd(['mkdir', '-p', dst.parent])
                    dst.parent.mkdir(parents=True, exist_ok=True)

                logger.cmd(['mv', src, dst])
                src.rename(dst)

                rmdir_p(src)

        except Exception as e:
            logger.error(e)
            return False
