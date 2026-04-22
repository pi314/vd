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
            if not self.src.exists():
                # Delete failed but it's gone so ok
                logger.warning(e)
            else:
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


class RelinkCommand:
    def __init__(self, lnk, ref):
        self.lnk = lnk
        self.ref = ref
        self.res = None

    def __call__(self):
        try:
            self.preview()
            if self.lnk.exists() or self.lnk.is_symlink():
                self.lnk.unlink()
            self.lnk.symlink_to(self.ref)
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False
            self.preview()

        return self.res

    def preview(self):
        if self.lnk.exists() or self.lnk.is_symlink():
            logger.cmd(['rm', self.lnk], res=self.res)
        logger.cmd(['ln', '-s', self.ref, self.lnk], res=self.res)


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
        logger.info(cyan('Track:') + cyan('[') + self.src.txt + cyan(']'))


class NoAction(MetaAction):
    pass


class ResolveLinkAction(MetaAction):
    def preview(self):
        logger.info(cyan('Resolve:') + cyan('[') + self.src.txt + cyan(']'))


class UntrackAction(MetaAction):
    def preview(self):
        logger.info(cyan('Untrack:') + cyan('[') + self.src.txt + cyan(']'))


class GlobAction(MetaAction):
    def preview(self):
        logger.info(cyan('Expand:') + cyan('[') + self.src.txt + cyan(']'))


class GlobAllAction(MetaAction):
    def preview(self):
        logger.info(cyan('ExpandAll:') + cyan('[') + self.src.txt + cyan(']'))


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
        logger.info(red('Delete:') + red('[') + self.src.txt + red(']'))

    def apply(self):
        try:
            return DeleteCommand(self.src.path)()
        except Exception as e:
            logger.error(e)
            return False


class CopyAction(FSAction):
    def preview(self):
        logger.info(yellow('Copy:') + yellow('[') + self.src.txt + yellow(']'))
        logger.info(yellow('└───►') + yellow('[') + self.dst.txt + yellow(']'))

    def apply(self):
        try:
            return CopyCommand(self.src.path, self.dst.path)()
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
                            yellow('[') + target + yellow(']'))

    def apply(self):
        try:
            for src, dst in list(zip(self.targets, self.targets[1:]))[::-1]:
                return MoveCommand(src.path, dst.path)()

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
                mv_list.append((src.path, dst.path))
            mv_list.append((tmpdst, self.targets[0]))

            for src, dst in mv_list:
                if not dst.parent.exists():
                    logger.cmd(['mkdir', '-p', dst.parent])
                    dst.parent.mkdir(parents=True, exist_ok=True)

                logger.cmd(['mv', src, dst])
                src.rename(dst)

                rmdir_p(src)

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
            return RelinkCommand(self.src.path, self.dst.path)()

        except Exception as e:
            logger.error(e)
            return False


class CompressAction(FSAction):
    pass


class UncompressAction(FSAction):
    pass
