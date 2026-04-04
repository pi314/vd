import shutil

from pathlib import Path

from . import logger


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


class ResolveLinkAction(MetaAction):
    pass
    # os.readlink(target)


class UntrackAction(MetaAction):
    pass


class GlobAction(MetaAction):
    pass


class GlobAllAction(MetaAction):
    pass


class DeleteAction(FSAction):
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
            return

        try:
            cwd = Path.cwd().resolve()
            for probe in Path(self.src).resolve().parents:
                # Delete .DS_Store if present
                try:
                    (probe / '.DS_Store').unlink()
                except:
                    pass

                if probe == cwd:
                    # dont delete cwd
                    return

                if not probe.is_dir():
                    # something weird happen
                    return

                for child in probe.iterdir():
                    # if probe/ is not empty, return
                    return

                # probe/ is empty, delete it
                probe.rmdir()

        except:
            return


class RenameAction(FSAction):
    pass


class CopyAction(FSAction):
    pass


class DominoRenameAction(FSAction):
    pass


class RotateRenameAction(FSAction):
    pass
