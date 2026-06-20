import shutil
import subprocess

from . import logger
from .utils import *
from .vdpath import *


def gen_tmp_file_name(path, postfix='.vdtmp'):
    import time
    now = time.time()
    tmp_file_name = '{orig_path}{postfix}[{timestamp}][{getpid}]'.format(
            orig_path=path,
            postfix=postfix,
            timestamp=now,
            getpid=os.getpid(),
            )
    return VDPath(tmp_file_name)


class MkdirsCommand:
    def __init__(self, who):
        self.who = who

    def __call__(self, echo=True):
        try:
            if not self.who.exists:
                if echo:
                    self.echo()
                self.who.mkdir()
        except:
            pass

        return self.who.exists

    def echo(self):
        logger.cmd(['mkdir', '-p', self.who])


class RmdirsCommand:
    def __init__(self, who):
        self.who = who

    def __call__(self, echo=True):
        who = self.who
        if isinstance(who, VDPath):
            who = who.path
        if isinstance(who, VDLink):
            who = who.lnk.path

        cwd = Path.cwd().resolve()
        targets = [(who, True)]
        for probe in who.resolve().parents:
            targets.append((probe, False))

        for probe, ignore_errors in targets:
            try:
                # Delete .DS_Store if present
                try:
                    (probe / '.DS_Store').unlink()
                except:
                    pass

                if probe == cwd:
                    # dont delete cwd
                    return True

                for child in probe.iterdir():
                    # if probe/ is not empty, return
                    return True

                # Try to delete it
                logger.cmd(['rmdir', probe])
                probe.rmdir()

            except:
                if ignore_errors:
                    pass
                else:
                    return

    def echo(self):
        logger.cmd(['rmdir', '-p', self.who])


class ShellCommand:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.p = iroiro.command(self.cmd, *self.kwargs)
        self.res = None

    def __call__(self, echo=True):
        if echo:
            self.echo()
        self.p.run()
        self.res = (self.p.returncode == 0)
        return self.res

    def echo(self):
        logger.cmd(self.cmd, res=self.res)


class CopyCommand:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.res = None

        if self.src.isdir:
            self.cp_cmd = ['cp', '-r', self.src.path, self.dst.path]
            self.do_copy = lambda: shutil.copytree(self.src.path, self.dst.path,
                                                    symlinks=True,
                                                    copy_function=shutil.copy,
                                                    ignore_dangling_symlinks=True)
        else:
            self.cp_cmd = ['cp', self.src.path, self.dst.path]
            self.do_copy = lambda: shutil.copy(self.src.path, self.dst.path, follow_symlinks=False)

    def __call__(self):
        try:
            if self.dst.exists:
                raise FileExistsError(self.dst)

            self.res = MkdirsCommand(self.dst.parent)()
            if not self.res:
                return self.res

            self.echo()
            self.do_copy()

            self.res = True

        except Exception as e:
            logger.error(e)
            raise e
            self.res = False

        return self.res

    def echo(self):
        logger.cmd(self.cp_cmd, res=self.res)


class MoveCommand:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.res = None

    def __call__(self):
        try:
            src = self.src
            dst = self.dst

            if src == dst.parent:
                tmpsrc = gen_tmp_file_name(src)
                MoveCommand(src, tmpsrc)()
                src = tmpsrc

            MkdirsCommand(dst.parent)()
            self.echo()
            src.rename(dst)
            RmdirsCommand(src)()
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False

        return self.res

    def echo(self):
        logger.cmd(['mv', self.src.path, self.dst.path], res=self.res)


class DeleteCommand:
    def __init__(self, who):
        self.who = who
        self.res = None
        if self.who.isdir:
            self.rm_cmd = ['rm', '-r', self.who.path]
            self.do_rm = lambda: shutil.rmtree(self.who.path)
        else:
            self.rm_cmd = ['rm', self.who.path]
            self.do_rm = lambda: self.who.unlink()

    def __call__(self):
        try:
            self.echo()
            self.do_rm()
            RmdirsCommand(self.who)()
            self.res = True

        except Exception as e:
            if not self.who.exists:
                # Delete failed but it's gone so ok
                logger.warning(e)
            else:
                logger.error(e)
                self.res = False

        return self.res

    def echo(self):
        logger.cmd(self.rm_cmd, res=self.res)


class RelinkCommand:
    def __init__(self, lnk, ref):
        self.lnk = lnk
        self.ref = ref
        if self.lnk.islink:
            self.rm_cmd = ['rm', self.lnk.path]
            self.do_rm = lambda: self.lnk.unlink()
        else:
            self.rm_cmd = []
            self.do_rm = lambda: True

        self.symlink_cmd = ['ln', '-s', self.ref.path, self.lnk.path]
        self.do_symlink = lambda: self.lnk.symlink_to(self.ref)

        self.res = None

    def __call__(self):
        try:
            self.echo()
            self.do_rm()
            self.do_symlink()
            self.res = True

        except Exception as e:
            logger.error(e)
            self.res = False

        return self.res

    def echo(self):
        logger.cmd(self.rm_cmd, res=self.res)
        logger.cmd(self.symlink_cmd, res=self.res)


class CompressCommand:
    def __init__(self, src, dst, keep=True):
        self.src = src
        self.dst = dst
        self.keep = keep
        self.res = None

        util = 'tar'
        flags = ['-c', '-v', '-f']
        if dst.name.endswith(('.tar.xz', '.xz')):
            flags.insert(0, '--xz')

        elif dst.name.endswith(('.tar.bz', '.tar.bz2', '.tbz', '.tbz2')):
            flags.insert(0, '--bzip2')

        elif dst.name.endswith(('.tar.gz', '.gz', '.tgz')):
            flags.insert(0, '--gzip')

        elif dst.name.endswith(('.tar.Z', '.Z')):
            flags.insert(0, '--compress')

        elif dst.name.endswith('.zip'):
            util = 'zip'
            flags = ['--symlinks', '-r']

        elif dst.name.endswith('.7z'):
            util = '7z'
            flags = ['a']

        if not shutil.which(util):
            logger.error('Command not found:', util)

        self.tar_cvf = ShellCommand([util] + flags + [self.dst.path, self.src.path])

    def __call__(self):
        try:
            if self.dst.exists:
                raise FileExistsError(self.dst.path)

            self.res = all((func() != False
                            for func in (
                                MkdirsCommand(self.dst.parent),
                                self.tar_cvf
                                )
                            ))
            if not self.res:
                return self.res

            if not self.keep:
                return DeleteCommand(self.src)()

        except Exception as e:
            logger.error(e)
            self.res = False

        return self.res


class UncompressCommand:
    def __init__(self, src, dst, keep=True):
        self.src = src
        self.dst = dst
        self.keep = keep
        self.res = None
        self.mkdir = MkdirsCommand(self.dst)

        util = 'tar'
        flags = ['-x', '-v', '-f', self.src.path]
        dir_flags = ['--cd', self.dst.path]
        if src.name.endswith('.zip'):
            util = 'unzip'
            flags = [self.src.path]
            dir_flags = ['-d', self.dst.path]
        self.tar_xvf = ShellCommand([util] + flags + dir_flags)

    def __call__(self):
        try:
            if self.dst.exists:
                raise FileExistsError(self.dst.path)

            self.res = all(func() for func in (self.mkdir, self.tar_xvf))
            if not self.res:
                return self.res

            ls = [f for f in self.dst.listdir(True) if f.name != '.DS_Store']
            if len(ls) == 1 and ls[0].name == self.dst.name:
                tmpdir = gen_tmp_file_name(self.dst)
                self.res = all((
                    func()!=False for func in (
                        MoveCommand(ls[0], tmpdir),
                        # DeleteCommand(self.dst),
                        MoveCommand(tmpdir, self.dst))
                    ))
                if not self.res:
                    return self.res

            if not self.keep:
                self.res = DeleteCommand(self.src)()

        except Exception as e:
            logger.error(e)
            self.res = False

        return self.res
