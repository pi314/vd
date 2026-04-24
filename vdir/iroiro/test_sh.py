import os
import shutil

from pathlib import Path

from .lib_test_utils import *

import iroiro as iro


class TestSh(TestCase):
    def setUp(self):
        self.root = Path(__file__).parent
        self.cwd = self.root
        iro.sh.dir_stack = []
        self.patch('os.chdir', self.mock_chdir)
        self.patch('os.getcwd', self.mock_getcwd)

    def mock_getcwd(self):
        if isinstance(self.cwd, Exception):
            raise self.cwd

        if self.cwd == 'invalid':
            raise FileNotFoundError('invalid')

        return str(self.cwd)

    def mock_chdir(self, path):
        if path == 'invalid':
            raise FileNotFoundError('invalid')

        path = Path(path)
        if path.is_absolute():
            self.cwd = path
        else:
            self.cwd = self.cwd / path

    def test_cwd(self):
        self.eq(iro.cwd(), Path.cwd())
        self.eq(iro.cwd('tmp'), self.root / 'tmp')
        self.eq(iro.cwd('invalid'), False)
        self.eq(iro.cwd(0), False)
        self.eq(iro.cwd(True), False)

    def test_pushd_invalid_path(self):
        here = iro.cwd()

        p = iro.pushd('invalid')
        self.false(p)

        p = iro.pushd(3)
        self.false(p)

        self.eq(iro.cwd(), here)

    def test_pushd_invalid_cwd(self):
        self.cwd = FileNotFoundError('deleted')

        self.eq(iro.cwd(), False)

        p = iro.pushd('tmp')
        self.false(p)

    def test_popd_to_invalid_path(self):
        here = iro.cwd()

        iro.pushd('tmp')

        def mock_chdir(path):
            raise FileNotFoundError('deleted')
        self.patch('os.chdir', mock_chdir)

        self.false(iro.popd())

        self.eq(iro.cwd(), here / 'tmp')

    def test_dirs_clear(self):
        here = iro.cwd()
        iro.pushd('tmp')
        iro.pushd('tmp2')

        self.eq(iro.dirs(), [here, here / 'tmp', here / 'tmp' / 'tmp2'])

        iro.dirs(clear=True)

        self.eq(iro.dirs(), [here / 'tmp' / 'tmp2'])

    def test_popd_all(self):
        here = iro.cwd()
        iro.pushd('tmp')
        iro.pushd('tmp2')

        self.eq(iro.dirs(), [here, here / 'tmp', here / 'tmp' / 'tmp2'])

        iro.popd(all=True)

        self.eq(iro.dirs(), [here])

    def test_pushd_popd_dirs(self):
        here = iro.cwd()

        with iro.pushd('tmp'):
            self.eq(iro.cwd(), here / 'tmp')

            iro.pushd('a')
            self.eq(iro.cwd(), here / 'tmp' / 'a')
            self.eq(iro.dirs(),
                    [
                        here,
                        here / 'tmp',
                        here / 'tmp' / 'a',
                    ])

            iro.cwd(here / 'tmp' / 'b')
            self.eq(iro.cwd(), here / 'tmp' / 'b')

            self.eq(iro.popd(), True)

            self.eq(iro.cwd(), here / 'tmp')
            self.eq(iro.dirs(),
                    [
                        here,
                        here / 'tmp',
                    ])

        self.eq(iro.cwd(), here)
        self.eq(iro.dirs(), [here])

        self.eq(iro.popd(), False)

    def test_home(self):
        from os.path import expanduser
        self.eq(str(iro.home()), expanduser('~'))

    def test_shrinkuser(self):
        HOME = iro.home()
        self.eq(iro.shrinkuser(HOME), '~')
        self.eq(iro.shrinkuser(str(HOME) + '/'), '~/')

        self.eq(iro.shrinkuser(HOME / 'bana'), '~/bana')
        self.eq(iro.shrinkuser(HOME / 'bana/'), '~/bana')
        self.eq(iro.shrinkuser(str(HOME) + '/bana/'), '~/bana/')

        self.eq(iro.shrinkuser('bana/na/'), 'bana/na/')
        self.eq(iro.shrinkuser('bana/na'), 'bana/na')
