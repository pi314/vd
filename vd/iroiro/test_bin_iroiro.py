import sys

import unittest.mock

from .lib_test_utils import *

import iroiro as iro


class TestBinIroiro(TestCase):
    def setUp(self):
        self.prints = []
        self.patch('builtins.print', self.mock_print)

    def tearDown(self):
        pass

    def mock_print(self, *args, **kwargs):
        self.prints.append((args, kwargs))

    @property
    def stdout(self):
        return [' '.join(args)
                for args, kwargs in self.prints
                if kwargs.get('file', sys.stdout) == sys.stdout]

    @property
    def stderr(self):
        return [' '.join(args)
                for args, kwargs in self.prints
                if kwargs.get('file') == sys.stderr]

    def test_bin_iroiro(self):
        # Run 'iroiro' and check output
        with self.raises(SystemExit):
            sys.argv = ['iroiro']
            iro.bin.iroiro.main()

        modules = self.stdout

        import os
        import re
        files = list(
                map(
                    lambda x: re.fullmatch(r'bin_(\w+).py', x).group(1),
                    filter(lambda x: x.startswith('bin') and x.endswith('.py'),
                           os.listdir('iroiro'))
                    )
                )
        self.eq(sorted(modules), sorted(files))

    def test_bin_iroiro_subcmd_rainbow(self):
        with self.raises(SystemExit):
            sys.argv = ['iroiro', 'rainbow']
            iro.bin.iroiro.main()

        sys.argv = ['iroiro', 'rainbow', 'murasaki']
        iro.bin.iroiro.main()

    def test_bin_iroiro_subcmd_unknown(self):
        with self.raises(SystemExit):
            sys.argv = ['iroiro', 'wow']
            iro.bin.iroiro.main()

        self.true('Unknown' in '\n'.join(self.stderr))

    def test_bin_iroiro_recursion_error(self):
        with self.raises(SystemExit):
            sys.argv = ['iroiro', 'iroiro', 'iroiro', 'iroiro']
            iro.bin.iroiro.main()

        self.true('RecursionError' in '\n'.join(self.stderr))
