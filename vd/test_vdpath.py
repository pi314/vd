import unittest

import os
import shutil

from os.path import exists, join

from .inventory import VDPath, VDGlob

from .testutils import *


class TestVDPath(unittest.TestCase):

    def setUp(self):
        self.test_chamber = TestChamber()
        self.test_chamber.create({
                'test_file': EmptyFile(),
                'test_link': SymlinkTo('test_file'),
                'test_fifo': FIFO(),
                'empty_dir': Dir(),
                'test_dir': Dir({
                    'test_file': EmptyFile(),
                    'test_file2': EmptyFile(),
                    }),
                })

        pushd(self.test_chamber.path)

    def tearDown(self):
        popd()
        self.test_chamber.destroy()


    def test_vdpath_file(self):
        p = VDPath('test_file')
        self.assertEqual(p.text, 'test_file')
        self.assertEqual(p.isfile, True)

    def test_vdpath_fifo(self):
        p = VDPath('test_fifo')
        self.assertEqual(p.text, 'test_fifo')
        self.assertEqual(p.isfifo, True)

    def test_vdpath_rpath(self):
        rpath = '../' + self.test_chamber.path.name + '/test_file'
        p = VDPath(rpath)
        self.assertEqual(p.text, rpath)
        self.assertEqual(p.realpath, join(os.getcwd(), 'test_file'))
        self.assertEqual(p.isfile, True)

    def test_vdpath_symlink(self):
        p = VDPath('test_link')
        self.assertEqual(p.text, 'test_link')
        self.assertEqual(p.islink, True)
        self.assertEqual(p.realpath, join(os.getcwd(), 'test_link'))

    def test_vdpath_dir(self):
        p = VDPath('test_dir')
        self.assertEqual(p.text, 'test_dir/')
        self.assertEqual(p.isdir, True)
        self.assertEqual(p.listdir(False), ['test_dir/test_file', 'test_dir/test_file2'])

    def test_vdpath_not_exist(self):
        p = VDPath('test_notexist')
        self.assertEqual(p.text, 'test_notexist')
        self.assertEqual(p.exists, False)


class TestVDGlob(unittest.TestCase):

    def setUp(self):
        self.test_chamber = TestChamber()
        self.test_chamber.create({
                'test_file': EmptyFile(),
                'test_link': SymlinkTo('test_file'),
                'test_dir/test_file': EmptyFile(),
                'test_dir/test_file2': EmptyFile(),
                })

        pushd(self.test_chamber.path)


    def tearDown(self):
        popd()
        self.test_chamber.destroy()

    def test_vdglob_glob(self):
        p = VDGlob('test_dir/*')
        self.assertEqual(p.glob(), ['test_dir/test_file', 'test_dir/test_file2'])
