import unittest.mock

from .lib_test_utils import *

import iroiro as iro


class TestOpen(TestCase):
    def test_open_wb(self):
        mock_open = unittest.mock.mock_open()
        self.patch('builtins.open', mock_open)
        with iro.open('iroiro.txt', 'wb') as f:
            mock_open.assert_called_once_with('iroiro.txt', mode='wb')

    def test_open_rb(self):
        mock_open = unittest.mock.mock_open()
        self.patch('builtins.open', mock_open)
        with iro.open('iroiro.txt', 'rb') as f:
            mock_open.assert_called_once_with('iroiro.txt', mode='rb')

    def test_open_without_mode(self):
        mock_open = unittest.mock.mock_open()
        self.patch('builtins.open', mock_open)
        with iro.open('iroiro.txt') as f:
            mock_open.assert_called_once_with(
                    'iroiro.txt', mode='rt',
                    encoding='utf-8', errors='backslashreplace')

    def test_write(self):
        mock_open = unittest.mock.mock_open()
        self.patch('builtins.open', mock_open)
        with iro.open('iroiro.txt', 'w') as f:
            mock_open.assert_called_once_with(
                    'iroiro.txt', mode='w',
                    encoding='utf-8', errors='backslashreplace')

            f.write('normal write\n')
            f.writeline()
            f.writeline('line1')
            f.writelines(['line2', 'line3', 'line4'])

        handle = mock_open()
        handle.write.assert_has_calls([
            unittest.mock.call('normal write\n'),
            unittest.mock.call('\n'),
            unittest.mock.call('line1\n'),
            unittest.mock.call('line2\n'),
            unittest.mock.call('line3\n'),
            unittest.mock.call('line4\n'),
            ])

    def test_read(self):
        answer = [
                'line1',
                'line2',
                'line3',
                'and line4',
                'and line5',
                ]
        mock_open = unittest.mock.mock_open(read_data='\n'.join(answer))
        self.patch('builtins.open', mock_open)

        lines = []
        with iro.open('iroiro.txt', 'r') as f:
            mock_open.assert_called_once_with(
                    'iroiro.txt', mode='r',
                    encoding='utf-8', errors='backslashreplace')

            lines.append(f.readline())
            lines.append(f.readline())
            lines += f.readlines()

        self.eq(lines, answer)

    def test_read_without_rstrip(self):
        answer = [
                'line1',
                'line2',
                'line3',
                'and line4',
                'and line5',
                ]
        mock_open = unittest.mock.mock_open(read_data='\n'.join(answer))
        self.patch('builtins.open', mock_open)

        lines = []
        with iro.open('iroiro.txt', mode='r', rstrip='') as f:
            mock_open.assert_called_once_with(
                    'iroiro.txt', mode='r',
                    encoding='utf-8', errors='backslashreplace')

            lines.append(f.readline())
            lines.append(f.readline())
            lines += f.readlines()

        self.eq(lines, [
                'line1\n',
                'line2\n',
                'line3\n',
                'and line4\n',
                'and line5',
            ])


class TestNatsorted(TestCase):
    def test_natsorted(self):
        self.eq(
                iro.natsorted([
                    'apple1',
                    'apple10',
                    'banana10',
                    'apple2',
                    'banana1',
                    'banana3',
                    ]),
                [
                    'apple1',
                    'apple2',
                    'apple10',
                    'banana1',
                    'banana3',
                    'banana10',
                    ])

        self.eq(
                iro.natsorted([
                    'version-1.9',
                    'version-2.0',
                    'version-1.11',
                    'version-1.10',
                    ]),
                [
                    'version-1.9',
                    'version-1.10',
                    'version-1.11',
                    'version-2.0',
                    ])

        from pathlib import Path
        self.eq(
                iro.natsorted([
                    Path('version-1.9'),
                    Path('version-2.0'),
                    Path('version-1.11'),
                    Path('version-1.10'),
                    ]),
                [
                    Path('version-1.9'),
                    Path('version-1.10'),
                    Path('version-1.11'),
                    Path('version-2.0'),
                    ])
