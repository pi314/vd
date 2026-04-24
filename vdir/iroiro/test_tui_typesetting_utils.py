from .lib_test_utils import *

from iroiro import *


class TestTypesettingUtils(TestCase):
    def test_charwidth(self):
        self.eq(charwidth('t'), 1)
        self.eq(charwidth('哇'), 2)
        self.eq(charwidth('嗚'), 2)
        self.eq(charwidth('😂'), 2)

        self.eq(charwidth(chr(0x2028)), 0) # Line Separator
        self.eq(charwidth(chr(0x2060)), 0) # Word Joiner

        with self.raises(TypeError):
            charwidth('test')

    def test_strwidth(self):
        self.eq(strwidth('test'), 4)
        self.eq(strwidth(orange('test')), 4)
        self.eq(strwidth('哇嗚'), 4)

    def test_wrap(self):
        # Basic cases
        self.eq(wrap('iroiro', 1), ('i', 'roiro'))
        self.eq(wrap('iroiro', 2), ('ir', 'oiro'))
        self.eq(wrap('iroiro', 3), ('iro', 'iro'))
        self.eq(wrap('iroiro', 4), ('iroi', 'ro'))
        self.eq(wrap('iroiro', 5), ('iroir', 'o'))
        self.eq(wrap('iroiro', 6), ('iroiro', ''))
        self.eq(wrap('iroiro', 7), ('iroiro', ''))

        # Mixed with CJK
        self.eq(wrap('いろiろ', 1), ('', 'いろiろ'))
        self.eq(wrap('いろiろ', 2), ('い', 'ろiろ'))
        self.eq(wrap('いろiろ', 3), ('い', 'ろiろ'))
        self.eq(wrap('いろiろ', 4), ('いろ', 'iろ'))
        self.eq(wrap('いろiろ', 5), ('いろi', 'ろ'))
        self.eq(wrap('いろiろ', 6), ('いろi', 'ろ'))
        self.eq(wrap('いろiろ', 7), ('いろiろ', ''))
        self.eq(wrap('いろiろ', 8), ('いろiろ', ''))

        # Clip
        self.eq(wrap('いろいろ', 1, clip='>'), ('>', 'いろいろ'))
        self.eq(wrap('いろいろ', 2, clip='>'), ('い', 'ろいろ'))
        self.eq(wrap('いろいろ', 3, clip='>'), ('い>', 'ろいろ'))
        self.eq(wrap('いろいろ', 4, clip='>'), ('いろ', 'いろ'))
        self.eq(wrap('いろいろ', 5, clip='>'), ('いろ>', 'いろ'))
        self.eq(wrap('いろいろ', 6, clip='>'), ('いろい', 'ろ'))
        self.eq(wrap('いろいろ', 7, clip='>'), ('いろい>', 'ろ'))
        self.eq(wrap('いろいろ', 8, clip='>'), ('いろいろ', ''))
        self.eq(wrap('いろいろ', 9, clip='>'), ('いろいろ', ''))

        # Clip with color string
        self.eq(wrap('いろいろ', 1, clip='\033[1;35m>\033[m'), ('\033[1;35m>\033[m', 'いろいろ'))
        self.eq(wrap('いろいろ', 2, clip='\033[1;35m>\033[m'), ('い', 'ろいろ'))
        self.eq(wrap('いろいろ', 3, clip='\033[1;35m>\033[m'), ('い\033[1;35m>\033[m', 'ろいろ'))
        self.eq(wrap('いろいろ', 4, clip='\033[1;35m>\033[m'), ('いろ', 'いろ'))
        self.eq(wrap('いろいろ', 5, clip='\033[1;35m>\033[m'), ('いろ\033[1;35m>\033[m', 'いろ'))
        self.eq(wrap('いろいろ', 6, clip='\033[1;35m>\033[m'), ('いろい', 'ろ'))
        self.eq(wrap('いろいろ', 7, clip='\033[1;35m>\033[m'), ('いろい\033[1;35m>\033[m', 'ろ'))
        self.eq(wrap('いろいろ', 8, clip='\033[1;35m>\033[m'), ('いろいろ', ''))
        self.eq(wrap('いろいろ', 9, clip='\033[1;35m>\033[m'), ('いろいろ', ''))

        # String with color sequence
        self.eq(wrap('い\033[1;35mろい\033[mろ', 1), ('', 'い\033[1;35mろい\033[mろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 2), ('い', '\033[1;35mろい\033[mろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 3), ('い', '\033[1;35mろい\033[mろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 4), ('い\033[1;35mろ', 'い\033[mろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 5), ('い\033[1;35mろ', 'い\033[mろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 6), ('い\033[1;35mろい\033[m', 'ろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 7), ('い\033[1;35mろい\033[m', 'ろ'))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 8), ('い\033[1;35mろい\033[mろ', ''))
        self.eq(wrap('い\033[1;35mろい\033[mろ', 9), ('い\033[1;35mろい\033[mろ', ''))

        self.eq(wrap('いろいろ\033[1;35m', 8), ('いろいろ', '\033[1;35m'))
        self.eq(wrap('いろいろ\033[1;35m', 9), ('いろいろ\033[1;35m', ''))

        # String with multiple color sequences
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 1), ('', 'い\033[1;35mろ\033[m\033[1;35mい\033[0mろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 2), ('い', '\033[1;35mろ\033[m\033[1;35mい\033[0mろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 3), ('い', '\033[1;35mろ\033[m\033[1;35mい\033[0mろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 4), ('い\033[1;35mろ\033[m', '\033[1;35mい\033[0mろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 5), ('い\033[1;35mろ\033[m', '\033[1;35mい\033[0mろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 6), ('い\033[1;35mろ\033[m\033[1;35mい\033[0m', 'ろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 7), ('い\033[1;35mろ\033[m\033[1;35mい\033[0m', 'ろ'))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 8), ('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', ''))
        self.eq(wrap('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', 9), ('い\033[1;35mろ\033[m\033[1;35mい\033[0mろ', ''))

        # String with unknown escape sequence
        self.eq(wrap('い\033ろいろ', 4), ('い\033ろいろ', ''))

        with self.raises(ValueError):
            wrap('whatever', 1, clip=1)

        with self.raises(ValueError):
            wrap('whatever', 1, clip='wa')

        with self.raises(ValueError):
            wrap('whatever', 1, clip='蛤')

    def test_ljust_str(self):
        self.eq(ljust('test', 10), 'test      ')
        self.eq(rjust('test', 10), '      test')

        padding = ' ' * 6
        self.eq(ljust(orange('test'), 10), orange('test') + padding)
        self.eq(rjust(orange('test'), 10), padding + orange('test'))

        padding = '#' * 6
        self.eq(ljust(orange('test'), 10, '#'), orange('test') + padding)
        self.eq(rjust(orange('test'), 10, '#'), padding + orange('test'))

    def test_just_rect(self):
        data = [
                ('column1', 'col2'),
                ('word1', 'word2'),
                ('word3', 'word4 long words'),
                ]

        self.eq(ljust(data), [
            ('column1', 'col2            '),
            ('word1  ', 'word2           '),
            ('word3  ', 'word4 long words'),
            ])

        self.eq(rjust(data), [
            ('column1', '            col2'),
            ('  word1', '           word2'),
            ('  word3', 'word4 long words'),
            ])

    def test_just_with_fillchar(self):
        data = [
                ('column1', 'col2'),
                ('word1', 'word2'),
                ('word3', 'word4 long words'),
                ]

        self.eq(ljust(data, fillchar='#'), [
            ('column1', 'col2############'),
            ('word1##', 'word2###########'),
            ('word3##', 'word4 long words'),
            ])

    def test_just_with_fillchar_func(self):
        data = [
                ('up left',   'up',   'up right'),
                ('left',      '',     'right'),
                ('down left', 'down', 'down r'),
                ]

        def fillchar(row, col, text):
            if row + col == 2:
                return '%'
            if text == 'right':
                return '$'
            return '#' if (row % 2) else '@'

        self.eq(ljust(data, fillchar=fillchar, width=10), [
            ('up left@@@', 'up@@@@@@@@', 'up right%%'),
            ('left######', '%%%%%%%%%%', 'right$$$$$'),
            ('down left%', 'down@@@@@@', 'down r@@@@'),
            ])

        self.eq(rjust(data, fillchar=fillchar, width=10), [
            ('@@@up left', '@@@@@@@@up', '%%up right'),
            ('######left', '%%%%%%%%%%', '$$$$$right'),
            ('%down left', '@@@@@@down', '@@@@down r'),
            ])

    def test_just_with_width(self):
        data = [
                ('column1', 'col2'),
                ('word1', 'word2'),
                ('word3', 'word4 long words'),
                ]

        self.eq(ljust(data, width=20), [
            ('column1             ', 'col2                '),
            ('word1               ', 'word2               '),
            ('word3               ', 'word4 long words    '),
            ])

        self.eq(ljust(data, width=(10, 20)), [
            ('column1   ', 'col2                '),
            ('word1     ', 'word2               '),
            ('word3     ', 'word4 long words    '),
            ])

    def test_just_with_generator(self):
        data = [
                ('column1', 'col2'),
                ('word1', 'word2'),
                ('word3', 'word4 long words'),
                ]

        ret = ljust((vector for vector in data), width=(10, 20))
        self.false(isinstance(ret, (tuple, list)))

        self.eq(list(ret), [
            ('column1   ', 'col2                '),
            ('word1     ', 'word2               '),
            ('word3     ', 'word4 long words    '),
            ])

    def test_just_rect_lack_columns(self):
        self.eq(
                ljust([
                    ('column1', 'col2'),
                    ('word1',),
                    tuple(),
                    ('', 'multiple words'),
                    tuple(),
                    ]),
                [
                    ('column1', 'col2          '),
                    ('word1  ', '              '),
                    ('       ', '              '),
                    ('       ', 'multiple words'),
                    ('       ', '              '),
                    ])

    def test_just_rect_more_columns(self):
        self.eq(
                ljust([
                    ('column1', 'col2'),
                    tuple(),
                    ('word1', 'word2', 'word4'),
                    ('word3', 'multiple words'),
                    ]),
                [
                    ('column1', 'col2          ', '     '),
                    ('       ', '              ', '     '),
                    ('word1  ', 'word2         ', 'word4'),
                    ('word3  ', 'multiple words', '     '),
                    ])
