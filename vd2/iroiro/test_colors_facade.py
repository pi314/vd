from .lib_test_utils import *

from iroiro import Emphasis, bold, lowint, underline, blink, reverse, invisible
from iroiro import Color, Color8, Color256, ColorRGB, ColorHSV
from iroiro import color, paint
from iroiro import nocolor


class TestColorFacade(TestCase):
    def test_forbit_instantiate_color_class(self):
        with self.raises(TypeError):
            Color()

    def test_color_facade(self):
        # no argument unpack
        with self.raises(TypeError):
            color((208,))

        with self.raises(TypeError):
            color([208])

        with self.raises(TypeError):
            color((0xC0, 0xFF, 0xEE))

        with self.raises(TypeError):
            color([0xC0, 0xFF, 0xEE])

        with self.raises(TypeError):
            color(0xC0, 0xFF)

        # copy_ctor
        self.eq(color(color(208)), color(208))

        # subclass
        self.true(issubclass(Color256, Color))
        self.true(issubclass(ColorRGB, Color))

        # Color256
        orange = color(208)
        self.true(isinstance(orange, Color256))
        self.true(isinstance(orange, Color))

        # ColorRGB
        coffee = color(0xC0, 0xFF, 0xEE)
        self.true(isinstance(coffee, ColorRGB))
        self.true(isinstance(coffee, Color))

        # ColorRGB
        coffee = color('#C0FFEE')
        self.true(isinstance(coffee, ColorRGB))
        self.true(isinstance(coffee, Color))

        # ColorHSV
        lime = color('@120,100,100')
        self.true(isinstance(lime, ColorHSV))
        self.true(isinstance(lime, Color))

    def test_empty(self):
        self.eq(color(), '')

    def test_color_invalid_value(self):
        with self.raises(TypeError):
            color(True)

        with self.raises(TypeError):
            Color256(True)

        with self.raises(TypeError):
            ColorRGB(True)

    def test_color__parse_tokenize(self):
        import iroiro

        # function under test
        fut = iroiro.colors._tokenize

        self.eq(fut('\033'), None)
        self.eq(fut('\033{'), None)
        self.eq(fut('\033['), None)
        self.eq(fut('\033[38z'), None)
        self.eq(fut('\033[38m'), [38])
        self.eq(fut('\033[m'), [0])
        self.eq(fut('\033[0m'), [0])
        self.eq(fut('\033[1m'), [1])
        self.eq(fut('\033[1;31;25;47;0;38;5;214m'), [1, 31, 25, 47, 0, 38, 5, 214])
        self.eq(fut('\033[1;38;5;208morange \033[48;5;208morange\033[m '), [1, 38, 5, 208, 48, 5, 208, 0])
        self.eq(fut('\033[1;38;5; 208morange \033[48;5;208morange\033[m '), [48, 5, 208, 0])

    def test_color_parse(self):
        # empty seq
        self.eq(color(''), paint(fg=None, bg=None))

        # reset attr
        self.eq(color('\033[m'), paint(reset=True))

        # fg
        self.eq(color('\033[38;5;214m'), paint(fg=color(214)))

        # bg
        self.eq(color('\033[48;5;214m'), paint(bg=color(214)))

        # reset + fg
        self.eq(color('\033[0;38;5;214m'), paint(reset=True, fg=color(214)))

        # fg + bg and then override by next seq
        self.eq(color('\033[38;5;214;48;5;208m\033[38;5;2;48;5;10m'), paint(fg=color(2), bg=color(10)))

        # em + fg (256) + bg (RGB)
        self.eq(
                color('\033[1;4;5;38;5;214;48;2;192;255;238m'),
                paint(em=Emphasis(bold=True, underline=True, blink=True),
                      fg=Color256(214),
                      bg=ColorRGB(192, 255, 238))
                )

        # em + fg
        self.eq(color('\033[1;31m'), Color8(1) | bold)

        # fg override by fg
        self.eq(color('\033[31;35;37m'), Color8(7))

        # Valid seq and reset
        self.eq(color('\033[31;35;37;0m'), '\033[m')

        # Valid seq and reset and more seq
        self.eq(color('\033[31;35;37;0;32m'), paint(reset=True, fg=Color8(2)))

        # Incomplete seq
        self.eq(color('\033[38m'), color())

        # Unknown color type
        self.eq(color('\033[38;3m'), color())

        # Incomplete seq
        self.eq(color('\033[38;5m'), color())

        # Incomplete seq
        self.eq(color('\033[38;2m'), color())

        # Incomplete seq, match bold
        self.eq(color('\033[38;2;1m'), paint(em=Emphasis(bold=True)))

        # Incomplete seq, match lowint
        self.eq(color('\033[38;2;2m'), paint(em=Emphasis(lowint=True)))

        # Incomplete seq, match lowint
        self.eq(color('\033[38;2;3;2m'), paint(em=Emphasis(lowint=True)))

        # Unknown leading code
        self.eq(color('\033[39m'), color())
