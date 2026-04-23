from .lib_test_utils import *

from iroiro import ColorCompound
from iroiro import color, paint
from iroiro import nocolor, orange, bold


orange = color(214)
coffee = color('#C0FFEE')
purple = color('@300,100,50')


class TestColorTraits(TestCase):
    def test_repr(self):
        self.eq(repr(orange), 'Color256(214)')
        self.eq(' '.join(repr(coffee).split()), 'ColorRGB(192, 255, 238)')
        self.eq(' '.join(repr(purple).split()), 'ColorHSV(300deg, 100%, 50%)')

    def test_int(self):
        self.eq(int(orange), 214)
        self.eq(int(coffee), 0xC0FFEE)
        self.eq(int(purple), 300100050)

    def test_fg(self):
        self.eq(orange('text'), '\033[38;5;214mtext\033[m')
        self.eq(coffee('text'), '\033[38;2;192;255;238mtext\033[m')
        self.eq(purple('text'), '\033[38;2;128;0;128mtext\033[m')
        self.eq(orange.fg('text'), '\033[38;5;214mtext\033[m')
        self.eq(coffee.fg('text'), '\033[38;2;192;255;238mtext\033[m')
        self.eq(purple.fg('text'), '\033[38;2;128;0;128mtext\033[m')

    def test_bg(self):
        self.eq(orange.bg('text'), '\033[48;5;214mtext\033[m')
        self.eq(coffee.bg('text'), '\033[48;2;192;255;238mtext\033[m')
        self.eq(purple.bg('text'), '\033[48;2;128;0;128mtext\033[m')

    def test_str(self):
        self.eq(str(orange), '\033[38;5;214m')
        self.eq(str(coffee), '\033[38;2;192;255;238m')
        self.eq(str(purple), '\033[38;2;128;0;128m')
        self.eq('{}'.format(orange), str(orange))
        self.eq('{}'.format(coffee), str(coffee))
        self.eq('{}'.format(purple), str(purple))

    def test_invert(self):
        self.eq(str(~orange), '\033[48;5;214m')
        self.eq(str(~coffee), '\033[48;2;192;255;238m')
        self.eq(str(~purple), '\033[48;2;128;0;128m')
        self.true(isinstance(~orange, ColorCompound))
        self.true(isinstance(~coffee, ColorCompound))
        self.true(isinstance(~purple, ColorCompound))

    def test_div(self):
        colors_under_test = (orange, coffee, purple)
        for A in colors_under_test:
            for B in colors_under_test:
                self.eq(A / B, paint(fg=A, bg=B))

        for A in colors_under_test:
            with self.raises(TypeError):
                A / 1

    def test_or(self):
        colors_under_test = (orange, coffee, purple)
        for A in colors_under_test:
            self.eq(nocolor | A, A)
            self.eq(A | nocolor, nocolor)

        for A in colors_under_test:
            for B in colors_under_test:
                self.eq(A | B, B)

    def test_or_with_emphasis(self):
        ob = orange | bold
        self.eq(ob('wah'), '\033[1;38;5;214mwah\033[m')
