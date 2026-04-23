from .lib_test_utils import *

import iroiro

from iroiro import Emphasis, bold, lowint, underline, blink, reverse, invisible
from iroiro import orange, yellow


class TestEmphasis(TestCase):
    def test_builtin_emphasis(self):
        self.eq(bold, Emphasis(bold=True))
        self.eq(lowint, Emphasis(lowint=True))
        self.eq(underline, Emphasis(underline=True))
        self.eq(blink, Emphasis(blink=True))
        self.eq(reverse, Emphasis(reverse=True))
        self.eq(invisible, Emphasis(invisible=True))

    def test_builtin_from_int(self):
        self.eq(bold, Emphasis(1))
        self.eq(lowint, Emphasis(2))
        self.eq(underline, Emphasis(4))
        self.eq(blink, Emphasis(5))
        self.eq(reverse, Emphasis(7))
        self.eq(invisible, Emphasis(8))

    def test_emphasis_seq(self):
        self.eq(bold.seq, '\033[1m')
        self.eq(lowint.seq, '\033[2m')
        self.eq(underline.seq, '\033[4m')
        self.eq(blink.seq, '\033[5m')
        self.eq(reverse.seq, '\033[7m')
        self.eq(invisible.seq, '\033[8m')

        self.eq(str(bold), bold.seq)
        self.eq(str(lowint), lowint.seq)
        self.eq(str(underline), underline.seq)
        self.eq(str(blink), blink.seq)
        self.eq(str(reverse), reverse.seq)
        self.eq(str(invisible), invisible.seq)

    def test_emphasis_int(self):
        self.eq(int(bold), 1 << 0)
        self.eq(int(lowint), 1 << 1)
        self.eq(int(underline), 1 << 3)
        self.eq(int(blink), 1 << 4)
        self.eq(int(reverse), 1 << 6)
        self.eq(int(invisible), 1 << 7)

    def test_emphasis_call(self):
        self.eq(bold(), bold.seq)
        self.eq(lowint(), lowint.seq)
        self.eq(underline(), underline.seq)
        self.eq(blink(), blink.seq)
        self.eq(reverse(), reverse.seq)
        self.eq(invisible(), invisible.seq)

        self.eq(bold('test'), '\033[1mtest\033[m')
        self.eq(lowint('test'), '\033[2mtest\033[m')
        self.eq(underline('test'), '\033[4mtest\033[m')
        self.eq(blink('test'), '\033[5mtest\033[m')
        self.eq(reverse('test'), '\033[7mtest\033[m')
        self.eq(invisible('test'), '\033[8mtest\033[m')

    def test_emphasis_or(self):
        bu = bold | underline
        self.eq(bu, Emphasis(bold=True, underline=True))
        self.eq(bu(), '\033[1;4m')
        self.eq(bu('iro'), '\033[1;4miro\033[m')

    def test_emphasis_or_with_color(self):
        bo = bold | orange
        self.eq(bo('iro'), '\033[1;38;2;255;165;0miro\033[m')

    def test_emphasis_or_with_paints(self):
        boy = bold | (orange / yellow)
        self.eq(boy('iro'), '\033[1;38;2;255;165;0;48;5;11miro\033[m')

    def test_emphasis_or_with_invalid_types(self):
        with self.raises(TypeError):
            boy = bold | 'wah'

    def test_emphasis_repr(self):
        self.true(repr(Emphasis()).startswith('Emphasis'))

        bu = bold | underline
        self.eq(repr(bu), 'Emphasis(bold=True, underline=True)')
