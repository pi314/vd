from .lib_test_utils import *

from iroiro import Emphasis
from iroiro import color, paint
from iroiro import black, maroon, green, olive, navy, purple, teal, silver
from iroiro import grey, red, lime, yellow, blue, fuchsia, magenta, cyan, aqua, white
from iroiro import orange, darkorange, murasaki

from iroiro import bold, underline


class TestPaint(TestCase):
    def test_repr(self):
        self.true(repr(paint()).startswith('ColorCompound'))

    def test_empty(self):
        self.eq(paint(), '')
        self.eq(paint().seq, '')
        self.eq(paint(em=None, fg=None, bg=None), '')
        self.eq(paint(em=Emphasis(), fg=None, bg=None), '')
        self.eq(paint(em=None, fg=color(), bg=None), '')
        self.eq(paint(em=None, fg=None, bg=color()), '')

    def test_eq(self):
        self.eq(paint(fg=42), color(42))
        self.eq(paint(fg=0, bg=11), paint(fg=0, bg=11))
        self.ne(paint(fg=0, bg=11), None)
        self.ne(paint(fg=0, bg=11), 42)

    def test_div(self):
        ry = red / yellow
        bg = blue / green
        rybg = ry / bg
        self.eq(rybg, paint(fg=red, bg=blue))
        self.eq(rybg('text'), '\033[38;5;9;48;5;12mtext\033[m')

    def test_or(self):
        self.eq(black | (~yellow), paint(fg=0, bg=11))

        ry = red / yellow
        ig = ~green
        ryig = ry | ig
        self.eq(ryig, paint(fg=red, bg=green))

        my = ry | magenta
        self.eq(my, magenta / yellow)

        myb = my | bold
        self.eq(myb, '\033[1;38;5;13;48;5;11m')

        with self.raises(TypeError):
            ry | None

        with self.raises(TypeError):
            ry | 42

        self.eq(ry | bold, '\033[1;38;5;9;48;5;11m')
        self.eq(ry | bold | underline, '\033[1;4;38;5;9;48;5;11m')

        self.eq(ry | paint(reset=True, bg=lime), '\033[48;5;10m')

    def test_invert(self):
        ry = red / yellow
        bg = blue / green
        rybg = ry / bg
        self.eq(~rybg, paint(fg=blue, bg=red))
        self.eq((~rybg)('text'), '\033[38;5;12;48;5;9mtext\033[m')

    def test_select_partial_attr(self):
        ry = (red / yellow) | bold
        self.eq(paint(fg=ry), paint(fg=red))
        self.eq(paint(bg=ry), paint(bg=yellow))
        self.eq(paint(em=ry), paint(em=bold))

    def test_type_check(self):
        with self.raises(TypeError):
            paint(em='wah', fg=None, bg=None)
