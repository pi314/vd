from .lib_test_utils import *

from iroiro import ColorRGB, ColorHSV
from iroiro import color


class TestColorHSV(TestCase):
    def test_copy_ctor(self):
        red = ColorHSV('@0,100,100')
        other = ColorHSV(red)
        self.eq(red, other)

    def test_empty(self):
        self.eq(ColorHSV(), '')
        self.eq(ColorHSV().seq, '')

    def test_parse_str(self):
        self.eq(ColorHSV('@0,100,100'), ColorHSV(0, 100, 100))
        self.eq(ColorHSV('@41,100,100'), ColorHSV(41, 100, 100))
        self.eq(ColorHSV('@401,100,100'), ColorHSV(41, 100, 100))
        self.eq(repr(ColorHSV('@0,100,100')), 'ColorHSV(0deg, 100%, 100%)')
        self.eq(repr(ColorHSV('@360,100,100')), 'ColorHSV(0deg, 100%, 100%)')

    def test_to_rgb(self):
        self.eq(ColorHSV('@0,100,100').to_rgb(), ColorRGB(255, 0, 0))

    def test_to_hsv(self):
        red = ColorHSV('@0,100,100')
        self.true(red.to_hsv() is red)

    def test(self):
        some_color = ColorHSV(420, 90.5, 0.5)
        self.eq(some_color.h, 60)
        self.eq(some_color.s, 90.5)
        self.eq(some_color.v, 0.5)
        self.eq(some_color.hsv, (60, 90.5, 0.5))

        self.eq(some_color.H, 60)
        self.eq(some_color.S, 90)
        self.eq(some_color.V, 0)
        self.eq(some_color.HSV, (60, 90, 0))

        self.eq(int(some_color), 60090000)

    def test_value_range_check(self):
        with self.raises(TypeError):
            ColorHSV(0, 101, 0)

        with self.raises(TypeError):
            ColorHSV(0, 0, 101)

        with self.raises(TypeError):
            ColorHSV(0, 0, 101, 0)

    def test_mul(self):
        some_color = ColorHSV(180, 100, 100) * 2
        self.eq(some_color.h, 0)
        self.eq(some_color.s, 200)
        self.eq(some_color.v, 200)
        self.eq(str(some_color), '\033[38;2;255;0;0m')

        some_color = ColorHSV(0, 100, 100) * 0.8
        self.eq(str(some_color), '\033[38;2;204;41;41m')

    def test_div(self):
        some_color = ColorHSV(160, 90, 0) // 2
        self.eq(some_color, ColorHSV(80, 45, 0))

    def test_overflow(self):
        some_color = ColorHSV(160, 90, 100) * 2
        self.eq(str(some_color), '\033[38;2;255;0;170m')
        self.eq(int(some_color), 320100100)

    def test_average(self):
        red = ColorHSV(360, 100, 100)
        yellow = ColorHSV(60, 100, 100)

        d_orange = red + yellow
        self.eq(d_orange.h, 60)
        self.eq(d_orange.s, 200)
        self.eq(d_orange.v, 200)

        orange = d_orange // 2
        self.eq(orange, ColorHSV(30, 100, 100))

    def test_format(self):
        lime = ColorHSV(120, 100, 100)
        self.eq('{}'.format(lime), str(ColorRGB(0, 255, 0)))
        self.eq('{:#}'.format(lime), '(@120, 100%, 100%)')

        with self.raises(TypeError):
            '{:d}'.format(lime)
