from .lib_test_utils import *

from iroiro import ColorRGB, ColorHSV
from iroiro import color


class TestColorRGB(TestCase):
    def test_copy_ctor(self):
        red = ColorRGB(160, 90, 0)
        other = ColorRGB(red)
        self.eq(red, other)

    def test_empty(self):
        self.eq(ColorRGB(), '')
        self.eq(ColorRGB().seq, '')

    def test(self):
        some_color = ColorRGB(160.2, 90, 0)

        self.eq(some_color.r, 160.2)
        self.eq(some_color.g, 90)
        self.eq(some_color.b, 0)
        self.eq(some_color.rgb, (160.2, 90, 0))

        self.eq(some_color.R, 160)
        self.eq(some_color.G, 90)
        self.eq(some_color.B, 0)
        self.eq(some_color.RGB, (160, 90, 0))

        self.eq(int(some_color), 0xA05A00)

    def test_value_range_check(self):
        with self.raises(TypeError):
            ColorRGB(300, 300, 300)

    def test_mul(self):
        some_color = ColorRGB(160, 90, 0) * 2
        self.eq(some_color.r, 320)
        self.eq(some_color.g, 180)
        self.eq(some_color.b, 0)

        self.eq(some_color.R, 255)
        self.eq(some_color.G, 180)
        self.eq(some_color.B, 0)

        some_color = ColorRGB(160, 90, 0) * 0.8
        self.eq(str(some_color), '\033[38;2;128;72;0m')

    def test_div(self):
        some_color = ColorRGB(160, 90, 0) // 2
        self.eq(some_color, ColorRGB(80, 45, 0))

    def test_overflow(self):
        some_color = ColorRGB(160, 90, 0) * 2
        self.eq(str(some_color), '\033[38;2;255;180;0m')
        self.eq(int(some_color), 0xFFB400)

    def test_average(self):
        d_red = ColorRGB(255, 0, 0) * 2
        self.eq(d_red.r, 255 * 2)
        self.eq(d_red.g, 0)
        self.eq(d_red.b, 0)

        d_green = ColorRGB(0, 255, 0) * 2
        self.eq(d_green.r, 0)
        self.eq(d_green.g, 255 * 2)
        self.eq(d_green.b, 0)

        d_yellow = d_red + d_green
        self.eq(d_yellow.r, 255 * 2)
        self.eq(d_yellow.g, 255 * 2)
        self.eq(d_yellow.b, 0)

        yellow = d_yellow // 2
        self.eq(yellow, ColorRGB(255, 255, 0))

    def test_to_rgb(self):
        red = ColorRGB(255, 0, 0)
        self.true(red.to_rgb() is red)

    def test_to_hsv(self):
        red = ColorRGB(255, 0, 0)
        self.eq(red.to_hsv(), ColorHSV(0, 100, 100))

    def test_format(self):
        red = ColorRGB(255, 0, 0)
        self.eq('{:#x}'.format(red), '#ff0000')
        self.eq('{:#X}'.format(red), '#FF0000')
        with self.raises(TypeError):
            self.eq('{:d}'.format(red), str(red))
