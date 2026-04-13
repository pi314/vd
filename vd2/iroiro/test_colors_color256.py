from .lib_test_utils import *

from iroiro import Color256, ColorRGB
from iroiro import color


class TestColor256(TestCase):
    def test_copy_ctor(self):
        red = Color256(1)
        other = Color256(red)
        self.eq(red, other)

    def test_empty(self):
        self.eq(Color256(), '')
        self.eq(Color256().seq, '')

    def test_index(self):
        for i in range(256):
            self.eq(color(i).index, i)

        for i in range(256):
            self.eq(int(color(i)), i)

    def test_value_type_check(self):
        with self.raises(TypeError):
            Color256('wah')

    def test_value_range_check(self):
        with self.raises(ValueError):
            Color256(300)

    def test_repr(self):
        for i in range(256):
            self.eq(repr(Color256(i)), 'Color256({})'.format(i))

    def test_seq(self):
        for i in range(256):
            self.eq(Color256(i).seq, '\033[38;5;{}m'.format(i))

    def test_to_rgb(self):
        self.eq(color(0).to_rgb(), ColorRGB(0x00, 0x00, 0x00))
        self.eq(color(1).to_rgb(), ColorRGB(0x80, 0x00, 0x00))
        self.eq(color(2).to_rgb(), ColorRGB(0x00, 0x80, 0x00))
        self.eq(color(3).to_rgb(), ColorRGB(0x80, 0x80, 0x00))
        self.eq(color(4).to_rgb(), ColorRGB(0x00, 0x00, 0x80))
        self.eq(color(5).to_rgb(), ColorRGB(0x80, 0x00, 0x80))
        self.eq(color(6).to_rgb(), ColorRGB(0x00, 0x80, 0x80))
        self.eq(color(7).to_rgb(), ColorRGB(0xC0, 0xC0, 0xC0))
        self.eq(color(8).to_rgb(), ColorRGB(0x80, 0x80, 0x80))
        self.eq(color(9).to_rgb(), ColorRGB(0xFF, 0x00, 0x00))
        self.eq(color(10).to_rgb(), ColorRGB(0x00, 0xFF, 0x00))
        self.eq(color(11).to_rgb(), ColorRGB(0xFF, 0xFF, 0x00))
        self.eq(color(12).to_rgb(), ColorRGB(0x00, 0x00, 0xFF))
        self.eq(color(13).to_rgb(), ColorRGB(0xFF, 0x00, 0xFF))
        self.eq(color(14).to_rgb(), ColorRGB(0x00, 0xFF, 0xFF))
        self.eq(color(15).to_rgb(), ColorRGB(0xFF, 0xFF, 0xFF))
        self.eq(color(208).to_rgb(), ColorRGB(0xFF, 0x87, 0x00))
        self.eq(color(232).to_rgb(), ColorRGB(0x08, 0x08, 0x08))
        self.eq(color(237).to_rgb(), ColorRGB(0x3A, 0x3A, 0x3A))

    def test_to_hsv(self):
        for i in range(256):
            self.eq(color(i).to_hsv(), color(i).to_rgb().to_hsv())
