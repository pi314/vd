from .lib_test_utils import *

from iroiro import Color8, Color256, ColorRGB


class TestColor8(TestCase):
    def test_copy_ctor(self):
        red = Color8(1)
        other = Color8(red)
        self.eq(red, other)

    def test_empty(self):
        self.eq(Color8(), '')
        self.eq(Color8().seq, '')

    def test_index(self):
        for i in range(8):
            self.eq(Color8(i).index, i)

        for i in range(8):
            self.eq(int(Color8(i)), i)

    def test_value_type_check(self):
        with self.raises(TypeError):
            Color8('wah')

    def test_value_range_check(self):
        with self.raises(ValueError):
            Color8(10)

    def test_repr(self):
        for i in range(8):
            self.eq(repr(Color8(i)), 'Color8({})'.format(i))

    def test_seq(self):
        for i in range(8):
            self.eq(Color8(i).seq, '\033[3{}m'.format(i))

    def test_to_color256(self):
        for i in range(8):
            self.eq(Color8(i).to_256(), Color256(i))

    def test_to_rgb(self):
        self.eq(Color8(0).to_rgb(), ColorRGB(0x00, 0x00, 0x00))
        self.eq(Color8(1).to_rgb(), ColorRGB(0x80, 0x00, 0x00))
        self.eq(Color8(2).to_rgb(), ColorRGB(0x00, 0x80, 0x00))
        self.eq(Color8(3).to_rgb(), ColorRGB(0x80, 0x80, 0x00))
        self.eq(Color8(4).to_rgb(), ColorRGB(0x00, 0x00, 0x80))
        self.eq(Color8(5).to_rgb(), ColorRGB(0x80, 0x00, 0x80))
        self.eq(Color8(6).to_rgb(), ColorRGB(0x00, 0x80, 0x80))
        self.eq(Color8(7).to_rgb(), ColorRGB(0xC0, 0xC0, 0xC0))

    def test_to_hsv(self):
        for i in range(8):
            self.eq(Color8(i).to_hsv(), Color8(i).to_rgb().to_hsv())
