from .lib_test_utils import *

from iroiro import ColorRGB, ColorHSV
from iroiro import color
from iroiro import gradient


class TestGradient(TestCase):
    def test_invalid_values(self):
        with self.raises(TypeError):
            gradient(True, False)

        A = color()
        B = color()

        with self.raises(TypeError):
            gradient(A, B, 1.5)

        with self.raises(ValueError):
            gradient(A, B, 1)

    def test_trivial(self):
        # N=2 trivial case
        A = color(39)
        B = color(214)
        self.eq(gradient(A, B, 2), (A, B))

        # Color256() and ColorRGB() case
        A = color(39)
        B = color('#C0FFEE')
        self.eq(gradient(A, B), (A, B))

        # Color256() rgb6 and gray case
        A = color(39)
        B = color(255)
        self.eq(gradient(A, B), (A, B))

    def test_color256_gray(self):
        A = color(235)
        B = color(245)

        # default length
        res = gradient(A, B)
        ans = tuple(range(235, 246))
        self.eq(res, tuple(map(color, ans)))

        # shorter length
        res = gradient(A, B, N=5)
        ans = (235, 238, 241, 243, 245)
        self.eq(res, tuple(map(color, ans)))

        # longer length
        res = gradient(A, B, N=15)
        ans = (235, 235, 236, 236, 237, 237, 238, 238, 239, 240, 241, 242, 243, 244, 245)
        self.eq(res, tuple(map(color, ans)))


    def test_color256_rgb(self):
        A = color(39)
        B = color(214)

        # default length
        res = gradient(A, B)
        ans = (39 ,74 ,109 ,144 ,179 ,214)
        self.eq(res, tuple(map(color, ans)))

        # shorter length
        res = gradient(A, B, N=4)
        ans = (39, 109, 179, 214)
        self.eq(res, tuple(map(color, ans)))

        # longer length
        res = gradient(A, B, N=15)
        ans = (39, 39, 39, 74, 74, 74, 109, 109, 109, 144, 144, 179, 179, 214, 214)
        self.eq(res, tuple(map(color, ans)))

    def test_rgb(self):
        A = color(242, 5, 148)
        B = color(146, 219, 189)

        # default length
        res = gradient(A, B)
        ans = (ColorRGB(242, 5, 148),
               ColorRGB(218, 58, 158),
               ColorRGB(194, 112, 168),
               ColorRGB(170, 166, 179),
               ColorRGB(146, 219, 189))
        for i in range(len(res)):
            self.eq(res[i], ans[i])

        # shorter length
        res = gradient(A, B, N=4)
        ans = (ColorRGB(242, 5, 148),
               ColorRGB(210, 76, 162),
               ColorRGB(178, 148, 175),
               ColorRGB(146, 219, 189))
        for i in range(len(res)):
            self.eq(res[i], ans[i])

        # longer length
        res = gradient(A, B, N=15)
        ans = (ColorRGB(242, 5, 148),
               ColorRGB(235, 20, 151),
               ColorRGB(228, 36, 154),
               ColorRGB(221, 51, 157),
               ColorRGB(215, 66, 160),
               ColorRGB(208, 81, 163),
               ColorRGB(201, 97, 166),
               ColorRGB(194, 112, 168),
               ColorRGB(187, 127, 171),
               ColorRGB(180, 143, 174),
               ColorRGB(173, 158, 177),
               ColorRGB(167, 173, 180),
               ColorRGB(160, 188, 183),
               ColorRGB(153, 204, 186),
               ColorRGB(146, 219, 189))
        for i in range(len(res)):
            self.eq(res[i], ans[i])

        A = color('#FF1100')
        B = color('#FF0011')
        res = gradient(A, B, N=3)
        self.eq(res, (A, color('#FF0808'), B))

    def test_hsv(self):
        A = ColorHSV(0, 100, 100)
        B = ColorHSV(300, 50, 100)

        # clockwise
        res = gradient(A, B, clockwise=True)
        res2 = gradient(A, B)
        ans = (ColorHSV(0, 100, 100),
               ColorHSV(33, 94, 100),
               ColorHSV(66, 88, 100),
               ColorHSV(100, 83, 100),
               ColorHSV(133, 77, 100),
               ColorHSV(166, 72, 100),
               ColorHSV(200, 66, 100),
               ColorHSV(233, 61, 100),
               ColorHSV(266, 55, 100),
               ColorHSV(300, 50, 100),)

        self.eq(len(res), len(ans))
        for a, b, c in zip(res, res2, ans):
            # Check if the colors are close enough
            self.le(abs(sum(a.hsv) - sum(b.hsv)), 2)
            self.le(abs(sum(a.hsv) - sum(c.hsv)), 2)

        res = gradient(A, B, clockwise=True, N=5)
        res2 = gradient(A, B)
        ans = (ColorHSV(0, 100, 100),
               ColorHSV(75, 88, 100),
               ColorHSV(150, 75, 100),
               ColorHSV(225, 62, 100),
               ColorHSV(300, 50, 100),)

        for a, b in zip(res, ans):
            # Check if the colors are close enough
            self.le(abs(sum(a.hsv) - sum(b.hsv)), 2)

        # counter-clockwise
        res = gradient(A, B, clockwise=False)
        ans = (ColorHSV(0, 100, 100),
               ColorHSV(345, 87, 100),
               ColorHSV(330, 75, 100),
               ColorHSV(315, 62, 100),
               ColorHSV(300, 50, 100),)

        self.eq(len(res), len(ans))
        for a, b in zip(res, ans):
            # Check if the colors are close enough
            self.le(abs(sum(a.hsv) - sum(b.hsv)), 2)

        # reverse
        res = gradient(A, B, clockwise=True)
        rev = gradient(A, B, clockwise=True, reverse=True)

        for a, b in zip(res, rev[::-1]):
            # Check if the colors are close enough
            self.le(abs(sum(a.hsv) - sum(b.hsv)), 2)

        # reverse
        res = gradient(A, B, clockwise=False)
        rev = gradient(A, B, clockwise=False, reverse=True)

        for a, b in zip(res, rev[::-1]):
            # Check if the colors are close enough
            self.le(abs(sum(a.hsv) - sum(b.hsv)), 2)
