from .lib_test_utils import *

from iroiro import decolor
from iroiro import orange


class TestDecolor(TestCase):
    def test_decolor(self):
        self.eq(decolor(orange('test')), 'test')
        self.eq(decolor('\033[1;31mred\033[m'), 'red')
