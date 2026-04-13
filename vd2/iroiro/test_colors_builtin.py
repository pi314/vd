from .lib_test_utils import *

from iroiro import paint
from iroiro import nocolor
from iroiro import black, maroon, green, olive, navy, purple, teal, silver
from iroiro import grey, red, lime, yellow, blue, fuchsia, magenta, cyan, aqua, white
from iroiro import orange, darkorange, murasaki


class TestBuiltInColors(TestCase):
    def test_nocolor(self):
        self.eq(nocolor(), '')
        self.eq(nocolor('text'), 'text')
        self.eq(str(nocolor), '\033[m')
        self.eq('{}'.format(nocolor), '\033[m')

    def test_str(self):
        self.eq(str(black),     '\033[38;5;0m')
        self.eq(str(maroon),    '\033[38;5;1m')
        self.eq(str(green),     '\033[38;5;2m')
        self.eq(str(olive),     '\033[38;5;3m')
        self.eq(str(navy),      '\033[38;5;4m')
        self.eq(str(purple),    '\033[38;5;5m')
        self.eq(str(teal),      '\033[38;5;6m')
        self.eq(str(silver),    '\033[38;5;7m')
        self.eq(str(grey),      '\033[38;5;8m')
        self.eq(str(red),       '\033[38;5;9m')
        self.eq(str(lime),      '\033[38;5;10m')
        self.eq(str(yellow),    '\033[38;5;11m')
        self.eq(str(blue),      '\033[38;5;12m')
        self.eq(str(fuchsia),   '\033[38;5;13m')
        self.eq(str(magenta),   '\033[38;5;13m')
        self.eq(str(cyan),      '\033[38;5;14m')
        self.eq(str(aqua),      '\033[38;5;14m')
        self.eq(str(white),     '\033[38;5;15m')
        self.eq(str(darkorange),'\033[38;2;255;140;0m')
        self.eq(str(murasaki),  '\033[38;5;135m')

    def test_invert(self):
        self.eq(~red,      paint(bg=red))
        self.eq(~green,    paint(bg=green))
        self.eq(~yellow,   paint(bg=yellow))
        self.eq(~blue,     paint(bg=blue))
        self.eq(~magenta,  paint(bg=magenta))
        self.eq(~cyan,     paint(bg=cyan))
        self.eq(~white,    paint(bg=white))
        self.eq(~orange,   paint(bg=orange))

    def test_call(self):
        self.eq(black('text'),     '\033[38;5;0mtext\033[m')
        self.eq(maroon('text'),    '\033[38;5;1mtext\033[m')
        self.eq(green('text'),     '\033[38;5;2mtext\033[m')
        self.eq(olive('text'),     '\033[38;5;3mtext\033[m')
        self.eq(navy('text'),      '\033[38;5;4mtext\033[m')
        self.eq(purple('text'),    '\033[38;5;5mtext\033[m')
        self.eq(teal('text'),      '\033[38;5;6mtext\033[m')
        self.eq(silver('text'),    '\033[38;5;7mtext\033[m')
        self.eq(grey('text'),      '\033[38;5;8mtext\033[m')
        self.eq(red('text'),       '\033[38;5;9mtext\033[m')
        self.eq(lime('text'),      '\033[38;5;10mtext\033[m')
        self.eq(yellow('text'),    '\033[38;5;11mtext\033[m')
        self.eq(blue('text'),      '\033[38;5;12mtext\033[m')
        self.eq(fuchsia('text'),   '\033[38;5;13mtext\033[m')
        self.eq(magenta('text'),   '\033[38;5;13mtext\033[m')
        self.eq(cyan('text'),      '\033[38;5;14mtext\033[m')
        self.eq(aqua('text'),      '\033[38;5;14mtext\033[m')
        self.eq(white('text'),     '\033[38;5;15mtext\033[m')
        self.eq(darkorange('text'),'\033[38;2;255;140;0mtext\033[m')
        self.eq(murasaki('text'),  '\033[38;5;135mtext\033[m')
