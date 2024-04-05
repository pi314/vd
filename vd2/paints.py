import re


class paint:
    def __init__(self, fg=None, bg=None, blue=None):
        if blue is not None:
            self.fg = (fg, bg, blue)
            self.bg = None
        else:
            self.fg = fg
            self.bg = bg

        fg_seq = self._seq(self.fg, True)
        bg_seq = self._seq(self.bg, False)
        seq = ';'.join(filter(None, [fg_seq, bg_seq]))
        self.seq = '' if not seq else ('\033[' + seq + 'm')

    def __repr__(self):
        return f'paint(fg={self.fg}, bg={self.bg})'

    def __call__(self, s=''):
        return s if not self.seq else f'{self.seq}{s}\033[m'

    def __str__(self):
        return self.seq

    def __or__(self, other):
        fg = self.fg if other.fg is None else other.fg
        bg = self.bg if other.bg is None else other.bg
        return paint(fg=fg, bg=bg)

    def __add__(self, other):
        return self | other

    def __truediv__(self, other):
        return paint(fg=self.fg, bg=other.fg)

    def __rtruediv__(self, left):
        if left == 1:
            left = paint()

        return left | paint(fg=self.bg, bg=self.fg)

    def _seq(self, code, is_fg):
        if code in (None, ''):
            return ''

        if isinstance(code, str):
            return code

        ret = ('38' if is_fg else '48') + ';'

        if isinstance(code, int):
            ret += f'5;{code}'

        elif len(code) == 3:
            ret += '2;' + ';'.join(str(c) for c in code)

        else:
            raise ValueError(f'Invalid color code: {code}')

        return ret

    def __eq__(self, other):
        return self.seq == other.seq


decolor_regex = re.compile('\033' + r'\[[\d;]*m')
def decolor(s):
    return decolor_regex.sub('', s)


nocolor = paint()
black = paint(236)
red = paint(1)
green = paint(2)
yellow = paint(3)
blue = paint(4)
magenta = paint(5)
cyan = paint(6)
white = paint(7)
orange = paint(208)
