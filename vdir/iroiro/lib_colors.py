import re
import abc
import itertools

from .lib_math import sgn
from .lib_math import vector
from .lib_math import lerp
from .lib_math import interval
from .lib_math import resample
from .lib_math import is_uint8
from .lib_math import clamp

from .internal_utils import exporter
export, __all__ = exporter()


def _apply(em, fg, bg, *args, reset=False):
    s = ' '.join(str(arg) for arg in args)

    code = ';'.join(filter(None, (
        em.code if em is not None and em.code else None,
        ('3' + fg.code) if fg is not None and fg.code else None,
        ('4' + bg.code) if bg is not None and bg.code else None,
        )))

    if reset and code:
        code = '0;' + code

    start = ('\033[' + code + 'm') if code or reset else ''
    end = '\033[m' if start else ''

    if not args:
        return start

    return start + s + end


class AbstractColor(abc.ABC):
    @property
    @abc.abstractmethod
    def seq(self): # pragma: no cover
        pass

    def __eq__(self, other):
        if isinstance(other, AbstractColor):
            return self.seq == other.seq
        return self.seq == other


@export
class NoColor(AbstractColor):
    @property
    def seq(self):
        return '\033[m'

    @property
    def code(self):
        return ''

    def __repr__(self):
        return 'NoColor()'

    def __eq__(self, other):
        return isinstance(other, self.__class__) or other == '\033[m'

    def __str__(self):
        return '\033[m'

    def __or__(self, other):
        return other

    def __call__(self, *args):
        return _apply(self, None, None, *args)


@export
class Emphasis(AbstractColor):
    ATTR_CODE = {
            'bold': 1,
            'lowint': 2,
            'underline': 4,
            'blink': 5,
            'reverse': 7,
            'invisible': 8,
            }

    def __init__(self, *codes, bold=False, lowint=False, underline=False,
                 blink=False, reverse=False, invisible=False):
        if codes:
            for name, code in Emphasis.ATTR_CODE.items():
                setattr(self, name, code in codes)
        else:
            self.bold = bold
            self.lowint = lowint
            self.underline = underline
            self.blink = blink
            self.reverse = reverse
            self.invisible = invisible

    @property
    def code(self):
        return ';'.join(str(Emphasis.ATTR_CODE[attr])
                        for attr in Emphasis.ATTR_CODE
                        if getattr(self, attr)
                        )

    @property
    def seq(self):
        return _apply(self, None, None)

    def __repr__(self):
        attrs = []
        return '{name}({attrs})'.format(
                name=self.__class__.__name__,
                attrs=', '.join('{}=True'.format(attr)
                                for attr in Emphasis.ATTR_CODE
                                if getattr(self, attr)
                                )
                )

    def __int__(self):
        ret = 0
        for attr, code in Emphasis.ATTR_CODE.items():
            ret |= (1 if getattr(self, attr) else 0) << (code - 1)
        return ret

    def __call__(self, *args):
        return _apply(self, None, None, *args)

    def __str__(self):
        return self.seq or '\033[m'

    def __or__(self, rhs):
        if rhs is None:
            return self

        if isinstance(rhs, NoColor):
            return rhs

        elif isinstance(rhs, self.__class__):
            attrs = {attr: (getattr(rhs, attr) or getattr(self, attr))
                     for attr in Emphasis.ATTR_CODE}
            return Emphasis(**attrs)

        elif isinstance(rhs, Color):
            return ColorCompound(em=self, fg=rhs, bg=None)

        elif isinstance(rhs, ColorCompound):
            return ColorCompound(em=self | rhs.em, fg=rhs.fg, bg=rhs.bg)

        raise TypeError('unsupported operand types for |: {} and {}'.format(
            type(self).__name__, type(rhs).__name__))


named_emphasis = ('bold', 'lowint', 'underline', 'blink', 'reverse', 'invisible')
bold = Emphasis(bold=True)
lowint = Emphasis(lowint=True)
underline = Emphasis(underline=True)
blink = Emphasis(blink=True)
reverse = Emphasis(reverse=True)
invisible = Emphasis(invisible=True)

def _setup_named_emphasis():
    for name in named_emphasis:
        em = Emphasis(**{name: True})
        globals()[name] = em
        export(name)
_setup_named_emphasis()
del _setup_named_emphasis


@export
class Color(AbstractColor):
    @abc.abstractmethod
    def __init__(self, *args,
                 bold=False, lowint=False, underline=False,
                 blink=False, reverse=False, invisible=False,
                 **kwargs): # pragma: no cover
        self.bold = bold
        self.lowint = lowint
        self.underline = underline
        self.blink = blink
        self.reverse = reverse
        self.invisible = invisible

    @property
    @abc.abstractmethod
    def code(self): # pragma: no cover
        raise NotImplementedError

    @property
    def seq(self):
        return _apply(None, self, None)

    @abc.abstractmethod
    def __repr__(self): # pragma: no cover
        raise NotImplementedError

    @abc.abstractmethod
    def __int__(self): # pragma: no cover
        raise NotImplementedError

    def __call__(self, *args):
        return self.fg(*args)

    def fg(self, *args):
        return _apply(None, self, None, *args)

    def bg(self, *args, **kwargs):
        return _apply(None, None, self, *args)

    def __str__(self):
        return _apply(None, self, None) or '\033[m'

    def __invert__(self):
        return ColorCompound(bg=self)

    def __truediv__(self, other):
        if not isinstance(other, Color):
            raise TypeError('Only Color() / Color() is allowed')
        return ColorCompound(fg=self, bg=other)

    def __or__(self, other):
        if isinstance(other, NoColor):
            return other
        elif isinstance(other, Color):
            return other if other.code else self
        return ColorCompound(fg=self) | other


@export
def color(*args, **kwargs):
    nargs = len(args)
    arg1 = args[0] if len(args) == 1 else None

    # empty
    if not args:
        return Color256(None)

    if nargs != 1:
        # ColorRGB ctor
        if len(args) == 3 and all(is_uint8(i) for i in args):
            return ColorRGB(*args, **kwargs)

    # Copy ctor
    elif issubclass(type(arg1), Color):
        return type(arg1)(*args, **kwargs)

    # Color256 ctor
    elif arg1 is None or is_uint8(arg1):
        return Color256(*args, **kwargs)

    # ColorRGB ctor #RRGGBB
    elif isinstance(arg1, str) and re.fullmatch(r'#[0-9A-Fa-f]{6}', arg1):
        return ColorRGB(*args, **kwargs)

    # ColorHSV @H,S,V format
    elif isinstance(arg1, str) and re.fullmatch(r'@[0-9]+,[0-9]+,[0-9]+', arg1):
        return ColorHSV(*args, **kwargs)

    elif isinstance(arg1, str):
        return _parse(arg1)

    raise TypeError('Invalid arguments: {}'.format(args))


@export
class Color8(Color):
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(index, self.__class__):
            index = index.index

        no = None
        if index is None:
            pass
        elif isinstance(index, bool) or not isinstance(index, int):
            no = TypeError
        elif not (0 <= index <= 7):
            no = ValueError
        if no:
            raise no('Invalid color index: {}'.format(index))

        self.index = index

    def __repr__(self):
        return '{name}({index})'.format(
                name=self.__class__.__name__,
                index=self.index)

    def __int__(self):
        return self.index

    @property
    def code(self):
        if self.index is None:
            return ''
        return str(self.index)

    def to_256(self):
        return Color256(self.index)

    def to_rgb(self):
        return self.to_256().to_rgb()

    def to_hsv(self):
        return self.to_rgb().to_hsv()


@export
class Color256(Color):
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(index, self.__class__):
            index = index.index

        no = None
        if index is None:
            pass
        elif isinstance(index, bool) or not isinstance(index, int):
            no = TypeError
        elif not is_uint8(index):
            no = ValueError
        if no:
            raise no('Invalid color index: {}'.format(index))

        self.index = index

    def __repr__(self):
        return '{name}({index})'.format(
                name=self.__class__.__name__,
                index=self.index)

    def __int__(self):
        return self.index

    @property
    def code(self):
        if self.index is None:
            return ''
        return '8;5;{}'.format(self.index)

    def to_rgb(self):
        if self.index < 16:
            base = 0xFF if (self.index > 7) else 0x80
            is_7 = (self.index == 7)
            is_8 = (self.index == 8)
            R = base * ((self.index & 0x1) != 0) + (0x40 * is_7) + (0x80 * is_8)
            G = base * ((self.index & 0x2) != 0) + (0x40 * is_7) + (0x80 * is_8)
            B = base * ((self.index & 0x4) != 0) + (0x40 * is_7) + (0x80 * is_8)

        elif self.index < 232:
            base = self.index - 16
            index_R = (base // 36)
            index_G = ((base % 36) // 6)
            index_B = (base % 6)
            R = (55 + index_R * 40) if index_R > 0 else 0
            G = (55 + index_G * 40) if index_G > 0 else 0
            B = (55 + index_B * 40) if index_B > 0 else 0

        else:
            R = G = B = (self.index - 232) * 10 + 8

        return ColorRGB(R, G, B)

    def to_hsv(self):
        return self.to_rgb().to_hsv()


@export
class ColorRGB(Color):
    def __init__(self, *args, overflow=False, **kwargs):
        super().__init__(**kwargs)

        nargs = len(args)
        arg1 = args[0] if len(args) else None

        type_check = (lambda x: 0 <= x < 256
                      if not overflow
                      else lambda x: isinstance(x, (int, float)))

        self.r = None
        self.g = None
        self.b = None

        if not args:
            return

        elif nargs != 1:
            # (num, num, num) format
            if len(args) == 3 and all(type_check(i) for i in args):
                (self.r, self.g, self.b) = args

        # Copy ctor
        elif isinstance(arg1, self.__class__):
            (self.r, self.g, self.b) = arg1.rgb

        # #RRGGBB format
        elif isinstance(arg1, str) and re.fullmatch(r'#[0-9A-Fa-f]{6}', arg1):
            rgb_str = arg1[1:]
            self.r = int(rgb_str[0:2], 16)
            self.g = int(rgb_str[2:4], 16)
            self.b = int(rgb_str[4:6], 16)

        if None in self.rgb:
            raise TypeError('Invalid RGB value: {}'.format(args))

    def __repr__(self):
        return '{name}({self.r}, {self.g}, {self.b})'.format(
                name=self.__class__.__name__,
                self=self)

    @property
    def R(self):
        return 0 if self.r is None else clamp(0, round(self.r), 255)

    @property
    def G(self):
        return 0 if self.g is None else clamp(0, round(self.g), 255)

    @property
    def B(self):
        return 0 if self.b is None else clamp(0, round(self.b), 255)

    @property
    def RGB(self):
        return (self.R, self.G, self.B)

    @property
    def rgb(self):
        return (self.r, self.g, self.b)

    @property
    def code(self):
        if None in self.rgb:
            return ''
        return '8;2;{};{};{}'.format(self.R, self.G, self.B)

    def __add__(self, other):
        rgb = vector(self.rgb) + vector(other.rgb)
        return ColorRGB(*rgb, overflow=True)

    def __mul__(self, num):
        return ColorRGB(*vector(self.rgb) * num, overflow=True)

    def __floordiv__(self, num):
        return ColorRGB(*vector(self.rgb) // num, overflow=True)

    def __int__(self):
        return (self.R << 16) | (self.G << 8) | (self.B)

    def __format__(self, spec):
        if not spec:
            return str(self)

        if spec in ('#', '#x', '#X'):
            x = (spec + 'X')[1]
            return '#{r:0>2{x}}{g:0>2{x}}{b:0>2{x}}'.format(r=self.R, g=self.G, b=self.B, x=x)

        return format(self.RGB, spec)

    def to_rgb(self):
        return self

    def to_hsv(self, overflow=False):
        import colorsys
        hsv = colorsys.rgb_to_hsv(self.R / 255, self.G / 255, self.B / 255)
        return ColorHSV(hsv[0] * 360, hsv[1] * 100, hsv[2] * 100, overflow=overflow)


@export
class ColorHSV(Color):
    def __init__(self, *args, overflow=False, **kwargs):
        super().__init__(**kwargs)

        arg1 = args[0] if len(args) else None

        self.h = None
        self.s = None
        self.v = None

        h = None
        s = None
        v = None

        if not args:
            return

        # Copy ctor
        elif len(args) == 1 and isinstance(arg1, self.__class__):
            other = arg1
            (h, s, v) = (other.h, other.s, other.v)

        # @H,S,V format
        elif len(args) == 1 and isinstance(arg1, str) and re.fullmatch(r'@[0-9]+,[0-9]+,[0-9]+', arg1):
            (h, s, v) = list(map(lambda x: int(x, 10), arg1[1:].split(',')))

        # (num, num, num) format
        elif len(args) == 3:
            (h, s, v) = args

        # Value range check
        if (all(isinstance(x, (int, float)) for x in (h, s, v)) and
            overflow or
            ((s is not None and 0 <= s <= 100) and
             (v is not None and 0 <= v <= 100))):
            (self.h, self.s, self.v) = (h % 360, s, v)
            self._rgb = self.to_rgb(overflow)

        else:
            raise TypeError('Invalid HSV value: {}'.format(args))

    def __repr__(self):
        return '{name}({h}deg, {s}%, {v}%)'.format(
                name=self.__class__.__name__,
                h=self.h, s=self.s, v=self.v)

    @property
    def H(self):
        return 0 if self.h is None else ((round(self.h) + 360) % 360)

    @property
    def S(self):
        return 0 if self.s is None else clamp(0, round(self.s), 100)

    @property
    def V(self):
        return 0 if self.v is None else clamp(0, round(self.v), 100)

    @property
    def HSV(self):
        return (self.H, self.S, self.V)

    @property
    def hsv(self):
        return (self.h, self.s, self.v)

    @property
    def code(self):
        if None in self.hsv:
            return ''
        return self._rgb.code

    def __add__(self, other):
        hsv = vector(self.hsv) + vector(other.hsv)
        return ColorHSV(*hsv, overflow=True)

    def __mul__(self, num):
        return ColorHSV(*vector(self.hsv) * num, overflow=True)

    def __floordiv__(self, num):
        return ColorHSV(*vector(self.hsv) // num, overflow=True)

    def __int__(self):
        return (self.H * 1000000) + (self.S * 1000) + (self.V)

    def __format__(self, spec):
        if not spec:
            return str(self)
        if spec == '#':
            return '(@{}, {}%, {}%)'.format(self.H, self.S, self.V)
        return format(self.hsv, spec)

    def to_rgb(self, overflow=False):
        import colorsys
        return ColorRGB(*vector(colorsys.hsv_to_rgb(
            self.H / 360,
            self.S / 100,
            self.V / 100)) * 255, overflow=overflow)

    def to_hsv(self):
        return self


@export
class ColorCompound(AbstractColor):
    def __init__(self, *, reset=False, em=None, fg=None, bg=None):
        self.reset = reset

        if em is None:
            self.em = None
        elif isinstance(em, Emphasis):
            self.em = em
        elif isinstance(em, self.__class__):
            self.em = em.em
        else:
            raise TypeError('Invalid em: {}'.format(em))

        if fg is None:
            self.fg = None
        elif isinstance(fg, self.__class__):
            self.fg = fg.fg
        else:
            self.fg = color(fg)

        if bg is None:
            self.bg = None
        elif isinstance(bg, self.__class__):
            self.bg = bg.bg
        else:
            self.bg = color(bg)

    @property
    def seq(self):
        return _apply(self.em, self.fg, self.bg, reset=self.reset)

    def __repr__(self):
        return '{clsname}(reset={reset}, em={em}, fg={fg}, bg={bg})'.format(
                clsname=self.__class__.__name__,
                reset=repr(self.reset),
                em=repr(self.em),
                fg=repr(self.fg), bg=repr(self.bg))

    def __call__(self, *args):
        return _apply(self.em, self.fg, self.bg, *args, reset=self.reset)

    def __str__(self):
        return self.seq or '\033[m'

    def __or__(self, other):
        if isinstance(other, NoColor):
            return other
        elif isinstance(other, self.__class__):
            reset = other.reset
            em = other.em
            fg = other.fg
            bg = other.bg
        elif isinstance(other, Color):
            reset = False
            em = None
            fg = other
            bg = None
        elif isinstance(other, Emphasis):
            reset = False
            em = other
            fg = None
            bg = None
        else:
            raise TypeError('unsupported operand types for |: {} and {}'.format(
                type(self).__name__, type(other).__name__))

        if not reset:
            if em is None:
                em = self.em
            elif isinstance(self.em, Emphasis):
                em = em | self.em
            fg = fg or self.fg
            bg = bg or self.bg

        return self.__class__(em=em, fg=fg, bg=bg)

    def __truediv__(self, other):
        return self.__class__(fg=self.fg, bg=other.fg)

    def __invert__(self):
        return self.__class__(fg=self.bg, bg=self.fg)


@export
def paint(*, reset=False, em=None, fg=None, bg=None):
    return ColorCompound(reset=reset, em=em, fg=fg, bg=bg)


export('nocolor')
nocolor = NoColor()

named_colors = [
        (0, ('black',)),
        (1, ('maroon',)),
        (2, ('green',)),
        (3, ('olive',)),
        (4, ('navy',)),
        (5, ('purple',)),
        (6, ('teal',)),
        (7, ('silver',)),
        (8, ('gray', 'grey',)),
        (9, ('red',)),
        (10, ('lime',)),
        (11, ('yellow',)),
        (12, ('blue',)),
        (13, ('fuchsia', 'magenta',)),
        (14, ('aqua', 'cyan',)),
        (15, ('white',)),
        (135, ('murasaki',)),
        ((240,248,255), ('aliceblue',)),
        ((250,235,215), ('antiquewhite',)),
        ((127,255,212), ('aquamarine',)),
        ((240,255,255), ('azure',)),
        ((245,245,220), ('beige',)),
        ((255,228,196), ('bisque',)),
        ((255,235,205), ('blanchedalmond',)),
        ((138,43,226), ('blueviolet',)),
        ((165,42,42), ('brown',)),
        ((222,184,135), ('burlywood',)),
        ((95,158,160), ('cadetblue',)),
        ((127,255,0), ('chartreuse',)),
        ((210,105,30), ('chocolate',)),
        ((233,110,0), ('clementine',)),
        ((255,127,80), ('coral',)),
        ((100,149,237), ('cornflowerblue',)),
        ((255,248,220), ('cornsilk',)),
        ((220,20,60), ('crimson',)),
        ((0,0,139), ('darkblue',)),
        ((0,139,139), ('darkcyan',)),
        ((184,134,11), ('darkgoldenrod',)),
        ((169,169,169), ('darkgray',)),
        ((0,100,0), ('darkgreen',)),
        ((169,169,169), ('darkgrey',)),
        ((189,183,107), ('darkkhaki',)),
        ((139,0,139), ('darkmagenta',)),
        ((85,107,47), ('darkolivegreen',)),
        ((255,140,0), ('darkorange',)),
        ((153,50,204), ('darkorchid',)),
        ((139,0,0), ('darkred',)),
        ((233,150,122), ('darksalmon',)),
        ((143,188,143), ('darkseagreen',)),
        ((72,61,139), ('darkslateblue',)),
        ((47,79,79), ('darkslategray',)),
        ((47,79,79), ('darkslategrey',)),
        ((0,206,209), ('darkturquoise',)),
        ((148,0,211), ('darkviolet',)),
        ((255,20,147), ('deeppink',)),
        ((0,191,255), ('deepskyblue',)),
        ((105,105,105), ('dimgray',)),
        ((105,105,105), ('dimgrey',)),
        ((30,144,255), ('dodgerblue',)),
        ((178,34,34), ('firebrick',)),
        ((255,250,240), ('floralwhite',)),
        ((34,139,34), ('forestgreen',)),
        ((220,220,220), ('gainsboro',)),
        ((248,248,255), ('ghostwhite',)),
        ((255,215,0), ('gold',)),
        ((218,165,32), ('goldenrod',)),
        ((173,255,47), ('greenyellow',)),
        ((240,255,240), ('honeydew',)),
        ((255,105,180), ('hotpink',)),
        ((205,92,92), ('indianred',)),
        ((75,0,130), ('indigo',)),
        ((255,255,240), ('ivory',)),
        ((240,230,140), ('khaki',)),
        ((230,230,250), ('lavender',)),
        ((255,240,245), ('lavenderblush',)),
        ((124,252,0), ('lawngreen',)),
        ((255,250,205), ('lemonchiffon',)),
        ((173,216,230), ('lightblue',)),
        ((240,128,128), ('lightcoral',)),
        ((224,255,255), ('lightcyan',)),
        ((250,250,210), ('lightgoldenrodyellow',)),
        ((211,211,211), ('lightgray',)),
        ((144,238,144), ('lightgreen',)),
        ((211,211,211), ('lightgrey',)),
        ((255,182,193), ('lightpink',)),
        ((255,160,122), ('lightsalmon',)),
        ((32,178,170), ('lightseagreen',)),
        ((135,206,250), ('lightskyblue',)),
        ((119,136,153), ('lightslategray',)),
        ((119,136,153), ('lightslategrey',)),
        ((176,196,222), ('lightsteelblue',)),
        ((255,255,224), ('lightyellow',)),
        ((50,205,50), ('limegreen',)),
        ((250,240,230), ('linen',)),
        ((102,205,170), ('mediumaquamarine',)),
        ((0,0,205), ('mediumblue',)),
        ((186,85,211), ('mediumorchid',)),
        ((147,112,219), ('mediumpurple',)),
        ((60,179,113), ('mediumseagreen',)),
        ((123,104,238), ('mediumslateblue',)),
        ((0,250,154), ('mediumspringgreen',)),
        ((72,209,204), ('mediumturquoise',)),
        ((199,21,133), ('mediumvioletred',)),
        ((25,25,112), ('midnightblue',)),
        ((245,255,250), ('mintcream',)),
        ((255,228,225), ('mistyrose',)),
        ((255,228,181), ('moccasin',)),
        ((255,222,173), ('navajowhite',)),
        ((253,245,230), ('oldlace',)),
        ((107,142,35), ('olivedrab',)),
        ((255,165,0), ('orange',)),
        ((255,69,0), ('orangered',)),
        ((218,112,214), ('orchid',)),
        ((238,232,170), ('palegoldenrod',)),
        ((152,251,152), ('palegreen',)),
        ((175,238,238), ('paleturquoise',)),
        ((219,112,147), ('palevioletred',)),
        ((255,239,213), ('papayawhip',)),
        ((255,218,185), ('peachpuff',)),
        ((205,133,63), ('peru',)),
        ((255,192,203), ('pink',)),
        ((221,160,221), ('plum',)),
        ((176,224,230), ('powderblue',)),
        ((188,143,143), ('rosybrown',)),
        ((65,105,225), ('royalblue',)),
        ((139,69,19), ('saddlebrown',)),
        ((250,128,114), ('salmon',)),
        ((244,164,96), ('sandybrown',)),
        ((46,139,87), ('seagreen',)),
        ((255,245,238), ('seashell',)),
        ((160,82,45), ('sienna',)),
        ((135,206,235), ('skyblue',)),
        ((106,90,205), ('slateblue',)),
        ((112,128,144), ('slategray',)),
        ((112,128,144), ('slategrey',)),
        ((255,250,250), ('snow',)),
        ((0,255,127), ('springgreen',)),
        ((70,130,180), ('steelblue',)),
        ((210,180,140), ('tan',)),
        ((216,191,216), ('thistle',)),
        ((255,99,71), ('tomato',)),
        ((64,224,208), ('turquoise',)),
        ((238,130,238), ('violet',)),
        ((245,222,179), ('wheat',)),
        ((245,245,245), ('whitesmoke',)),
        ((154,205,50), ('yellowgreen',)),
]
export('names')
names = tuple(name for index, names in named_colors for name in names)
def _setup_named_colors():
    for index, names in named_colors:
        c = color(index) if isinstance(index, int) else color(*index)
        for name in names:
            globals()[name] = c
            export(name)
_setup_named_colors()
del _setup_named_colors


color_esc_seq_regex = re.compile('\033' + r'\[[\d;]*m')

@export
def decolor(s):
    return color_esc_seq_regex.sub('', s)


def _tokenize(seq):
    tokens = []
    matching = 0
    buf = ''
    for char in seq:
        if char == '\033':
            matching = 1
            buf = ''
            continue

        if matching == 1 and char == '[':
            matching = 2
            buf = ''
            continue

        elif matching == 2:
            if char not in '0123456789;m':
                buf = ''
                matching = 0
                continue

            if char == 'm':
                empty = True
                for token in buf.split(';'):
                    if token:
                        empty = False
                        tokens.append(int(token, 10))
                if empty:
                    tokens.append(0)
                buf = ''
                matching = 0
                continue

            else:
                buf += char

        else:
            matching = 0

    return tokens or None


def _parse(seq):
    attr = {}

    tokens = _tokenize(seq)
    if tokens is None:
        return ColorCompound()

    codes = []
    while tokens:
        if tokens[0] == 0:
            tokens.pop(0)
            attr = {'reset': True}
            continue

        if tokens[0] in (1, 2, 4, 5, 7, 8):
            attr['em'] = attr.get('em', Emphasis()) | Emphasis(tokens.pop(0))
            continue

        if (30 <= tokens[0] <= 37) or (40 <= tokens[0] <= 47):
            t = tokens.pop(0)
            attr['fg' if t < 40 else 'bg'] = Color8(t % 10)
            continue

        if tokens[0] not in (38, 48):
            tokens.pop(0)
            continue
        ground = tokens.pop(0)

        if not tokens or tokens[0] not in (2, 5):
            if tokens:
                tokens.pop(0)
            continue
        color_type = tokens.pop(0)

        if color_type == 5:
            if tokens:
                attr['fg' if ground < 40 else 'bg'] = Color256(tokens.pop(0))
            continue

        # color_type == 2
        if tokens and len(tokens) >= 3:
            attr['fg' if ground < 40 else 'bg'] = ColorRGB(*tokens[:3])
            tokens = tokens[3:]

    return ColorCompound(**attr)


@export
def gradient(A, B, N=None, reverse=False, clockwise=None):
    if not isinstance(A, Color) or not isinstance(B, Color):
        raise TypeError('Can only calculate gradient() on Color objects')

    if N is not None and not isinstance(N, int):
        raise TypeError('N must be a integer')

    if N is not None and N < 2:
        raise ValueError('N={} is too small'.format(N))

    ret = None
    if N == 2:
        ret = (A, B)

    elif isinstance(A, Color256) and isinstance(B, Color256):
        ret = gradient_color256(A, B, N=N)

    elif isinstance(A, ColorRGB) and isinstance(B, ColorRGB):
        ret = gradient_rgb(A, B, N=N)

    elif isinstance(A, ColorHSV) and isinstance(B, ColorHSV):
        ret = gradient_hsv(A, B, N=N, clockwise=clockwise)

    else:
        ret = (A, B)

    if reverse:
        return ret[::-1]
    else:
        return ret


def gradient_color256(A, B, N=None):
    if A.index in range(232, 256) and B.index in range(232, 256):
        return gradient_color256_grayscale_range(A, B, N)

    if A.index in range(16, 232) and B.index in range(16, 232):
        return gradient_color256_rgb_range(A, B, N)

    return (A, B)


def gradient_color256_grayscale_range(A, B, N=None):
    a, b = A.index, B.index
    direction = sgn(b - a)
    n = abs(b - a) + 1
    return tuple(Color256(c) for c in resample(interval(a, b), N or n))


def gradient_color256_rgb_range(A, B, N=None):
    def color_to_rgb6(p):
        c = int(p) - 16
        r = c // 36
        g = (c % 36) // 6
        b = c % 6
        return vector(r, g, b)

    def rgb6_to_color(rgb6):
        return Color256(rgb6[0] * 36 + rgb6[1] * 6 + rgb6[2] + 16)

    rgb_a = color_to_rgb6(A)
    rgb_b = color_to_rgb6(B)

    delta = rgb_b - rgb_a
    cont_step_count = max(abs(d) for d in delta)

    if N is None or N > cont_step_count:
        # N >= minimum contiguous path
        steps = []
        for n in range(cont_step_count):
            step = delta.map(sgn)
            steps.append(step)
            delta = delta.map(lambda x: x - sgn(x))

        ret = resample(list(itertools.accumulate([rgb_a] + steps)), N)

    else:
        # N is shorter than minimum contiguous path
        ret = zip(
                resample(interval(rgb_a[0], rgb_b[0]), N),
                resample(interval(rgb_a[1], rgb_b[1]), N),
                resample(interval(rgb_a[2], rgb_b[2]), N),
                )

    return tuple(rgb6_to_color(i) for i in ret)


def gradient_rgb(A, B, N):
    # Calculate gradient in RGB
    a = vector(A.rgb)
    b = vector(B.rgb)
    if N is None:
        import math
        dist_r = math.ceil(abs(a[0] - b[0]) // 40)
        dist_g = math.ceil(abs(a[1] - b[1]) // 40)
        dist_b = math.ceil(abs(a[2] - b[2]) // 40)
        N = max(dist_r, dist_g, dist_b)

    ret = [A]
    for t in (i / (N - 1) for i in range(1, N - 1)):
        ret.append(ColorRGB(*tuple(lerp(a, b, t))))
    ret.append(B)
    return tuple(ret)


def gradient_hsv(A, B, N, clockwise):
    # Calculate gradient in HSV
    a = vector(A.hsv)
    b = vector(B.hsv)

    if clockwise == True:
        b[0] += 360 if (a[0] > b[0]) else 0
    elif clockwise == False:
        a[0] += 360 if (a[0] < b[0]) else 0

    if N is None:
        import math
        dist_hue = math.ceil(abs(a[0] - b[0]) / 30)
        dist_sat = math.floor(abs(a[1] - b[1]) / 10)
        dist_val = math.floor(abs(a[2] - b[2]) / 10)
        N = max(dist_hue, dist_sat, dist_val)

    ret = [A]
    for t in (i / (N - 1) for i in range(1, N - 1)):
        c = lerp(a, b, t)
        ret.append(ColorHSV(*c))

    ret.append(B)

    return tuple(ret)
