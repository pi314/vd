import sys
import shutil
import argparse
import textwrap
import re

from os.path import basename

from . import lib_colors

from .lib_colors import paint, color
from .lib_colors import Color, Color256, ColorRGB, ColorHSV
from .lib_regex import rere
from .lib_math import resample
from .lib_math import is_uint8
from .lib_math import lerp
from .lib_itertools import lookahead
from .lib_tui import getch


errors = []
def add_error(*errmsg):
    errors.append(errmsg)


def check_error():
    if not errors:
        return
    for error in errors:
        print(*error)
    sys.exit(1)


def parse_target(arg):
    if not isinstance(arg, str):
        return

    ret = None

    to = []
    while True:
        m = rere(arg)
        if m.fullmatch(r'^(.+)\.(rgb|RGB|hsv|HSV)$'):
            to.append(m.group(2))
            arg = m.group(1)
            continue
        break

    if arg in lib_colors.names:
        ret = getattr(lib_colors, arg)

    # #RRGGBB format
    elif m.fullmatch(r'#?([0-9a-fA-Z]{6})'):
        ret = color('#' + m.group(1))

    # #RRR,GGG,BBB format
    elif m.fullmatch(r'#([0-9]+),([0-9]+),([0-9]+)'):
        r, g, b = map(lambda x: int(x, 10), arg[1:].split(','))
        ret = ColorRGB(r, g, b)

    # @HHH,SSS,VVV format
    elif m.fullmatch(r'@([0-9]+),([0-9]+),([0-9]+)'):
        ret = ColorHSV(arg)

    # int
    elif m.fullmatch(r'[0-9]+'):
        try:
            i = int(arg, 10)
            if is_uint8(i):
                ret = color(i)
        except:
            ret = None

    tr_path = arg
    for t in to[::-1]:
        try:
            if t.lower() == 'rgb':
                ret = ret.to_rgb()
                tr_path += '.rgb'
            elif t.lower() == 'hsv':
                ret = ret.to_hsv()
                tr_path += '.hsv'
        except AttributeError:
            add_error('Error: Cannot transform color', tr_path, 'to', t)

    return ret


def spell_suggestions(word):
    import difflib
    return difflib.get_close_matches(word, lib_colors.names, cutoff=0)


def spell_suggestion_err_msg(word):
    if word is None:
        return

    err_msg = 'Unknown color name "{}"'.format(word)
    suggestions = spell_suggestions(word)[:3]
    if suggestions:
        err_msg += ', did you mean '
        if len(suggestions) == 1:
            err_msg += '"{}"'.format(suggestions[0])
        elif len(suggestions) == 2:
            err_msg += '"{}", or "{}"'.format(suggestions[0], suggestions[1])
        elif len(suggestions) == 3:
            err_msg += '"{}", "{}", or "{}"'.format(*suggestions)
        err_msg += '?'
    add_error(err_msg)


class Inventory:
    def __init__(self):
        self.data = []

    def __bool__(self):
        return bool(self.data)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, color):
        return self[color] is not None

    def __getitem__(self, idx):
        if isinstance(idx, Color):
            for item in self.data:
                if item[0] == idx:
                    return item
        else:
            return self.data[idx]

    def add(self, color, name=None):
        if not isinstance(name, (list, tuple)):
            namelist = [name]
        else:
            namelist = name

        item = self[color]
        if not item:
            item = self.append(color, name=name)
        for n in namelist:
            if n and n not in item[1]:
                item[1].append(n)
        return item

    def append(self, color, name=None):
        if not isinstance(name, (list, tuple)):
            namelist = [name]
        else:
            namelist = name

        item = (color, [])
        self.data.append(item)
        for n in namelist:
            if n and n not in item[1]:
                item[1].append(n)
        return item

    def sort(self, key):
        if key == 'index':
            key = 'i'
        elif key == 'name':
            key = 'n'
        elif key == 'hue':
            key = 'h'

        def sort_key(item):
            ret = []
            for ch in key:
                try:
                    if ch == 'n':
                        ret.append(item[1])
                    elif ch == 'i':
                        ret.append(item[0].index)
                    else:
                        ret.append(getattr(item[0], ch))
                except AttributeError:
                    if ch in 'rgbRGB':
                        ret.append(getattr(item[0].to_rgb(), ch))
                    if ch in 'hsvHSV':
                        ret.append(getattr(item[0].to_hsv(), ch))
                    if ch == 'i':
                        ret.append(int(item[0]) + (isinstance(item[0], ColorRGB) << 8))
            return ret

        self.data.sort(key=sort_key)

    def grep(self, keywords):
        import string

        if not keywords:
            return

        tmp = self.data
        self.data = []
        for clr, name_list in tmp:
            for name in name_list:
                for keyword in keywords:
                    if set(string.ascii_letters) - set(keyword):
                        if re.search(keyword, name):
                            self.add(clr, name)
                    else:
                        if keyword in name:
                            self.add(clr, name)


def expand_macro_all():
    ret = []
    for i in range(256):
        ret.append((parse_target(str(i)), []))

    for name in lib_colors.names:
        c = parse_target(name)
        if hasattr(c, 'index'):
            ret[c.index][1].append(name)
        else:
            ret.append((c, [name]))

    return ret


def expand_macro_named():
    ret = []
    for name in lib_colors.names:
        ret.append((parse_target(name), [name]))
    return ret


def main_256cube():
    # Print color cube palette
    print('Format: ESC[30;48;5;{}m')

    tiles = []
    for c in range(0, 256):
        bg = color(c).to_rgb()

        # Approximation of brightness
        if c < 16:
            brightness = 32 * (c != 0)
        elif c < 232:
            brightness = 0.21 * bg.R + 0.72 * bg.G + 0.07 * bg.B
        else:
            brightness = 32 * (c >= 238)

        # Too dark
        if brightness < 32:
            if brightness == 0:
                fg = 8
            elif max(bg.RGB) > 192:
                fg = 0
            else:
                fg = bg * 4
        else:
            fg = bg.to_rgb() // 4

        tiles.append(paint(fg=fg, bg=c)(' ' + str(c).rjust(3)))

    import shutil
    term_size = shutil.get_terminal_size()
    term_width = term_size.columns

    # Basic ANSI 8 color section
    for i in range(0, 16):
        print(tiles[i], end='')
        if (i + 1) % 8 == 0:
            print()
    print()

    cubes = [[], [], [], [], [], []]

    for i in range(16, 232):
        base = i - 16
        index_R = (base // 36)
        index_G = ((base % 36) // 6)
        index_B = (base % 6)
        cubes[index_G].append(tiles[i])

    # 6x6x6 color cube section
    for rows in range(1, 6):
        cols, remaining = divmod(len(cubes), rows)
        if remaining:
            continue
        if 4 * 6 * cols + 2 * (cols - 1) <= term_width:
            break

    for cube_row in range(rows):
        for tile_row in range(6):
            for cube_col in range(cols):
                cube_idx = cube_row * cols + cube_col
                if cube_col:
                    print('  ', end='')
                for tile_col in range(6):
                    print(cubes[cube_idx][tile_row * 6 + tile_col], end='')
            print()
        print()

    # Grayscale section
    tile_num = 256 - 232
    for lines in range(1, tile_num):
        cols, rem = divmod(tile_num, lines)
        if rem:
            continue
        if 4 * cols <= term_width:
            break

    for i in range(232, 256):
        print(tiles[i], end='')
        if (i - 232 + 1) % cols == 0:
            print()
    print()

    sys.exit()


def main():
    prog = basename(sys.argv[0])
    argv = sys.argv[1:]

    if not argv:
        main_256cube()

    colorful = ''.join(
            map(
                lambda x: color(x[0])(x[1]),
                zip(
                    ['#FF2222', '#FFC000', '#FFFF00',
                     '#C0FF00', '#00FF00', '#00FFC0',
                     '#00FFFF', '#00C0FF', '#3333FF', '#C000FF', '#FF00FF'],
                    'coooolorful'
                    )
                )
            )

    class YesNoToBoolOption(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values == 'yes')

    parser = argparse.ArgumentParser(
            prog=prog,
            description=('Query pre-defined colors from iroiro, ' +
                         'or produce ' + colorful + ' strips/tiles to fill the screen.'),
            epilog=textwrap.dedent('''
                    Example usages:
                    {s} {prog}
                    {s} {prog} all
                    {s} {prog} all --sort svh
                    {s} {prog} named --grep orange --hex
                    {s} {prog} FFD700 --rgb
                    {s} {prog} tile --lines=2 --cols=8 salmon white
                    {s} {prog} gradient AF5EFF 0000FF +7
                    '''.strip('\n')).format(prog=prog, s=lib_colors.murasaki('$')),
            allow_abbrev=False, add_help=False,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')

    parser.add_argument('--grep',
                        action='append',
                        help='''Filter colors that have name contains the specified sub-string
This argument can be specified multiple times for multiple keywords''')

    parser.add_argument('-a', '--aliases',
                        action='store_true',
                        help='Show aliases of specified colors')

    parser.add_argument('--hex', dest='val_fmt',
                        action='append_const', const='hex',
                        help='Show RGB value in hex number')

    parser.add_argument('--rgb', dest='val_fmt',
                        action='append_const', const='rgb',
                        help='Show RGB value in 3-tuple')

    parser.add_argument('--hsv', dest='val_fmt',
                        action='append_const', const='hsv',
                        help='Show HSV value in 3-tuple')

    def SortKey(arg):
        if arg in ('index', 'name', 'rgb', 'hsv', 'hue'):
            return arg
        if all(ch in 'rgbRGBhsvHSVni' for ch in arg):
            return arg
        raise ValueError(arg)
    parser.add_argument('--sort',
                        nargs='?', type=SortKey, const='index', metavar='key',
                        default='',
                        help='Sort output by the specified attribute.\nAvailable keys: index, name, rgb, hue, [rgbRGBhsvHSVni]')

    parser.add_argument('-r', '--reverse',
                        action='store_true',
                        help='Reverse output sequence')

    parser.add_argument('-m', '--merge',
                        action=YesNoToBoolOption, nargs='?', choices=['yes', 'no'], const='yes',
                        help='Merge colors that have same index')

    parser.add_argument('-M', '--no-merge',
                        action='store_false', dest='merge',
                        help='Dont merge colors that have same index')

    parser.add_argument('-c', '--clockwise',
                        action=YesNoToBoolOption, nargs='?', choices=['yes', 'no'], const='yes',
                        help='Calculate clockwise color gradient for HSV')

    parser.add_argument('--cols', '--columns',
                        type=int,
                        help='Specify terminal columns')

    parser.add_argument('--lines',
                        type=int,
                        help='Specify terminal lines')

    parser.add_argument('subcommand', nargs='?',
                        help='Sub-command: list, gradient, tile, hsv, help')

    parser.add_argument('targets', nargs='*', help='''Names / indexs / RGB hex values / HSV values to query
"all" and "named" macros could be used in "list" mode''')

    parser.set_defaults(val_fmt=[])

    args = parser.parse_intermixed_args(argv)

    if args.subcommand == 'help':
        parser.print_help()
        parser.exit(1)

    if args.subcommand in ('list', 'gradient', 'tile', 'hsv'):
        pass
    else:
        args.targets.insert(0, args.subcommand)

    if args.subcommand == 'tile':
        main_tile(args)
    elif args.subcommand == 'gradient':
        main_list(args, gradient=True)
    elif args.subcommand == 'hsv':
        main_hsv(args)
    else:
        main_list(args)


def main_list(args, gradient=False):
    inventory = Inventory()

    if gradient:
        # argument handling for gradient
        if not args.targets:
            add_error('No colors to gradient')

        check_error()

        targets = list(args.targets)
        path = []
        while targets:
            if not path:
                path.append([None, None, None])

            arg = targets.pop(0)
            if re.fullmatch(r'\+\d+', arg):
                path[-1][2] = arg

            elif path[-1][0] is None:
                path[-1][0] = arg

            elif path[-1][1] is None:
                path[-1][1] = arg

            else:
                path.append([path[-1][1], arg, None])

        def color_text(this_color):
            if isinstance(this_color, Color256):
                return str(this_color.index)
            elif isinstance(this_color, ColorRGB):
                return '{:#X}'.format(this_color)
            elif isinstance(this_color, ColorHSV):
                return '@{:},{:},{:}'.format(this_color.H, this_color.S, this_color.V)
            else:
                line.append('(?)')

        src = parse_target(path[0][0])
        if not src:
            spell_suggestion_err_msg(src)

        expanded = [(src, color_text(src))]
        for step in path:
            src = step[0]
            dst = step[1]

            if dst is None:
                continue

            arg_n = step[2]

            src = parse_target(src)
            if not src:
                spell_suggestion_err_msg(src)

            dst = parse_target(dst)
            if not dst:
                spell_suggestion_err_msg(dst)

            try:
                n = int(arg_n, 10) if arg_n else None
            except:
                add_error('Invalid number: {}'.format(arg_n))

            check_error()

            expanded += [(g, color_text(g)) for g in lib_colors.gradient(src, dst, n, clockwise=args.clockwise)[1:]]

    else:
        # argument handling for list
        if args.merge is None:
            if 'all' in args.targets or 'named' in args.targets:
                args.merge = True
            elif args.aliases:
                args.merge = True
            else:
                args.merge = False

        if args.grep and not args.targets:
            args.targets = ['all']

        for arg in args.targets:
            if arg in ('all', 'named'):
                if arg == 'all':
                    for i in expand_macro_all():
                        inventory.add(i[0], i[1])

                elif arg == 'named':
                    for i in expand_macro_named():
                        inventory.add(i[0], i[1])
                continue

            t = parse_target(arg)
            if t:
                inventory.append(t, arg)
            else:
                spell_suggestion_err_msg(arg)

        check_error()

    try:
        inventory.grep(args.grep)
    except re.PatternError as e:
        add_error('Invalid pattern:', e.pattern)
        check_error()

    inventory.sort(args.sort)

    if not inventory:
        print('No colors to show')
        sys.exit(1)

    unmentioned_names = set(lib_colors.names)
    for _, names in inventory:
        unmentioned_names -= set(names)

    aliases = [[] for i in range(256)]
    if args.aliases:
        for name in unmentioned_names:
            c = getattr(lib_colors, name).index
            aliases[c].append(name)

    for this_color, names in inventory[::(-1 if args.reverse else 1)]:
        line = []
        rgb = this_color if isinstance(this_color, ColorRGB) else this_color.to_rgb()
        hsv = rgb.to_hsv()

        if isinstance(this_color, Color256):
            line.append('{:>3}'.format(this_color.index))
        elif isinstance(this_color, ColorRGB):
            line.append('(#)')
        elif isinstance(this_color, ColorHSV):
            line.append('(@)')
        else:
            line.append('(?)')

        for val_fmt in args.val_fmt:
            if val_fmt == 'rgb':
                line.append('({:>3}, {:>3}, {:>3})'.format(*rgb.RGB))

            elif val_fmt == 'hex':
                line.append('{:#X}'.format(rgb))

            elif val_fmt == 'hsv':
                line.append('(@{:>3}, {:>3}%, {:>3}%)'.format(*hsv.HSV))

        line.append(paint(fg=this_color, bg=this_color)('iroiro'))

        line.append(', '.join(names))

        if isinstance(this_color, Color256):
            aliases[this_color.index] = [name
                                         for name in aliases[this_color.index]
                                         if name not in names]
            if aliases[this_color.index]:
                a = ('(' + ', '.join(aliases[this_color.index]) + ')')
                line[-1] = line[-1] + (' ' if line[-1] else '') + a
                aliases[this_color.index] = []

        print(' '.join(line))


def main_hsv(args):
    term_size = shutil.get_terminal_size()
    term_width = term_size.columns
    term_height = term_size.lines

    height = 5
    if term_width < 80:
        width = term_width
    elif term_width > 120:
        width = 120
    else:
        width = 80

    for y in range(0, height):
        line = ''
        for x in range(width):
            hue = (x / width) * 360
            sat = 100
            val1 = lerp(30, 100, (y * 2) / (height*2))
            val2 = lerp(30, 100, (y * 2 + 1) / (height*2))

            bgcolor = ColorHSV(hue, sat, val1)
            color = ColorHSV(hue, sat, val2)
            line += (color / bgcolor)('▄')

        print(line)


def main_tile(args):
    if not args.targets:
        add_error('No colors to tile')

    check_error()

    tiles = [[]]
    for arg in args.targets:
        for token in arg.split('/'):
            if token in ('all', 'named'):
                add_error('"{}" cannot be used in tile mode'.format(token))
                continue

            t = parse_target(token)
            if not t:
                spell_suggestion_err_msg(token)
                continue

            tiles[-1].append((token, t))

        tiles.append([])

    check_error()

    if not tiles[-1]:
        tiles.pop()

    cols, lines = shutil.get_terminal_size()
    cols = args.cols or cols
    lines = args.lines or lines

    if lines < 0:
        lines = len(tiles)

    for idx, is_last in lookahead(resample(range(len(tiles)), lines)):
        colors = tiles[idx]
        widths = []
        quo, rem = divmod(cols, len(colors))
        widths = [quo + (i < rem) for i, elem in enumerate(colors)]
        line = ''
        for idx, textcolor in enumerate(colors):
            text, c = textcolor
            text = text[:widths[idx]]
            line += paint(fg=c, bg=c)(text) + (~c)(' ' * (widths[idx] - len(text)))

        if args.lines:
            print(line)
        else:
            print(line, end='\n' if not is_last else '')

    sys.stdout.flush()

    getch()
