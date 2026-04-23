import argparse
import os
import sys

from os.path import basename, dirname

from . import __version__
from . import bin


def main():
    prog = basename(sys.argv[0])
    sys.argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='iroiro', prog='iroiro')
    parser.add_argument('-v', '--version', action='version', help='print version and exit', version=__version__)
    parser.add_argument('-w', '--which', '--where', action='version', help='print package path and exit', version=dirname(__file__))
    parser.add_argument('command', nargs='*', help='sub-command and args')
    args = parser.parse_args(sys.argv)

    arg_idx = None
    for idx, arg in enumerate(sys.argv):
        if arg != 'iroiro':
            arg_idx = idx
            break

    if arg_idx is None and len(sys.argv) > 2:
        print(
r'''
        ╭────────────────────────────────╮
        │       ╭──────────────────────╮ │
        │       │   ╭────────────────╮ │ │
        │       │   │ RecursionError │ │ │
        │       │   ╰─────┬──────────╯ │ │
        │       │    .__. ╯            │ │
        │       │ .(=('')              │ │
        │       │  ||-||               │ │
        │       ╰────┬─────────────────╯ │
        │       ,__, ╯                   │
        │    ___(..)                     │
        │  /(   (__)                     │
        │ ' ||--||                       │
        ╰────────┬───────────────────────╯
            (__) ╯
    _______/(..)
  /(       /(__)
 * | w----||
   ||     ||
''', file=sys.stderr)
        sys.exit(1)

    sys.argv = sys.argv[(arg_idx or 0):]

    if not sys.argv:
        for f in sorted(os.listdir(os.path.dirname(__file__))):
            if f.startswith('bin_') and f.endswith('.py'):
                m = os.path.splitext(f[4:])[0]
                print(m)
        sys.exit(1)

    subcmd = sys.argv[0]

    try:
        getattr(bin, subcmd).main()
    except (AttributeError, ModuleNotFoundError):
        print(f'Unknown subcommand: {subcmd}', file=sys.stderr)
        sys.exit(1)
