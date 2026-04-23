import os
import sys
import argparse
import shutil

from . import lib_tui as tui
from . import lib_colors as colors


def main():
    prog = sys.argv[0]
    argv = sys.argv[1:]

    term_size = shutil.get_terminal_size()

    parser = argparse.ArgumentParser(description='nowrap', prog='nowrap')
    parser.add_argument('-w', '--width', type=int, help='Width limit')

    args = parser.parse_args(argv)

    try:
        for line in sys.stdin:
            line = line.rstrip()
            a, b = tui.wrap(line, args.width or term_size.columns)
            print(a + str(colors.color(b)), flush=True)
    except BrokenPipeError:
        # https://docs.python.org/3/library/signal.html#note-on-sigpipe
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
