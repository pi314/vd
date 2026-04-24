import argparse
import shlex
import sys


from . import lib_subproc as subproc
from . import lib_colors as paints


def print_cmd(cmd):

    def color_token(arg):
        token = shlex.quote(arg)
        color = paints.nocolor
        if token.startswith(('"', "'")):
            color = paints.orange
        elif token.startswith('-'):
            color = paints.cyan

        return color(token)

    print(paints.magenta('$'), ' '.join(
        color_token(arg) for arg in cmd
        ))


def notify(title, lines):
    def quote(text):
        return '"' + text.replace('"', r'\"') + '"'

    args = ['display', 'notification', quote(r'\n'.join(lines))]

    if title is not None:
        args += ['with', 'title', quote(title)]

    cmd = ['osascript', '-e', ' '.join(args)]
    print_cmd(cmd)
    exit(subproc.run(cmd).returncode)


def main():
    prog = sys.argv[0]
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='ntfy', prog='ntfy')
    parser.add_argument('-t', '--title', help='Notification title')
    parser.add_argument('lines', nargs='*', help='Notification text', default=[])

    args = parser.parse_args(argv)

    notify(args.title, args.lines)
