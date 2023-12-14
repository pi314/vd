#!/usr/bin/env python3

""; hint_text = '''
" =============================================================================
" vd vimrc
" =============================================================================

" Turn off line number for not interfere with item key
set nonu

" Set a wide gap between item key and path
set tabstop=8

set listchars=tab:¦¦

" Rename item
nnoremap cc ^WC
nnoremap S ^WC

""'''.lstrip()

""; '''
finish
""; '''
print(hint_text.strip('""').strip())



# Mandatary
#TODO: Think in Path, define new Inventory interface

#TODO: Rethink about folder and files

# Vim related
#TODO: Polyglot this file as the default vimrc
#TODO: Respect LS_COLORS by utilizing bits in III


__version__ = '0.0.1'


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import datetime
import difflib
import inspect
import io
import os
import re
import readline
import shlex
import shutil
import subprocess as sub
import sys
import tempfile
import unicodedata

from pathlib import Path


# =============================================================================
# Global option
# -----------------------------------------------------------------------------

options = argparse.Namespace(
        debug=False,
        )

VD_VIMRC_PATH = expanduser(join('~/.config', 'vd', 'vd.vimrc'))


# =============================================================================
# Generalized Utilities
# -----------------------------------------------------------------------------

class RegexCache:
    def __init__(self, text):
        self.text = text
        self.m = None

    def match(self, regex):
        self.m = re.match(regex, self.text)
        return self.m

    def group(self, *args, **kwargs):
        return self.m.group(*args, **kwargs)

    def groups(self, *args, **kwargs):
        return self.m.groups(*args, **kwargs)


class paint:
    def __init__(self, color_code):
        self.escape_seq = f'\033[{color_code}m'

    def __call__(self, s):
        return f'{self.escape_seq}{s}\033[m'

    def __str__(self):
        return self.escape_seq

black = paint('38;2;22;22;29') # eigengrau, or brain gray
red = paint('31')
green = paint('32')
yellow = paint('33')
blue = paint('34')
magenta = paint('35')
cyan = paint('36')
white = paint('37')

red_bg = paint('41')
green_bg = paint('42')
yellow_bg = paint('30;43')
blue_bg = paint('44')
magenta_bg = paint('45')
cyan_bg = paint('46')
white_bg = paint('47')
nocolor = paint('')

orange = paint('38;2;160;90;0')
orange_bg = paint('30;48;2;160;90;0')

RLB = red('[')
RRB = red(']')


decolor_regex = re.compile('\033' + r'\[[\d;]*m')
def decolor(s):
    return decolor_regex.sub('', s)


def print_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def print_msg(tag, print_func, *args, **kwargs):
    with io.StringIO() as buffer:
        print(*args, file=buffer, end='', **kwargs)

        for line in buffer.getvalue().split('\n'):
            print_func(tag, line.rstrip('\n'))


def debug(*args, **kwargs):
    if options.debug:
        print_msg(magenta('[Debug]'), print_stderr, *args, **kwargs)


def info(*args, **kwargs):
    print_msg(cyan('[Info]'), print, *args, **kwargs)


def warning(*args, **kwargs):
    print_msg(yellow('[Warning]'), print_stderr, *args, **kwargs)


error_lines = []
def errorq(*args, **kwargs):
    error_lines.append((args, kwargs))


def errorflush():
    for a, ka in error_lines:
        print_msg(red('[Error]'), print_stderr, *a, **ka)

    error_lines.clear()


def error(*args, **kwargs):
    errorflush()
    print_msg(red('[Error]'), print_stderr, *args, **kwargs)


def has_error():
    return bool(error_lines)


def str_width(s):
    return sum(1 + (unicodedata.east_asian_width(c) in 'WF') for c in decolor(s))


def shrinkuser(path):
    homepath = expanduser('~').rstrip('/') + '/'
    if path.startswith(homepath):
        return join('~', path[len(homepath):])

    return path


def xxxxpath(path):
    # It's basically realpath() except it treats the trailing symlink as file

    # Empty path is specially treated, for avoiding realpath('') -> cwd
    if not path:
        return

    path = path.rstrip('/')
    if islink(path):
        head, tail = split(path)
        return join(realpath(head), tail)

    return realpath(path)


def parent_dir(path):
    return dirname(path.rstrip('/'))


def inode(path):
    # It's IMPORTANT to remove the trailing slash before calling lstat().
    # Because the trailing slash causes the last component to be resolved
    # unconditionally.

    path = path.rstrip('/')
    if exists(path):
        return os.stat(path, follow_symlinks=False).st_ino


def splitpath(path):
    return path.split('/')


def fsorted(iterable, key=None):
    def filename_as_key(name):
        def int_or_not(x):
            if x and x[0] in '1234567890':
                return int(x)
            return x
        return tuple(int_or_not(x) for x in re.split(r'([0-9]+)', name))

    if key is None:
        sort_key = filename_as_key
    else:
        sort_key = lambda x: filename_as_key(key(x))

    return sorted(iterable, key=sort_key)


class UserSelection:
    def __init__(self, options):
        self.options = dict()
        for o in options:
            self.options[o[0]] = o
            self.options[o] = o
            self.options[o.lower()] = o

        self.selected = None

    def select(self, o):
        o = o.lower()

        if o not in self.options:
            raise ValueError('Invalid option: ' + o)

        self.selected = self.options[o]

    def __eq__(self, other):
        other = other.lower()

        if other not in self.options:
            raise ValueError('Invalid option: ' + other)

        return self.selected == other

    def __str__(self):
        return self.selected


def prompt_confirm(prompt_text, options, allow_empty_input=True):
    options = [o.lower() for o in options]

    us = UserSelection(options)

    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    sys.stdin = open('/dev/tty')
    sys.stdout = open('/dev/tty', 'w')
    sys.stderr = open('/dev/tty', 'w')

    try:
        if allow_empty_input:
            options[0] = options[0][0].upper() + options[0][1:]

        while True:
            print(prompt_text + ' '
                    + '['
                    + ' / '.join('({}){}'.format(o[0], o[1:]) for o in options)
                    + ']', end=' ')

            try:
                i = input().strip()
                if (not i) and allow_empty_input:
                    i = options[0]

                us.select(i)
            except ValueError as e:
                continue

            break

    except KeyboardInterrupt:
        print(black('KeyboardInterrupt'))
        exit(1)

    except EOFError:
        print(black('^D'))
        if allow_empty_input:
            us.select(options[0])

    print()

    sys.stdin = stdin_backup
    sys.stdout = stdout_backup
    sys.stderr = stderr_backup

    return us


def gen_tmp_file_name(path, postfix='.vdtmp.'):
    now = datetime.datetime.now()
    tmp_file_name = '{orig_path}{postfix}{timestamp}[{getpid}]'.format(
            orig_path=path.lstrip('/'),
            postfix=postfix,
            timestamp=now.strftime('%Y-%m-%d.%H:%M:%S.%f'),
            getpid=os.getpid(),
            )
    return tmp_file_name


def FUNC_LINE():
    cf = inspect.currentframe()
    bf = cf.f_back
    return '[{}:{}]'.format(bf.f_code.co_name, bf.f_lineno)

# -----------------------------------------------------------------------------
# Generalized Utilities
# =============================================================================


# =============================================================================
# Containers
# -----------------------------------------------------------------------------

class InvalidIiiError(Exception):
    pass

class DuplicatedIiiError(Exception):
    pass

class ConflictedPathError(Exception):
    pass

class UnknownPathError(Exception):
    pass

class WTF(Exception):
    pass


class InventoryItem:
    def __init__(self, iii, path, is_untrack=False):
        # III = Inventory Item Identifier
        self.iii = iii
        self.path = Path(path)
        self.is_untrack = is_untrack

    @property
    def display_text(self):
        ...

    @property
    def realpath(self):
        ...

    @property
    def exists(self):
        ...

    @property
    def is_dir(self):
        ...

    @property
    def is_file(self):
        ...

    @property
    def is_link(self):
        ...


class Inventory:
    def __init__(self):
        ...

    def append(self):
        ...

    def freeze(self):
        ...


class VirtualOperation:
    def __hash__(self):
        return id(self)


class VirtualSingleTargetOperation:
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<{} [{}]>'.format(self.__class__.__name__, self.target)


class VirtualMultiTargetOperation:
    def __init__(self, targets):
        self.targets = targets

    def __repr__(self):
        return '<{} {}>'.format(
                self.__class__.__name__,
                ','.join('[' + t + ']' for t in self.targets))


class TrackOperation(VirtualSingleTargetOperation):
    pass


class ResolveLinkOperation(VirtualSingleTargetOperation):
    def __init__(self, target):
        super().__init__(target)
        self.resolve_to = os.readlink(target)


class UntrackOperation(VirtualSingleTargetOperation):
    pass


class DeleteOperation(VirtualSingleTargetOperation):
    pass


class RenameOperation(VirtualMultiTargetOperation):
    def __init__(self, src, dst):
        super().__init__((src, dst))

    @property
    def src(self):
        return self.targets[0]

    @property
    def dst(self):
        return self.targets[1]


class DominoRenameOperation(VirtualMultiTargetOperation):
    pass


class RotateRenameOperation(VirtualMultiTargetOperation):
    pass

# -----------------------------------------------------------------------------
# Containers
# =============================================================================

# =============================================================================
# Specialized Utilities
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Specialized Utilities
# =============================================================================


# =============================================================================
# "Step" functions
# -----------------------------------------------------------------------------
# Step functions have to return a tuple containing
# [0] the next step function to be invoked, and
# [1:] function arguments
#
# Some step functions have to relay arguments for the next step function,
# although they are not going to use it at all.
# -----------------------------------------------------------------------------

def step_vim_edit_inventory(base, inventory):
    debug(FUNC_LINE())
    return (exit, 0)


def step_calculate_inventory_diff(base, new):
    debug(FUNC_LINE())
    return (exit, 0)


def step_ask_fix_it(base, new):
    debug(FUNC_LINE())
    return (exit, 0)


def step_check_change_list(base, new, change_list_raw):
    debug(FUNC_LINE())
    return (exit, 0)


def step_confirm_change_list(base, new, change_list_raw):
    debug(FUNC_LINE())
    return (exit, 0)


def step_apply_change_list(base, new, change_list):
    debug(FUNC_LINE())
    return (exit, 0)


def step_expand_inventory(new):
    debug(FUNC_LINE())
    return (exit, 0)

# -----------------------------------------------------------------------------
# "Step" functions
# =============================================================================


# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='vd',
        description='\n'.join((
            '# ======================================= #',
            '# A modern Python implementation of vidir #',
            '# ======================================= #',
            )),
        epilog='\n'.join((
            'examples:',
            '  $ vd',
            '  $ vd -a',
            '  $ find . -type f | vd',
            )),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    parser.add_argument('-a', '--all', action='store_true',
            default=False,
            help='Include hidden paths')

    parser.add_argument('--vimrc', action='store_true',
            default=False,
            help='Edit or create the vimrc for vd')

    parser.add_argument('--debug', action='store_true',
            default=False,
            help='Print debug messages')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit. Directories are expanded')

    args = parser.parse_args()

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('Both stdout and stderr must be tty')
        exit(1)

    if args.vimrc:
        exit(open_vd_vimrc())

    options.debug = args.debug
    debug(options)

    # =========================================================================
    # Collect initial targets
    # -------------------------------------------------------------------------
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded
    # -------------------------------------------------------------------------

    targets = []

    for target in fsorted(args.targets):
        targets.append((target, True))

    if not sys.stdin.isatty():
        for line in sys.stdin:
            targets.append((line.rstrip('\n'), False))

    if not targets:
        targets.append(('.', True))

    # inventory = Inventory()
    # inventory.ignore_hidden = not args.all
    # inventory.ignore_duplicated_path = True

    # for t, e in targets:
    #     if t:
    #         inventory.append(t, expand=e)

    # has_error = False
    # for _, path in inventory.content:
    #     if not islink(path) and not exists(path):
    #         print_path_with_prompt(error, red, 'File does not exist:', path)
    #         has_error = True

    if has_error:
        exit(1)

    if not inventory:
        info('No targets to edit')
        exit(0)

    inventory.build_index()

    # =========================================================================
    # Main loop
    # -------------------------------------------------------------------------
    # 1. Construct the stage: inventory => (seq num, tab, ./file/path/)
    # 2. Invoke vim with current stage content
    # 3. Parse and get new stage content
    # 4. Compare new/old stage content and generate OP list
    # 5. Confirm with user
    # 5.q. if user say "q" (quit), quit
    # 5.e. if user say "e" (edit), invoke vim with new stage content
    # 5.r. if user say "r" (redo), invoke vim with old stage content
    # 5.y. if user say "y" (yes) or enter, apply the OP list
    # 5.*. keep asking until recognized option is selected or Ctrl-C is pressed
    # -------------------------------------------------------------------------

    def name(a):
        try:
            return a.__name__
        except AttributeError:
            return a

    prev_call = None
    next_call = (step_vim_edit_inventory, inventory, inventory)
    while next_call:
        func, *args = next_call
        try:
            next_call = func(*args)
            prev_call = (func, *args)
        except TypeError as e:
            errorq(e)
            errorq(f'prev_call.func = {name(prev_call[0])}')
            errorq(f'prev_call.args = (')
            for a in prev_call[1:]:
                errorq(f'    {repr(a)}')
            errorq(')')

            errorq()
            errorq(f'next_call.func = {name(next_call[0])}')
            errorq(f'next_call.args = (')
            for a in next_call[1:]:
                errorq(f'    {repr(a)}')
            errorq(')')
            errorflush()

            raise e


if __name__ == '__main__':
    exit(main())
