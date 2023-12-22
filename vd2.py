#!/usr/bin/env python3

""; vimrc = r'''
" =============================================================================
" vd vimrc
" =============================================================================

let g:vd_disable_default_options = get(g:, 'vd_disable_default_options', v:false)
let g:vd_disable_key_mappings = get(g:, 'vd_disable_key_mappings', v:false)
let g:vd_disable_syntax_highlight = get(g:, 'vd_disable_syntax_highlight', v:false)

" ============================================================================='''.strip()

""; r'''
if !g:vd_disable_default_options
    " Turn off line number for not interferring with item index
    set nonu

    set nowrap
endif

if !g:vd_disable_key_mappings
    " Rename item
    nnoremap cc ^WC
    nnoremap S ^WC
endif

if !g:vd_disable_syntax_highlight
    set listchars=tab:\\x20│\\x20

    highlight link vdComment Comment
    syntax region vdComment start=/^#/ end=/$/ oneline

    " Parse $LS_COLORS for basic highlighting
    for s:token in split($LS_COLORS, ':')
        let s:m = matchlist(s:token, '\v^(\w+)\=([0-9;]+)$')
        let [s:unused, s:glob, s:code; s:unused] = s:m
        let s:cterm = ''
        let s:ctermfg = ''
        let s:ctermbg = ''
        for s:code_token in split(s:code, ';')
            if s:code_token == '1' || s:code_token == '01'
                let s:cterm = 'cterm=bold'
            elseif s:code_token[0] == '3'
                let s:ctermfg = 'ctermfg='. s:code_token[1:]
            elseif s:code_token[0] == '4'
                let s:ctermbg = 'ctermbg='. s:code_token[1:]
            endif
        endfor
        let s:hlgroup = 'vdHlGroup_'. s:glob
        exec join(['highlight', s:hlgroup, s:cterm, s:ctermfg, s:ctermbg])
    endfor

    syntax region vdInventoryItem start=/^\v[+*@]?\d+/ end=/$/ oneline

    highlight link vdIII Number
    syntax match vdIII /\v(^\D?)@<=\zs(\d+)\ze\t/ containedin=VdInventoryItem

    syntax match vdHlGroup_di /\v(^\D?\d+1\t)@<=\zs(.+)\ze$/ containedin=VdInventoryItem
    syntax match vdHlGroup_ex /\v(^\D?\d+2\t)@<=\zs(.+)\ze$/ containedin=VdInventoryItem
    syntax match vdHlGroup_ln /\v(^\D?\d+3\t)@<=\zs(.+)\ze$/ containedin=VdInventoryItem
    syntax match vdHlGroup_pi /\v(^\D?\d+4\t)@<=\zs(.+)\ze$/ containedin=VdInventoryItem

    highlight vdHlGroup_err ctermfg=black ctermbg=red
    syntax match vdHlGroup_err /\v(^\D?\d+9\t)@<=\zs(.+)\ze$/ containedin=VdInventoryItem

endif

finish
'''


# Mandatary
#TODO: Rethink about folder and files

# Vim related
#TODO: Refine syntax highlighting,
#TODO: Check if iii.type enum definition could be reused for vim and python
#TODO: Put "widgets" on the top area, move cursor onto them and press to activate

# Enhancement
#TODO: Define some commands like ":sort by date" ?

__version__ = '0.0.1'


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import datetime
import difflib
import functools
import glob
import inspect
import io
import os
import os.path
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
# Global variables {
# -----------------------------------------------------------------------------

options = argparse.Namespace(
        debug=False,
        )

VD_VIMRC_PATH = Path.home() / '.config' / 'vd' / 'vd2.vimrc'

# -----------------------------------------------------------------------------
# Global variables }
# =============================================================================


# =============================================================================
# Generalized Utilities {
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


def FUNC_LINE():
    cf = inspect.currentframe()
    bf = cf.f_back
    return '[{}:{}]'.format(bf.f_code.co_name, bf.f_lineno)


def str_width(s):
    return sum(1 + (unicodedata.east_asian_width(c) in 'WF') for c in decolor(s))


def shrinkuser(path):
    homepath = os.path.expanduser('~').rstrip('/') + '/'
    if path.startswith(homepath):
        return join('~', path[len(homepath):])
    return path


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


def gen_unique_file_name(path, postfix='.vdtmp.'):
    now = datetime.datetime.now()
    tmp_file_name = '{orig_path}{postfix}{timestamp}[{getpid}]'.format(
            orig_path=path.lstrip('/'),
            postfix=postfix,
            timestamp=now.strftime('%Y-%m-%d.%H:%M:%S.%f'),
            getpid=os.getpid(),
            )
    return tmp_file_name

# -----------------------------------------------------------------------------
# Generalized Utilities }
# =============================================================================


# =============================================================================
# Containers {
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


class VDGlob:
    def __init__(self, text):
        self.txt = os.path.expanduser(text)

    def __repr__(self):
        return f'VDGlob({self.text})'

    @property
    def text(self):
        return shrinkuser(self.txt)

    def glob(self):
        ret = glob.glob(self.txt, recursive=True)
        return fsorted(ret)


class VDPath:
    def __init__(self, text):
        self.txt = text
        self.path = Path(text).expanduser()

    def __repr__(self):
        return f'VDPath({self.text})'

    @property
    def text(self):
        if not self.txt:
            return '.'

        ret = self.txt.rstrip('/')

        # Add postfix to display text
        if self.isdir and not self.txt.endswith('*'):
            ret += '/'

        return shrinkuser(ret)

    @property
    def inode(self):
        if self.exists:
            return self.path.stat(follow_symlinks=False).st_ino

    @property
    def realpath(self):
        if self.islink:
            return str(self.path.parent.resolve() / self.path.name)

        return str(self.path.resolve())

    @property
    def exists(self):
        return self.path.exists() or self.islink

    @property
    def isdir(self):
        return self.path.is_dir() and not self.islink

    @property
    def isfile(self):
        return self.path.is_file() and not self.islink

    @property
    def isfifo(self):
        return self.path.is_fifo() and not self.islink

    @property
    def isexecutable(self):
        return os.access(self.path, os.X_OK)

    @property
    def islink(self):
        return self.path.is_symlink()

    def listdir(self, include_hidden):
        if not self.exists:
            return []

        if not self.isdir:
            return ['.'] if not self.txt else [self.text]

        ret = []

        children = fsorted(p.name for p in self.path.iterdir())
        for child in children:
            if child.startswith('.') and not include_hidden:
                continue

            ret.append(child if not self.txt
                    else os.path.join(self.text, child)
                    )

        if not ret:
            ret = ['.'] if not self.txt else [self.text]

        return ret


class TrackingItem:
    def __init__(self, iii, text, mark=None):
        # III = Inventory Item Index
        self.iii = iii

        if not mark or not isinstance(mark, str) or mark not in '#*+@':
            self.mark = '.'
        else:
            self.mark = mark

        self.path = VDPath(text)

    def __repr__(self):
        return f'{self.mark.ljust(1)} {self.iii} [{self.text}]'

    def __getattr__(self, attr):
        if hasattr(self.path, attr):
            return getattr(self.path, attr)

    @property
    def type(self):
        if not self.exists:
            return 9
        if self.isdir:
            return 1
        if self.isfile and self.isexecutable:
            return 2
        if self.islink:
            return 3
        if self.isfifo:
            return 4
        return 0


class Inventory:
    def __init__(self):
        self.content = []

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index):
        return self.content[index]

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return False

        return self.content == other.content

    def append(self, text, iii=None, mark=None):
        if text is None:
            self.content.append(None)

        elif isinstance(text, TrackingItem):
            self.content.append(text)

        elif iii is not None:
            self.content.append(TrackingItem(iii, text, mark=mark))

        elif '*' in text.replace('[*]', '_'):
            self.content.append(VDGlob(text))

        else:
            self.content.append(VDPath(text.replace('[*]', '*')))

    def freeze(self):
        while self.content[0] is None:
            self.content.pop(0)
        while self.content[-1] is None:
            self.content.pop(-1)

        offset = 10 ** (len(str(len(self.content))))
        iii = 1
        for item in self.content:
            if not isinstance(item, TrackingItem):
                continue

            if item.iii is not None:
                continue

            item.iii = (offset + iii) * 10 + item.type
            iii += 1


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
# Containers }
# =============================================================================

# =============================================================================
# Specialized Utilities {
# -----------------------------------------------------------------------------

@functools.lru_cache
def screen_width():
    return shutil.get_terminal_size().columns


def hint_text():
    sepline = '# ' + ('=' * 77)

    ret = [
            sepline,
            '# - Add a path to track it.',
            '# - Sort the paths as you want.',
            "# - Add a '#' before the id to untrack an item.",
            "# - Add a '+' before the id to expand non-hidden items under the directory.",
            "# - Add a '*' before the id to expand all items under the directory.",
            "# - Add a '@' before the id to resolve the soft link.",
            ]

    if os.path.isfile(VD_VIMRC_PATH):
        ret.append('# - Configure hotkeys in ~/.config/vd/vd.vimrc')
    else:
        ret.append('# - Setup default vd.vimrc with')
        ret.append('#   $ vd --vimrc')

    ret.append(sepline)

    return ret

# -----------------------------------------------------------------------------
# Specialized Utilities }
# =============================================================================


# =============================================================================
# "Step" functions {
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

    with tempfile.NamedTemporaryFile(prefix='vd', suffix='.vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            def writeline(line=''):
                f.write(line + '\n')

            def writelines(lines=[]):
                for line in lines:
                    writeline(line)

            writelines(hint_text())
            writeline()

            if isinstance(inventory, Inventory):
                for item in inventory:
                    if item is None:
                        writeline()
                    elif item.iii is None:
                        writeline(f'{item.text}')
                    else:
                        writeline(f'{item.iii}\t{item.text}')
            else:
                for line in inventory:
                    writeline(f'{line}')
            f.flush()

        # Invoke vim to edit item list
        cmd = ['vim', tf.name]

        cmd.append('+set ft=vd')

        # Source vd.vimrc
        if os.path.isfile(VD_VIMRC_PATH):
            cmd += ['+source ' + str(VD_VIMRC_PATH)]

        cmd += ['+source ' + __file__]

        # Set proper tabstop for my (arguably) perfect vertical separation line
        if len(inventory):
            cmd.append('+set tabstop=' + str(len(str(inventory[0].iii)) + 4))

        # Move cursor to the line above first inventory item
        cmd.append('+normal }')

        sub.call(cmd, stdin=open('/dev/tty'))
        print()

        # Parse tempfile content
        new = Inventory()
        with open(tf.name, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip('\n')

                if not line:
                    new.append(None)
                    continue

                rec = RegexCache(line)

                if rec.match(r'^([#+*@]?) *(\d+)\t+(.*)$'):
                    mark, iii, text = rec.groups()
                    new.append(text, iii=iii, mark=mark)

                elif line.startswith('#'):
                    continue

                else:
                    new.append(line)

        new.freeze()

    return (step_calculate_inventory_diff, base, new)


def step_calculate_inventory_diff(base, new):
    debug(FUNC_LINE())
    for item in new:
        print(item)

        if isinstance(item, VDGlob):
            for i in item.glob():
                print(f'   [{i}]')

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


def edit_vd_vimrc():
    debug(FUNC_LINE())

    VD_VIMRC_PATH.parent.mkdir(parents=True, exist_ok=True)

    if VD_VIMRC_PATH.exists() and not VD_VIMRC_PATH.is_file():
        error(VD_VIMRC_PATH, 'exists and it\'s not a file')
        return 1

    print(VD_VIMRC_PATH)

    if VD_VIMRC_PATH.exists():
        return sub.call(['vim', VD_VIMRC_PATH])

    # Deploy vd vimrc if user didn't have one
    with tempfile.NamedTemporaryFile() as tf:
        # Write it to a temp file first
        with open(tf.name, mode='w', encoding='utf8') as f:
            f.write(vimrc)

        return sub.call([
            'vim', VD_VIMRC_PATH,
            '+0read ' + tf.name, # Read the content into buffer
            '+$d_', # Remove the extra empty line with the black hole register
            '+normal gg', # Move cursor back to 1st line
            ])


# -----------------------------------------------------------------------------
# "Step" functions }
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
        exit(edit_vd_vimrc())

    options.debug = args.debug
    debug(FUNC_LINE(), options)

    # =========================================================================
    # Collect initial targets
    # -------------------------------------------------------------------------
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded
    # -------------------------------------------------------------------------
    targets = []

    for target in fsorted(args.targets):
        for i in VDPath(target).listdir(args.all):
            targets.append(i)

    if not sys.stdin.isatty():
        for line in sys.stdin:
            targets.append(line.rstrip('\n'))

    if not targets:
        for i in VDPath('').listdir(args.all):
            targets.append(i)

    inventory = Inventory()

    for target in targets:
        if target:
            inventory.append(TrackingItem(None, target))

    has_error = False
    for item in inventory:
        if not item.exists:
            print_path_with_prompt(error, red, 'File does not exist:', item.text)
            has_error = True

    if has_error:
        exit(1)

    if not inventory:
        info('No targets to edit')
        exit(0)

    inventory.freeze()

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
