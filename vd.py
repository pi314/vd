#!/usr/bin/env python3

# Mandatory
#TODO: Generate OP list
#TODO: Refine parse error message, because vim's content disappears after exit
#TODO: Comment ('#') a line to untrack an entry
#TODO: Expand dir, '*' for all and '+' for non-hidden entries
#TODO: -r/--recursive, with depth limit?
#TODO: Refine error() API, centralize common handling

# Vim related
#TODO: nnoremap J/K to move entry downward/upward
#TODO: vim default config
#TODO: Print some hints on top
#TODO: Respect LS_COLORS by utilizing bits in PITI

# Enhancement
#TODO: If user change dir to .tar (by removing the trailing slash), tar it
#TODO: Add -e/-E: always expand/not expand path for the first parsing
#TODO: Provide directives for adding new entries like {file} {dir}

__version__ = '0.0.0'


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import collections
import difflib
import itertools
import math
import os
import re
import subprocess as sub
import sys
import tempfile

from os.path import basename, join, exists, isdir, isfile, split, realpath, abspath, relpath
from math import ceil, log10


# =============================================================================
# Generalized Utilities
# =============================================================================

class RegexCache:
    def __init__(self, text):
        self.text = text
        self.m = None

    def match(self, regex):
        self.m = re.match(regex, self.text)
        return self.m

    def group(self, *args, **kwargs):
        return self.m.group(*args, **kwargs)


def color_to(color_code):
    def colorer(s):
        if not s:
            return ''
        return f'\033[{color_code}m{s}\033[m'
    return colorer

black = color_to('38;2;22;22;29') # eigengrau, or brain gray
red = color_to(31)
green = color_to(32)
yellow = color_to(33)
blue = color_to(34)
magenta = color_to(35)
cyan = color_to(36)
white = color_to(37)
nocolor = lambda s: '\033[m' + s

RLB = red('[')
RRB = red(']')


def print_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def debug(*args, **kwargs):
    print_stderr(magenta('[Debug]'), *args, **kwargs)


def info(*args, **kwargs):
    print(cyan('[Info]'), *args, **kwargs)


def error(*args, **kwargs):
    print_stderr(red('[Error]'), *args)


# =============================================================================
# Containers
# =============================================================================

class EmptyPathError(Exception):
    pass


class TrackingEntry:
    def __init__(self, piti, path, linenum=None):
        # PITI = Path Item Temporary Identifier
        # A temporary unique number that associated to a path for path operation

        self.piti = piti
        self.path = path
        self.linenum = linenum

    def __eq__(self, other):
        return (self.piti, self.path) == (other.piti, other.path)

    def __iter__(self):
        return iter((self.piti, self.path))

    def __getitem__(self, index):
        return (self.piti, self.path)[index]

    def __repr__(self):
        return repr((self.piti, self.path))



class Inventory:
    def __init__(self):
        self.ignore_hidden = False
        self.ignore_duplicated_path = False
        self.content = []
        self.real_path_set = set()
        self.piti_index = dict()

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index):
        return self.content[index]

    def __contains__(self, path):
        return path in self.real_path_set

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return False

        return self.content == other.content

    def append_entry(self, piti, path, linenum=None):
        if not path:
            raise EmptyPathError()

        npath = normalize_path(path)

        if piti and piti not in self.piti_index:
            self.piti_index[piti] = npath

        if realpath(npath) not in self.real_path_set:
            self.real_path_set.add(realpath(npath))
        elif self.ignore_duplicated_path:
            return

        self.content.append(TrackingEntry(piti, npath, linenum=linenum))

    def _append_path(self, path):
        if basename(path).startswith('.') and self.ignore_hidden:
            return

        self.append_entry(None, path)

    def append(self, path, expand=False, keep_empty_dir=False):
        if not expand:
            self._append_path(path)

        elif not exists(path):
            self._append_path(path)

        elif isdir(path):
            ls = sorted(os.listdir(path))
            for i in ls:
                self._append_path(join(path, i))

            if not ls and keep_empty_dir:
                self._append_path(path)

        elif isfile(path):
            self._append_path(path)

        else:
            raise Exception('Unknown file type')

    def build_index(self):
        if not self.content:
            return

        self.piti_index = dict()

        start = 10 ** (len(str(len(self.content)))) + 1
        for i, entry in enumerate(self.content, start=start):
            entry.piti = str(i)
            self.piti_index[entry.piti] = entry.path

    def get_path_by_piti(self, piti):
        return self.piti_index.get(piti)


# =============================================================================
# Specialized Utilities
# =============================================================================

def normalize_path(path):
    if not path:
        return ''

    npath = ''
    if not path.startswith(('/', './', '../')) and path not in ('.', '..'):
        npath = './'

    npath += path

    if not path.endswith('/') and isdir(npath):
        npath += '/'

    return npath


class UserSelection:
    def __init__(self, options):
        self.options = dict()
        self.options[''] = options[0]
        for o in options:
            self.options[o[0]] = o
            self.options[o[0].upper()] = o
            self.options[o] = o
            self.options[o.upper()] = o

        self.selected = None

    def select(self, o):
        if o not in self.options:
            raise ValueError('Invalid option: ' + o)

        self.selected = self.options[o]

    def __eq__(self, other):
        if other not in self.options:
            raise ValueError('Invalid option: ' + other)

        return self.selected == other

    def __str__(self):
        return self.selected


def prompt_confirm(prompt_text, options):
    options = [o.lower() for o in options]

    us = UserSelection(options)

    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    sys.stdin = open('/dev/tty')
    sys.stdout = open('/dev/tty', 'w')
    sys.stderr = open('/dev/tty', 'w')

    try:
        options[0] = options[0][0].upper() + options[0][1:]

        while True:
            print(prompt_text + ' '
                    + '['
                    + ' / '.join('({}){}'.format(o[0], o[1:]) for o in options)
                    + ']', end=' ')

            try:
                us.select(input().strip())
            except ValueError as e:
                continue

            break

    except KeyboardInterrupt:
        print(black('KeyboardInterrupt'))
        exit(1)

    except EOFError:
        us.select('')

    sys.stdin = stdin_backup
    sys.stdout = stdout_backup
    sys.stderr = stderr_backup

    return us


# =============================================================================
# Chaos
# =============================================================================

def pretty_print_operand(level ,color, prompt, path):
    level(color(prompt) + color('[') + path + color(']'))


def apply_op_list(op_list):
    for op in op_list:
        if op[0] == 'remove':
            # dir: shutil.rmtree()
            # file: os.remove()
            pretty_print_operand(info, red, '(dry)Removing:', red(op[1] + ('/' if isdir(op[1]) else '')))

        elif op[0] == 'rename':
            # shutil.move()
            pretty_print_operand(info, yellow, '(dry)Renaming:', yellow(op[1]))
            pretty_print_operand(info, yellow, '(dry)========>', yellow(op[2]))

        else:
            debug('(dry)', op)


# =============================================================================
# "Step" functions
# -----------------------------------------------------------------------------
# Step functions have to return a tuple containing
# [0] the next step function to be invoked, and
# [1:] function arguments
#
# Some step functions have to relay arguments for the next step function,
# although they are not going to use it at all.
# =============================================================================

def step_vim_edit_inventory(base, inventory):
    with tempfile.NamedTemporaryFile(prefix='vd', suffix='vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            for piti, path in inventory.content:
                f.write(f'{piti}\t{path}\n')
            f.flush()

        # Invoke vim to edit item list
        sub.call([
            'vim', tf.name,
            '+set nonu',
            '+set syntax=python',
            '+set tabstop=8'],
            stdin=open('/dev/tty')
            )

        # Parse tempfile content
        new_inventory = Inventory()
        errors = []
        with open(tf.name, mode='r', encoding='utf8') as f:
            for linenum, line in enumerate(f, start=1):
                line = line.rstrip('\n')

                if not line:
                    continue

                rec = RegexCache(line)

                if rec.match(r'^(#?\d+)\t(.*)$'):
                    piti, path = rec.group(1), rec.group(2)

                    try:
                        new_inventory.append_entry(piti, path, linenum=linenum)
                    except EmptyPathError:
                        errors.append(f'Line {linenum}: file path cannot be empty')

                elif exists(line):
                    new_inventory.append_entry(None, line, linenum=linenum)

                else:
                    errors.append(f'Line {linenum}: parsing error: {RLB}{line}{RRB}')

        if errors:
            for e in errors:
                error(e)

            user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'])
            if user_confirm == 'edit':
                print()
                return (step_vim_edit_inventory, base, new_inventory)

            if user_confirm == 'redo':
                print()
                return (step_vim_edit_inventory, base, base)

            return (exit, 1)

    return (step_calculate_inventory_diff, base, new_inventory)


def step_calculate_inventory_diff(base, new):
    debug(magenta('==== inventory (base) ===='))
    for opiti, opath in base.content:
        debug((opiti, opath))
    debug('-------------------------')
    for npiti, npath in new.content:
        debug((npiti, npath))
    debug(magenta('==== inventory (new) ===='))

    piti_set = set()
    path_set = set()
    errors = []
    # Sanity check of new inventory content
    for npiti, path in new.content:
        if (npiti is None) or (npiti.startswith('#')):
            continue

        # Check duplicated piti
        if npiti in piti_set:
            errors.append(f'Duplicated key: {npiti}')
        else:
            piti_set.add(npiti)

        # Check duplicated path
        if realpath(path) in path_set:
            errors.append(f'Duplicated path: {RLB}{path}{RRB}')
            if realpath(path) != abspath(path):
                nrrp = normalize_path(relpath(realpath(path)))
                errors.append(f'Real path ====== {RLB}{nrrp}{RRB}')
        else:
            path_set.add(realpath(path))

        # Check inexisting piti
        if base.get_path_by_piti(npiti) is None:
            errors.append(f'Invalid key: {npiti}')

    if errors:
        for e in errors:
            error(e)

        user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'])
        if user_confirm == 'edit':
            print()
            return (step_vim_edit_inventory, base, new)

        if user_confirm == 'redo':
            print()
            return (step_vim_edit_inventory, base, base)

        return (exit, 1)

    op_list = []

    for opiti, opath in base.content:
        npath = new.get_path_by_piti(opiti)
        if not npath:
            if new.get_path_by_piti('#' + opiti) is None:
                op_list.append(('remove', opath))

        elif realpath(opath) != realpath(npath):
            op_list.append(('rename', opath, npath))

    return (step_print_op_list, base, new, op_list)


def step_print_op_list(base, new, op_list):
    if base == new:
        info('No change')
        return (exit, 0)

    if not op_list:
        newnew = Inventory()
        for piti, path in new:
            if piti is None or not piti.startswith('#'):
                newnew.append(path)
        newnew.build_index()

        if not newnew:
            print('No targets to edit')
            return (exit, 1)

        return (step_vim_edit_inventory, newnew, newnew)

    for op in op_list:
        if op[0] == 'remove':
            p = op[1]
            pretty_print_operand(info, red, 'Remove:', p)

        elif op[0] == 'rename':
            a, b = (op[1], op[2])
            s = difflib.SequenceMatcher(None, a, b)
            A, B = ('', '')
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'equal':
                    A += a[i1:i2]
                    B += b[j1:j2]

                elif tag == 'delete':
                    A += red(a[i1:i2])

                elif tag == 'insert':
                    B += green(b[j1:j2])

                elif tag == 'replace':
                    A += yellow(a[i1:i2])
                    B += yellow(b[j1:j2])

            pretty_print_operand(info, yellow, 'Rename:', A)
            pretty_print_operand(info, yellow, '======>', B)

        else:
            info(op)

    user_confirm = prompt_confirm('Continue?', ['yes', 'edit', 'redo', 'quit'])
    if user_confirm == 'quit':
        return (exit, 1)

    if user_confirm == 'edit':
        print()
        return (step_vim_edit_inventory, base, new)

    elif user_confirm == 'redo':
        print()
        return (step_vim_edit_inventory, base, base)

    info('WIP, exit')


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

    parser.add_argument('-n', '--dry-run', action='store_true',
            default=False,
            help='Perform a trial run with no changes made')

    parser.add_argument('-a', '--all', action='store_true',
            default=False,
            help='Include hidden paths')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit, directories are expanded')

    options = parser.parse_args()

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('Both stdout and stderr must be tty')
        exit(1)

    # =========================================================================
    # Collect initial targets
    # =========================================================================
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded

    targets = []

    for target in options.targets:
        targets.append((target, True))

    if not sys.stdin.isatty():
        for line in sys.stdin:
            targets.append((line.rstrip('\n'), False))

    if not targets:
        targets.append(('.', True))

    inventory = Inventory()
    inventory.ignore_hidden = not options.all
    inventory.ignore_duplicated_path = True

    for t, e in targets:
        inventory.append(t, expand=e)

    has_error = False
    for _, path in inventory.content:
        if not exists(path):
            pretty_print_operand(error, red, 'File does not exist:', path)
            has_error = True

    if has_error:
        exit(1)

    if not inventory:
        info('No targets to edit')
        exit(0)

    inventory.build_index()

    # =========================================================================
    # Main loop
    # =========================================================================
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
            error(e)
            error(f'prev_call.func = {name(prev_call[0])}')
            error(f'prev_call.args = (')
            for a in prev_call[1:]:
                error(f'    {repr(a)}')
            error(')')

            error()
            error(f'next_call.func = {name(next_call[0])}')
            error(f'next_call.args = (')
            for a in next_call[1:]:
                error(f'    {repr(a)}')
            error(')')

            raise e


if __name__ == '__main__':
    exit(main())
