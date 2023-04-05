#!/usr/bin/env python3

# Mandatory
#TODO: Generate OP list
#TODO: Prefer hardlink over symlink when path duplicated?
#TODO: Review symlink handling, should I treat a symlink to dir as file or dir?
#TODO: Refine parse error message, because vim's content disappears after exit
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
#TODO: Add -e/-E: always/not expand path for the first parsing
#TODO: Provide directives for adding new entries like {file} {dir}

__version__ = '0.0.0'


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import collections
import difflib
import functools
import itertools
import math
import os
import re
import readline
import shutil
import subprocess as sub
import sys
import tempfile
import types

from enum import Enum
from math import ceil, log10
from os.path import basename, join, exists, isdir, isfile, split, realpath, abspath, relpath
from unicodedata import east_asian_width


# =============================================================================
# Global option
# -----------------------------------------------------------------------------

opt_diff_style = None


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
red_bg = color_to('41')
green_bg = color_to('42')
yellow_bg = color_to('43')
blue_bg = color_to('44')
magenta_bg = color_to('45')
cyan_bg = color_to('46')
white_bg = color_to('47')
nocolor = lambda s: '\033[m' + s

RLB = red('[')
RRB = red(']')


decolor_regex = re.compile('\033' + r'\[[\d;]*m')
def decolor(s):
    return decolor_regex.sub('', s)


def print_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def debug(*args, **kwargs):
    print_stderr(magenta('[Debug]'), *args, **kwargs)


def info(*args, **kwargs):
    print(cyan('[Info]'), *args, **kwargs)


def error(*args, **kwargs):
    print_stderr(red('[Error]'), *args)


def str_width(s):
    return sum(1 + (east_asian_width(c) in 'WF') for c in decolor(s))


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

# -----------------------------------------------------------------------------
# Generalized Utilities
# =============================================================================


# =============================================================================
# Containers
# =============================================================================

class InvalidPitiError(Exception):
    pass

class DuplicatedPitiError(Exception):
    pass

class EmptyPathError(Exception):
    pass

class DuplicatedPathError(Exception):
    pass

class UnknownPathError(Exception):
    pass


class TrackingEntry:
    def __init__(self, piti, path):
        # PITI = Path Item Temporary Identifier
        # A temporary unique number that associated to a path for path operation

        self.piti = piti
        self.path = path
        self.errors = set()

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
        return realpath(path) in self.real_path_set

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return False

        return self.content == other.content

    def append_entry(self, piti, path):
        npath = normalize_path(path)

        if piti and piti not in self.piti_index:
            self.piti_index[piti] = npath

        if npath and realpath(npath) not in self.real_path_set:
            self.real_path_set.add(realpath(npath))
        elif self.ignore_duplicated_path:
            return

        self.content.append(TrackingEntry(piti, npath))

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

    def get_entry_by_piti(self, piti):
        for entry in self:
            if entry.piti == piti:
                return entry


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


def print_path_with_prompt(level ,color, prompt, path):
    level(color(prompt) + color('[') + path + color(']'))


@functools.lru_cache
def screen_width():
    return shutil.get_terminal_size().columns


DiffStyle = Enum('DiffStyle', ['aligned', 'compact'])


def pretty_diff_strings(a, b):
    diff_segments = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        diff_segments.append((tag, a[i1:i2], b[j1:j2]))

    tag_counter = collections.Counter(equal=0, delete=0, insert=0, replace=0)
    tag_counter.update(s[0] for s in diff_segments)
    tag_count = types.SimpleNamespace(**tag_counter)

    diff_style = opt_diff_style

    if diff_style is None:
        if tag_count.replace == 1 and tag_count.delete + tag_count.insert == 0:
            # Only one replace, the diff is simple
            diff_style = DiffStyle.compact

        elif tag_count.replace == 0 and tag_count.delete * tag_count.insert == 0:
            # Only deletes or only inserts, the diff is simple
            diff_style = DiffStyle.compact

        elif tag_count.replace == 0 and tag_count.delete == 1 and tag_count.insert == 1:
            # Only one delete and one insert, the diff is ... simple, probably
            diff_style = DiffStyle.compact

        else:
            # By default, aligned style is used
            diff_style = DiffStyle.aligned

            align_width = sum(max(str_width(s[1]), str_width(s[2])) for s in diff_segments)
            if align_width + str_width('[Info] Rename:[]') > screen_width():
                # The screen is not wide enough
                diff_style = DiffStyle.compact

    A = ''
    B = ''
    for tag, sa, sb in diff_segments:
        if tag == 'equal':
            A += sa
            B += sb

        elif tag == 'delete':
            if diff_style == DiffStyle.aligned:
                A += red_bg(sa)
                B += ' ' * str_width(sa)
            else:
                A += red_bg(sa)

        elif tag == 'insert':
            if diff_style == DiffStyle.aligned:
                A += ' ' * str_width(sb)
                B += green_bg(sb)
            else:
                B += green_bg(sb)

        elif tag == 'replace':
            if diff_style == DiffStyle.aligned:
                wa = str_width(sa)
                wb = str_width(sb)
                w = max(wa, wb)
                A += red_bg(sa) + (' ' * (w - wa))
                B += green_bg(sb) + (' ' * (w - wb))
            else:
                A += red_bg(sa)
                B += green_bg(sb)

    return (A, B)


def aggregate_changes(change_list_raw):
    # Collect and aggregate raw 'rename' operations into the following
    # high level change types:
    # - 'domino': sequential renaming
    # - 'rotate': 'domino', plus the final path is renamed to the first path
    #
    # Original 'rename' is a special case of 'domino' where N=2

    change_list_untrack = []
    change_list_track = []
    change_list_remove = []
    change_list_rename = []

    rename_chains = set()

    for change in change_list_raw:
        if change[0] == 'untrack':
            change_list_untrack.append(change)

        elif change[0] == 'track':
            change_list_track.append(change)

        elif change[0] == 'remove':
            change_list_remove.append(change)

        else:
            src = change[1]
            dst = change[2]
            rename_chains.add((src, dst))

    while rename_chains:
        this_chain = rename_chains.pop()

        add = None
        remove = None
        for other_chain in rename_chains:
            if realpath(this_chain[-1]) == realpath(other_chain[0]):
                remove = other_chain
                add = this_chain + other_chain[1:]
                break

            elif realpath(other_chain[-1]) == realpath(this_chain[0]):
                remove = other_chain
                add = other_chain + this_chain[1:]
                break

        if remove:
            rename_chains.remove(remove)

        if add:
            rename_chains.add(add)

        if add is None:
            if this_chain[0] == this_chain[-1]:
                rotate_chain = this_chain[:-1]
                pivot = rotate_chain.index(min(rotate_chain))
                rotate_chain = rotate_chain[pivot:] + rotate_chain[:pivot]
                change_list_rename.append(('rotate', *rotate_chain))

            else:
                change_list_rename.append(('domino', *this_chain))

    return (change_list_untrack +
            change_list_track +
            change_list_remove +
            sorted(change_list_rename, key=lambda x: x[1]))


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
    with tempfile.NamedTemporaryFile(prefix='vd', suffix='vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            if isinstance(inventory, Inventory):
                for piti, path in inventory:
                    if piti is None:
                        f.write(f'{path}\n')
                    else:
                        f.write(f'{piti}\t{path}\n')
            else:
                for line in inventory:
                    f.write(f'{line}\n')
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
        lines = []
        errors = []
        with open(tf.name, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip('\n')
                lines.append(line)

                if not line:
                    continue

                rec = RegexCache(line)

                if rec.match(r'^(#? *\d+)\t(.*)$'):
                    piti, path = rec.group(1), rec.group(2)
                    new_inventory.append_entry(piti, path)

                elif line.startswith('#'):
                    continue

                else:
                    new_inventory.append_entry(None, line)

    if errors:
        for e in errors:
            error(e)

        user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'], allow_empty_input=False)
        if user_confirm == 'edit':
            return (step_vim_edit_inventory, base, lines)

        if user_confirm == 'redo':
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

    # =========================================================================
    # Calculate inventory diff
    # -------------------------------------------------------------------------
    # 1. Construct bucket (indexed by piti)
    # 1.1 Put base inventory into bucket
    # 1.2 Put new inventory info bucket, do sanity check at the same time
    # 2. Construct change list from bucket
    # -------------------------------------------------------------------------

    # 1.1 Put base inventory into bucket
    bucket = {'new':[]}
    for opiti, opath in base:
        bucket[opiti] = opath

    # 1.2 Put new inventory info bucket, do sanity check at the same time

    # Collect duplicated pitis and paths beforehand
    npiti_list = [npiti for npiti,npath in new]
    dup_piti_set = set(
            npiti for npiti in npiti_list
            if npiti is not None and npiti_list.count(npiti) > 1)

    npath_list = [realpath(npath) for npiti,npath in new]
    dup_path_set = set(
            npath for npath in npath_list
            if npath_list.count(npath) > 1)

    for entry in new:
        npiti, npath = entry

        # not allow empty path
        if not npath:
            entry.errors.add(EmptyPathError)

        # not allow duplicated path
        if realpath(npath) in dup_path_set:
            entry.errors.add(DuplicatedPathError)

        # handle newly-tracked path
        if npiti is None:
            if not exists(npath):
                entry.errors.add(FileNotFoundError)
            else:
                bucket['new'].append(npath)
            continue

        if npiti is not None:
            untrack = npiti.startswith('#')
            npiti = npiti.lstrip('#')

        # not allow duplicated piti
        if npiti in dup_piti_set:
            entry.errors.add(DuplicatedPitiError)

        # not allow writing to existing path
        if exists(npath) and npath not in base:
            entry.errors.add(FileExistsError)

        # not allow invalid piti
        if npiti not in bucket.keys():
            entry.errors.add(InvalidPitiError)

        if entry.errors:
            continue

        opath = bucket[npiti]

        if untrack:
            bucket[npiti] = (opath, None)
        else:
            bucket[npiti] = (opath, npath)

    has_error = False
    if set(e for entry in new for e in entry.errors) & {InvalidPitiError, DuplicatedPitiError}:
        piti_error_left_padding = '   '
    else:
        piti_error_left_padding = ''

    for entry in new:
        if entry.errors:
            line = ''
            if entry.piti is not None:
                if entry.errors & {InvalidPitiError, DuplicatedPitiError}:
                    line += red('─►') + ' ' + red_bg(entry.piti)
                else:
                    line += piti_error_left_padding + entry.piti

                line += '  '

            if DuplicatedPathError in entry.errors:
                line += red_bg(entry.path) + red(' ◄─ Duplicated path')
            elif FileNotFoundError in entry.errors:
                line += red_bg(entry.path) + red(' ◄─ Unknown path')
            elif FileExistsError in entry.errors:
                line += yellow_bg(entry.path) + yellow(' ◄─ {} already exists'.format(
                    'Dir' if isdir(entry.path) else 'File'))
            elif EmptyPathError in entry.errors:
                line += red('() ◄─ Empty path')
            else:
                line += entry.path

            error(line)
            has_error = True

    if has_error:
        user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'], allow_empty_input=False)
        if user_confirm == 'edit':
            return (step_vim_edit_inventory, base, new)

        if user_confirm == 'redo':
            return (step_vim_edit_inventory, base, base)

        return (exit, 1)

    # 2. Construct change list from bucket
    # ('track', path)
    # ('untrack', path)
    # ('remove', path)
    # ('rename', src, dst)
    change_list_raw = []
    for piti in bucket:
        if piti == 'new':
            for path in bucket['new']:
                if path not in base:
                    change_list_raw.append(('track', path))

        elif isinstance(bucket[piti], str) and bucket[piti] not in bucket['new']:
            change_list_raw.append(('remove', bucket[piti]))

        elif isinstance(bucket[piti], tuple):
            opath = bucket[piti][0]
            npath = bucket[piti][1]

            if bucket[piti][1] is None:
                change_list_raw.append(('untrack', opath))

            elif realpath(opath) != realpath(npath):
                change_list_raw.append(('rename', opath, npath))

    return (step_print_change_list, base, new, change_list_raw)


def step_print_change_list(base, new, change_list_raw):
    # If base inventory and new inventory is exactly the same, exit
    if base == new:
        info('No change')
        return (exit, 0)

    change_list = aggregate_changes(change_list_raw)

    for change in change_list:
        if change[0] == 'untrack':
            print_path_with_prompt(info, cyan, 'Untrack:', change[1])

        elif change[0] == 'track':
            print_path_with_prompt(info, cyan, 'Track:', change[1])

        elif change[0] == 'remove':
            print_path_with_prompt(info, red, 'Remove:', change[1])

        elif change[0] == 'domino':
            if len(change) == 3:
                A, B = pretty_diff_strings(change[1], change[2])
                print_path_with_prompt(info, yellow, 'Rename:', A)
                print_path_with_prompt(info, yellow, '└─────►', B)

            else:
                for idx, path in enumerate(change[1:]):
                    print_path_with_prompt(info, yellow,
                            'Rename:' + ('┌─' if idx == 0 else '└►'),
                            path)

        elif change[0] == 'rotate':
            if len(change) == 3:
                print_path_with_prompt(info, yellow, 'Swap:┌►', change[1])
                print_path_with_prompt(info, yellow, 'Swap:└►', change[2])
            else:
                rotate_chain_len = len(change[1:])
                for idx, path in enumerate(change[1:]):
                    if idx == 0:
                        arrow = '┌►┌─'
                    elif idx == rotate_chain_len - 1:
                        arrow = '└───'
                    else:
                        arrow = '│ └►'

                    print_path_with_prompt(info, yellow, 'Rotate:' + arrow, path)

        else:
            info(change)

    # If all changes are 'track' and 'untrack', apply the change directly
    if all({c[0] in ('track', 'untrack') for c in change_list}):
        newnew = Inventory()
        for piti, path in new:
            if piti is None or not piti.startswith('#'):
                newnew.append(path)
        newnew.build_index()

        if not newnew:
            print('No targets to edit')
            return (exit, 0)

        return (step_vim_edit_inventory, newnew, newnew)

    user_confirm = prompt_confirm('Continue?', ['yes', 'edit', 'redo', 'quit'])
    if user_confirm == 'yes':
        return (step_calculate_op_list, base, new, change_list)

    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    elif user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    elif user_confirm == 'quit':
        return (exit, 1)

    error('Unexpected flow, exit')
    return (exit, 1)


def step_calculate_op_list(base, new, change_list_raw):
    for change in change_list_raw:
        if change[0] == 'remove':
            # dir: shutil.rmtree(path)
            # file: os.remove()
            print_path_with_prompt(info, red, '(dry)Removing:', change[1])

        elif change[0] == 'rename':
            # shutil.move(src, dst) (calls os.rename(src, dst) on same fs)
            print_path_with_prompt(info, yellow, '(dry)Renaming:', change[1])
            print_path_with_prompt(info, yellow, '(dry)========>', change[2])

        else:
            debug('(dry)', change)

    info('WIP, exit')


# =============================================================================
# Main function
# =============================================================================

def main():
    global opt_diff_style

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

    parser.add_argument('-d', '--diff-style', choices=DiffStyle,
            type=lambda x: DiffStyle[x],
            help='Specify diff style')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit, directories are expanded')

    options = parser.parse_args()

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('Both stdout and stderr must be tty')
        exit(1)

    opt_diff_style = options.diff_style

    # =========================================================================
    # Collect initial targets
    # -------------------------------------------------------------------------
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded
    # -------------------------------------------------------------------------

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
        if t:
            inventory.append(t, expand=e)

    has_error = False
    for _, path in inventory.content:
        if not exists(path):
            print_path_with_prompt(error, red, 'File does not exist:', path)
            has_error = True

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
