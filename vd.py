#!/usr/bin/env python3

# Mandatory
#TODO: Refine simple diff
#TODO: meta_change_list
#TODO: Expand symlink with '@'
#TODO: Cancel out untrack + track
#TODO: -r/--recursive, with depth limit?
#TODO: Refine error() API, centralize common handling

# Vim related
#TODO: nnoremap J/K to move item downward/upward
#TODO: vim default config
#TODO: Print some hints on top
#TODO: Respect LS_COLORS by utilizing bits in PITI

# Enhancement
#TODO: If user change dir trailing slash to .tar, tar it
#TODO: Add -e/-E: always/not expand path for the first parsing
#TODO: Provide directives for adding new entries like {file} {dir}
#====> let user use :r !find themself?

__version__ = '0.0.0'


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import collections
import datetime
import difflib
import functools
import io
import itertools
import math
import os
import re
import readline
import shlex
import shutil
import subprocess as sub
import sys
import tempfile
import types

from enum import Enum
from math import ceil, log10
from os.path import split, dirname, basename, join, exists, isdir, islink, abspath, realpath, expanduser
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

    def groups(self, *args, **kwargs):
        return self.m.groups(*args, **kwargs)


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


def print_msg(tag, print_func, *args, **kwargs):
    with io.StringIO() as buffer:
        print(*args, file=buffer, end='', **kwargs)

        for line in buffer.getvalue().split('\n'):
            print_func(tag, line.rstrip('\n'))


def debug(*args, **kwargs):
    print_msg(magenta('[Debug]'), print_stderr, *args, **kwargs)


def info(*args, **kwargs):
    print_msg(cyan('[Info]'), print, *args, **kwargs)


def warning(*args, **kwargs):
    print_msg(yellow('[Warning]'), print_stderr, *args, **kwargs)


def error(*args, **kwargs):
    print_msg(red('[Error]'), print_stderr, *args, **kwargs)


def str_width(s):
    return sum(1 + (east_asian_width(c) in 'WF') for c in decolor(s))


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
    if exists(path):
        return os.stat(path, follow_symlinks=False).st_ino


def splitpath(path):
    return path.split('/')


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

# -----------------------------------------------------------------------------
# Generalized Utilities
# =============================================================================


# =============================================================================
# Containers
# -----------------------------------------------------------------------------

class InvalidPitiError(Exception):
    pass

class DuplicatedPitiError(Exception):
    pass

class ConflictedPathError(Exception):
    pass

class UnknownPathError(Exception):
    pass

class WTF(Exception):
    pass


class InventoryItem:
    def __init__(self, is_untrack, piti, path):
        # PITI = Path Item Temporary Identifier
        # A temporary unique number that associated to a path for path operation

        self.is_untrack = bool(is_untrack)
        self.piti = piti
        self.path = path
        self.errors = set()

    def __eq__(self, other):
        return (self.is_untrack, self.piti, self.path) == (other.is_untrack, other.piti, other.path)

    def __iter__(self):
        return iter((self.piti, self.path))

    def __getitem__(self, index):
        return (self.piti, self.path)[index]

    def __repr__(self):
        return repr((self.piti, self.path))

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '<{}{} {}>'.format(
                '#' if self.is_untrack else '',
                self.piti,
                self.path
                )


class Inventory:
    def __init__(self):
        self.ignore_hidden = False
        self.ignore_duplicated_path = False
        self.content = []
        self.xxxxpath_set = set()
        self.piti_index = dict()

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index):
        return self.content[index]

    def __contains__(self, path):
        return xxxxpath(path) in self.xxxxpath_set

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return False

        return self.content == other.content

    def append_item(self, is_untrack, piti, path):
        # Empty paths are directly ignored
        if not path:
            return

        npath = normalize_path(expanduser(path))

        if piti and piti not in self.piti_index:
            self.piti_index[piti] = npath

        rpath = xxxxpath(path)

        if rpath not in self.xxxxpath_set:
            self.xxxxpath_set.add(rpath)
        elif self.ignore_duplicated_path:
            return

        self.content.append(InventoryItem(is_untrack, piti, npath))

    def _append_path(self, path):
        if basename(path).startswith('.') and self.ignore_hidden:
            return

        self.append_item(False, None, path)

    def append(self, path, expand=False, keep_empty_dir=False):
        if not expand:
            self._append_path(path)

        elif not exists(path):
            self._append_path(path)

        elif isdir(path) and not islink(path):
            ls = sorted(os.listdir(path))
            for i in ls:
                self._append_path(join(path, i))

            if not ls and keep_empty_dir:
                self._append_path(path)

        else:
            self._append_path(path)

    def build_index(self):
        if not self.content:
            return

        self.piti_index = dict()

        start = 10 ** (len(str(len(self.content)))) + 1
        for i, item in enumerate(self.content, start=start):
            item.piti = str(i)
            self.piti_index[item.piti] = item.path

    def get_path_by_piti(self, piti):
        return self.piti_index.get(piti)


class VirtualOperation:
    def __hash__(self):
        return id(self)


class TrackOperation(VirtualOperation):
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<Track {}>'.format(self.target)


class ExpandOperation(VirtualOperation):
    def __init__(self, target):
        if target.endswith('*'):
            expansion_mark = '*'
        elif target.endswith('+'):
            expansion_mark = '+'

        self.target = target.rstrip(expansion_mark)

        self.expand_to = []
        for f in os.listdir(self.target):
            newpath = join(self.target, f)
            if expansion_mark == '+' and f.startswith('.'):
                continue

            self.expand_to.append(newpath)

    def __repr__(self):
        return '<Expand {}>'.format(self.target)


class UntrackOperation(VirtualOperation):
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<Untrack {}>'.format(self.target)


class DeleteOperation(VirtualOperation):
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<Delete {}>'.format(self.target)


class RenameOperation(VirtualOperation):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def __repr__(self):
        return '<Rename {}>'.format(', '.join((self.src, self.dst)))


class DominoRenameOperation(VirtualOperation):
    def __init__(self, targets):
        self.targets = targets

    def __repr__(self):
        return '<Domino {}>'.format(', '.join(self.targets))


class RotateRenameOperation(VirtualOperation):
    def __init__(self, targets):
        self.targets = targets

    def __repr__(self):
        return '<Rotate {}>'.format(', '.join(self.targets))


class ReferencedPathTree:
    def __init__(self, node_name):
        self.name = node_name or '/'
        self.children = dict()
        self.tags = dict()
        self.changes = set()

    def _add(self, node_list, userpath, tag, change):
        if not node_list:
            if tag is not None and None in self.tags:
                del self.tags[None]

            if tag not in self.tags:
                self.tags[tag] = set()

            self.tags[tag].add(userpath)

            if change:
                self.changes.add(change)

            return

        if node_list[0] not in self.children:
            self.children[node_list[0]] = ReferencedPathTree(node_list[0])

        self.children[node_list[0]]._add(node_list[1:],
                userpath, tag, change)

    def add(self, userpath, tag, change):
        if not userpath or not isinstance(userpath, str):
            return

        self._add(splitpath(xxxxpath(userpath).lstrip('/')),
                userpath, tag, change)

    def _get(self, node_list):
        if not node_list:
            return self

        if node_list[0] not in self.children:
            return

        return self.children[node_list[0]]._get(node_list[1:])

    def get(self, path):
        if not isinstance(path, str):
            return

        node_list = splitpath(xxxxpath(path).lstrip('/'))
        return self._get(node_list)

    def walk(self):
        yield self

        for child in self.children.values():
            for n in child.walk():
                yield n

    def __repr__(self):
        return '<ReferencedPathTree {}>'.format(self.name)

    def to_lines(self):
        ret = []
        ret.append(self.name + ' (' + ','.join((tag or 'tracking') for tag in self.tags) + ')')

        children = sorted(self.children)
        for idx, child in enumerate(children):
            if idx == len(children) - 1:
                prefix = ['└─ ', '   ']
            else:
                prefix = ['├─ ', '│  ']

            lines = self.children[child].to_lines()
            ret.append(prefix[0] + lines[0])
            for line in lines[1:]:
                ret.append(prefix[1] + line)

        return ret

    def print(self, print_func=print):
        print_func('\n'.join(self.to_lines()))

# -----------------------------------------------------------------------------
# Containers
# =============================================================================

# =============================================================================
# Specialized Utilities
# -----------------------------------------------------------------------------

def normalize_path(path):
    if not path:
        return ''

    npath = ''
    if not path.startswith(('/', './', '../', '~/')) and path not in ('.', '..', '~'):
        npath = './'

    npath += path.rstrip('/')

    if isdir(path) and not islink(path):
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
    change_list_delete = []
    change_list_rename = []

    rename_chains = set()

    for change in change_list_raw:
        if isinstance(change, UntrackOperation):
            change_list_untrack.append(change)

        elif isinstance(change, TrackOperation):
            change_list_track.append(change)

        elif isinstance(change, ExpandOperation):
            change_list_track.append(change)

        elif isinstance(change, DeleteOperation):
            change_list_delete.append(change)

        elif isinstance(change, RenameOperation):
            rename_chains.add((change.src, change.dst))

        else:
            error('Unknown change:', change)

    while rename_chains:
        this_chain = rename_chains.pop()

        add = None
        delete = None
        for other_chain in rename_chains:
            if realpath(this_chain[-1]) == realpath(other_chain[0]):
                delete = other_chain
                add = this_chain + other_chain[1:]
                break

            elif realpath(other_chain[-1]) == realpath(this_chain[0]):
                delete = other_chain
                add = other_chain + this_chain[1:]
                break

        if delete:
            rename_chains.remove(delete)

        if add:
            rename_chains.add(add)

        if add is None:
            if this_chain[0] == this_chain[-1]:
                rotate_chain = this_chain[:-1]
                pivot = rotate_chain.index(min(rotate_chain))
                rotate_chain = rotate_chain[pivot:] + rotate_chain[:pivot]
                change_list_rename.append(RotateRenameOperation(rotate_chain))

            else:
                change_list_rename.append(DominoRenameOperation(this_chain))

    return (change_list_untrack +
            change_list_track +
            change_list_delete +
            sorted(change_list_rename, key=lambda x: x.targets[0]))

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
    if exists('exit'):
        return (exit, 1)

    with tempfile.NamedTemporaryFile(prefix='vd', suffix='vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            if isinstance(inventory, Inventory):
                for piti, path in inventory:
                    if piti is None:
                        f.write(f'{path}\n')
                    else:
                        f.write(f'{piti}\t{shrinkuser(path)}\n')
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
        new = Inventory()
        lines = []
        with open(tf.name, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip('\n')
                lines.append(line)

                if not line:
                    continue

                rec = RegexCache(line)

                if rec.match(r'^(#?) *(\d+)\t(.*)$'):
                    untrack_mark, piti, path = rec.groups()
                    new.append_item(untrack_mark == '#', piti, path)

                elif line.startswith('#'):
                    continue

                else:
                    new.append_item(False, None, line)

    return (step_calculate_inventory_diff, base, new)


def step_calculate_inventory_diff(base, new):
    debug(magenta('==== inventory (base) ===='))
    for oitem in base.content:
        debug(oitem)
    debug('-------------------------')
    for nitem in new.content:
        debug(nitem)
    debug(magenta('==== inventory (new) ===='))

    # =========================================================================
    # Calculate inventory diff
    # -------------------------------------------------------------------------
    # 1. Count piti before hand
    # 2. Construct bucket (indexed by piti)
    # 2.1 Put base inventory into bucket
    # 2.2 Put new inventory info bucket
    # 3. Check piti errors
    # 4. Construct change list from bucket
    # -------------------------------------------------------------------------

    # 1. Count piti before hand
    piti_count = dict()

    for nitem in new:
        npiti, npath = nitem

        if npiti is not None:
            piti_count[npiti] = piti_count.get(npiti, 0) + 1

    # 2. Construct bucket (indexed by piti)
    bucket = {'new':[]}

    # 2.1 Put base inventory into bucket
    for opiti, opath in base:
        bucket[opiti] = opath

    # 2.2 Put new inventory info bucket
    for nitem in new:
        npiti, npath = nitem

        # Handle newly-tracked path
        if npiti is None:
            bucket['new'].append(npath)
            continue

        # Piti checks
        else:
            # Not allow invalid piti
            if npiti not in bucket:
                nitem.errors.add(InvalidPitiError)
                continue

            # Not allow duplicated piti
            if piti_count[npiti] > 1:
                nitem.errors.add(DuplicatedPitiError)
                continue

        opath = bucket[npiti]

        if nitem.is_untrack:
            bucket[npiti] = (opath, None)
        else:
            bucket[npiti] = (opath, npath)

    # 3. Check piti errors
    error_lines = []
    for nitem in new:
        if nitem.errors:
            line = ''
            if nitem.piti is not None:
                line += (red_bg if nitem.errors else lambda x: x)(
                        ('#' if nitem.is_untrack else '') + nitem.piti
                        )

                line += '  '

            if InvalidPitiError in nitem.errors:
                line += nitem.path + red(' ◄─ Invalid key')
            elif DuplicatedPitiError in nitem.errors:
                line += nitem.path + red(' ◄─ Duplicated key')
            elif nitem.errors:
                line += nitem.path + red(' ◄─ ' + ','.join(e.__name__ for e in path_errors))
            else:
                line += nitem.path

            error_lines.append(line)

    if error_lines:
        for line in error_lines:
            error(line)

        user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'],
                allow_empty_input=False)
        if user_confirm == 'edit':
            return (step_vim_edit_inventory, base, new)

        if user_confirm == 'redo':
            return (step_vim_edit_inventory, base, base)

        return (exit, 1)

    # 4. Construct change list from bucket
    # ('track', target)
    # ('untrack', target)
    # ('delete', target)
    # ('rename', src, dst)
    change_list_raw = []
    for piti in bucket:
        if piti == 'new':
            for path in bucket['new']:
                change_list_raw.append(TrackOperation(path))

        elif isinstance(bucket[piti], str):
            change_list_raw.append(DeleteOperation(bucket[piti]))

        elif isinstance(bucket[piti], tuple):
            opath = bucket[piti][0]
            npath = bucket[piti][1]

            if bucket[piti][1] is None:
                change_list_raw.append(UntrackOperation(opath))

            elif isdir(opath) and (npath == join(opath, '*') or npath == join(opath, '+')):
                change_list_raw.append(ExpandOperation(npath))

            elif realpath(opath) != realpath(npath):
                change_list_raw.append(RenameOperation(opath, npath))

    return (step_check_change_list, base, new, change_list_raw)


def step_check_change_list(base, new, change_list_raw):
    # =========================================================================
    # Check change list
    # -------------------------------------------------------------------------
    # 1. Put all referenced paths into tree
    # 1.1 Put paths of new inventory items into tree
    # 1.2 Put paths of changes into tree
    # 2. Conflict operations and path checks
    # 2.1 Check if there are multiple operations targets a single path
    # 2.2 or change targets on a path that have children
    # -------------------------------------------------------------------------

    # 1. Put path(s) of changes into tree
    tree = ReferencedPathTree(None)

    # 1.1 Put paths of new inventory items into tree
    for nitem in new:
        if nitem.is_untrack:
            continue

        if not nitem.path.endswith(('*', '+')):
            tree.add(nitem.path, None, None)

    # 1.2 Put paths of changes into tree
    for change in change_list_raw:
        if isinstance(change, UntrackOperation):
            tree.add(change.target, 'untrack', change)

        elif isinstance(change, TrackOperation):
            tree.add(change.target, 'track', change)

        elif isinstance(change, ExpandOperation):
            for f in change.expand_to:
                tree.add(f, 'track', change)

        elif isinstance(change, DeleteOperation):
            tree.add(change.target, 'delete', change)

        elif isinstance(change, RenameOperation):
            tree.add(change.src, 'rename/from', change)
            tree.add(change.dst, 'rename/to', change)

    tree.print(debug)

    # 2. Conflict operations and path checks
    # 2. A risky policy is used: only explicit errors are forbidden
    change_list_filtered = set()
    error_groups = []
    for node in tree.walk():
        ok_changes = node.changes.copy()
        error_group = []

        # Cancel out track and untrack/delete, results in track
        if set(node.tags.keys()) in [{'track', 'untrack'}, {'track', 'delete'}]:
            ok_changes = set(c for c in ok_changes if isinstance(c, TrackOperation))

        elif set(node.tags.keys()) == {'rename/to', 'rename/from'}:
            pass

        # Multiple operations on same path are not allowed
        elif len(node.tags) > 1:
            for tag, refers in node.tags.items():
                for refer in refers:
                    error_group.append((tag, refer))

        # Check if the modify target has children
        elif (set(node.tags.keys()) & {'delete', 'rename/from', 'rename/to'}
                and node.children):
            for tag, refers in node.tags.items():
                for refer in refers:
                    error_group.append((tag, refer))

            for child_node in node.children.values():
                for tag, refers in child_node.tags.items():
                    for refer in refers:
                        error_group.append((tag, refer))

        if error_group:
            error_groups.append(error_group)
        else:
            change_list_filtered |= ok_changes

    for idx, error_group in enumerate(error_groups):
        if idx:
            print_stderr()

        error('Conflicted operations:')
        tag_column_width = max(len(tag or '(tracking)') for tag, path in error_group)
        for tag, path in sorted(error_group,
                key=lambda x: (x[0] is None, x[0], x[1])):
            error(yellow((tag or '(tracking)').ljust(tag_column_width)), path)

    if error_groups:
        user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'],
                allow_empty_input=False)
        if user_confirm == 'edit':
            return (step_vim_edit_inventory, base, new)

        if user_confirm == 'redo':
            return (step_vim_edit_inventory, base, base)

        return (exit, 1)

    return (step_confirm_change_list, base, new, list(change_list_filtered))


def step_confirm_change_list(base, new, change_list_raw):
    # If base inventory and new inventory is exactly the same, exit
    if base == new:
        info('No change')
        return (exit, 0)

    change_list = aggregate_changes(change_list_raw)

    for change in change_list:
        if isinstance(change, UntrackOperation):
            print_path_with_prompt(info, cyan, 'Untrack:', change.target)

        elif isinstance(change, TrackOperation):
                print_path_with_prompt(info, cyan, 'Track:', change.target)

        elif isinstance(change, ExpandOperation):
            print_path_with_prompt(info, cyan, 'Expand:', change.target)

        elif isinstance(change, DeleteOperation):
            print_path_with_prompt(info, red, 'Delete:', change.target)

        elif isinstance(change, DominoRenameOperation):
            if len(change.targets) == 2:
                A, B = pretty_diff_strings(change.targets[0], change.targets[1])
                print_path_with_prompt(info, yellow, 'Rename:', A)
                print_path_with_prompt(info, yellow, '└─────►', B)

            else:
                for idx, path in enumerate(change.targets):
                    print_path_with_prompt(info, yellow,
                            'Rename:' + ('┌─' if idx == 0 else '└►'),
                            path)

        elif isinstance(change, RotateRenameOperation):
            if len(change.targets) == 2:
                print_path_with_prompt(info, yellow, 'Swap:┌►', change.targets[0])
                print_path_with_prompt(info, yellow, 'Swap:└►', change.targets[1])
            else:
                rotate_chain_len = len(change.targets)
                for idx, path in enumerate(change.targets):
                    if idx == 0:
                        arrow = '┌►┌─'
                    elif idx == rotate_chain_len - 1:
                        arrow = '└───'
                    else:
                        arrow = '│ └►'

                    print_path_with_prompt(info, yellow, 'Rotate:' + arrow, path)

        else:
            warning(f'Unknown change: {change}')

    # If all changes are 'track' and 'untrack', apply the change directly
    if (not change_list) or all({type(c) in {TrackOperation, UntrackOperation, ExpandOperation} for c in change_list}):
        return (step_expand_inventory, new)

    user_confirm = prompt_confirm('Continue?', ['yes', 'edit', 'redo', 'quit'])
    if user_confirm == 'yes':
        return (step_apply_change_list, base, new, change_list)

    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    elif user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    elif user_confirm == 'quit':
        return (exit, 1)

    error('Unexpected scenario, abort')
    return (exit, 1)


def step_apply_change_list(base, new, change_list):
    cmd_list = []
    for change in change_list:
        if isinstance(change, UntrackOperation):
            print_path_with_prompt(info, cyan, 'Untrack:', change.target)

        elif isinstance(change, TrackOperation):
            print_path_with_prompt(info, cyan, 'Track:', change.target)

        elif isinstance(change, ExpandOperation):
            print_path_with_prompt(info, cyan, 'Expand:', change.target)

        elif isinstance(change, DeleteOperation):
            cmd_list.append(('rm', change.target))

        elif isinstance(change, DominoRenameOperation):
            for src, dst in list(zip(change.targets, change.targets[1:]))[::-1]:
                cmd_list.append(('mv', src, dst))

        elif isinstance(change, RotateRenameOperation):
            tmpdst = gen_tmp_file_name(change.targets[-1])

            cmd_list.append(('mv', change.targets[-1], tmpdst))
            for src, dst in list(zip(change.targets, change.targets[1:]))[::-1]:
                cmd_list.append(('mv', src, dst))

            cmd_list.append(('mv', tmpdst, change.targets[0]))

        else:
            warning(f'Unknown change: {change}')

    def clean_up_empty_dir(path):
        path = path.rstrip('/')
        while path and path != '.':
            try:
                ls = os.listdir(path)
                if ls and ls != ['.DS_Store']:
                    return

                dot_ds_store = join(path, '.DS_Store')
                if exists(dot_ds_store):
                    print(red('$'), 'rm', magenta(dot_ds_store))
                    os.remove(join(dot_ds_store))

                print(red('$'), 'rmdir', magenta(path))
                os.rmdir(path)
            except FileNotFoundError:
                return
            except OSError:
                return

            path = parent_dir(path)

    rmdirset = set()

    for cmd in cmd_list:
        if cmd[0] == 'rm':
            if not isdir(cmd[1]) or islink(cmd[1]):
                print(red('$'), 'rm', magenta(shlex.quote(cmd[1])))
                os.remove(cmd[1])
                rmdirset.add(parent_dir(cmd[1]))
            else:
                print(red('$'), 'rm', '-r', magenta(shlex.quote(cmd[1])))
                shutil.rmtree(cmd[1])
                rmdirset.add(parent_dir(cmd[1]))

        elif cmd[0] == 'mv':
            if not exists(parent_dir(cmd[2])):
                print(yellow('$'), 'mkdir', '-p', magenta(parent_dir(cmd[2])))
                os.makedirs(parent_dir(cmd[2]), exist_ok=True)
            print(yellow('$'), 'mv',
                    magenta(shlex.quote(cmd[1])),
                    magenta(shlex.quote(cmd[2])))

            if islink(cmd[1]):
                linkto = os.readlink(cmd[1])
                os.symlink(linkto, cmd[2])
                os.remove(cmd[1])

            else:
                shutil.move(cmd[1], cmd[2])

            p1 = parent_dir(cmd[1])
            p2 = parent_dir(cmd[2])
            if p1 != p2:
                rmdirset.add(p1)

        else:
            warning(f'Unknown cmd: {cmd}')

    for d in rmdirset:
        clean_up_empty_dir(d)

    if any({type(c) in (TrackOperation, UntrackOperation, ExpandOperation) for c in change_list}):
        return (step_expand_inventory, new)

    return (exit, 0)


def step_expand_inventory(new):
    newnew = Inventory()
    for item in new:
        if not item.is_untrack:
            if item.path.endswith(('*', '+')):
                operation = ExpandOperation(item.path)

                for f in operation.expand_to:
                    newnew.append(f)
            else:
                newnew.append(item.path)
    newnew.build_index()

    if not newnew:
        info('No targets to edit')
        return (exit, 0)

    return (step_vim_edit_inventory, newnew, newnew)

# -----------------------------------------------------------------------------
# "Step" functions
# =============================================================================


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
        if not islink(path) and not exists(path):
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
