#!/usr/bin/env python3

# Mandatory
#TODO: Refine simple diff
#TODO: -r/--recursive, with depth limit?
#TODO: Change the ./ prefix?

# Vim related
#TODO: Respect LS_COLORS by utilizing bits in PITI

# Enhancement
#TODO: If user change dir trailing slash to .tar, tar it
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
import unicodedata

from enum import Enum
from math import ceil, log10
from os.path import split, dirname, basename, join, exists, isdir, islink, isfile, abspath, realpath, expanduser


# =============================================================================
# Global option
# -----------------------------------------------------------------------------

options = argparse.Namespace(
        diff_style=None,
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
red = paint(31)
green = paint(32)
yellow = paint(33)
blue = paint(34)
magenta = paint(35)
cyan = paint(36)
white = paint(37)
red_bg = paint('41')
green_bg = paint('42')
yellow_bg = paint('30;43')
blue_bg = paint('44')
magenta_bg = paint('45')
cyan_bg = paint('46')
white_bg = paint('47')
nocolor = paint('')

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


def sorted_as_filename(ls):
    def filename_key(name):
        def int_or_not(x):
            if x and x[0] in '1234567890':
                return int(x)
            return x
        return tuple(int_or_not(x) for x in re.split(r'(\d+)', name))

    return sorted(ls, key=filename_key)


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
            ls = sorted_as_filename(os.listdir(path))
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


class GlobbingOperation(VirtualSingleTargetOperation):
    def __init__(self, target):
        super().__init__(target)

        if target.endswith('*'):
            wildcard = '*'
        elif target.endswith('+'):
            wildcard = '+'

        self.target = target.rstrip(wildcard)

        self.expand_to = []
        for f in sorted_as_filename(os.listdir(self.target)):
            newpath = join(self.target, f)
            if wildcard == '+' and f.startswith('.'):
                continue

            self.expand_to.append(newpath)


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


class ReferencedPathTree:
    def __init__(self, node_name):
        self.name = node_name or '/'
        self.children = dict()
        self.changes = set()

        self.inv_inward_flows = []
        self.inv_outward_flows = []
        self.fs_inward_flows = []
        self.fs_outward_flows = []

    @property
    def flows(self):
        return (
                self.inv_inward_flows +
                self.inv_outward_flows +
                self.fs_inward_flows +
                self.fs_outward_flows
                )

    @property
    def flow_type_list(self):
        return [flow for flow,_,_ in self.flows]

    def _add(self, node_list, referpath, flow, change):
        if not node_list:
            if self.flow_type_list == ['(tracking)']:
                self.inv_inward_flows = []

            if flow in {'(tracking)', 'track', 'resolve/real'}:
                self.inv_inward_flows.append((flow, referpath, change))

            if flow in {'untrack'}:
                self.inv_outward_flows.append((flow, referpath, change))

            if flow in {'rename', 'rename/to'}:
                self.inv_inward_flows.append((flow, referpath, change))
                self.fs_inward_flows.append((flow, referpath, change))

            if flow in {'rename', 'delete', 'rename/from'}:
                self.inv_outward_flows.append((flow, referpath, change))
                self.fs_outward_flows.append((flow, referpath, change))

            if change:
                self.changes.add(change)

            return

        if node_list[0] not in self.children:
            self.children[node_list[0]] = ReferencedPathTree(node_list[0])

        self.children[node_list[0]]._add(node_list[1:],
                referpath, flow, change)

    def add(self, referpath, flow, change):
        if not referpath or not isinstance(referpath, str):
            return

        self._add(splitpath(xxxxpath(referpath).lstrip('/')),
                referpath, flow, change)

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
        return f'<ReferencedPathTree {self.name}>'

    def to_lines(self):
        ret = []
        ret.append(self.name + ' (' + ','.join(
            (flow or 'tracking')
            for flow, _, _ in self.flows) + ')')

        children = sorted_as_filename(self.children)
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


def fancy_diff_strings(a, b):
    diff_segments = []

    diff_aligned_A = ''
    diff_aligned_B = ''
    diff_compact_A = ''
    diff_compact_B = ''
    diff_oneline = ''

    tag_counter = collections.Counter(equal=0, delete=0, insert=0, replace=0)

    NFKC = lambda s: unicodedata.normalize('NFKC', s)

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        seg_a = a[i1:i2]
        seg_b = b[j1:j2]

        tag_counter[tag] += 1

        if tag == 'equal':
            diff_aligned_A += seg_a
            diff_aligned_B += seg_b

            diff_compact_A += seg_a
            diff_compact_B += seg_b

            if diff_oneline is not None:
                diff_oneline += seg_a

        elif tag == 'delete':
            diff_aligned_A += red_bg(seg_a)
            diff_aligned_B += ' ' * str_width(seg_a)

            diff_compact_A += red_bg(seg_a)

            if diff_oneline is not None:
                diff_oneline += red_bg(seg_a)

        elif tag == 'insert':
            diff_aligned_A += ' ' * str_width(seg_b)
            diff_aligned_B += green_bg(seg_b)

            diff_compact_B += green_bg(seg_b)

            if diff_oneline is not None:
                diff_oneline += green_bg(seg_b)

        elif tag == 'replace':
            wa = str_width(seg_a)
            wb = str_width(seg_b)
            w = max(wa, wb)

            diff_aligned_A += (' ' * (w - wa)) + red_bg(seg_a)
            diff_aligned_B += (' ' * (w - wb)) + green_bg(seg_b)

            diff_compact_A += red_bg(seg_a)
            diff_compact_B += green_bg(seg_b)

            if diff_oneline is None:
                pass
            elif NFKC(seg_a.strip()) == NFKC(seg_b.strip()):
                diff_oneline += yellow_bg(seg_b)
            else:
                diff_oneline = None

    tag_count = types.SimpleNamespace(**tag_counter)

    diff_style = options.diff_style

    if options.debug:
        debug(diff_style)

    if diff_style is None:
        if diff_oneline:
            return (diff_oneline, None)

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

    elif diff_style == DiffStyle.aligned:
        return (diff_aligned_A, diff_aligned_B)

    else:
        return (diff_compact_A, diff_compact_B)


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

        elif isinstance(change, GlobbingOperation):
            change_list_track.append(change)

        elif isinstance(change, ResolveLinkOperation):
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

hint_text_sepline = '# ' + ('=' * (screen_width() - 5))

hint_text = '''
# - Prefix an item with '#' to untrack it.
# - Add a path to track it.
# - Sort it as you want.
# - path/to/directory/+ to expand non-hidden items
# - path/to/directory/* to expand all items
# - path/to/link@ to resolve the soft link
'''.strip()

def hint_text_vimrc():
    if isfile(VD_VIMRC_PATH):
        return '# - Configure hotkeys in ~/.config/vd/vd.vimrc'
    else:
        return '''
# - Setup default vd.vimrc with
#   $ vd --vimrc
'''.strip()


def step_vim_edit_inventory(base, inventory):
    if exists('exit'):
        return (exit, 1)

    with tempfile.NamedTemporaryFile(prefix='vd', suffix='vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            def writeline(line=''):
                f.write(line + '\n')

            writeline(hint_text_sepline)
            writeline(hint_text)
            writeline(hint_text_vimrc())
            writeline(hint_text_sepline)
            writeline()

            if isinstance(inventory, Inventory):
                for piti, path in inventory:
                    if piti is None:
                        writeline(f'{path}')
                    else:
                        writeline(f'{piti}\t{shrinkuser(path)}')
            else:
                for line in inventory:
                    writeline(f'{line}')
            f.flush()

        # Invoke vim to edit item list
        cmd = ['vim', tf.name, '+normal }']

        if isfile(VD_VIMRC_PATH):
            cmd += ['+source ' + VD_VIMRC_PATH]
        else:
            cmd += ['+set nonu']

        sub.call(cmd, stdin=open('/dev/tty'))
        print()

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
    if options.debug:
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
    # 2. Construct a piti-indexed bucket to join base and new inventories
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

    # 2. Construct a piti-indexed bucket to join base and new inventories
    bucket = {'new':[]}

    # 2.1 Put base inventory into bucket
    for opiti, opath in base:
        bucket[opiti] = opath

    # 2.2 Put new inventory info bucket
    for nitem in new:
        npiti, npath = nitem

        # Handle newly-tracked path
        if npiti is None:
            bucket['new'].append((nitem, npath))
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
            bucket[npiti] = (nitem, opath, None)
        else:
            bucket[npiti] = (nitem, opath, npath)

    # 3. Check piti errors
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

            errorq(line)

    if has_error():
        return (step_ask_fix_it, base, new)

    # 4. Construct change list from bucket
    # ('track', target)
    # ('untrack', target)
    # ('delete', target)
    # ('rename', src, dst)
    change_list_raw = []
    for piti in bucket:
        if piti == 'new':
            for _, path in bucket['new']:
                if exists(path):
                    change_list_raw.append(TrackOperation(path))
                else:
                    errorq(red(nitem.path + ' ◄─ Not exists'))

        elif isinstance(bucket[piti], str):
            change_list_raw.append(DeleteOperation(bucket[piti]))

        elif isinstance(bucket[piti], tuple):
            nitem = bucket[piti][0]
            opath = bucket[piti][1]
            npath = bucket[piti][2]

            if npath is None:
                # Untrack
                change_list_raw.append(UntrackOperation(opath))

            elif npath == join(opath, '*') or npath == join(opath, '+'):
                # Expand
                if isdir(opath):
                    change_list_raw.append(GlobbingOperation(npath))
                else:
                    errorq(nitem.piti + '  ' +
                            red(nitem.path + ' ◄─ Cannot expand file'))

            elif npath == (opath + '@'):
                if islink(opath):
                    change_list_raw.append(ResolveLinkOperation(opath))
                else:
                    errorq(nitem.piti + '  ' +
                            red(nitem.path + ' ◄─ Can only resolve link'))

            elif xxxxpath(opath) != xxxxpath(npath):
                # Rename
                change_list_raw.append(RenameOperation(opath, npath))

            else:
                # Nothing changed for this piti
                pass

    if has_error():
        return (step_ask_fix_it, base, new)

    return (step_check_change_list, base, new, change_list_raw)


def step_ask_fix_it(base, new):
    errorflush()

    user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'],
            allow_empty_input=False)
    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    if user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    return (exit, 1)


def step_check_change_list(base, new, change_list_raw):
    # =========================================================================
    # Check change list
    # -------------------------------------------------------------------------
    # 1. Put all referenced paths into tree
    # 1.1 Put paths of base inventory items into tree
    # 1.2 Put paths of changes into tree
    # 2. Conflict operations and path checks
    # 2.1 Check if there are multiple operations targets a single path
    # 2.2 Or changes target on paths that have children
    # -------------------------------------------------------------------------

    # 1. Put path(s) of changes into tree
    tree = ReferencedPathTree(None)

    # 1.1 Put paths of new inventory items into tree
    for oitem in base:
        if oitem.is_untrack:
            continue

        if not oitem.path.endswith(('/*', '/+')):
            tree.add(oitem.path, '(tracking)', None)

    # 1.2 Put paths of changes into tree
    for change in change_list_raw:
        if isinstance(change, UntrackOperation):
            tree.add(change.target, 'untrack', change)

        elif isinstance(change, TrackOperation):
            tree.add(change.target, 'track', change)

        elif isinstance(change, GlobbingOperation):
            for f in change.expand_to:
                tree.add(f, 'track', change)

        elif isinstance(change, ResolveLinkOperation):
            tree.add(change.target, 'untrack', change)
            tree.add(change.resolve_to, 'resolve/real', change)

        elif isinstance(change, DeleteOperation):
            tree.add(change.target, 'delete', change)

        elif isinstance(change, RenameOperation):
            if inode(change.src) == inode(change.dst):
                # If both inode are the same, try to make it non-conflict.
                # One possibility is the file system treats different
                # unicode normal form as the same file
                tree.add(change.src, 'rename', change)
                tree.add(change.dst, 'rename', change)
            else:
                tree.add(change.src, 'rename/from', change)
                tree.add(change.dst, 'rename/to', change)

    if options.debug:
        tree.print(debug)

    # 2. Conflict operations and path checks
    # 2. A risky policy is used: only explicit errors are forbidden
    change_list_filtered = set()
    error_groups = []
    for node in tree.walk():
        error_group = set()

        # Special case: Override track + delete with a single track
        if node.flow_type_list == ['track', 'delete', 'delete']:
            node.inv_inward_flows = [node.inv_inward_flows[0]]
            node.inv_outward_flows = []
            node.fs_outward_flows = []

        # Forbid multiple operations on a same path
        if any(len(flows) > 1 for flows in (
            node.inv_inward_flows,
            node.inv_outward_flows,
            node.fs_inward_flows,
            node.fs_outward_flows
            )):
            error_group |= set(node.flows)

        if not error_group:
            # Count net flow of the node
            fs_net_flow_count = len(node.fs_inward_flows) - len(node.fs_outward_flows)

            # Forbid overriding existing path
            if fs_net_flow_count > 0 and exists(node.fs_inward_flows[0][1]):
                error_group.add(('(existing)', node.fs_inward_flows[0][1], None))
                error_group |= set(node.fs_inward_flows)

        if not error_group:
            # Forbid modifying/deleting targets that has children
            if len(node.fs_outward_flows) and node.children:
                error_group |= set(node.fs_outward_flows)

                for child_node in node.children.values():
                    error_group |= set(child_node.flows)

        if error_group:
            error_groups.append(error_group)
        else:
            change_list_filtered |= set(
                    change for _,_,change in node.flows
                    if change is not None)

    for idx, error_group in enumerate(error_groups):
        if idx:
            errorq()

        err_msg = []
        for idx, (flow, refer, change) in enumerate(error_group):
            if idx:
                err_msg.append(('', None))

            if change is None:
                err_msg.append((flow, refer))

            elif isinstance(change, GlobbingOperation):
                err_msg.append(('track', refer))

            elif isinstance(change, RenameOperation):
                err_msg.append(('rename/from', change.src))
                err_msg.append(('rename/to', change.dst))

            elif isinstance(change, VirtualSingleTargetOperation):
                err_msg.append((flow, refer))

        errorq('Conflicted operations:')
        tag_column_width = max(len(flow) for flow, _ in err_msg)
        for idx, (flow, refer) in enumerate(err_msg):
            if not flow:
                errorq()
                continue

            errorq(f'  {yellow}{flow:<{tag_column_width}}{nocolor} {refer}')

    if error_groups:
        return (step_ask_fix_it, base, new)

    return (step_confirm_change_list, base, new, list(change_list_filtered))


def step_confirm_change_list(base, new, change_list_raw):
    # If base inventory and new inventory is exactly the same, exit
    if base == new:
        info('No change')
        return (exit, 0)

    change_list = aggregate_changes(change_list_raw)

    deleted_item_list = set(
            change.target for change in change_list
            if isinstance(change, DeleteOperation))

    for change in change_list:
        if isinstance(change, UntrackOperation):
            print_path_with_prompt(info, cyan, 'Untrack:', change.target)

        elif isinstance(change, TrackOperation):
            print_path_with_prompt(info, cyan, 'Track:', change.target)

        elif isinstance(change, GlobbingOperation):
            print_path_with_prompt(info, cyan, 'Expand:', change.target)

        elif isinstance(change, ResolveLinkOperation):
            print_path_with_prompt(info, cyan, 'Resolve:', change.target)

        elif isinstance(change, DeleteOperation):
            print_path_with_prompt(info, red, 'Delete:', change.target)

        elif isinstance(change, DominoRenameOperation):
            if len(change.targets) == 2:
                if change.targets[1] in deleted_item_list:
                    A, B = change.targets
                else:
                    A, B = fancy_diff_strings(change.targets[0], change.targets[1])

                print_path_with_prompt(info, yellow, 'Rename:', A)

                if B:
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
    if (not change_list) or all({type(c) in {
        TrackOperation, UntrackOperation,
        GlobbingOperation, ResolveLinkOperation
        } for c in change_list}):
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

        elif isinstance(change, GlobbingOperation):
            print_path_with_prompt(info, cyan, 'Expanding:', change.target)

        elif isinstance(change, ResolveLinkOperation):
            print_path_with_prompt(info, cyan, 'Resolving:', change.target)

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

    if any({type(c) in (TrackOperation, UntrackOperation, GlobbingOperation) for c in change_list}):
        return (step_expand_inventory, new)

    return (exit, 0)


def step_expand_inventory(new):
    newnew = Inventory()
    for item in new:
        if not item.is_untrack:
            if item.path.endswith(('*', '+')):
                operation = GlobbingOperation(item.path)

                for f in operation.expand_to:
                    newnew.append(f)

            elif item.path.endswith('@'):
                newnew.append(os.readlink(item.path.rstrip('@')))

            else:
                newnew.append(item.path)
    newnew.build_index()

    if not newnew:
        info('No targets to edit')
        return (exit, 0)

    print()
    print()

    return (step_vim_edit_inventory, newnew, newnew)

# -----------------------------------------------------------------------------
# "Step" functions
# =============================================================================


def open_vd_vimrc():
    os.makedirs(dirname(VD_VIMRC_PATH), exist_ok=True)

    if exists(VD_VIMRC_PATH) and not isfile(VD_VIMRC_PATH):
        error(VD_VIMRC_PATH, 'exists and is not a file')
        return 1

    if not exists(VD_VIMRC_PATH):
        # deploy a new one
        with open(VD_VIMRC_PATH, 'w') as vd_vimrc:
            vd_vimrc.write('''
" =============================================================================
" vd vimrc
" =============================================================================

" Turn off line number for not interfere with item key
set nonu

" Set a wide gap between item key and path
set tabstop=8

" Rename item
nnoremap cc ^WC
nnoremap S ^WC

set listchars=tab:¦¦
'''.lstrip())

    print(VD_VIMRC_PATH)
    return sub.call(['vim', VD_VIMRC_PATH])


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

    parser.add_argument('-s', '--diff-style',
            choices=DiffStyle.__members__.keys(),
            help='Specify diff style')

    parser.add_argument('--vimrc', action='store_true',
            default=False,
            help='Edit or create the vimrc for vd')

    parser.add_argument('--debug', action='store_true',
            default=False,
            help='Print debug messages')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit, directories are expanded')

    args = parser.parse_args()

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('Both stdout and stderr must be tty')
        exit(1)

    if args.vimrc:
        exit(open_vd_vimrc())

    options.debug = args.debug
    if args.diff_style:
        options.diff_style = DiffStyle[args.diff_style]

    if options.debug:
        print(options)

    # =========================================================================
    # Collect initial targets
    # -------------------------------------------------------------------------
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded
    # -------------------------------------------------------------------------

    targets = []

    for target in sorted_as_filename(args.targets):
        targets.append((target, True))

    if not sys.stdin.isatty():
        for line in sys.stdin:
            targets.append((line.rstrip('\n'), False))

    if not targets:
        targets.append(('.', True))

    inventory = Inventory()
    inventory.ignore_hidden = not args.all
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
