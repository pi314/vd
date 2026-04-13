import os
import os.path
import sys
import unicodedata

from pathlib import Path

from .paints import *


def str_width(s):
    return sum(1 + (unicodedata.east_asian_width(c) in 'WF') for c in paints.decolor(s))


def shrinkuser(path):
    homepath = os.path.expanduser('~').rstrip('/') + '/'
    if path.startswith(homepath):
        return os.path.join('~', path[len(homepath):])
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


def uniq(lst):
    added = set()
    ret = []
    for elem in lst:
        if elem not in added:
            added.add(elem)
            ret.append(elem)

    return ret


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
        sys.exit(1)

    except EOFError:
        print(black('^D'))
        if allow_empty_input:
            us.select(options[0])

    print()

    sys.stdin = stdin_backup
    sys.stdout = stdout_backup
    sys.stderr = stderr_backup

    return us


_open = open
def open(*args, **kwargs):
    f = _open(*args, **kwargs)

    def writeline(line=''):
        f.write(line + '\n')

    def writelines(lines=[]):
        for line in lines:
            writeline(line)

    def readlines():
        lines = []
        for line in f:
            lines.append(line.rstrip('\n'))
        return lines

    f.writeline = writeline
    f.writelines = writelines
    f.readlines = readlines

    return f


def gen_tmp_file_name(path, postfix='.vdtmp'):
    import time
    now = time.time()
    tmp_file_name = '{orig_path}{postfix}[{getpid}].{timestamp}'.format(
            orig_path=path.lstrip('/'),
            postfix=postfix,
            timestamp=now,
            getpid=os.getpid(),
            )
    return Path(tmp_file_name)


def fancy_diff_strings(a, b):
    import collections
    import unicodedata
    import difflib

    red_bg = paint('41')
    green_bg = paint('42')
    yellow_bg = paint('30;43')

    diff_segments = []

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
            diff_compact_A += seg_a
            diff_compact_B += seg_b

            if diff_oneline is not None:
                diff_oneline += seg_a

        elif tag == 'delete':
            diff_compact_A += red_bg(seg_a)

            if diff_oneline is not None:
                diff_oneline += red_bg(seg_a)

        elif tag == 'insert':
            diff_compact_B += green_bg(seg_b)

            if diff_oneline is not None:
                diff_oneline += green_bg(seg_b)

        elif tag == 'replace':
            wa = str_width(seg_a)
            wb = str_width(seg_b)
            w = max(wa, wb)

            diff_compact_A += red_bg(seg_a)
            diff_compact_B += green_bg(seg_b)

            if diff_oneline is None:
                pass
            elif NFKC(seg_a.strip()) == NFKC(seg_b.strip()):
                diff_oneline += yellow_bg(seg_b)
            else:
                diff_oneline = None

    if diff_oneline:
        return (diff_oneline, None)

    return (diff_compact_A, diff_compact_B)
