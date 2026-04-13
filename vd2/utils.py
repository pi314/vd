import os
import os.path
import sys
import unicodedata

from pathlib import Path

from .iroiro import *
from .paints import *


def uniq(lst):
    added = set()
    ret = []
    for elem in lst:
        if elem not in added:
            added.add(elem)
            ret.append(elem)

    return ret


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
