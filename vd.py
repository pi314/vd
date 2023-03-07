#!/usr/bin/env python3

# Mandatory
#TODO: implement PathList to merge inventory and stage

# Vim related
#TODO: Set default mapping for J/K to move entry downward/upward
#TODO: vim default config

# Misc
#TODO: If user change dir to .tar (by removing the trailing slash), tar it
#TODO: expand dir by appending a asterisk to a dir

__version__ = '0.0.0'

import argparse
import collections
import difflib
import os
import re
import subprocess as sub
import sys
import tempfile

from os.path import basename, join, exists, isdir, relpath


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
        return '\033[{}m{}\033[m'.format(color_code, s)
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


def print_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def debug(*args, **kwargs):
    print_stderr(magenta('[Debug]'), *args, **kwargs)


def info(*args, **kwargs):
    print('[Info]', *args, **kwargs)


def error(*args, **kwargs):
    print_stderr(red('[Error]'), *args)


def inventory_to_stage(inventory):
    stage = collections.OrderedDict()

    for idx, fpath in enumerate(inventory, start=10**len(str(len(inventory)))):
        stage[str(idx)] = relpath(fpath)

    return stage


def stage_to_inventory(stage):
    return list(stage.values())


def diff_stages(stage_now, stage_new):
    ret = []

    debug('stage current')
    for idx, fpath in stage_now.items():
        debug(idx, fpath)

    debug()
    debug('stage new')
    for idx, fpath in stage_new.items():
        debug(idx, fpath)

    for idx in stage_now:
        if idx not in stage_new:
            ret.append(('remove', stage_now[idx]))
        elif stage_now[idx] != stage_new[idx]:
            ret.append(('rename', stage_now[idx], stage_new[str(idx)]))

    return ret


def pretty_print_operand(color, prompt, path):
    print(color(prompt) + color('[') + ' ' + path + ' ' + color(']'))


def print_op_list(op_list):
    for op in op_list:
        if op[0] == 'remove':
            p = op[1] + ('/' if isdir(op[1]) else '')
            pretty_print_operand(red, 'Remove:', p)

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

            pretty_print_operand(yellow, 'Rename:', A)
            pretty_print_operand(yellow, '======>', B)

        else:
            debug(op)


def apply_op_list(op_list):
    for op in op_list:
        if op[0] == 'remove':
            pretty_print_operand(red, '(dry)Removing:', red(op[1] + ('/' if isdir(op[1]) else '')))

        elif op[0] == 'rename':
            pretty_print_operand(yellow, '(dry)Renaming:', yellow(op[1]))
            pretty_print_operand(yellow, '(dry)========>', yellow(op[2]))

        else:
            debug('(dry)', op)


def user_edit_stage(stage_now):
    while True:
        with tempfile.NamedTemporaryFile(prefix='vd', suffix='vd') as tf:

            # Write current stage into tempfile
            with open(tf.name, mode='w', encoding='utf8') as f:
                for idx, fpath in stage_now.items():
                    f.write('{idx}\t{dot_slash}{fpath}{trailing_slash}\n'.format(
                        idx=idx,
                        dot_slash='./' if not fpath.startswith(('/', './')) else '',
                        fpath=fpath,
                        trailing_slash='/' if isdir(fpath) else '',
                        ))
                f.flush()

            # Invoke vim to edit item list
            sub.call([
                'vim', tf.name,
                '+set nonu',
                '+set syntax=python',
                '+set tabstop=8'],
                stdin=open('/dev/tty')
                )

            # Parse tempfile and retrieve new stage content
            stage_new = collections.OrderedDict()
            has_parse_error = False
            with open(tf.name, mode='r', encoding='utf8') as f:
                for linenum, line in enumerate(f, start=1):
                    line = line.rstrip('\n')

                    if not line:
                        continue

                    rec = RegexCache(line)

                    if rec.match(r'^(#?\d+)\t(.*)$'):
                        idx, item = rec.group(1), rec.group(2)
                        if not item:
                            error('Line {}: file path cannot be empty'.format(linenum))
                            has_parse_error = True

                        else:
                            stage_new[idx] = relpath(item)

                    else:
                        error('Line {}, parsing error: {}'.format(linenum, red(line)))
                        has_parse_error = True

            if has_parse_error:
                user_confirm = prompt_confirm('Edit again?', ['edit', 'quit'])
                if user_confirm == 'edit':
                    continue

                exit(1)

        break

    return stage_new


class UserConfirm:
    def __init__(self, options):
        self.options = set(options)
        self.selected = None

    def __eq__(self, other):
        if other not in self.options:
            raise ValueError('Invalid option: ' + other)

        return self.selected == other


def prompt_confirm(prompt_text, options):
    options = [o.lower() for o in options]

    uc = UserConfirm(options)

    options_abbr_map = dict()
    options_abbr_map.update({o[0] : o for o in options})
    options_abbr_map.update({o[0].upper() : o for o in options})

    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    sys.stdin = open('/dev/tty')
    sys.stdout = open('/dev/tty', 'w')
    sys.stderr = open('/dev/tty', 'w')

    try:
        user_confirm = None

        options[0] = options[0][0].upper() + options[0][1:]

        while user_confirm is None:
            print(prompt_text + ' '
                    + '['
                    + ' / '.join('({}){}'.format(o[0], o[1:]) for o in options)
                    + ']', end=' ')
            user_confirm = input().strip().lower()

            if not user_confirm:
                user_confirm = options[0][0].lower()

            if user_confirm not in options_abbr_map:
                user_confirm = None
                continue

    except KeyboardInterrupt:
        print(black('KeyboardInterrupt'))
        exit(1)

    except EOFError:
        user_confirm = 'y'

    sys.stdin = stdin_backup
    sys.stdout = stdout_backup
    sys.stderr = stderr_backup

    uc.selected = options_abbr_map[user_confirm]

    return uc


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
            help='Perform a trial run with no changes made')

    parser.add_argument('-a', '--all', action='store_true',
            help='Include hidden files')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit, directories are expanded')

    args = parser.parse_args()

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('stdout and stderr must be tty')
        exit(1)

    # =========================================================================
    # Collect initial targets
    # =========================================================================
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If non-provided, '.' is expanded

    initial_targets = []

    for arg in args.targets:
        initial_targets.append((arg, True))

    if not sys.stdin.isatty():
        for line in sys.stdin:
            initial_targets.append((line.rstrip('\n'), False))

    if not initial_targets:
        initial_targets.append(('.', True))

    # =========================================================================
    # Construct the inventory
    # =========================================================================
    # The inventory contains the watching file/dir path list, that will be
    # maintained over iterations

    inventory = []
    has_error = False
    for target, expand in initial_targets:
        if not exists(target):
            error('File does not exist: [{}]'.format(target));
            has_error = True

        if has_error:
            continue

        if expand and isdir(target):
            inventory += [
                    relpath(join(target, i))
                    for i in sorted(os.listdir(target))
                    if args.all or not i.startswith('.')
                    ]

        else:
            inventory.append(relpath(target))

    if has_error:
        exit(1)

    if not inventory:
        print('No targets to edit')
        exit(0)

    inventory = [i[2:] if i.startswith('./') else i for i in inventory]

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

    stage_now = inventory_to_stage(inventory)
    stage_new = stage_now.copy()

    while True:
        stage_new = user_edit_stage(stage_new)

        op_list = diff_stages(stage_now, stage_new)

        if not op_list:
            info('No change')
            break

        print_op_list(op_list)

        user_confirm = prompt_confirm('Continue?', ('yes', 'edit', 'redo', 'quit'))
        if user_confirm == 'quit':
            break

        if user_confirm == 'edit':
            continue

        if user_confirm == 'redo':
            stage_new = stage_now
            continue

        apply_op_list(op_list)
        break


if __name__ == '__main__':
    exit(main())
