#!/usr/bin/env python3

#TODO: Set default mapping for J/K to move entry downward/upward
#TODO: vim default config
#TODO: If user change dir to .tar (by removing the trailing slash), tar it

__version__ = '0.0.0'

import os
import re
import subprocess as sub
import sys
import tempfile
import collections

from os.path import basename, join, exists, isdir


def color_to(color_code):
    def colorer(s):
        if not s:
            return ''
        return '\033[3{}m{}\033[m'.format(color_code, s)
    return colorer

red = color_to(1)
green = color_to(2)
yellow = color_to(3)
blue = color_to(4)
magenta = color_to(5)
cyan = color_to(6)
white = color_to(7)
nocolor = lambda s: '\033[m' + s


def print_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def debug(*args, **kwargs):
    print_stderr(magenta('[Debug]'), *args, **kwargs)


def error(*args, **kwargs):
    print_stderr(red('[Error]'), *args, **kwargs)


def usage(prog):
    print_stderr('Usage:')
    print_stderr()
    print_stderr('    $ {} [-h|--help]'.format(prog))
    print_stderr('    $ {}'.format(prog))
    print_stderr('    $ {} files...'.format(prog))
    print_stderr('    $ ls -1 | {}'.format(prog))
    exit(1)


def inventory_to_stage(inventory):
    stage = collections.OrderedDict()

    for idx, fpath in enumerate(inventory, start=10**len(str(len(inventory)))):
        stage[idx] = fpath.rstrip('/')

    return stage


def stage_to_inventory(stage):
    return list(stage.values())


def diff_stages(stage_current, stage_new):
    return set()


def prompt_to_vim(stage_current):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', prefix='vd', suffix='vd') as tf:
        # Write current content into tempfile
        for idx, fpath in stage_current.items():
            tf.write('{idx}\t{dot_slash}{fpath}{trailing_slash}\n'.format(
                idx=idx,
                dot_slash='./' if not fpath.startswith(('/', './')) else '',
                fpath=fpath,
                trailing_slash='/' if isdir(fpath) else '',
                ))
        tf.flush()

        # Invoke vim to edit item list
        sub.call(['vim', tf.name, '+set nonu', '+set syntax=python', '+set tabstop=8'], stdin=open('/dev/tty'))

        stage_new = collections.OrderedDict()
        with open(tf.name, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip('\n')
                m = re.match(r'^(#?)(\d+)\t(.*)$', line)
                #TODO: handle incorrect formatted lines
                idx, item = m.group(2), m.group(3)
                stage_new[idx] = item.rstrip('/')

        return stage_new


def prompt_confirm():
    stdin_backup = sys.stdin
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr

    sys.stdin = open('/dev/tty')
    sys.stdout = open('/dev/tty', 'w')
    sys.stderr = open('/dev/tty', 'w')

    try:
        user_confirm = None

        while user_confirm is None:
            print_stderr('Continue? [(Y)es / (e)dit / (r)edo / (q)uit]', end=' ')
            user_confirm = input().strip().lower()

            if not user_confirm:
                user_confirm = 'y'

            if user_confirm not in 'yerq':
                user_confirm = None
                continue

    except KeyboardInterrupt:
        print_stderr('KeyboardInterrupt')
        exit(1)

    except EOFError:
        print_stderr('Y')
        user_confirm = 'y'

    sys.stdin = stdin_backup
    sys.stdout = stdout_backup
    sys.stderr = stderr_backup

    return user_confirm


def main():
    prog = basename(sys.argv[0])
    argv = sys.argv[1:]

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        error('stdout and stderr must be tty')
        exit(1)

    if '-h' in argv or '--help' in argv:
        usage(prog)

    initial_targets = collections.OrderedDict()

    # =========================================================================
    # Collect initial targets and construct the inventory
    # =========================================================================
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If non-provided, '.' is expanded
    #
    # The inventory contains the watching file/dir list without trailing slash

    for arg in argv:
        initial_targets[arg] = True

    if not sys.stdin.isatty():
        for line in sys.stdin:
            initial_targets[line.rstrip('\n')] = False

    if not initial_targets:
        initial_targets['.'] = True

    inventory = []
    has_error = False
    for target, expand in initial_targets.items():
        if not exists(target):
            error('File does not exist: [{}]'.format(target));
            has_error = True

        if has_error:
            continue

        if expand and isdir(target):
            inventory += [join(target, i) for i in sorted(os.listdir(target))]

        else:
            inventory.append(target.rstrip('/'))

    if has_error:
        exit(1)

    # =========================================================================
    # Main loop
    # =========================================================================
    # 1. Construct the stage: inventory => (seq num, tab, ./file/path/)
    # 2. Invoke vim with current stage content
    # 3. Parse and get new stage content
    # 4. Compare new/old stage content and generate OP list
    # 5. Confirm with user
    # 5.y. if user say "y" (yes) or enter,
    # 5.e. if user say "e" (edit), invoke vim with new stage content
    # 5.r. if user say "r" (redo), invoke vim with old stage content
    # 5.q. if user say "q" (quit), quit
    # 5.*. keep asking until recognized option is selected or Ctrl-C is pressed
    # 6. Apply the OP list

    while True:
        stage_current = inventory_to_stage(inventory)

        stage_new = prompt_to_vim(stage_current)

        inventory_new = stage_to_inventory(stage_new)

        #TODO: diff them and generate op list
        print('==')
        for idx in sorted(stage_new.keys()):
            print('(', idx, stage_new[idx], ')')

        # Just for testing
        if (not inventory_new) or (inventory == inventory_new):
            user_confirm = prompt_confirm()
            if user_confirm in 'yq':
                break

        inventory = inventory_new


if __name__ == '__main__':
    exit(main())
