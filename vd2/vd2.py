#!/usr/bin/env python3

# Mandatary
#TODO: Rethink about folder and files

# Vim related
#TODO: Refine syntax highlighting,
#TODO: Check if iii.type enum definition could be reused for vim and python
#TODO: Put "widgets" on the top area, move cursor onto them and press to activate

# Enhancement
#TODO: Define some commands like ":sort by date" ?
###==> ":uniq" ?


# =============================================================================
# Package Imports
# =============================================================================

import argparse
import datetime
import difflib
import functools
import glob
import inspect
import os
import os.path
import re
import readline
import shlex
import shutil
import subprocess as sub
import sys
import tempfile

from pathlib import Path

from . import regex
from . import logger

from .paints import *
from .utils import *
from .inventory import *
from .actions import *


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

RLB = red('[')
RRB = red(']')


def FUNC_LINE():
    cf = inspect.currentframe()
    bf = cf.f_back
    return '[{}:{}]'.format(bf.f_code.co_name, bf.f_lineno)

# -----------------------------------------------------------------------------
# Generalized Utilities }
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
        ret.append('# - Configure hotkeys in ' + shrinkuser(VD_VIMRC_PATH))
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
    logger.debug(FUNC_LINE())

    with tempfile.NamedTemporaryFile(prefix='vd', suffix='.vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            f.writelines(hint_text())
            f.writeline()

            if isinstance(inventory, Inventory):
                for item in inventory:
                    if item is None:
                        f.writeline()
                    elif item.iii is None:
                        f.writeline(f'{item.text}')
                    else:
                        f.writeline(f'{item.iii}\t{item.text}')
            else:
                for line in inventory:
                    f.writeline(f'{line}')

            f.flush()

        # Invoke vim to edit item list
        cmd = ['vim', tf.name]

        cmd.append('+set ft=vd')

        # Source vd.vimrc
        if os.path.isfile(VD_VIMRC_PATH):
            cmd += ['+source ' + str(VD_VIMRC_PATH)]

        cmd += ['+source ' + str(Path(__file__).parent / 'vimrc')]

        # Set proper tabstop for my (arguably) perfect vertical separation line
        if len(inventory):
            cmd.append('+set tabstop=' + str(len(str(inventory[0].iii)) + 4))

        # Move cursor to the line above first inventory item
        cmd.append('+normal }')

        logger.cmd(cmd, tag='debug')
        sub.call(cmd, stdin=open('/dev/tty'))
        print()

        # Parse tempfile content
        new = Inventory()
        with open(tf.name, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                if not line:
                    new.append(None)
                    continue

                rec = regex.rere(line)

                if rec.match(r'^([#+*@]?) *(\d+)\t+(.*)$'):
                    mark, iii, path = rec.groups()
                    new.append(path, iii=iii, mark=mark)

                elif line.startswith('#'):
                    continue

                else:
                    new.append(line)

        new.freeze()

    return (step_calculate_inventory_diff, base, new)


def step_calculate_inventory_diff(base, new):
    logger.debug(FUNC_LINE())

    logger.debug(magenta('==== inventory (base) ===='))
    for item in base:
        logger.debug(item)
    logger.debug(magenta('-------------------------'))
    for item in new:
        logger.debug(item)
    logger.debug(magenta('==== inventory (new) ===='))

    item_changes = dict()
    item_added = []

    # Put items from base inventory into item mapping
    for item in base:
        if not isinstance(item, TrackingItem):
            item_added.append(item)
            continue

        item_changes[item.iii] = ItemChange(item)

    # Attach items from new inventory into item mapping
    for item in new:
        if item is None:
            continue

        if not isinstance(item, TrackingItem):
            item_added.append(item)
            continue

        if item.iii not in item_changes:
            logger.errorq('{iii}  {text} {red}◄─ Invalid index{nocolor}'.format(
                iii=red(item.iii),
                text=item.text,
                red=red,
                nocolor=nocolor,
                ))
            continue

        item_changes[item.iii].append(item)

    return (step_compile_inventory_diff, base, new, item_changes, item_added)


def step_compile_inventory_diff(base, new, item_changes, item_added):
    logger.debug()
    logger.debug('Mapping')
    actions = []
    for iii, change in item_changes.items():
        if not change.changed:
            continue
        logger.debug('{iii} [{src}] => [{dsts}]'.format(
            iii=iii,
            src=change.src.text,
            dsts=", ".join(repr(i) if not isinstance(i, TrackingItem) else i.text for i in change.dst)
            ))

        if not change.dst:
            actions.append(DeleteAction(change.src.text))
            continue

        raw_actions = []
        for dst in change.dst:
            if dst.mark == '#':
                raw_actions.append(UntrackAction(change.src.text))
            elif dst.mark == '+':
                raw_actions.append(GlobAction(change.src.text))
            elif dst.mark == '*':
                raw_actions.append(GlobAllAction(change.src.text))
            elif dst.mark == '@':
                raw_actions.append(ResolveLinkAction(change.src.text))
            else:
                raw_actions.append(CopyAction(change.src.text, dst.text))

        if len(set(isinstance(action, MetaAction) for action in raw_actions)) != 1:
            logger.error('Meta changes could not be used with FS changes.')
            logger.error('Conflicted change:')
            logger.error('base:')
            logger.error(' ', change.src)
            logger.error('new:')
            for dst in change.dst:
                logger.error(' ', dst)
            logger.error()

        # merge fs actions
        contains = False
        fs_actions = [action for action in raw_actions if not isinstance(action, MetaAction)]
        for action in fs_actions:
            if action.src == action.dst:
                contains = True

        if contains:
            # All actions are CopyAction
            for action in fs_actions:
                if action.src != action.dst:
                    actions.append(action)
        else:
            for idx, action in enumerate(fs_actions):
                logger.debug(change.dst)
                if idx == len(fs_actions) - 1:
                    actions.append(RenameAction(action.src, action.dst))
                else:
                    actions.append(action)

    for item in item_added:
        logger.debug(f'track [{item.text}]')
        actions.append(TrackAction(item.text))

    logger.debug()
    for action in actions:
        logger.debug(action)

    if logger.has_error():
        logger.errorflush()
        return (exit, 1)

    if not actions:
        logger.info('No change')
        return (exit, 0)

    return (step_confirm_change_list, base, new, actions)


# def step_merge_actions(base, new, actions):
#     logger.debug()
#     logger.debug(FUNC_LINE())
#     for action in actions:
#         logger.debug(action)
#
#     if logger.has_error():
#         logger.errorflush()
#         return (step_ask_fix_it, base, new)
#
#     return (exit, 1)


def step_ask_fix_it(base, new):
    logger.debug(FUNC_LINE())

    logger.errorflush()

    user_confirm = prompt_confirm('Fix it?', ['edit', 'redo', 'quit'],
            allow_empty_input=False)

    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    if user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    return (exit, 1)


def step_check_change_list(base, new, change_list_raw):
    logger.debug(FUNC_LINE())
    return (exit, 0)


def step_confirm_change_list(base, new, change_list_raw):
    logger.debug(FUNC_LINE())

    user_confirm = prompt_confirm('Continue?', ['yes', 'edit', 'redo', 'quit'])

    if user_confirm == 'yes':
        return (step_apply_change_list, base, new, change_list_raw)

    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    if user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    if user_confirm == 'quit':
        return (exit, 0)

    logger.error(FUNC_LINE())
    return (exit, 1)


def step_apply_change_list(base, new, change_list):
    logger.debug(FUNC_LINE())
    for change in change_list:
        if hasattr(change, 'apply'):
            change.apply()

    return (exit, 0)


def step_expand_inventory(new):
    logger.debug(FUNC_LINE())
    return (exit, 0)


def edit_vd_vimrc():
    logger.debug(FUNC_LINE())

    VD_VIMRC_PATH.parent.mkdir(parents=True, exist_ok=True)

    if VD_VIMRC_PATH.exists() and not VD_VIMRC_PATH.is_file():
        logger.error(VD_VIMRC_PATH, 'exists and it\'s not a file')
        return 1

    print(VD_VIMRC_PATH)

    if VD_VIMRC_PATH.exists():
        return sub.call(['vim', VD_VIMRC_PATH])

    # Deploy vd vimrc if user didn't have one
    # Use tempfile so if user don't save the file, it won't exist
    with tempfile.NamedTemporaryFile() as tf:
        # Write it to a temp file first
        with open(tf.name, mode='w', encoding='utf8') as f:
            with open(Path(__file__).parent / 'vimrc') as vimrc:
                f.write(vimrc.read())

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
        logger.error('Both stdout and stderr must be tty')
        exit(1)

    if args.vimrc:
        exit(edit_vd_vimrc())

    # options.debug = args.debug
    options.debug = True
    logger.debug(FUNC_LINE(), options)

    # =========================================================================
    # Collect initial targets
    # -------------------------------------------------------------------------
    # Targets from commnad line arguments are expanded
    # Targets from stdin are not expanded
    # If none provided, '.' is expanded
    # -------------------------------------------------------------------------
    targets = []

    for target in args.targets:
        for i in VDPath(target).listdir(args.all):
            targets.append(i)

    targets = fsorted(targets)

    if not sys.stdin.isatty():
        for line in sys.stdin:
            targets.append(line.rstrip('\n'))

    if not targets:
        for i in VDPath('').listdir(args.all):
            targets.append(i)

    targets = uniq(targets)

    inventory = Inventory()

    for target in targets:
        if target:
            inventory.append(TrackingItem(None, target))

    for item in inventory:
        if not item.exists:
            logger.error('File does not exist:', item.text)

    if logger.has_error():
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
    # 4. Compare new/old stage content and generate action list
    # 5. Confirm with user
    # 5.q. if user say "q" (quit), quit
    # 5.e. if user say "e" (edit), invoke vim with new stage content
    # 5.r. if user say "r" (redo), invoke vim with old stage content
    # 5.y. if user say "y" (yes) or enter, apply the action list
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
            logger.errorclear()
            prev_call = (func, *args)
            next_call = func(*args)
        except TypeError as e:
            logger.errorq(e)
            logger.errorq(f'prev_call.func = {name(prev_call[0])}')
            logger.errorq(f'prev_call.args = (')
            for a in prev_call[1:]:
                logger.errorq(f'    {repr(a)}')
            logger.errorq(')')

            logger.errorq()
            logger.errorq(f'next_call.func = {name(next_call[0])}')
            logger.errorq(f'next_call.args = (')
            for a in next_call[1:]:
                logger.errorq(f'    {repr(a)}')
            logger.errorq(')')
            logger.errorflush()

            raise e


if __name__ == '__main__':
    exit(main())
