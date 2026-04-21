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

from . import logger

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
                    elif isinstance(item, (VDPath, VDGlob, VDLink)):
                        f.writeline(f'{item.text}')
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
        cmd.append('+normal ' + chr(0x7d))

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

                rec = rere(line)

                if rec.match(r'^([#+*@]?) *(\d+)\t+(.*)$'):
                    mark, iii, path = rec.groups()

                    if '*' in path:
                        path = VDGlob(path)
                    elif '->' in path:
                        a, b = path.split('->')
                        a = a.rstrip()
                        b = b.lstrip()
                        path = VDLink(a, b)
                    else:
                        path = VDPath(path)

                    new.append(path, iii=iii, mark=mark)

                elif line.startswith('#'):
                    continue

                else:
                    if '*' in line:
                        path = VDGlob(line)
                    elif '->' in line:
                        path = VDLink(line.split('->')[0].rstrip())
                    else:
                        path = VDPath(line)
                    new.append(path)

        new.freeze()

    return (step_collect_inventory_delta, base, new)


def step_collect_inventory_delta(base, new):
    logger.debug(FUNC_LINE())

    logger.debug(magenta('==== inventory (base) ===='))
    for item in base:
        logger.debug(item)
    logger.debug(magenta('-------------------------'))
    for item in new:
        logger.debug(item)
    logger.debug(magenta('==== inventory (new) ===='))

    delta_by_iii = {}
    delta_by_iii[None] = []

    # Put items from base inventory into item mapping
    for item in base:
        if not isinstance(item, TrackingItem):
            delta_by_iii[None].append(item)
        else:
            delta_by_iii[item.iii] = ItemChange(item)

    # Attach items from new inventory into item mapping
    for item in new:
        if item is None:
            continue

        if isinstance(item, (VDGlob, VDPath, VDLink)):
            delta_by_iii[None].append(item)
            continue

        if item.iii not in delta_by_iii:
            logger.errorq('{iii}  {text} {red}◄─ Invalid index{nocolor}'.format(
                iii=red(item.iii),
                text=item.text,
                red=red,
                nocolor=nocolor,
                ))
            continue

        delta_by_iii[item.iii].append(item)

    if logger.has_error():
        logger.errorflush()
        return (step_ask_fix_it, base, new)

    return (step_construct_raw_actions, base, new, delta_by_iii)


def step_construct_raw_actions(base, new, delta_by_iii):
    logger.debug()
    logger.debug(FUNC_LINE())
    logger.debug('Mapping')

    ticket_pool = TicketPool()

    # Index everything from base inventory pathlib.Path()
    for item in base:
        if not isinstance(item, TrackingItem):
            logger.errorq('base inventory should not have this:', item)
            continue
        ticket_pool.reserve(item)

    if logger.has_error():
        logger.errorflush()
        return (sys.exit, 1)

    # Index newly added paths as TrackAction
    for item in delta_by_iii[None]:
        ticket_pool.register(
                ('track', Path(item.txt)),
                TrackAction(item.txt))

    del delta_by_iii[None]

    # Index dst as raw Actions with help of delta_by_iii
    for iii, change in delta_by_iii.items():
        logger.debug('{iii} [{src}] => [{dsts}]'.format(
            iii=iii,
            src=repr(change.src),
            dsts=', '.join(repr(i) if not isinstance(i, TrackingItem) else repr(i) for i in change.dst)
            ))

        src = change.src

        if not change.dst:
            ticket_pool.register(
                    ('delete', src),
                    DeleteAction(src.txt))

        for dst in change.dst:
            if dst.mark not in '.#' and src.path != dst.path:
                logger.error('Conflict: path and mark changed at the same time:', dst.path)
                continue

            if dst.mark in '#*+@':
                action_cls, tag = {
                        '#': (UntrackAction, 'untrack'),
                        '*': (GlobAllAction, 'glob_all'),
                        '+': (GlobAction, 'glob'),
                        '@': (ResolveLinkAction, 'resolve'),
                        }.get(dst.mark)
                ticket_pool.register(
                        (tag, src),
                        action_cls(src.txt))

            elif src == dst:
                ticket_pool.register(
                        ('nop', src),
                        NoAction(src.txt))

            else:
                ticket_pool.register(
                        ('from', src),
                        ('to', dst),
                        CopyAction(src.txt, dst.txt))

    if logger.has_error():
        return (sys.exit, 1)

    if not ticket_pool:
        base_iii_order = [getattr(item, 'iii', 0) for item in base]
        new_iii_order = [getattr(item, 'iii', 0) for item in new]

        if sorted(base_iii_order) == sorted(new_iii_order) and base_iii_order != new_iii_order:
            return (step_vim_edit_inventory, new, new)

        else:
            logger.info('No change')
            return (sys.exit, 0)

    return (step_merge_actions, base, new, ticket_pool)


def step_merge_actions(base, new, ticket_pool):
    logger.debug()
    logger.debug(FUNC_LINE())

    def dump():
        logger.debug('delta by path:')
        for path, action_list in ticket_pool.by_path.items():
            logger.debug(f'{path}: {action_list}')
        logger.debug()
        logger.debug('ticket list:')
        for index, action in enumerate(ticket_pool.ticket_list):
            logger.debug(f'[{index}] {action}')

    # Multi-pass action merge

    # dump
    logger.debug(magenta('==== before merge ===='))
    dump()
    logger.debug(magenta('-------------------------'))

    # Pass 1, conflict check
    for path, actions in ticket_pool.by_path.items():
        if len(actions.get('to', [])) > 1:
            logger.errorq('Conflict: multiple copy/move into single destination')
            for ticket in actions['to']:
                logger.errorq(f'From: {ticket.action.src}')
            logger.errorq(f'To  : {path}')

        elif (len(actions.get('nop', [])) + len(actions.get('to', []))) > 1:
            logger.errorq('Conflict: override tracking item')
            for ticket in actions['to']:
                logger.errorq(f'From: {ticket.action.src}')
            logger.errorq(f'To  : {path}')

    if logger.has_error():
        logger.errorflush()
        return (step_ask_fix_it, base, new)

    # Pass 2, transform (CopyAction && !NoAction) into RenameAction
    for path, actions in ticket_pool.by_path.items():
        if 'from' in actions and 'nop' not in actions:
            for ticket in actions['from']:
                if isinstance(ticket.action, CopyAction):
                    ticket.action = RenameAction(ticket.action.src, ticket.action.dst)
                    break
    dump()
    logger.debug(magenta('---- pass 2 fin ---------'))

    # Pass 3, fuse contiguous RenameActions into Rotate RenameAction
    has_fuse = True
    while has_fuse:
        logger.debug()
        logger.debug('loop')
        has_fuse = False
        for ticket in ticket_pool.ticket_list:
            if not isinstance(ticket.action, RenameAction):
                continue

            start = ticket.participants[0]
            end = ticket.participants[-1]

            candidates = ticket_pool.by_path[end].get('from')
            if not candidates:
                continue

            if candidates[0] is not ticket and isinstance(candidates[0].action, RenameAction):
                fusee = candidates[0]

                if fusee.participants[-1] == start:
                    new_ticket = Ticket(
                            RotateRenameAction(*ticket.action.targets, *fusee.action.targets[1:-1]),
                            *ticket.participants, *fusee.participants[1:-1])
                    ticket_pool.replace(ticket, new_ticket)
                    ticket_pool.replace(fusee, new_ticket)

                else:
                    new_ticket = Ticket(
                            RenameAction(*ticket.action.targets, *fusee.action.targets[1:]),
                            *ticket.participants, *fusee.participants[1:])
                    ticket_pool.replace(ticket, new_ticket)
                    ticket_pool.replace(fusee, new_ticket)

                has_fuse = True

    # dump
    logger.debug(magenta('-------------------------'))
    dump()
    logger.debug(magenta('==== after merge ===='))

    if logger.has_error():
        logger.errorflush()
        return (step_ask_fix_it, base, new)

    return (step_confirm_action_list, base, new, ticket_pool)


def step_ask_fix_it(base, new):
    logger.debug()
    logger.debug(FUNC_LINE())

    logger.errorflush()
    logger.errorclear()

    user_confirm = prompt('Fix it?', ['edit', 'redo', 'quit'],
            allow_empty_input=False)

    if user_confirm == 'edit':
        return (step_vim_edit_inventory, base, new)

    if user_confirm == 'redo':
        return (step_vim_edit_inventory, base, base)

    return (sys.exit, 1)


def step_confirm_action_list(base, new, ticket_pool):
    logger.debug()
    logger.debug(FUNC_LINE())

    action_list = [ticket.action
                   for ticket in ticket_pool
                   if ticket.action is not None]

    if not action_list:
        logger.info('No change')
        return (sys.exit, 0)

    def action_sort_key(action):
        if isinstance(action, DeleteAction):
            return (1, action.targets[0])
        elif isinstance(action, CopyAction):
            return (2, action.targets[0])
        elif isinstance(action, RenameAction):
            return (3, action.targets[0])
        elif isinstance(action, UntrackAction):
            return (4, action.targets[0])
        elif isinstance(action, TrackAction):
            return (5, action.targets[0])
        return (99, action.targets[0])

    action_list = sorted(action_list, key=action_sort_key)

    for action in action_list:
        if hasattr(action, 'preview'):
            action.preview()
        else:
            logger.debug(repr(action))

    if all(isinstance(action, (TrackAction, UntrackAction)) for action in action_list):
        yes = True
    else:
        yes = False

    yn = prompt('Continue?', ['yes', 'no', 'edit', 'redo'], yes='' if yes else None)

    if yn == 'yes':
        return (step_apply_change_list, base, new, action_list, yn)

    if yn == 'edit':
        return (step_vim_edit_inventory, base, new)

    if yn == 'redo':
        return (step_vim_edit_inventory, base, base)

    if yn == 'no':
        return (sys.exit, 0)

    logger.error(FUNC_LINE())
    return (sys.exit, 1)


def step_apply_change_list(base, new, action_list, yn):
    logger.debug()
    logger.debug(FUNC_LINE())
    has_error = False

    for action in action_list:
        if has_error:
            logger.info(action)
            continue

        if hasattr(action, 'apply'):
            ret = action.apply()
            if ret is False:
                logger.error('Action failed')
                logger.info()
                logger.info('Skipped:')
                has_error = True

    if has_error:
        return (sys.exit, 1)

    return (step_expand_inventory, new, action_list, yn)


def step_expand_inventory(new, action_list, yn):
    logger.debug()
    logger.debug(FUNC_LINE())
    logger.debug(magenta('==== inventory ===='))
    for item in new:
        logger.debug(item)
    logger.debug(magenta('==== ========= ===='))

    has_meta = False
    for action in action_list:
        if isinstance(action, MetaAction):
            has_meta = True

    newnew = Inventory()
    for item in new:
        if isinstance(item, TrackingItem):
            if item.mark == '#':
                pass

            elif item.mark in ('*', '+'):
                for p in item.path.listdir(item.mark == '*'):
                    if not new.contains(p):
                        newnew.append(TrackingItem(None, p))

            elif item.mark == '@':
                if not new.contains(item.path.ref):
                    newnew.append(TrackingItem(None, item.path.ref))

            else:
                newnew.append(TrackingItem(None, item.path))

        elif isinstance(item, (VDPath, VDLink)):
            if not new.contains(item):
                newnew.append(TrackingItem(None, item))

        elif isinstance(item, VDGlob):
            logger.debug('expand', item)
            for p in item.glob():
                if not new.contains(p):
                    newnew.append(TrackingItem(None, p))

    newnew.freeze()

    logger.debug('has_meta', has_meta)
    if yn.selected == '' and has_meta == 0:
        return (sys.exit, 0)
    else:
        return (step_vim_edit_inventory, newnew, newnew)

    return (sys.exit, 0)


def edit_vd_vimrc():
    logger.debug()
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
        description='An (arguably) better vidir',
        epilog='\n'.join((
            'Examples:',
            magenta('$') + ' vd',
            magenta('$') + ' vd -a',
            magenta('$') + ' find . -type f | vd',
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
        sys.exit(1)

    if args.vimrc:
        sys.exit(edit_vd_vimrc())

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

    targets = natsorted(targets)

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
            logger.error(item)
            logger.error('File does not exist: ' + red('[') + item.text + red(']'))

    if logger.has_error():
        sys.exit(1)

    if not inventory:
        info('No targets to edit')
        sys.exit(0)

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

            if logger.has_error():
                logger.errorflush()
                sys.exit(1)

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
    sys.exit(main())
