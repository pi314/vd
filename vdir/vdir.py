#!/usr/bin/env python3

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

from .version import __version__
from .vdpath import *
from .utils import *
from .inventory import *
from .actions import *


# =============================================================================
# Global variables {
# -----------------------------------------------------------------------------

options = argparse.Namespace(
        debug=False,
        )

VDIR_DEFAULT_VIMRC_PATH = Path(__file__).parent / 'vimrc.vdir'
VDIR_USER_VIMRC_BLUEPRINT_PATH = Path(__file__).parent / 'vimrc.user'
VDIR_USER_VIMRC_PATH = Path.home() / '.config' / 'vdir' / 'vdir.vimrc'

SEPLINE = ('═' * 77)

# -----------------------------------------------------------------------------
# Global variables }
# =============================================================================


# =============================================================================
# Generalized Utilities {
# -----------------------------------------------------------------------------

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

def hint_banner():
    sepline = '# ' + SEPLINE

    ret = [
            sepline,
            '# - Add paths to stage them. Globs are recognized.',
            "# - Add a '#' before id to untrack an item.",
            "# - Add a '+' before id to expand non-hidden items under the directory.",
            "# - Add a '*' before id to expand all items under the directory.",
            "# - Add a '@' before id to resolve the soft link.",
            '# - Stage items by shell command output: (globs are not supported here)',
            '#   $ find . -type f | grep py',
            '# - Sort with:',
            '#   :sort [-][type|isdir|isfile|isfifo|islink|path|basename|name|dirname|size|atime|mtime|ctime|birthtime] ...',
            ]

    if os.path.isfile(VDIR_USER_VIMRC_PATH):
        ret.append('# - Configure hotkeys in ' + shrinkuser(VDIR_USER_VIMRC_PATH))
    else:
        ret.append('# - Setup default vdir vimrc with:')
        ret.append('#   $ vdir --vimrc')

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
    logger.debug()
    logger.debug(FUNC_LINE())

    if not inventory:
        logger.info('No targets to edit')
        return (sys.exit, 0)

    if not isinstance(inventory, Inventory):
        logger.error(f'TypeError: {repr(inventory)}')
        return (sys.exit, 1)

    with tempfile.NamedTemporaryFile(prefix='vd', suffix='.vd') as tf:
        # Write inventory into tempfile
        with open(tf.name, mode='w', encoding='utf8') as f:
            f.writelines(hint_banner())
            f.writeline()

            for item in inventory:
                if item is None:
                    f.writeline()
                elif isinstance(item, (VDPath, VDGlob, VDLink, VDComment, VDShCmd)):
                    f.writeline(f'{item.text}')
                elif item.iii is None:
                    f.writeline(f'{item.text}')
                else:
                    f.writeline(f'{item.iii}\t{item.text}')

            f.flush()

        # Invoke vim to edit item list
        cmd = ['vim', tf.name]

        cmd.append('+set ft=vdir')

        # Source vdir user vimrc
        if os.path.isfile(VDIR_USER_VIMRC_PATH):
            cmd += ['+source ' + str(VDIR_USER_VIMRC_PATH)]

        # Source vdir default vimrc
        cmd += ['+source ' + str(VDIR_DEFAULT_VIMRC_PATH)]

        # Set proper tabstop for my (arguably) perfect vertical separation line
        for item in inventory:
            if not hasattr(item, 'iii'):
                continue
            cmd.append('+set tabstop=' + str(len(str(item.iii)) + 4))
            break

        # Move cursor to the line above first inventory item
        cmd.append('+normal ' + chr(0x7d))

        logger.cmd(cmd)
        sub.call(cmd, stdin=open('/dev/tty'))
        print()

        # Parse tempfile content
        new = Inventory()
        hint_banner_lines = set(hint_banner())
        with open(tf.name, mode='r', encoding='utf8') as f:
            in_banner = True
            for line in f.readlines():
                if line and not line.startswith('#'):
                    in_banner = False

                if in_banner and line in hint_banner_lines:
                    continue

                if not line:
                    new.append(None)
                    continue

                rec = rere(line)

                if in_banner and rec.fullmatch(r'# *' + SEPLINE[0] + '{4,}'):
                    new.clear()

                elif rec.fullmatch(r'([#+*@]?) *(\d+)\t+(.*)'):
                    mark, iii, path = rec.groups()
                    iii = int(iii, 10)

                    if '->' in path:
                        a, b = path.split('->')
                        a = a.rstrip()
                        b = b.lstrip()
                        path = VDLink(a, b)
                    else:
                        path = VDPath(path)

                    new.append(TrackingItem(iii, path, mark))

                elif rec.fullmatch(r'\$ +(.+)'):
                    new.append(VDShCmd(rec.group(1)))

                elif rec.fullmatch(r':sort( +.*)?'):
                    new.append(VDInvSortCmd(rec.group(1)))

                elif line.startswith('#'):
                    new.append(VDComment(line.lstrip('# ')))

                else:
                    if '*' in line:
                        path = VDGlob(line)
                    else:
                        path = VDPath(line)
                        if path.islink:
                            path = VDLink(line)

                    new.append(path)

        new.freeze()

    return (step_collect_inventory_delta, base, new)


def step_collect_inventory_delta(base, new):
    logger.debug(FUNC_LINE())

    logger.debug(magenta('==== inventory (base) ===='))
    for item in base:
        logger.debug(repr(item))
    logger.debug(magenta('-------------------------'))
    for item in new:
        logger.debug(repr(item))
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
        if item is None or isinstance(item, VDComment):
            continue

        if isinstance(item, (VDGlob, VDPath, VDLink, VDShCmd, VDInvSortCmd)):
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
        return (step_ask_fix_it, base, new)

    return (step_construct_raw_actions, base, new, delta_by_iii)


def step_construct_raw_actions(base, new, delta_by_iii):
    logger.debug()
    logger.debug(FUNC_LINE())

    ticket_pool = TicketPool()

    # Index everything from base inventory pathlib.Path()
    for item in base:
        if not isinstance(item, TrackingItem):
            continue
        ticket_pool.reserve(item)

    if logger.has_error():
        logger.errorflush()
        return (sys.exit, 1)

    # Register newly added items into ticket_pool
    for item in delta_by_iii[None]:
        if item is None:
            continue
        elif isinstance(item, VDComment):
            continue
        elif isinstance(item, VDInvSortCmd):
            ticket_pool.register(SortInventoryAction(item))
        elif isinstance(item, (VDGlob, VDShCmd)):
            ticket_pool.register(TrackAction(item))
        elif isinstance(item, VDPath):
            ticket_pool.register(
                    ('track', item.path),
                    TrackAction(item))
        elif isinstance(item, VDLink):
            ticket_pool.register(
                    ('track', item.path),
                    TrackAction(item.lnk))
        else:
            ticket_pool.register(
                    ('track', Path(item.path)),
                    TrackAction(item.path))

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
                    DeleteAction(src.path))

        for dst in change.dst:
            if dst.mark not in '.#' and src.path != dst.path:
                logger.error('Conflict: path and mark changed at the same time:', dst.path)
                continue

            elif type(src.path) != type(dst.path):
                logger.error('Conflict: Cannot change item type:')
                logger.error(f'{src}')
                logger.error(f'{dst}')
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
                        action_cls(src.path))

            elif src == dst:
                ticket_pool.register(
                        ('nop', src),
                        NoAction(src.path))

            elif isinstance(src.path, VDLink) and isinstance(dst.path, VDLink):
                if src.lnk != dst.lnk:
                    ticket_pool.register(
                            ('from', src),
                            ('to', dst),
                            CopyAction(src.path, dst.path))
                if src.ref != dst.ref:
                    ticket_pool.register(
                            ('relink', dst),
                            RelinkAction(VDLink(dst.lnk, src.ref), dst.ref))

            else:
                ticket_pool.register(
                        ('from', src),
                        ('to', dst),
                        CopyAction(src.path, dst.path))

    if logger.has_error():
        return (sys.exit, 1)

    if not ticket_pool:
        base_iii_order = [getattr(item, 'iii', 0) for item in base]
        new_iii_order = [getattr(item, 'iii', 0) for item in new]

        if sorted(base_iii_order) != sorted(new_iii_order) or base_iii_order != new_iii_order:
            yn = prompt('Next round', ['yes'], yes='yes')
            return (step_expand_inventory, new, [], yn)

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

    # Conflict check
    logger.debug(magenta('---- Conflict check -----'))
    for path, actions in ticket_pool.by_path.items():
        to_list = actions.get('to', [])
        from_list = actions.get('from', [])
        nop_list = actions.get('nop', [])

        if len(to_list) > 1:
            logger.errorq('Conflict: multiple copy/move into single destination')
            for ticket in to_list:
                logger.errorq(f'From: {ticket.action.src}')
            logger.errorq(f'To  : {path}')

        elif path.exists() and to_list and not from_list:
            problems = [ticket
                        for ticket in to_list
                        if ticket.action.src.inode != ticket.action.dst.inode
                        and not ticket_pool[ticket.action.dst].get('delete')]
            if problems:
                logger.errorq('Conflict: has risk overwrite existing file')
                for ticket in problems:
                    logger.errorq(f'From: {ticket.action.src}')
                logger.errorq(f'To: {path}')

        elif (len(nop_list) + len(to_list)) > 1:
            logger.errorq('Conflict: overwrite tracking item')
            for ticket in nop_list + to_list:
                logger.errorq(f'From: {ticket.action.src}')
            logger.errorq(f'To  : {path}')

    if logger.has_error():
        return (step_ask_fix_it, base, new)

    # Cancel DeleteAction if TrackAction exists
    logger.debug(magenta('---- Reduce DeleteAction ----'))
    for path, actions in ticket_pool.by_path.items():
        if 'delete' in actions and not path.exists():
            for ticket in actions['delete']:
                ticket.action = NoAction(ticket.action.src)
        elif 'delete' in actions and 'track' in actions:
            for ticket in actions['delete']:
                ticket.action = NoAction(ticket.action.src)
    dump()

    # Transform (CopyAction && !NoAction) into RenameAction
    logger.debug(magenta('---- Construct RenameAction ----'))
    for path, actions in ticket_pool.by_path.items():
        if 'from' in actions and 'nop' not in actions:
            for ticket in actions['from']:
                if isinstance(ticket.action, CopyAction):
                    ticket.action = RenameAction(ticket.action.src, ticket.action.dst)
                    break
    dump()

    # Transform CopyAction(*, .tar) into CompressAction
    # Transform CopyAction(.tar, *) into UncompressAction
    suffix_list = ('.tar',
                   '.tar.xz', '.xz',
                   '.tar.bz', '.tar.bz2', '.tbz', '.tbz2', '.bz', '.bz2',
                   '.tar.gz', '.gz', '.tgz',
                   '.tar.Z', '.Z',
                   '.zip', '.7z',
                   )
    logger.debug(magenta('---- Construct CompressAction/UncompressAction ----'))
    for ticket in ticket_pool.ticket_list:
        if isinstance(ticket.action, (CopyAction, RenameAction)):
            is_copy = isinstance(ticket.action, CopyAction)
            src = ticket.action.src
            dst = ticket.action.dst
            if src.name.endswith(suffix_list) and dst.name.endswith(suffix_list):
                pass
            elif dst.name.endswith(suffix_list):
                ticket.action = CompressAction(src, dst, keep=is_copy)
            elif src.name.endswith(suffix_list):
                ticket.action = UncompressAction(src, dst, keep=is_copy)
    dump()

    # Check src/dst isdir/isfile/isfifo/islink consistency
    logger.debug(magenta('---- Check src/dst type consistency ----'))
    def file_type(f):
        if f.isdir:
            return ls_colors('di')(' (dir)')
        elif f.isfile:
            return '(file)'
        elif f.islink:
            return ls_colors('ln')('(link)')
        elif f.isfifo:
            return ls_colors('pi')('(fifo)')
        return red('what')
    for ticket in ticket_pool:
        if isinstance(ticket.action, (CopyAction, RenameAction)):
            if ticket.action.dst.exists and file_type(ticket.action.src) != file_type(ticket.action.dst):
                logger.errorq()
                logger.errorq('Conflict: file type changed')
                logger.errorq(file_type(ticket.action.src), '│', ticket.action.src)
                logger.errorq(file_type(ticket.action.dst), '│', ticket.action.dst)
    dump()

    # Fuse contiguous RenameActions into Rotate RenameAction
    logger.debug(magenta('---- Fuse RenameAction ----'))
    has_fuse = True
    while has_fuse:
        logger.debug()
        logger.debug('fuse iteration')
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
        return (step_ask_fix_it, base, new)

    return (step_confirm_action_list, base, new, ticket_pool)


def step_ask_fix_it(base, new):
    logger.debug()
    logger.debug(FUNC_LINE())

    logger.errorflush()
    logger.errorclear()

    user_confirm = prompt('Fix it?', ['edit', 'redo', 'no', 'quit'],
            accept_empty=False)

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
                   if ticket.action is not None and
                   not isinstance(ticket.action, NoAction)]

    if not action_list:
        logger.info('No change')
        return (sys.exit, 0)

    def action_sort_key(action):
        if isinstance(action, DeleteAction):
            action_type = 1
        elif isinstance(action, CopyAction):
            action_type = 2
        elif isinstance(action, RenameAction):
            action_type = 3
        elif isinstance(action, UntrackAction):
            action_type = 4
        elif isinstance(action, TrackAction):
            action_type = 5
        elif isinstance(action, RelinkAction):
            action_type = 6
        else:
            action_type = 99

        tgt = action.targets[0]
        if isinstance(tgt, VDPath):
            tgt = 1
        elif isinstance(tgt, (VDGlob, VDShCmd)):
            tgt = 2
        else:
            tgt = 99

        return (action_type, tgt)

    action_list = sorted(action_list, key=action_sort_key)

    for action in action_list:
        if hasattr(action, 'preview'):
            action.preview()
        else:
            logger.debug(repr(action))

    if logger.has_error():
        return (step_ask_fix_it, base, new)

    if all(isinstance(action, (TrackAction, UntrackAction, GlobAction, GlobAllAction)) for action in action_list):
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
            logger.debug(action)
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
    logger.debug(magenta('==================='))

    has_inv_cmd = False
    for action in action_list:
        if isinstance(action, InvAction):
            has_inv_cmd = True

    newnew = Inventory()
    for item in new:
        if item is None or isinstance(item, VDComment):
            newnew.append(item)

        elif isinstance(item, TrackingItem):
            if item.mark == '#':
                pass

            elif item.mark in ('*', '+'):
                # GlobAction and GlobAllAction take effect here
                for p in item.path.listdir(item.mark == '*'):
                    if not new.contains(p) and not newnew.contains(p):
                        newnew.append(TrackingItem(None, p))

            elif item.mark == '@':
                # ResolveLinkAction takes effect here
                ref = item.path.lnk.resolve()
                if not new.contains(ref) and not newnew.contains(ref):
                    newnew.append(TrackingItem(None, ref))

            else:
                if not newnew.contains(item.path):
                    newnew.append(TrackingItem(None, item.path))

        elif isinstance(item, (VDPath, VDLink)):
            if not new.contains(item) and not newnew.contains(item):
                newnew.append(TrackingItem(None, item))

        elif isinstance(item, VDGlob):
            logger.debug('expand', item)
            for p in item.glob():
                if not new.contains(p) and not newnew.contains(p):
                    newnew.append(TrackingItem(None, p))

        elif isinstance(item, VDShCmd):
            returncode, ran_cmd, stdout, stderr = item.run()
            if returncode:
                for idx, cmd_str in enumerate(ran_cmd):
                    newnew.append(VDComment('{} {}'.format('$' if idx == 0 else ' |', cmd_str)))
                newnew.append(VDComment(f'returncode={returncode}'))
            for line in stderr:
                newnew.append(VDComment(line))
            for line in stdout:
                if not new.contains(line) and not newnew.contains(line):
                    newnew.append(TrackingItem(None, line))

        elif isinstance(item, VDInvSortCmd):
            newnew.sort(item)
            newnew.append(VDComment(':sort ' + item.text))

    newnew.freeze()

    logger.debug(magenta('==== inventory ===='))
    for item in newnew:
        logger.debug(item)
    logger.debug(magenta('==================='))

    logger.debug('has_inv_cmd', has_inv_cmd)
    if yn.selected == '' and has_inv_cmd == 0:
        return (sys.exit, 0)
    else:
        return (step_vim_edit_inventory, newnew, newnew)

    return (sys.exit, 0)


def edit_vd_vimrc():
    logger.debug()
    logger.debug(FUNC_LINE())

    # vdir --vimrc | something
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        print(VDIR_USER_VIMRC_PATH)
        sys.exit(0 if VDIR_USER_VIMRC_PATH.exists() else 1)

    if VDIR_USER_VIMRC_PATH.exists() and not VDIR_USER_VIMRC_PATH.is_file():
        logger.error(VDIR_USER_VIMRC_PATH, 'exists and it\'s not a file')
        return 1

    VDIR_USER_VIMRC_PATH.parent.mkdir(parents=True, exist_ok=True)

    ret = None
    if VDIR_USER_VIMRC_PATH.exists():
        ret = sub.call(['vim', VDIR_USER_VIMRC_PATH])
    else:
        # Deploy vd vimrc if user didn't have one
        # Use tempfile so if user don't save the file, it won't exist
        with tempfile.NamedTemporaryFile() as tf:
            # Write it to a temp file first
            with open(tf.name, mode='w', encoding='utf8') as f:
                with open(VDIR_USER_VIMRC_BLUEPRINT_PATH) as vimrc_user:
                    uuvuu = f'Generated by vdir {__version__}'
                    f.writelines([
                        '" ' + '=' * len(uuvuu) + ' "',
                        '" ' + uuvuu + ' "',
                        '" ' + '=' * len(uuvuu) + ' "',
                        ])
                    f.writeline()
                    f.write(vimrc_user.read())

            ret = sub.call([
                'vim', VDIR_USER_VIMRC_PATH,
                '+0read ' + tf.name, # Read the content into buffer
                '+$d_', # Remove the extra empty line with the black hole register
                '+normal gg', # Move cursor back to 1st line
                ])

    if VDIR_USER_VIMRC_PATH.exists():
        print(VDIR_USER_VIMRC_PATH)
    return ret


# -----------------------------------------------------------------------------
# "Step" functions }
# =============================================================================


# =============================================================================
# Main function
# =============================================================================

def main():
    logger.options = options

    parser = argparse.ArgumentParser(
        prog='vdir',
        description='An (arguably) better vidir',
        epilog='\n'.join((
            'Examples:',
            magenta('$') + ' vdir',
            magenta('$') + ' vdir -a',
            magenta('$') + ' find . -type f | vdir',
            )),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    parser.add_argument('-a', '--all', action='store_true',
            default=False,
            help='Include hidden paths')

    parser.add_argument('--vimrc', action='store_true',
            default=False,
            help='Edit or create the vimrc for vdir')

    parser.add_argument('--debug', action='store_true',
            default=False,
            help='Print debug messages')

    parser.add_argument('targets', nargs='*',
            help='Paths to edit. Directories are expanded')

    args = parser.parse_args()

    if args.vimrc:
        sys.exit(edit_vd_vimrc())

    if not sys.stdout.isatty() or not sys.stderr.isatty():
        logger.error('Both stdout and stderr must be tty')
        sys.exit(1)

    options.debug = args.debug
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
        targets.extend(line.rstrip('\n') for line in sys.stdin)

    if not targets:
        targets.extend(VDPath('').listdir(args.all))

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
        logger.info('No targets to edit')
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

    # for key, value in ls_colors().items():
    #     print(key, repr(value), value('test'))
    # sys.exit(1)
    prev_call = None
    next_call = (step_vim_edit_inventory, inventory, inventory)
    while next_call:
        func, *args = next_call
        try:
            logger.errorclear()
            prev_call = (func, *args)
            next_call = func(*args)

            # if logger.has_error():
            #     logger.errorflush()
            #     sys.exit(1)

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
