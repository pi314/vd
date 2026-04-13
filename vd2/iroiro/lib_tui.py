import sys
import builtins

from collections import UserList, UserDict

from .lib_threading import Lock
from .lib_itertools import zip_longest, is_iterable
from .lib_lang import getter, setter
from .lib_lang import ResourceError, SignatureError, AlreadyRunningError

from .internal_utils import exporter
export, __all__ = exporter()


def builtin_print(*args, **kwargs): # pragma: no cover
    kwargs['file'] = sys.stdout
    builtins.print(*args, **kwargs)

builtin_flush = sys.stdout.flush

builtin_input = input


tui_print = builtin_print
tui_flush = builtin_flush
tui_input = builtin_input


@export
def charwidth(c):
    if not c.isprintable():
        return 0
    import unicodedata
    return 1 + (unicodedata.east_asian_width(c) in 'WF')


@export
def strwidth(s):
    from .lib_colors import decolor
    return sum(charwidth(c) for c in decolor(s))


@export
def wrap(s, width, clip=None):
    if clip is None:
        pass
    elif not isinstance(clip, str) or (strwidth(clip) != 1):
        raise ValueError('clip should be a single width string')

    acc = ''
    def accumulate(char):
        nonlocal acc
        if not acc:
            if char != '\033':
                return (char, charwidth(char))
            acc = char
            return (None, None)

        elif acc == '\033':
            acc += char
            return (None, None)

        else:
            acc += char
            if ((not acc.startswith('\033[')) or
                    (char not in '0123456789;')):
                ret, acc = acc, ''
                return (ret, 0)
        return (None, None)

    aw = 0
    to = 0
    pending = ''
    for idx, char in enumerate(s):
        char, cw = accumulate(char)
        if char is None:
            continue

        if cw == 0 and char not in ('\033[m', '\033[0m'):
            pending += char
            continue

        if aw + cw > width:
            if clip and aw + 1 <= width:
                return (s[:to] + clip, s[to:])
            return (s[:to], s[to:])
        aw += cw
        to = idx + 1

    if aw == width:
        return (s[:to], s[to:])
    else:
        return (s, '')


def lpad(text, padding):
    return text + padding


def rpad(text, padding):
    return padding + text


def just_elem(func):
    def wrapper(elem, width, fillchar):
        row, col, text = elem
        padding = (width - strwidth(text)) * fillchar(row=row, col=col, text=text)
        return func(text, padding)
    return wrapper


def just_generator(just_func, data, width, fillchar):
    for row, vector in enumerate(data):
        if isinstance(width, int):
            width = (width,) * len(vector)
        yield tuple(
                just_func((row, col, text), w, fillchar)
                for col, (text, w) in enumerate(zip_longest(vector, width[:len(vector)], fillvalues=('', 0)))
                )


def just(just_func, data, width, fillchar):
    if not callable(fillchar):
        _fillchar = fillchar
        fillchar = lambda row, col, text: _fillchar

    if isinstance(data, str):
        return just_func((0, 0, data), width, fillchar)

    if width:
        if isinstance(data, (tuple, list)):
            t = type(data)
        else:
            t = lambda x: x
        return t(just_generator(just_func, data, width, fillchar))

    maxwidth = []
    for vector in data:
        maxwidth = [
                max(w, strwidth(text))
                for text, w in zip_longest(vector, maxwidth, fillvalues=('', 0))
                ]

    return [
            tuple(
                just_func((row, col, text), w, fillchar)
                for col, (text, w) in enumerate(zip_longest(vector, maxwidth, fillvalues=('', 0)))
                )
            for row, vector in enumerate(data)
            ]


@export
def ljust(data, width=None, fillchar=' '):
    return just(just_elem(lpad), data, width, fillchar)


@export
def rjust(data, width=None, fillchar=' '):
    return just(just_elem(rpad), data, width, fillchar)


@export
class ThreadedSpinner:
    def __init__(self, *icon, delay=0.1):
        if not icon:
            self.icon_entry = '⠉⠛⠿⣿⠿⠛⠉⠙'
            self.icon_loop = '⠹⢸⣰⣤⣆⡇⠏⠛'
            self.icon_leave = '⣿'
        elif len(icon) == 1:
            self.icon_entry = tuple()
            self.icon_loop = icon
            self.icon_leave = '.'
        elif len(icon) == 2:
            self.icon_entry = icon[0]
            self.icon_loop = icon[1]
            self.icon_leave = '.'
        elif len(icon) == 3:
            self.icon_entry = icon[0]
            self.icon_loop = icon[1]
            self.icon_leave = icon[2]
        else:
            raise ValueError('Invalid value: ' + repr(icon))

        ok = True
        for name, icon in (('entry', self.icon_entry), ('loop', self.icon_loop), ('leave', self.icon_leave)):
            if isinstance(icon, str):
                ok = True
            elif isinstance(icon, (tuple, list)) and all(isinstance(c, str) for c in icon):
                ok = True
            else:
                raise ValueError('Invalid value of icon[{}]: {}'.format(name, icon))

        self.delay = delay
        self.is_end = False
        self.thread = None
        self._text = ''

        import itertools
        self.icon_iter = (
                itertools.chain(
                    self.icon_entry,
                    itertools.cycle(self.icon_loop)
                    ),
                iter(self.icon_leave)
                )
        self.icon_head = [None, None]

    def __enter__(self):
        if self.thread:
            return self

        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    @property
    def icon(self):
        idx = self.is_end
        if self.icon_head[idx] is None:
            self.icon_head[idx] = next(self.icon_iter[idx])
        return self.icon_head[idx]

    def text(self, *args):
        if not args:
            return self._text

        self._text = ' '.join(str(a) for a in args)
        if self.thread:
            self.refresh()

    def refresh(self):
        tui_print('\r' + self.icon + '\033[K ' + self._text, end='')
        tui_flush()

    def animate(self):
        import time

        while not self.is_end:
            self.refresh()
            time.sleep(self.delay)
            self.icon_head[0] = next(self.icon_iter[0])

        try:
            while True:
                self.refresh()
                self.icon_head[1] = next(self.icon_iter[1])
                time.sleep(self.delay)
        except StopIteration:
            pass

        tui_print()
        tui_flush()

    def start(self):
        if self.thread:
            return

        import threading
        self.thread = threading.Thread(target=self.animate)
        self.thread.daemon = True
        self.thread.start()

    def end(self, wait=True):
        self.is_end = True
        if wait:
            self.join()

    def join(self):
        self.thread.join()


def alt_if_none(A, B):
    if A is None:
        return B
    return A


class UserSelection:
    def __init__(self, options, accept_empty=None, abbr=None, sep=None, ignorecase=None):
        if not options:
            accept_empty = True
            abbr = False
            ignorecase = False

        self.accept_empty = alt_if_none(accept_empty, True)
        self.abbr = alt_if_none(abbr, True)
        self.ignorecase = alt_if_none(ignorecase, self.abbr)
        self.sep = alt_if_none(sep, ' / ')

        self.mapping = dict()
        self.options = [o for o in options]

        if self.options:
            if self.accept_empty:
                self.mapping[''] = self.options[0]

            for opt in self.options:
                for o in (opt,) + ((opt[0],) if self.abbr else tuple()):
                    self.mapping[o.lower() if self.ignorecase else o] = opt

        self.selected = None

    def select(self, o=''):
        if self.ignorecase:
            o = o.lower()

        if not self.options:
            self.selected = o
            return

        if o not in self.mapping:
            raise ValueError('Invalid option: ' + o)

        self.selected = o

    @property
    def prompt(self):
        if not self.options:
            return ''

        opts = [o for o in self.options]
        if self.accept_empty and self.ignorecase:
            opts[0] = opts[0].capitalize()

        if self.abbr:
            return ' [' + self.sep.join('({}){}'.format(o[0], o[1:]) for o in opts) + ']'
        else:
            return ' [' + self.sep.join(opts) + ']'

    def __eq__(self, other):
        if self.ignorecase and other is not None:
            other = other.lower()

        if self.selected == other:
            return True

        if self.selected in self.mapping:
            return self.mapping[self.selected] == self.mapping.get(other)

        return False

    def __str__(self):
        return str(self.selected)

    def __repr__(self):
        return '<iroiro.tui.UserSelection selected=[{}]>'.format(self.selected)


class HijackStdio:
    def __init__(self, replace_with='/dev/tty'):
        self.replace_with = replace_with

    def __enter__(self):
        self.stdin_backup = sys.stdin
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr

        sys.stdin = open(self.replace_with)
        sys.stdout = open(self.replace_with, 'w')
        sys.stderr = open(self.replace_with, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdin.close()
        sys.stdout.close()
        sys.stderr.close()

        sys.stdin = self.stdin_backup
        sys.stdout = self.stdout_backup
        sys.stderr = self.stderr_backup


class ExceptionSuppressor:
    def __init__(self, *exc_group):
        if isinstance(exc_group[0], tuple):
            self.exc_group = exc_group[0]
        else:
            self.exc_group = exc_group

    def __enter__(self, *exc_group):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type in (EOFError, KeyboardInterrupt):
            tui_print()
        return exc_type in self.exc_group


@export
def prompt(question, options=tuple(),
           accept_empty=True,
           abbr=True,
           ignorecase=None,
           sep=' / ',
           suppress=(EOFError, KeyboardInterrupt)):

    if isinstance(options, str) and ' ' in options:
        options = options.split()

    user_selection = UserSelection(options, accept_empty=accept_empty, abbr=abbr, sep=sep, ignorecase=ignorecase)

    with HijackStdio():
        with ExceptionSuppressor(suppress):
            while user_selection.selected is None:
                tui_print((question + (user_selection.prompt)), end=' ')

                import contextlib
                with contextlib.suppress(ValueError):
                    i = tui_input().strip()
                    user_selection.select(i)

    return user_selection


class Key:
    def __init__(self, seq, *aliases):
        if isinstance(seq, str):
            seq = seq.encode('utf8')

        if not isinstance(seq, bytes):
            raise TypeError('seq should be in type bytes, not {}'.format(type(seq)))

        if not all(isinstance(a, str) for a in aliases):
            raise TypeError('Aliases should be in type str')

        self.seq = seq
        self.aliases = []
        for name in aliases:
            self.nameit(str(name))

    def __hash__(self):
        return hash(self.seq)

    def __repr__(self):
        fmt = type(self).__name__ + '({})'
        if self.aliases:
            return fmt.format(self.aliases[0])
        try:
            return fmt.format(repr(self.seq.decode('utf8')))
        except UnicodeError:
            return fmt.format(repr(self.seq))

    def nameit(self, name):
        if name not in self.aliases:
            self.aliases.append(name)

    def __eq__(self, other):
        if type(self) == type(other):
            return self.seq == other.seq
        elif isinstance(other, bytes) and self.seq == other:
            return True
        elif isinstance(other, str) and self.seq == other.encode('utf8'):
            return True
        else:
            return other in self.aliases


KEY_ESCAPE = Key(b'\033', 'esc', 'escape')
KEY_BACKSPACE = Key(b'\x7f', 'backspace')
KEY_TAB = Key(b'\t', 'tab', 'ctrl-i', 'ctrl+i', '^I')
KEY_ENTER = Key(b'\r', 'enter', 'ctrl-m', 'ctrl+m', '^M')
KEY_SPACE = Key(b' ', 'space')

KEY_FS = Key(b'\x1c', 'fs', 'ctrl-\\', 'ctrl+\\', '^\\')

KEY_UP = Key(b'\033[A', 'up')
KEY_DOWN = Key(b'\033[B', 'down')
KEY_RIGHT = Key(b'\033[C', 'right')
KEY_LEFT = Key(b'\033[D', 'left')

KEY_HOME = Key(b'\033[1~', 'home')
KEY_END = Key(b'\033[4~', 'end')
KEY_PGUP = Key(b'\033[5~', 'pgup', 'pageup')
KEY_PGDN = Key(b'\033[6~', 'pgdn', 'pagedown')

KEY_F1 = Key(b'\033OP', 'F1')
KEY_F2 = Key(b'\033OQ', 'F2')
KEY_F3 = Key(b'\033OR', 'F3')
KEY_F4 = Key(b'\033OS', 'F4')
KEY_F5 = Key(b'\033[15~', 'F5')
KEY_F6 = Key(b'\033[17~', 'F6')
KEY_F7 = Key(b'\033[18~', 'F7')
KEY_F8 = Key(b'\033[19~', 'F8')
KEY_F9 = Key(b'\033[20~', 'F9')
KEY_F10 = Key(b'\033[21~', 'F10')
KEY_F11 = Key(b'\033[23~', 'F11')
KEY_F12 = Key(b'\033[24~', 'F12')

def _register_ctrl_n_keys():
    for c in 'abcdefghjklnopqrstuvwxyz':
        C = c.upper()
        idx = ord(c) - ord('a') + 1
        aliases = ('ctrl-' + c, 'ctrl+' + c, '^' + C)
        globals()['KEY_CTRL_' + C] = Key(chr(idx), *aliases)

_register_ctrl_n_keys()
del _register_ctrl_n_keys


def _export_all_keys():
    for key in globals().keys():
        if key.startswith('KEY_'):
            export(key)

_export_all_keys()
del _export_all_keys


key_seq_table = {}
key_alias_table = {}

def _init_key_table():
    for k, v in globals().items():
        if not k.startswith('KEY_'):
            continue
        key_seq_table[v.seq] = v

        for alias in v.aliases:
            key_alias_table[alias] = v

_init_key_table()
del _init_key_table


@export
def register_key(seq, *aliases):
    if isinstance(seq, Key):
        new_key = seq
        seq = new_key.seq
        aliases = new_key.aliases + list(aliases)

    elif isinstance(seq, str):
        seq = seq.encode('utf8')

    if not seq:
        raise ValueError('huh?')

    if seq not in key_seq_table:
        key_seq_table[seq] = Key(seq, *aliases)
        return key_seq_table[seq]

    key = key_seq_table[seq]
    for name in aliases:
        key.nameit(name)

    return key


@export
def deregister_key(seq):
    if isinstance(seq, Key):
        seq = seq.seq
    elif isinstance(seq, str):
        seq = seq.encode('utf8')

    key = key_seq_table.pop(seq, None)
    for alias in key.aliases:
        key_alias_table.pop(alias, None)
    return key


_getch_lock = Lock()

@export
def getch(*, timeout=None, encoding='utf8', capture=('ctrl+c', 'ctrl+z', 'fs')):
    import termios, tty
    import os
    import select
    import signal

    with _getch_lock.acquire(blocking=False) as locked:
        if not locked:
            raise ResourceError('Simultaneous getch() calls are not allowed')

        fd = sys.stdin.fileno()
        orig_term_attr = termios.tcgetattr(fd)
        when = termios.TCSADRAIN

        term_attr_cc = termios.tcgetattr(fd)[6]

        capture_table = [
                [KEY_CTRL_C, term_attr_cc[termios.VINTR], signal.SIGINT],
                [KEY_CTRL_Z, term_attr_cc[termios.VSUSP], signal.SIGTSTP],
                [KEY_FS,     term_attr_cc[termios.VQUIT], signal.SIGQUIT],
                ]

        if isinstance(capture, str):
            capture = [capture]

        for cap in capture or []:
            for entry in capture_table:
                if entry[0] == cap:
                    entry[2] = None

        def has_data(t=0):
            return select.select([fd], [], [], t)[0]

        def read_one_byte():
            return os.read(sys.stdin.fileno(), 1)

        try:
            tty.setraw(fd, when=when)

            # Wait for input until timeout
            if not has_data(timeout):
                return None

            acc = b''
            candidate_matches = set(key_seq_table.keys())
            while True:
                acc += read_one_byte()

                # Check special sequences that correspond to signals
                for entry in capture_table:
                    key, seq, sig = entry
                    if acc[-len(seq):] == seq:
                        if sig is not None:
                            os.kill(os.getpid(), sig)
                        else:
                            break

                if not has_data():
                    break

                # Still have chance to match in key table
                if candidate_matches:
                    # eliminate potential matches
                    candidate_matches = set(key_seq for key_seq in candidate_matches if key_seq.startswith(acc))

                    # Perfect match, return
                    if candidate_matches == {acc}:
                        break

                    # multiple prefix matchs: collect more byte
                    if candidate_matches:
                        continue

                # Input sequence does not match anything in key table
                # Collect enough bytes to decode at least one unicode char
                try:
                    acc.decode(encoding)
                    break
                except UnicodeError:
                    continue

            if acc in key_seq_table:
                return key_seq_table[acc]

            try:
                return acc.decode(encoding)
            except UnicodeError:
                return acc

        finally:
            termios.tcsetattr(fd, when, orig_term_attr)


class Pagee:
    def __init__(self, text, section, offset, visible):
        self.text = str(text)
        self.section = section
        self.offset = offset
        self.visible = visible


class Subpager:
    def __init__(self, parent, section):
        self.lines = []
        self.parent = parent
        self.section = section

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        for pagee in self.parent.data:
            if pagee.section == self.section:
                yield pagee

    def __getitem__(self, idx):
        return self.lines[idx]

    def __setitem__(self, idx, line):
        self.lines[idx] = line

    @property
    def empty(self):
        return not self.lines

    def append(self, line=''):
        self.lines.append(line)

    def extend(self, lines=[]):
        for line in lines:
            self.append(line)

    def insert(self, index, line):
        return self.lines.insert(index, line)

    def pop(self, index=-1):
        return self.lines.pop(index)

    def clear(self):
        self.lines.clear()


@export
class Pager:
    def __init__(self, max_height=None, max_width=None, flex=False):
        self._max_height = max_height
        self._max_width = max_width
        self.flex = flex

        self.header = Subpager(parent=self, section='header')
        self.body = Subpager(parent=self, section='body')
        self.footer = Subpager(parent=self, section='footer')

        self.reset()

    @property
    def term_size(self):
        import shutil
        return shutil.get_terminal_size()

    @property
    def term_height(self):
        return self.term_size.lines

    @property
    def term_width(self):
        return self.term_size.columns

    @getter
    def max_height(self):
        return self._max_height

    @setter
    def max_height(self, value):
        self._max_height = max(value or 0, 0)

    @getter
    def max_width(self):
        return self._max_width

    @setter
    def max_width(self, value):
        self._max_width = max(value, 0)

    @property
    def height(self):
        if self.flex and self.max_height:
            content_total_height = self.max_height
        else:
            content_total_height = len(self.header) + len(self.body) + len(self.footer)

        return min(
                self.max_height or self.term_height,
                self.term_height,
                content_total_height,
                )

    @property
    def width(self):
        return min(self.max_width or self.term_width, self.term_width)

    def __len__(self):
        return len(self.body)

    def __iter__(self):
        return iter(self.body)

    def __getitem__(self, idx):
        content_height = max(0, self.height - len(self.header) - len(self.footer))
        return Pagee(
                text=self.body[idx],
                section='body',
                offset=len(self.header) - self.scroll,
                visible=(self.scroll) <= idx <= (self.scroll + content_height - 1)
                )

    def __setitem__(self, idx, line):
        if isinstance(idx, slice):
            start = idx.start or 0
        else:
            start = idx

        for i in range(len(self), start + 1):
            self.append()

        self.body[idx] = line

    @property
    def data(self):
        from .lib_collections import namablelist
        alloc = namablelist(header=0, body=0, padding=0, footer=0)

        for i in range(self.height):
            if not self.header.empty and alloc.header == 0:
                section = 'header'
            elif not self.footer.empty and alloc.footer == 0:
                section = 'footer'
            elif alloc.header < len(self.header):
                section = 'header'
            elif alloc.footer < len(self.footer):
                section = 'footer'
            elif alloc.body < len(self.body):
                section = 'body'
            else:
                section = 'padding'

            alloc[section] += 1

        at_line = 0
        for section, lines, base in [
                ('header',  self.header.lines,    0),
                ('body',    self.body.lines,      len(self.header) - self.scroll),
                ('padding', [''] * alloc.padding, alloc.header + alloc.body),
                ('footer',  self.footer.lines,    alloc.header + alloc.body + alloc.padding),
                ]:
            for idx, line in enumerate(lines):
                pagee = Pagee(text=line,
                            section=section,
                            offset=idx + base,
                            visible=idx + base >= at_line and getattr(alloc, section) > 0,)
                yield pagee
                if pagee.visible:
                    alloc[section] -= 1
                    at_line += 1

    @property
    def lines(self):
        return tuple(item.text for item in self.data)

    @property
    def preview(self):
        return tuple(item.text for item in self.data if item.visible)

    @property
    def display(self):
        return tuple(self._display)

    @property
    def empty(self):
        for line in self.lines:
            return False
        return True

    def append(self, line=''):
        self.body.append(line)

    def extend(self, lines=[]):
        for line in lines:
            self.append(line)

    def insert(self, index, line):
        return self.body.insert(index, line)

    def pop(self, index=-1):
        return self.body.pop(index)

    def clear(self):
        self.header.clear()
        self.body.clear()
        self.footer.clear()

    def reset(self):
        self._scroll = 0
        self._display = []
        self.clear()

    @property
    def home(self):
        return 0

    @property
    def end(self):
        return len(self.body) - 1

    @getter
    def scroll(self):
        self.scroll = self._scroll
        return self._scroll

    @setter
    def scroll(self, value):
        self._scroll = value

        content_height = max(0, self.height - len(self.header) - len(self.footer))
        from .lib_math import clamp
        self._scroll = clamp(0, self._scroll, max(0, len(self.body)-content_height))

    def render(self, *, all=None):
        # Skip out-of-screen lines, i.e. canvas size-- if terminal size--
        self._display = self._display[-self.term_height:] or [None]

        visible_lines = self.preview

        cursor = len(self._display) - 1

        for i in range(cursor, max(len(visible_lines) - 1, 0), -1):
            tui_print('\r\033[K\033[A', end='')
            self._display.pop()
            cursor -= 1

        if not visible_lines:
            tui_print('\r\033[K', end='')
            self._display.pop()
            return

        # Assumed that cursor is always at the end of last line
        from .lib_itertools import lookahead
        for (idx, line), is_last in lookahead(enumerate(visible_lines)):
            # Append empty lines, i.e. canvas size++ if terminal size++
            for i in range(len(self._display), idx + 1):
                self._display.append(None)

            # Skip non-dirty lines, but always redraw the last line
            # for keeping cursor at end of the last line
            if not all and not is_last and self._display[idx] == line:
                continue

            # Align cursor position
            if cursor != idx:
                dist = min(abs(cursor - idx), len(self._display) - 1)
                if cursor > idx:
                    tui_print('\r\033[{}A'.format(dist), end='')
                else:
                    down = 0
                    nl = 0
                    for i in range(cursor + 1, idx + 1):
                        if self._display[i] is None:
                            nl += 1
                        else:
                            down += 1
                    o = ''
                    o += f'\r\033[{down}B' if down else ''
                    o += ('\n' * nl) if nl else ''
                    tui_print(o, end='')

            wline = wrap(line, self.width)[0]
            self._display[idx] = wline

            # Print content onto screen
            tui_print('\r{}\033[K'.format(wline),
                  end='' if is_last else '\n')

            # Estimate cursor position
            cursor = idx + (not is_last)

        tui_flush()


class MenuData:
    def __init__(self):
        super().__setattr__('dataset', {})

    def __repr__(self):
        return f'MenuData({repr(self.dataset)})'

    def __setitem__(self, key, value):
        self.dataset[key] = value
        if value is None:
            del self.dataset[key]

    def __getitem__(self, key):
        return self.dataset.get(key)

    def __delitem__(self, key):
        if key in self.dataset:
            del self.dataset[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


class MenuThread:
    def __init__(self, menu, target=None, name=None, args=(), kwargs={}):
        self.menu = menu
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.thread = None

    def start(self):
        import threading
        if self.is_alive():
            raise AlreadyRunningError(self.target.__name__ + '()')

        self.thread = threading.Thread(
                target=self.target, name=self.name,
                args=self.args, kwargs=self.kwargs,
                daemon=True)

        # register self to self.menu
        self.menu.notify_start(self)

        self.thread.start()
        return self

    def is_alive(self):
        return self.thread and self.thread.is_alive()

    def join(self):
        self.thread.join()


class MenuThreadList(UserList):
    def __init__(self):
        super().__init__()

    def join(self):
        while self.data:
            self.data[0].join()
            self.data.pop(0)


@export
class Menu:
    class DoneSelection(Exception):
        pass

    class GiveUpSelection(Exception):
        pass

    class StdoutIsNotAtty(Exception):
        def __init__(self):
            super().__init__('Stdout should be a tty for using interactive menu')

    @staticmethod
    def parse_checkbox(checkbox):
        if checkbox in ('()', 'single', 'radio'):
            checkbox = '(*)'
        elif checkbox in ('[]', 'multi', 'multiple', 'checkbox'):
            checkbox = '[*]'
        elif checkbox in ('{}', 'meta'):
            checkbox = '{*}'

        if not checkbox:
            check = None
            box = None
        elif checkbox.startswith('(') and checkbox.endswith(')'):
            check = checkbox[1:-1]
            box = '()'
        elif checkbox.startswith('[') and checkbox.endswith(']'):
            check = checkbox[1:-1]
            box = '[]'
        elif checkbox.startswith('{') and checkbox.endswith('}'):
            check = checkbox[1:-1]
            box = '{}'
        else:
            check = None
            box = None

        return check, box

    def __init__(self, title=None, options=None, *, message=None,
                 max_height=None, wrap=False,
                 format=None, cursor='>', checkbox=None,
                 onkey=None, term_cursor_invisible=None):
        if options is None:
            options, title = title, None

        if not options or not is_iterable(options) or isinstance(options, str):
            raise TypeError('options should be a non-empty iterable')

        self.pager = Pager(max_height=max_height)

        self.title = title
        self.options = [self.Item(meta=False, text=opt, cursor=None, checkbox=None) for opt in options]
        self.message = message
        self.data = MenuData()

        self.term_cursor_invisible = term_cursor_invisible
        if self.term_cursor_invisible is None:
            self.term_cursor_invisible = message is None

        self.check, self.box = self.parse_checkbox(checkbox)

        if format:
            self.format = format
        elif self.box:
            self.format = '{cursor} {box[0]}{check}{box[1]} {item.text}'
        else:
            self.format = '{cursor} {item.text}'

        self._onkey = MenuKeyHandler(self)
        self.onkey = onkey
        self._onevent = MenuEventDispatcher(self)

        self.cursor_symbol = cursor
        self._cursor = MenuCursor(self, wrap=wrap)

        self._active = False

        from .lib_threading import Throttler
        self._refresh_throttler = Throttler(self.do_render, 1/60)

        self.threads = MenuThreadList()

    def __iter__(self):
        return iter(self.options)

    def __len__(self):
        return len(self.options)

    def __getitem__(self, idx):
        if isinstance(idx, (MenuItem, MenuCursor)):
            if idx.menu is not self:
                raise ValueError('MenuCursor is from different Menu')
            idx = idx.index
        return self.options[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, (MenuItem, MenuCursor)):
            if idx.menu is not self:
                raise ValueError('MenuCursor is from different Menu')
            idx = idx.index
        self.options[idx].text = str(value)

    def Thread(self, target=None, name=None, args=(), kwargs={}):
        return MenuThread(menu=self, target=target, name=name, args=args, kwargs=kwargs)

    def Item(self, text='', cursor=None, checkbox=None, check=None, box=None, meta=False, onkey=None):
        ret = MenuItem(menu=self, meta=meta, text=text, cursor=cursor, checkbox=checkbox, check=check, box=box)
        if onkey:
            ret.onkey += onkey
        return ret

    def emit(self, event, **kwargs):
        return self.onevent.emit(event=event, menu=self, **kwargs)

    def notify_start(self, thread):
        self.threads.append(thread)

    @property
    def active(self):
        return self._active

    @getter
    def wrap(self):
        return self.cursor.wrap

    @setter
    def wrap(self, value):
        self.cursor.wrap = value

    @getter
    def max_height(self):
        return self.pager.max_height

    @setter
    def max_height(self, value):
        self.pager.max_height = value

    @getter
    def cursor(self):
        return self._cursor

    @setter
    def cursor(self, value):
        self._cursor.to(value)

    @getter
    def onkey(self):
        return self._onkey

    @setter
    def onkey(self, value):
        self._onkey.set_to(value)

    @getter
    def onevent(self):
        return self._onevent

    @setter
    def onevent(self, value):
        self._onevent.set_to(value)

    @getter
    def onselect(self):
        return self.onevent['select']

    @setter
    def onselect(self, value):
        self.onselect.set_to(value)

    @getter
    def onunselect(self):
        return self.onevent['unselect']

    @setter
    def onunselect(self, value):
        return self.onunselect.set_to(value)

    @getter
    def onsubmit(self):
        return self.onevent['submit']

    @setter
    def onsubmit(self, value):
        self.onsubmit.set_to(value)

    @getter
    def onquit(self):
        return self.onevent['quit']

    @setter
    def onquit(self, value):
        self.onquit.set_to(value)

    @property
    def first(self):
        return self.options[0]

    @property
    def last(self):
        return self.options[-1]

    @property
    def top(self):
        for idx, pagee in enumerate(self.pager):
            if pagee.visible:
                return self.options[idx]

    @property
    def bottom(self):
        ret = None
        for idx, pagee in enumerate(self.pager):
            if pagee.visible:
                ret = self.options[idx]
            elif ret is not None:
                break
        return ret

    @property
    def selected(self):
        selected_items = [item for item in self if item.selected and not item.meta]
        if self.box == '[]':
            return selected_items
        else:
            if selected_items:
                return selected_items[0]

    def index(self, value):
        for index, item in enumerate(self.options):
            if item is value or item.text == value:
                return index
        return -1

    def insert(self, index, text='', cursor=None, checkbox=None, check=None, box=None, meta=False, onkey=None):
        ret = self.Item(meta=meta, text=text, cursor=cursor, checkbox=checkbox, check=check, box=box, onkey=onkey)
        self.options.insert(index, ret)
        return ret

    def append(self, text='', cursor=None, checkbox=None, check=None, box=None, meta=False, onkey=None):
        ret = self.Item(meta=meta, text=text, cursor=cursor, checkbox=checkbox, check=check, box=box, onkey=onkey)
        self.options.append(ret)
        return ret

    def extend(self, options, cursor=None, checkbox=None, check=None, box=None, meta=False, onkey=None):
        ret = [self.Item(meta=meta, text=text, cursor=cursor, checkbox=checkbox, check=check, box=box, onkey=onkey)
               for text in options]
        self.options.extend(ret)
        return ret

    def swap(self, a, b):
        if isinstance(a, (MenuItem, MenuCursor)):
            a = a.index
        if isinstance(b, (MenuItem, MenuCursor)):
            b = b.index
        self.options[a], self.options[b] = self.options[b], self.options[a]

    def moveto(self, item, to):
        if not isinstance(item, MenuItem):
            raise TypeError('item should be a MenuItem')

        item = item.index
        if isinstance(to, (MenuItem, MenuCursor)):
            to = to.index

        if item < to: # move down
            self.options = (
                    self.options[:item] +
                    self.options[item+1:to] +
                    [self.options[to]] +
                    [self.options[item]] +
                    self.options[to+1:]
                    )

        if item > to: # move up
            self.options = (
                    self.options[:to] +
                    [self.options[item]] +
                    [self.options[to]] +
                    self.options[to+1:item] +
                    self.options[item+1:]
                    )

    def bind(self, *args, **kwargs):
        return self._onkey.bind(*args, **kwargs)

    def unbind(self, *args, **kwargs):
        return self._onkey.unbind(*args, **kwargs)

    def submit(self):
        if not self.box:
            self.cursor.select()

        ok = self.onevent.handle(event='submit', menu=self)
        if ok is not None and not ok:
            return False

        raise Menu.DoneSelection()

    def quit(self):
        self.onevent.handle(event='quit', menu=self)
        raise Menu.GiveUpSelection()

    def select(self, item):
        if item.selected:
            return

        ok = item.onevent.handle(event='select', item=item)
        if ok is not None and not ok:
            return False

        if not item.meta:
            if self.box == '()':
                self.unselect_all()
            item._selected = True
        return True

    def select_all(self):
        if self.box != '[]':
            return False

        res = []
        for item in self.options:
            if not item.meta:
                res.append(item.select())
        return any(filter(lambda x: x is not None, res))

    def unselect(self, item):
        if not item.selected:
            return

        ok = item.onevent.handle(event='unselect', item=item)
        if ok is not None and not ok:
            return False

        item._selected = False
        return True

    def unselect_all(self):
        res = []
        for item in self.options:
            res.append(item.unselect())
        return any(filter(lambda x: x is not None, res))

    def toggle(self, item):
        if item.selected:
            return item.unselect()
        else:
            return item.select()

    def feedkey(self, key):
        ret = self[self.cursor].onkey.handle(key)
        if ret:
            return ret
        return self.onkey.handle(key)

    def scroll_to_cursor(self):
        try:
            if self.pager[int(self.cursor)].visible:
                return

            if self.cursor < self.pager.scroll:
                self.pager.scroll = int(self.cursor)
                return

            for i in range(int(self.cursor), 0, -1): # pragma: no cover
                if self.pager[i].visible:
                    self.pager.scroll += int(self.cursor) - i
                    break
        except IndexError:
            return

    def pull_cursor(self):
        if self.pager[int(self.cursor)].visible:
            return

        if self.cursor < self.pager.scroll:
            self.cursor = self.pager.scroll
            return

        for i in range(int(self.cursor), 0, -1): # pragma: no cover
            if self.pager[i].visible:
                self.cursor = i
                break

    def scroll(self, count=1):
        self.pager.scroll += count
        self.pull_cursor()

    def do_render(self, force=False):
        if not self.active and not force:
            return

        self.pager.clear()

        if self.title is not None:
            self.pager.header.extend(self.title.split('\n'))

        def pad(s):
            if not s:
                return ''
            return strwidth(str(s)) * ' '

        for idx, item in enumerate(self.options):
            cursor = self.cursor
            check = item.check or self.check
            box = item.box or self.box
            fmt = item.format or self.format

            check = '' if check is None else check
            fmt = fmt if callable(fmt) else fmt.format
            self.pager[idx] = fmt(
                    menu=self,
                    cursor=cursor if self.cursor == idx else pad(cursor),
                    item=item,
                    check=check if item.selected or item.meta else pad(check),
                    box=box or ('', ''),
                    )

        if self.message is not None:
            self.pager.footer.extend(self.message.split('\n'))

        self.scroll_to_cursor()
        self.pager.render()

    def refresh(self, force=False):
        self._refresh_throttler(blocking=force, args=[], kwargs={'force': force})

    def interact_loop(self):
        self.unselect_all()
        self.pager.reset()
        try:
            self._active = True
            if self.term_cursor_invisible:
                tui_print('\033[?25l', end='')

            while True:
                self.refresh(force=True)
                ch = getch(capture='fs')
                try:
                    with self._refresh_throttler.main_lock:
                        self.feedkey(ch)
                except Menu.GiveUpSelection:
                    return None
                except Menu.DoneSelection:
                    return self.selected
        finally:
            self._active = False
            self.refresh(force=True)
            tui_print('\033[?25h' if self.term_cursor_invisible else '')

    def interact(self, *, suppress=(EOFError, KeyboardInterrupt, BlockingIOError)):
        if not sys.stdout.isatty():
            raise Menu.StdoutIsNotAtty()

        with HijackStdio():
            with ExceptionSuppressor(suppress):
                # Default key handlers
                if not bool(self.onkey):
                    self.onkey(KEY_UP, self.cursor.up)
                    self.onkey(KEY_DOWN, self.cursor.down)
                    self.onkey(KEY_SPACE, self.cursor.toggle)
                    if not self.box:
                        self.onkey(KEY_ENTER, self.submit)
                    else:
                        def select_if_didnt(menu):
                            menu.cursor.select() or menu.submit()
                        self.onkey(KEY_ENTER, select_if_didnt)
                    self.onkey('q', self.quit)

                return self.interact_loop()


class MenuItemRef:
    def __cmp__(self, other):
        if isinstance(other, MenuItem) and other.menu is self.menu:
            a = self.index
            b = other.index
        elif isinstance(other, str):
            a = self.text
            b = other
        else:
            a = self.index
            b = other
        try:
            return (a > b) - (a < b)
        except TypeError:
            return a != b

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return self.__cmp__(other) != 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def emit(self, event, **kwargs):
        return self.onevent.emit(event=event, item=self, **kwargs)


class MenuItem(MenuItemRef):
    def __init__(self, *, menu, meta, text, cursor, checkbox=None, check=None, box=None):
        self.menu = menu
        self.meta = bool(meta)
        self.text = str(text)
        self._selected = False
        self.data = MenuData()
        self.format = None

        self.cursor_symbol = cursor
        self._check, self._box = Menu.parse_checkbox('meta' if self.meta and checkbox is None else checkbox)

        if check:
            self._check = check
        if box:
            self._box = box

        self._onkey = MenuKeyHandler(self)
        self._onevent = MenuEventDispatcher(self)

    @getter
    def check(self):
        if callable(self._check):
            return self._check(self)
        return self._check if self.selected else None

    @setter
    def check(self, value):
        self._check = value

    @getter
    def box(self):
        if callable(self._box):
            return self._box(self)
        return self._box

    @setter
    def box(self, value):
        self._box = value

    def __repr__(self):
        return f'MenuItem(index={self.index}, selected={self.selected}, text={repr(self.text)})'

    @getter
    def onkey(self):
        return self._onkey

    @setter
    def onkey(self, value):
        self._onkey.set_to(value)

    @getter
    def onevent(self):
        return self._onevent

    @setter
    def onevent(self, value):
        self._onevent.set_to(value)

    @getter
    def onselect(self):
        return self.onevent['select']

    @setter
    def onselect(self, value):
        return self.onselect.set_to(value)

    @getter
    def onunselect(self):
        return self.onevent['unselect']

    @setter
    def onunselect(self, value):
        return self.onunselect.set_to(value)

    @property
    def index(self):
        return self.menu.index(self)

    @getter
    def selected(self):
        return self._selected and not self.meta

    @setter
    def selected(self, value):
        if value:
            self.select()
        else:
            self.unselect()
        return self.selected

    def bind(self, *args, **kwargs):
        return self._onkey.bind(*args, **kwargs)

    def unbind(self, *args, **kwargs):
        return self._onkey.unbind(*args, **kwargs)

    def select(self):
        return self.menu.select(self)

    def unselect(self):
        return self.menu.unselect(self)

    def toggle(self):
        return self.menu.toggle(self)

    def moveto(self, where):
        self.menu.moveto(self, where)

    def feedkey(self, key):
        return self.onkey.handle(key)


class MenuCursor(MenuItemRef):
    def __init__(self, menu, *, wrap=False):
        self.menu = menu
        self.wrap = wrap
        self.pos = 0

    @property
    def item(self):
        return self.menu[self.pos]

    @property
    def index(self):
        return self.item.index

    def __repr__(self):
        return f'MenuCursor(pos={self.pos}, wrap={self.wrap})'

    def __str__(self):
        return self.menu[self].cursor_symbol or self.menu.cursor_symbol

    def __int__(self):
        return self.pos

    def __add__(self, other):
        return self.cal_index(self.pos + other)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        self.to(self + other)
        return self

    def __sub__(self, other):
        return self.cal_index(self.pos - other)

    def __rsub__(self, other):
        return other - self.pos

    def __isub__(self, other):
        self.to(self - other)
        return self

    def __getattr__(self, attr):
        if not attr.startswith('_') and hasattr(self.item, attr):
            return getattr(self.item, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if attr in ('menu', 'wrap', 'pos'):
            return super().__setattr__(attr, value)

        if not attr.startswith('_') and hasattr(self.item, attr):
            return setattr(self.item, attr, value)
        raise AttributeError(attr)

    def cal_index(self, value):
        if isinstance(value, MenuItem):
            if value.menu is not self.menu:
                raise ValueError('MenuItem is in different Menu')
            return value.index

        value = int(value)
        N = len(self.menu)
        if self.wrap:
            return ((value % N) + N) % N
        else:
            from .lib_math import clamp
            return clamp(0, value, N - 1)

    def to(self, value):
        self.pos = self.cal_index(value)
        self.menu.scroll_to_cursor()

    def up(self, count=1):
        self -= count

    def down(self, count=1):
        self += count

    def select(self):
        return self.item.select()

    def unselect(self):
        return self.item.unselect()

    def toggle(self):
        return self.item.toggle()

    def feedkey(self, key):
        return self.item.feedkey(key)


class MenuKeyHandler:
    class MenuKeySubHandlerList(UserList):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __add__(self, other):
            if isinstance(other, (list, tuple, UserList)):
                handler_list = other
            else:
                handler_list = [other]

            data = list(self.data)
            for handler in handler_list:
                if handler not in self.data:
                    data.append(handler)
            return type(self)(data)

        def __iadd__(self, other):
            self.data = (self + other).data
            return self

        def __sub__(self, other):
            if isinstance(other, (list, tuple, UserList)):
                handler_list = other
            else:
                handler_list = [other]

            data = list(self.data)
            for handler in handler_list:
                try:
                    data.remove(handler)
                except:
                    pass
            return type(self)(data)

        def __isub__(self, other):
            self.data = (self - other).data
            return self

    def __init__(self, target):
        self.target = target
        self.clear()
        self.MenuKeySubHandlerList = self.__class__.MenuKeySubHandlerList

    def __bool__(self):
        return any(h for e, h in self.handlers.items())

    def clear(self):
        self.handlers = {None: self.MenuKeySubHandlerList()}

    def __iadd__(self, other):
        if isinstance(other, MenuKeyHandler):
            # Unpack the MenuKeyHandler
            for key in other.handlers.keys():
                for handler in other.handlers[key]:
                    self.bind(key, handler)
            return self

        elif isinstance(other, (list, tuple, dict, UserList, UserDict)):
            self.bind(other)
            return self

        else:
            return self.bind(other)

    def __isub__(self, other):
        if isinstance(other, (list, tuple, UserList)):
            return self.unbind(*other)
        else:
            return self.unbind(other)

    def __getitem__(self, key):
        key = key_alias_table.get(key, key)
        return self.handlers.get(key, self.MenuKeySubHandlerList())

    def __setitem__(self, key, value):
        key = key_alias_table.get(key, key)
        try:
            if not value:
                del self.handlers[key]
            else:
                self.handlers[key] = self.MenuKeySubHandlerList() + value
        except KeyError:
            pass

    def __call__(self, *args):
        return self.bind(*args)

    def set_to(self, value):
        if value is self:
            return
        self.clear()
        if value is not None:
            self += value

    def bind(self, *args):
        if len(args) == 1:
            if isinstance(args[0], (dict, UserDict)):
                for key, handlers in args[0].items():
                    if callable(handlers):
                        handlers = [handlers]
                    for h in handlers:
                        self.bind(key, h)
                return self

            if isinstance(args[0], (tuple, list, UserList)):
                return self.bind(*args[0])

        key_list = [arg for arg in args if not callable(arg)] or [None]
        handler_list = [arg for arg in args if callable(arg)]

        if not handler_list:
            raise ValueError('No handlers to bind')

        for key in key_list:
            key = key_alias_table.get(key, key)

            for handler in handler_list:
                if isinstance(self.target, Menu):
                    ok_args = ['key', 'menu']
                elif isinstance(self.target, MenuItem):
                    ok_args = ['key', 'item']
                else:
                    ok_args = ['key']

                import inspect
                sig = inspect.signature(handler).parameters
                nok_args = tuple(repr(key) for key, value in sig.items()
                            if value.default == value.empty and
                            value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD) and
                            key not in ok_args)
                if nok_args:
                    raise SignatureError(f'Unreachable parameters: {",".join(nok_args)}')

                if key not in self.handlers:
                    self.handlers[key] = self.MenuKeySubHandlerList()

                self.handlers[key] += handler

        return self

    def unbind(self, *args):
        if len(args) == 1:
            if isinstance(args[0], (dict, UserDict)):
                for key, handlers in args[0].items():
                    if callable(handlers):
                        handlers = [handlers]
                    for h in handlers:
                        self.unbind(key, h)
                return self

            if isinstance(args[0], (tuple, list, UserList)):
                return self.unbind(*args[0])

        key_list = [arg for arg in args if not callable(arg)] or self.handlers.keys()
        handler_list = [arg for arg in args if callable(arg)]

        for key in key_list:
            key = key_alias_table.get(key, key)

            if not handler_list:
                self.handlers.pop(key)

            if key not in self.handlers:
                continue

            for handler in handler_list:
                self.handlers[key] -= handler

        return self

    def handle(self, key):
        key = key_alias_table.get(key, key)
        for handler in self.handlers.get(key, []) + self.handlers[None]:
            kwargs = {}

            import inspect
            sig = inspect.signature(handler).parameters

            if 'key' in sig:
                kwargs['key'] = key
            if isinstance(self.target, Menu) and 'menu' in sig:
                kwargs['menu'] = self.target
            if isinstance(self.target, MenuItem) and 'item' in sig:
                kwargs['item'] = self.target

            ret = handler(**kwargs)

            if ret:
                return ret


class MenuEventDispatcher:
    def __init__(self, target):
        if not isinstance(target, (Menu, MenuItem)):
            raise TypeError('target should be a Menu or a MenuItem')

        super().__setattr__('target', target)
        super().__setattr__('handlers', {})
        super().__setattr__('installers', {})

    def __eq__(self, other):
        if tuple(self.handlers.keys()) == (None,):
            return self.handlers[None] == other
        return False

    def __bool__(self):
        return any(h for e, h in self.handlers.items())

    def __call__(self, event, handler=None):
        if callable(event) and handler is None:
            event, handler = None, event
        self[event] = handler

    def __getattr__(self, event):
        return self[event]

    def __setattr__(self, event, handler):
        self[event] = handler
        return self[event]

    def __getitem__(self, event):
        if event in self.handlers:
            return self.handlers[event]
        if event not in self.installers:
            self.installers[event] = MenuEventHandlerInstaller(self, event)
        return self.installers[event]

    def __setitem__(self, event, handler):
        self.bind(event, handler)

    def clear(self):
        self.handlers.clear()

    def bind(self, event, handler):
        if handler is None:
            self.unbind(event)
        elif isinstance(handler, MenuEventHandlerInstaller):
            self.handlers[event] = MenuEventHandler()
            self.handlers[event].set_to(handler.handler)
            del self.installers[event]
        else:
            self[event].set_to(handler)
        return self

    def unbind(self, event):
        if event in self.handlers:
            del self.handlers[event]
        return self

    def set_to(self, value):
        if value is self:
            return

        self.clear()
        if not value:
            return
        elif callable(value):
            self[None] = value
        else:
            self[value[0]] = value[1]

    def handle(self, event, **kwargs):
        handler = self.handlers.get(event, None)
        if handler:
            ret = handler.handle(event=event, **kwargs)
            if ret is not None:
                return ret

        if isinstance(self.target, MenuItem):
            return self.target.menu.emit(event, **kwargs)

    def emit(self, event, **kwargs):
        return self.handle(event, **kwargs)


class MenuEventHandlerInstaller:
    def __init__(self, dispatcher, event):
        self.dispatcher = dispatcher
        self.event = event
        self.handler = None

    def __repr__(self):
        return f'MenuEventHandlerInstaller(event={repr(self.event)})'

    def __eq__(self, value):
        return self.handler == value

    def __call__(self, handler):
        self.set_to(handler)

    def set_to(self, handler):
        self.handler = handler
        self.dispatcher.bind(self.event, self)


class MenuEventHandler:
    def __init__(self, handler=None):
        self.set_to(handler)

    def __bool__(self):
        return bool(self.handler)

    def __call__(self, handler):
        self.handler = handler

    def __eq__(self, other):
        return self.handler == other

    def set_to(self, value):
        if value is self:
            return
        if isinstance(value, MenuEventHandler):
            value = value.handler
        if value is not None and not callable(value):
            raise ValueError('Event handler should be a callable')
        self.handler = value

    def handle(self, **kwargs):
        import inspect
        sig = inspect.signature(self.handler).parameters
        if not any(True for param in sig.values() if param.kind == param.VAR_KEYWORD):
            for key in [key for key in kwargs.keys() if key not in sig]:
                del kwargs[key]
        return self.handler(**kwargs)
