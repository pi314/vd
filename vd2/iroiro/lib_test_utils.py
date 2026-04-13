import unittest
import threading

from collections import UserList

from .internal_utils import exporter
export, __all__ = exporter()

from .lib_regex import rere
from .lib_colors import color


__unittest = True


@export
class Checkpoint:
    def __init__(self, testcase):
        self.testcase = testcase
        self.checkpoint = threading.Event()

    def set(self):
        self.checkpoint.set()

    def clear(self):
        self.checkpoint.clear()

    def wait(self):
        self.checkpoint.wait()

    def is_set(self):
        return self.checkpoint.is_set()

    def verify(self, is_set=True):
        self.testcase.eq(
                self.checkpoint.is_set(),
                is_set,
                'Checkpoint was' + (' ' if self.checkpoint.is_set() else ' not ') + 'set')
        self.checkpoint.clear()

    def check(self, *args, **kwargs):
        self.verify(*args, **kwargs)

    def __bool__(self):
        return self.is_set()


@export
class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.almost_eq = self.assertAlmostEqual
        self.ne = self.assertNotEqual
        self.le = self.assertLessEqual
        self.lt = self.assertLess
        self.ge = self.assertGreaterEqual
        self.gt = self.assertGreater
        self.true = self.assertTrue
        self.false = self.assertFalse
        self.raises = self.assertRaises

    def eq(self, first, second, msg=None):
        if (not isinstance(first, (list, tuple, UserList)) or
            not isinstance(second, (list, tuple, UserList)) or
            (type(first) is tuple) != (type(second) is tuple) or
            first == second):
            return self.assertEqual(first, second, msg)

        else:
            from difflib import SequenceMatcher
            try:
                m = SequenceMatcher(None, first, second, False)
                opcodes = m.get_opcodes()
            except TypeError:
                m = SequenceMatcher(None, [repr(i) for i in first], [repr(i) for i in second], False)
                opcodes = m.get_opcodes()

            msg = ['Lists not equal:']
            msg.append('[')
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal':
                    ops = ((' ', first[i1:i2]),)
                elif tag == 'insert':
                    ops = (('+', second[j1:j2]),)
                elif tag == 'delete':
                    ops = (('-', first[i1:i2]),)
                else: # tag == 'replace'
                    ops = (
                            ('-', first[i1:i2]),
                            ('+', second[j1:j2])
                            )
                for deco, items in ops:
                    msg += [f'{deco} {repr(item)},' for item in items]

            msg.append(']')
            raise AssertionError('\n'.join(msg))

    def contains(self, a, b):
        self.assertIn(b, a)

    def contains_no(self, a, b):
        self.assertNotIn(b, a)

    def isinstance(self, first, second):
        return self.true(isinstance(first, second))

    def checkpoint(self):
        return Checkpoint(self)

    class run_in_thread:
        def __init__(self, func, args=tuple(), kwargs=dict()):
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.thread = None

        def __enter__(self, *args):
            if self.thread is not None:
                raise RuntimeError('Thread objects cannot be reused')
            self.thread = threading.Thread(target=self.func, args=self.args, kwargs=self.kwargs)
            self.thread.daemon = True
            self.thread.start()

        def __exit__(self, exc_type, exc_value, traceback):
            self.thread.join()

    def patch(self, name, side_effect):
        patcher = unittest.mock.patch(name, side_effect=side_effect)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing


@export
class RunMocker:
    def __init__(self):
        self.rules = {}

    def register(self, cmd, callback=None, *, stdout=None, stderr=None, returncode=None):
        if not isinstance(cmd, str):
            raise ValueError('cmd must be a str')

        if callback is not None and not isinstance(callback, Exception) and not callable(callback):
            raise TypeError('callback should be an Exception or a callable')

        by_callback = callback
        by_output = (stdout, stderr, returncode)

        if by_callback is None and by_output == (None, None, None):
            raise ValueError('Meaningless behavior')

        if by_callback is not None and by_output != (None, None, None):
            raise ValueError('Ambiguous behavior')

        if cmd not in self.rules:
            self.rules[cmd] = []

        if by_callback:
            if isinstance(by_callback, Exception):
                behavior = by_callback
            else:
                def behavior(proc, *args):
                    proc.cmd = (cmd, *args)
                    return by_callback(proc, *args)

        else:
            def behavior(proc, *args):
                proc.cmd = (cmd, *args)
                if by_output[0]:
                    proc.stdout.writelines(by_output[0])
                if by_output[1]:
                    proc.stderr.writelines(by_output[1])
                return by_output[2]

        self.rules[cmd].append(behavior)
        return self

    def __call__(self, cmd, *,
                 stdin=None, stdout=True, stderr=True,
                 encoding='utf8', rstrip='\r\n',
                 bufsize=-1,
                 env=None,
                 wait=True):
        if not cmd:
            raise ValueError('command is empty')

        if isinstance(cmd, str):
            cmd = [cmd]

        matched_callbacks = None

        if cmd[0] in self.rules:
            matched_callbacks = self.rules[cmd[0]]
        elif '*' in self.rules:
            matched_callbacks = self.rules['*']
        else:
            raise ValueError('Unregistered command: {}'.format(cmd))

        behavior = matched_callbacks[0]
        if len(matched_callbacks) > 1:
            matched_callbacks.pop(0)

        if isinstance(behavior, Exception):
            raise behavior

        from .lib_subproc import command
        p = command([behavior] + cmd[1:],
                    stdin=stdin, stdout=stdout, stderr=stderr,
                    encoding=encoding, rstrip=rstrip,
                    bufsize=bufsize,
                    env=env)
        p.run(wait=wait)
        return p


class FakeTerminalCell:
    def __init__(self, char, attr):
        self.char = char
        self.attr = attr

    @property
    def width(self):
        from .lib_tui import charwidth
        return charwidth(self.char)


class FakeTerminalCursor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = 0
        self.x = 0
        self.attr = color()
        self.visible = True

    def __eq__(self, other):
        return (self.y, self.x) == other

    def __repr__(self): # pragma: no cover
        return 'Cursor(y={}, x={}, attr={}, visible={})'.format(
                self.y, self.x, repr(self.attr), self.visible)


@export
class FakeTerminal:
    def __init__(self, *, columns=80, lines=24):
        if columns < 0:
            raise ValueError('columns must >= 0')
        if lines < 0:
            raise ValueError('lines must >= 0')
        self.width = columns
        self.height = lines
        self.canvas = [[]]
        self.cursor = FakeTerminalCursor()

        self.chewing = ''

        self.recording_history = False

    @property
    def recording(self):
        return self.recording_history

    @recording.setter
    def recording(self, enable):
        if not isinstance(enable, bool):
            raise TypeError('recording must be a boolean')

        self.recording_history = [] if enable else False

    def __getitem__(self, idx):
        return ''.join(cell.char for cell in self.canvas[idx] if cell is not None).rstrip(' ')

    def __len__(self):
        return len(self.canvas)

    def __eq__(self, other):
        return self.lines == other

    @property
    def lines(self):
        return [self[idx] for idx in range(len(self))]

    def reset(self):
        self.canvas = [[]]
        self.cursor.reset()

    def get_terminal_size(self, *args, **kwargs):
        from os import terminal_size
        return terminal_size((
            self.width or max(len(line) for line in self),
            self.height or len(self.canvas)
            ))

    def ensure_cursor_pos(self):
        from .lib_math import clamp
        self.cursor.y = clamp(0, self.cursor.y, self.height or self.cursor.y)
        self.cursor.x = clamp(0, self.cursor.x, self.width or self.cursor.x)

        # Ensure canvas has enough lines
        while self.cursor.y >= len(self.canvas):
            self.canvas.append([])

    def print(self, *args, sep=' ', end='\n', **kwargs):
        if end is None:
            end = '\n'
        self.puts(sep.join(str(arg) for arg in args) + end)

    def puts(self, text):
        for char in text:
            self.chewing += char
            if self.check_control_seq():
                continue
            if self.chewing and self.chewing.isprintable():
                self.putc(self.chewing)
                self.chewing = ''

        if isinstance(self.recording_history, list):
            self.recording_history.append(text)

    def putc(self, char):
        cell = FakeTerminalCell(char, attr=self.cursor.attr)

        self.ensure_cursor_pos()

        # Make sure canvas is wide enough
        # Pre-fill spaces to make index-calculation easier
        for i in range(len(self.canvas[self.cursor.y]), self.cursor.x + cell.width):
            self.canvas[self.cursor.y].append(FakeTerminalCell(' ', attr=self.cursor.attr))

        current_line = self.canvas[self.cursor.y]
        current_char = self.canvas[self.cursor.y][self.cursor.x]

        if current_char is None:
            # Override the right-half of a wide-char on the left
            self.canvas[self.cursor.y][self.cursor.x - 1] = FakeTerminalCell(' ', attr=self.cursor.attr)

        # Override char under cursor
        self.canvas[self.cursor.y][self.cursor.x] = cell

        if cell.width == 2:
            # For wide-char, check if it overrides the next char
            next_char = self.canvas[self.cursor.y][self.cursor.x + 1]
            if next_char is not None and next_char.width == 2:
                self.canvas[self.cursor.y][self.cursor.x + 2] = FakeTerminalCell(' ', attr=self.cursor.attr)

            self.canvas[self.cursor.y][self.cursor.x + 1] = None

        # wrap
        if self.width and self.cursor.x >= self.width:
            self.cursor.y += 1
            self.cursor.x = 0

        self.cursor.x += cell.width

    def check_control_seq(self):
        m = rere(self.chewing)

        if self.chewing == '\033c':
            # Reset terminal to initial state
            self.reset()
            return True

        elif self.chewing == '\r':
            # Carriage return
            self.cursor.x = 0

        elif self.chewing == '\n':
            # Newline
            self.cursor.x = 0
            self.cursor.y += 1

        elif m.fullmatch('\033' + r'\[(\d*)([AB])'):
            # move cursor up/down
            direction = (1 if m.group(2) == 'B' else -1)
            self.cursor.y += int(m.group(1) or 1) * direction
            if direction == 1:
                self.cursor.y = min(self.cursor.y, len(self.canvas) - 1)

        elif m.fullmatch('\033' + r'\[(\d*)([CD])'):
            # move cursor right/left
            direction = (1 if m.group(2) == 'C' else -1)
            self.cursor.x += int(m.group(1) or 1) * direction

        elif self.chewing == '\033[H':
            self.cursor.y = 0
            self.cursor.x = 0

        elif m.fullmatch('\033' + r'\[(\d*);(\d*)H'):
            self.cursor.y = int(m.group(1) or 1) - 1
            self.cursor.x = int(m.group(2) or 1) - 1

        elif self.chewing == '\033[K':
            self.canvas[self.cursor.y] = self.canvas[self.cursor.y][:self.cursor.x]
            # Check is last character is cut into half
            if (self.cursor.x > 0 and
                self.canvas[self.cursor.y][-1] is not None and
                self.canvas[self.cursor.y][-1].width == 2):
                # if yes, replace it with a space
                self.canvas[self.cursor.y][-1] = FakeTerminalCell(' ', attr=color())

        elif m.fullmatch('\033' + r'\[([\d;]*)m'):
            self.cursor.attr = color(self.cursor.attr.seq + self.chewing)

        elif self.chewing == '\033[?25h':
            self.cursor.visible = True

        elif self.chewing == '\033[?25l':
            self.cursor.visible = False

        else:
            import string
            if (self.chewing and
                self.chewing.startswith('\033') and
                self.chewing[-1] in string.ascii_letters):
                # Escape sequence is terminated but it's unknown, drop it
                self.chewing = ''
            return False

        # Consume the escape sequence
        self.chewing = ''

        self.ensure_cursor_pos()

        return True


@export
class FakeTime:
    def __init__(self):
        self.sys_time = 0
        self.event_list = []

        me = self
        class FakeTimerWrapper(self.FakeTimer):
            def __init__(s, *args, **kwargs):
                super().__init__(me, *args, **kwargs)

        self.FakeTimerWrapper = FakeTimerWrapper

    def patch(self):
        return (
                ('time.time', self.time_time),
                ('time.monotonic', self.time_time),
                ('time.sleep', self.time_sleep),
                ('threading.Timer', self.FakeTimerWrapper),
                )

    def time_time(self):
        return self.sys_time

    def time_sleep(self, secs):
        if secs < 0:
            raise ValueError('This Python implementation is not powerful enough to rewind time')

        if secs == 0:
            return

        self.sys_time += secs
        expired_list = []
        waiting_list = []

        for t, poke, ack in self.event_list:
            if t <= self.sys_time:
                dest = expired_list
            else:
                dest = waiting_list
            dest.append((t, poke, ack))

        self.event_list = waiting_list

        for _, poke, _ in expired_list:
            poke.set()

        for _, _, ack in expired_list:
            ack.wait()

    def pin(self, interval, poke, ack):
        self.event_list.append((self.sys_time + interval, poke, ack))
        self.event_list.sort(key=lambda x: x[0])

    class FakeTimer:
        def __init__(self, coordinator, interval, function, args=[], kwargs={}):
            self.coordinator = coordinator
            self.interval = interval
            self.function = function
            self.args = args
            self.kwargs = kwargs
            self.active = False

            import threading
            self.thread = threading.Thread(target=self.gogo, daemon=True)
            self.expired = threading.Event()
            self.canceled = False
            self.finished = threading.Event()

            self.poke = threading.Event()

        def gogo(self):
            self.poke.wait()
            if self.canceled:
                return
            self.expired.set()
            self.function(*self.args, **self.kwargs)
            self.finished.set()

        def start(self):
            self.active = True
            self.coordinator.pin(self.interval, self.poke, self.finished)
            self.thread.start()

        def cancel(self):
            self.active = False
            self.canceled = True
            self.finished.set()

        def join(self):
            assert self.active
            self.finished.wait()
