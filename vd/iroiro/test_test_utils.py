import threading

from .lib_test_utils import *

import iroiro as iro


class TestTestCase(TestCase):
    def test_bridged_methods(self):
        self.eq(self.almost_eq, self.assertAlmostEqual)
        self.eq(self.ne, self.assertNotEqual)
        self.eq(self.le, self.assertLessEqual)
        self.eq(self.lt, self.assertLess)
        self.eq(self.ge, self.assertGreaterEqual)
        self.eq(self.gt, self.assertGreater)
        self.eq(self.true, self.assertTrue)
        self.eq(self.false, self.assertFalse)
        self.eq(self.raises, self.assertRaises)

    def test_isinstance(self):
        self.isinstance(True, bool)
        self.isinstance(3, int)
        self.isinstance(3.1415926535897932384626433832795, float)

    def test_contains(self):
        self.contains([1, 2, 3], 1)
        self.contains([1, 2, 3], 2)
        self.contains([1, 2, 3], 3)
        self.contains_no([1, 2, 3], 4)

    def test_list_diff_msg(self):
        try:
            self.eq([1, 2, 3], [1, 2, 3, 4])
        except AssertionError as e:
            self.eq(str(e),
'''Lists not equal:
[
  1,
  2,
  3,
+ 4,
]''')

        try:
            self.eq([1, 2, 3, 4], [1, 2, 3])
        except AssertionError as e:
            self.eq(str(e),
'''Lists not equal:
[
  1,
  2,
  3,
- 4,
]''')

        try:
            self.eq([1, 2, 5, 4], [1, 2, 3, 4])
        except AssertionError as e:
            self.eq(str(e),
'''Lists not equal:
[
  1,
  2,
- 5,
+ 3,
  4,
]''')

        from collections import UserList
        try:
            self.eq([2, 3, 4], UserList([1, 2, 3, 4]))
        except AssertionError as e:
            self.eq(str(e),
'''Lists not equal:
[
+ 1,
  2,
  3,
  4,
]''')

    def test_list_of_list_diff_msg(self):
        try:
            self.eq([1, [2], 3], [1, [2, 3], 4])
        except AssertionError as e:
            self.eq(str(e),
'''Lists not equal:
[
  1,
- [2],
- 3,
+ [2, 3],
+ 4,
]''')


class TestRunInThread(TestCase):
    def test_run_in_thread(self):
        barrier = threading.Barrier(2)
        checkpoint = False

        def may_stuck():
            nonlocal checkpoint
            barrier.wait()
            checkpoint = True

        with self.run_in_thread(may_stuck):
            self.false(checkpoint)
            barrier.wait()

        self.true(checkpoint)

    def test_run_in_thread_reuse(self):
        def foo():
            pass

        p = self.run_in_thread(foo)
        with p:
            pass

        with self.raises(RuntimeError):
            with p:
                pass


class TestCheckPoint(TestCase):
    def test_checkpoint(self):
        checkpoint = self.checkpoint()

        # Test a set checkpoint
        checkpoint.set()
        self.true(checkpoint)

        # Test an unset checkpoint
        checkpoint.clear()
        self.false(checkpoint)

        # Verify a checkpoint, which resets it
        checkpoint.set()
        checkpoint.verify(True)
        checkpoint.verify(False)
        checkpoint.verify(False)

        # Verify a checkpoint is reusable
        checkpoint.set()
        checkpoint.verify(True)
        checkpoint.verify(False)
        checkpoint.verify(False)

        # Test a checkpoint with thread
        def set_checkpoint():
            checkpoint.wait()

        with self.run_in_thread(set_checkpoint):
            checkpoint.set()

        checkpoint.check()


class TestSubprocRunMocker(TestCase):
    def test_mock_basic(self):
        mock_run = RunMocker()
        def mock_wah(proc, *args):
            proc.stdout.writeline('mock wah')
            if args:
                proc.stdout.writeline(' '.join(args))
            return 0
        mock_run.register('wah', mock_wah)

        p = mock_run('wah'.split())
        self.eq(p.stdout.lines, ['mock wah'])

        p = mock_run('wah')
        self.eq(p.stdout.lines, ['mock wah'])

        p = mock_run('wah wah wah'.split())
        self.eq(p.stdout.lines, ['mock wah', 'wah wah'])

    def test_mock_meaningless_mock(self):
        mock_run = RunMocker()
        with self.raises(ValueError):
            mock_run.register('cmd')

    def test_mock_ambiguous_mock(self):
        mock_run = RunMocker()
        with self.raises(ValueError):
            mock_run.register('wah', lambda: None, stdout='wah')

    def test_mock_cmd_with_wrong_type(self):
        mock_run = RunMocker()
        with self.raises(ValueError):
            mock_run.register(['wah'], lambda proc: 0)

    def test_mock_callback_with_wrong_type(self):
        mock_run = RunMocker()
        with self.raises(TypeError):
            mock_run.register('wah', 0)

    def test_mock_run_empty_cmd(self):
        mock_run = RunMocker()
        with self.raises(ValueError):
            mock_run([])

    def test_mock_run_unregistered_cmd(self):
        mock_run = RunMocker()
        with self.raises(ValueError):
            p = mock_run('ls -a -l'.split())

    def test_mock_register_default_cmd(self):
        mock_run = RunMocker()
        def default_cmd(proc, *args):
            self.eq(args, ('-a', '-l'))
            return 42
        mock_run.register('*', default_cmd)
        p = mock_run('ls -a -l'.split())
        self.eq(p.returncode, 42)

    def test_mock_with_returncode(self):
        mock_run = RunMocker()
        mock_run.register('wah', returncode=1)
        p = mock_run(['wah'])
        self.eq(p.returncode, 1)

    def test_mock_with_stdout_stderr_returncode(self):
        mock_run = RunMocker()
        mock_run.register('wah',
                      stdout=['wah', 'wah wah', 'wah wah wah'],
                      stderr=['WAH', 'WAH WAH', 'WAH WAH WAH'],
                      returncode=520)

        p = mock_run(['wah'])
        self.eq(p.stdout.lines, ['wah', 'wah wah', 'wah wah wah'])
        self.eq(p.stderr.lines, ['WAH', 'WAH WAH', 'WAH WAH WAH'])
        self.eq(p.returncode, 520)

    def test_mock_with_stdout_stderr_returncode_side_effect(self):
        mock_run = RunMocker()
        mock_run.register('wah',
                      stdout=['wah1'],
                      stderr=['WAH1'],
                      returncode=42)
        mock_run.register('wah',
                      stdout=['wah2'],
                      stderr=['WAH2'],
                      returncode=43)

        p = mock_run(['wah'])
        self.eq(p.stdout.lines, ['wah1'])
        self.eq(p.stderr.lines, ['WAH1'])
        self.eq(p.returncode, 42)

        p = mock_run(['wah'])
        self.eq(p.stdout.lines, ['wah2'])
        self.eq(p.stderr.lines, ['WAH2'])
        self.eq(p.returncode, 43)

    def test_mock_side_effect(self):
        mock_run = RunMocker()

        def mock_ls_1st(proc, *args):
            self.eq(args, ('-a', '-l'))
            proc.stdout.writeline('file1')
        def mock_ls_3rd(proc, *args):
            self.eq(proc.cmd, ('ls', '-a', '-l', '--wah'))
            self.eq(args, ('-a', '-l', '--wah'))
            proc.stdout.writeline('file3')
        mock_run.register('ls', mock_ls_1st)
        mock_run.register('ls', stdout=['wah'], stderr=['WAH'], returncode=42)
        mock_run.register('ls', mock_ls_3rd)
        mock_run.register('ls', ValueError('ls is called too many times'))

        p = mock_run('ls -a -l'.split())
        self.eq(p.returncode, None)
        self.eq(p.stdout.lines, ['file1'])

        p = mock_run('ls -a -l'.split())
        self.eq(p.returncode, 42)
        self.eq(p.stdout.lines, ['wah'])
        self.eq(p.stderr.lines, ['WAH'])

        p = mock_run('ls -a -l --wah'.split())
        self.eq(p.returncode, None)
        self.eq(p.stdout.lines, ['file3'])

        with self.raises(ValueError):
            p = mock_run('ls -a -l --wah'.split())


class TestFakeTerminal(TestCase):
    def test_init(self):
        ft = iro.FakeTerminal()
        self.eq(ft, [''])
        self.eq(ft.lines, [''])
        self.eq(ft.cursor, (0, 0))

    def test_escape_seq_reset(self):
        ft = iro.FakeTerminal()
        text = 'occuboinkal'
        ft.puts(text)
        self.eq(ft.lines, [text])
        self.eq(ft.cursor.x, len(text))
        ft.puts('\033c')
        self.eq(ft.lines, [''])
        self.eq(ft.cursor.x, 0)

    def test_escape_seq_carriage_return_and_newline(self):
        ft = iro.FakeTerminal()
        text = 'occuboinkal'
        ft.puts(text)
        self.eq(ft.lines, [text])
        self.eq(ft.cursor.x, len(text))
        ft.puts('\r')
        self.eq(ft.cursor.x, 0)
        ft.puts('\n' + text + 'ly')
        self.eq(ft.lines, [text, text + 'ly'])
        self.eq(ft.cursor.y, 1)
        self.eq(ft.cursor.x, len(text) + 2)

    def test_escape_seq_cursorpos(self):
        ft = iro.FakeTerminal()
        text = 'occuboinkal'
        ft.puts('\n'.join([text, text, text, text, text]))
        self.eq(ft.lines, [text, text, text, text, text])
        self.ne(ft.cursor.y, 0)
        self.ne(ft.cursor.x, 0)

        ft.puts('\033[H')
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 0)

        ft.puts('\033[3;7H')
        self.eq(ft.cursor.y, 2)
        self.eq(ft.cursor.x, 6)

        ft.puts('\033[A')
        self.eq(ft.cursor.y, 1)
        self.eq(ft.cursor.x, 6)

        ft.puts('\033[2A')
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 6)

        ft.puts('\033[C')
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 7)

        ft.puts('\033[2C')
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 9)

        ft.puts('\033[B')
        self.eq(ft.cursor.y, 1)
        self.eq(ft.cursor.x, 9)

        ft.puts('\033[2B')
        self.eq(ft.cursor.y, 3)
        self.eq(ft.cursor.x, 9)

        ft.puts('\033[D')
        self.eq(ft.cursor.y, 3)
        self.eq(ft.cursor.x, 8)

        ft.puts('\033[5D')
        self.eq(ft.cursor.y, 3)
        self.eq(ft.cursor.x, 3)

        ft.puts('\033[100B')
        self.eq(ft.cursor.y, 4)

    def test_escape_seq_cleareol(self):
        ft = iro.FakeTerminal()
        text = 'occuboinkal'
        ft.puts('\n'.join([text, text, text, text, text]))
        ft.puts('\033[3;7H')
        self.eq(ft.lines, [text, text, text, text, text])
        self.eq(ft.cursor.y, 2)
        self.eq(ft.cursor.x, 6)

        ft.puts('\033[K')
        self.eq(ft.cursor.y, 2)
        self.eq(ft.cursor.x, 6)
        self.eq(ft.lines, [text, text, 'occubo', text, text])

        ft.puts('\r嗚啦呀哈')
        self.eq(ft.lines[2], '嗚啦呀哈')

        ft.puts('\033[3D\033[K')
        self.eq(ft.cursor.x, 5)
        self.eq(ft.lines[2], '嗚啦')

    def test_escape_seq_cursor_visibility(self):
        ft = iro.FakeTerminal()
        self.eq(ft.cursor.visible, True)

        ft.puts('\033[?25l')
        self.eq(ft.cursor.visible, False)

        ft.puts('嗚啦呀哈')
        self.eq(ft.lines, ['嗚啦呀哈'])
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 8)

        ft.puts('\033[?25h')
        self.eq(ft.cursor.visible, True)

        ft.puts('嗚啦呀哈')
        self.eq(ft.lines, ['嗚啦呀哈嗚啦呀哈'])
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 16)

    def test_escape_seq_unknown_seq(self):
        ft = iro.FakeTerminal()
        ft.puts('\033$%#^&*(a')
        self.eq(ft.lines, [''])
        self.eq(ft.cursor, (0, 0))

    def test_basic_output(self):
        ft = iro.FakeTerminal()

        ft.puts('ABCD\nEF')
        self.eq(ft[0], 'ABCD')
        self.eq(ft[1], 'EF')
        self.eq(ft.cursor, (1, 2))

        ft.puts('\rGH')
        self.eq(ft.lines, ['ABCD', 'GH'])
        self.eq(ft.cursor, (1, 2))

    def test_wide_chars(self):
        ft = iro.FakeTerminal()

        ft.puts('嗚  拉')
        self.eq(ft[0], '嗚  拉')
        self.eq(ft.cursor, (0, 6))

        ft.puts('\r\033[C')
        self.eq(ft.cursor, (0, 1))

        ft.puts('呀')
        self.eq(ft[0], ' 呀 拉')
        self.eq(ft.cursor, (0, 3))

        ft.puts('哈')
        self.eq(ft[0], ' 呀哈')
        self.eq(ft.cursor, (0, 5))

        ft.puts('\033[3D')
        self.eq(ft.cursor, (0, 2))

        ft.puts('.')
        self.eq(ft[0], '  .哈')
        self.eq(ft.cursor, (0, 3))

        ft.puts('#')
        self.eq(ft[0], '  .#')
        self.eq(ft.cursor, (0, 4))

    def test_puts_over_line_end(self):
        ft = iro.FakeTerminal()
        ft.puts('ABCD')
        self.eq(ft[0], 'ABCD')
        self.eq(ft.cursor, (0, 4))

        ft.puts('\033[5C哇')
        self.eq(ft[0], 'ABCD     哇')
        self.eq(ft.cursor, (0, 11))

    def test_invalid_size_limit(self):
        ft = iro.FakeTerminal()
        self.eq(ft.get_terminal_size(), (80, 24))

        ft = iro.FakeTerminal(columns=0, lines=0)
        self.eq(ft.get_terminal_size().lines, 1)
        self.eq(ft.get_terminal_size().columns, 0)

        with self.raises(ValueError):
            iro.FakeTerminal(columns=-1)

        with self.raises(ValueError):
            iro.FakeTerminal(lines=-1)

    def test_auto_size(self):
        ft = iro.FakeTerminal(columns=0, lines=0)
        ft.print('ABCD')
        ft.print('EFGHI', end=None)
        self.eq(ft.get_terminal_size().lines, 3)
        self.eq(ft.get_terminal_size().columns, 5)

    def test_size_limit(self):
        ft = iro.FakeTerminal()
        self.eq(ft.get_terminal_size(), (80, 24))
        self.eq(ft.cursor, (0, 0))

        ft.puts('.' * 80)
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 80)

        ft = iro.FakeTerminal()
        ft.puts('.' * 81)
        self.eq(ft.cursor.y, 1)
        self.eq(ft.cursor.x, 1)

        ft = iro.FakeTerminal()
        ft.puts('.' + '哇' * 40)
        self.eq(ft.cursor.y, 0)
        self.eq(ft.cursor.x, 81)

        ft.puts('哇')
        self.eq(ft.cursor.y, 1)
        self.eq(ft.cursor.x, 2)

    def test_recording(self):
        ft = iro.FakeTerminal()
        self.false(ft.recording)

        ft.recording = True
        self.eq(ft.recording, [])

        ft.puts('wah')
        self.eq(ft.recording, ['wah'])

        ft.puts('wow')
        self.eq(ft.recording, ['wah', 'wow'])

        ft.recording = False
        self.false(ft.recording)

        with self.raises(TypeError):
            ft.recording = 'wah'

    def test_color(self):
        ft = iro.FakeTerminal()
        self.false(ft.recording)

        # ft.recording = True
        ft.puts('\033[38;5;208mwah')
        self.eq(ft.canvas[0][0].attr, '\033[38;5;208m')
        self.eq(ft.canvas[0][1].attr, '\033[38;5;208m')
        self.eq(ft.canvas[0][2].attr, '\033[38;5;208m')
        self.eq(ft.canvas[0][0].char, 'w')
        self.eq(ft.canvas[0][1].char, 'a')
        self.eq(ft.canvas[0][2].char, 'h')

        ft.puts('\033[47;1mwow')
        self.eq(ft.canvas[0][3].attr, '\033[1;38;5;208;47m')
        self.eq(ft.canvas[0][4].attr, '\033[1;38;5;208;47m')
        self.eq(ft.canvas[0][5].attr, '\033[1;38;5;208;47m')
        self.eq(ft.canvas[0][3].char, 'w')
        self.eq(ft.canvas[0][4].char, 'o')
        self.eq(ft.canvas[0][5].char, 'w')

        ft.puts('\033[mlol')
        self.eq(ft.canvas[0][6].attr, '\033[m')
        self.eq(ft.canvas[0][7].attr, '\033[m')
        self.eq(ft.canvas[0][8].attr, '\033[m')
        self.eq(ft.canvas[0][6].char, 'l')
        self.eq(ft.canvas[0][7].char, 'o')
        self.eq(ft.canvas[0][8].char, 'l')


class TestFakeTime(TestCase):
    def test_get_current_time(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        import time
        self.eq(time.time(), 0)
        time.sleep(4.2)
        self.eq(time.time(), 4.2)

        time.sleep(42)
        self.eq(time.time(), 4.2 + 42)

        time.sleep(0)
        self.eq(time.time(), 4.2 + 42)

        with self.raises(ValueError):
            time.sleep(-1)

        self.eq(time.time(), 4.2 + 42)

    def test_timer_normal(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()
        def foo(bar):
            self.eq(bar, 42)
            checkpoint.set()

        import threading
        t = threading.Timer(10, foo, kwargs={'bar': 42})
        t.start()
        self.false(checkpoint)

        import time
        time.sleep(5)
        self.false(checkpoint)

        time.sleep(5)
        checkpoint.wait()

    def test_timer_cancel(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()
        def foo(bar):
            self.eq(bar, 42)
            checkpoint.set()

        import threading
        t = threading.Timer(10, foo, kwargs={'bar': 42})
        t.start()
        self.false(checkpoint)

        t.cancel()
        self.false(t.active)
        self.true(t.canceled)

        import time
        time.sleep(10)
        self.false(checkpoint)

    def test_timer_cancel_after_join(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()
        def foo(bar):
            self.eq(bar, 42)
            checkpoint.set()

        import threading
        t = threading.Timer(10, foo, kwargs={'bar': 42})
        t.start()
        self.false(checkpoint)

        t_canceled = self.checkpoint()
        def t_join():
            t.join()
            t_canceled.set()

        thread = threading.Thread(target=t_join, daemon=True)
        thread.start()

        t.cancel()
        self.false(t.active)
        self.true(t.canceled)

        import time
        time.sleep(10)
        self.false(checkpoint)

        thread.join()
        t_canceled.check()
