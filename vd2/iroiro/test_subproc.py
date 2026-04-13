import os
import threading
import queue

from .lib_test_utils import *

from iroiro import *

import iroiro
stream = iroiro.subproc.stream


def queue_to_list(Q):
    ret = []
    while not Q.empty():
        ret.append(Q.get())
    return ret


class TestEventBroadcaster(TestCase):
    def test_all(self):
        data1 = []
        def handler1(arg):
            data1.append(arg)

        data2 = []
        def handler2(arg):
            data2.append(arg)

        import iroiro
        hub = iroiro.subproc.EventBroadcaster()
        hub.broadcast('...')

        hub += handler1
        hub.broadcast('wah')

        hub += handler2
        hub.broadcast('Wah')

        hub += handler2
        hub.broadcast('WAAAAAH')

        hub -= handler1
        hub.broadcast('wah?')

        hub -= handler2
        hub.broadcast('wow')

        hub -= handler2
        hub.broadcast('bye')

        self.eq(data1, ['wah', 'Wah', 'WAAAAAH'])
        self.eq(data2, ['Wah', 'WAAAAAH', 'WAAAAAH', 'wah?', 'wah?', 'wow'])


class TestStream(TestCase):
    def test_stream_basic_io(self):
        s = stream()
        self.eq(s.keep, False)

        s.writeline('line1')
        s.writeline('line2')
        s.writeline('line3')
        self.false(s.closed)

        s.close()
        self.true(s.closed)

        self.eq(s.lines, [])
        self.eq(s.readline(), 'line1')
        self.eq(s.readline(), 'line2')
        self.eq(s.readline(), 'line3')

    def test_stream_iter(self):
        s = stream()
        self.eq(s.keep, False)
        s.writelines(['line1', 'line2', 'line3'])

        i = iter(s)
        self.eq(next(i), 'line1')
        self.eq(next(i), 'line2')
        self.eq(next(i), 'line3')

        s.close()

        with self.raises(StopIteration):
            next(iter(s))

    def test_stream_keep(self):
        s = stream()
        s.keep = True

        lines = ['line1', 'line2', 'line3']
        for line in lines:
            s.writeline(line)

        self.eq(s.lines, lines)
        self.eq(len(s), 3)

        self.false(s.empty)
        self.true(bool(s))

        self.eq(s.readline(), 'line1')
        self.eq(s.readline(), 'line2')
        self.eq(s.readline(), 'line3')

        self.eq(s.lines, lines)
        self.eq(len(s), 3)

    def test_stream_invalid_subscribers(self):
        s = stream()
        with self.raises(TypeError):
            s.welcome(s)
        with self.raises(TypeError):
            s.welcome(3)

        class WeirdQueue:
            def __init__(self):
                self.put = 'put'
        with self.raises(TypeError):
            s.welcome(WeirdQueue())

        class WeirdFile:
            def __init__(self):
                self.write = 'write'
        with self.raises(TypeError):
            s.welcome(WeirdFile())

    def test_stream_subscribers(self):
        data1 = []
        def handler1(line):
            data1.append(line)

        data2 = []
        def handler2(line):
            data2.append(line)

        Q = queue.Queue()

        s = stream()
        s.welcome([handler1, handler2])
        s.welcome(Q)
        s.welcome(True)

        lines = ['line1', 'line2', 'line3']
        s.writelines(lines)

        self.eq(data1, lines)
        self.eq(data2, lines)

    def test_stream_write_after_close(self):
        def should_not_be_called_handler(line):
            self.fail()

        # test in test
        with self.raises(AssertionError):
            should_not_be_called_handler('wah')

        s = stream()
        s.welcome(should_not_be_called_handler)
        s.close()

        s.writeline('line1')

        with self.raises(BrokenPipeError):
            s.writeline('line2', suppress=False)


class TestSubproc(TestCase):
    def tearDown(self):
        children().clear()

    def test_default_properties(self):
        def prog(proc):
            self.eq(proc[0].readline(), 'line')

        p = run(prog, stdin='line', stdout=False, stderr=False)

        p = command(['iroiro', 'arg1', 'arg2'])
        self.true('command' in repr(p))
        self.true(hex(id(p)) in repr(p))
        self.true('iroiro' in repr(p))

    def test_stdout_lines(self):
        p = run('seq 5'.split())
        self.eq(p.stdout.lines, '1 2 3 4 5'.split())

    def test_stdout_and_stderr_none(self):
        p = command('seq 5'.split(), stdout=None, stderr=None)
        self.true(p.stdout.closed)
        self.true(p.stderr.closed)

    def test_stdout_to_false(self):
        p = command('seq 5'.split(), stdout=False)
        p.run()
        self.eq(p.stdout.lines, [])

    def test_stdout_to_callback(self):
        lines = []
        def callback(line):
            lines.append(line)
        p = command('seq 5'.split(), stdout=callback)
        p.run()
        self.eq(lines, ['1', '2', '3', '4', '5'])

    def test_stdout_to_queue(self):
        Q = queue.Queue()
        p = command('seq 5'.split(), stdout=Q)
        self.false(p.stdout.keep)
        p.run()
        self.eq(p.stdout.lines, [])
        self.eq(queue_to_list(Q), ['1', '2', '3', '4', '5'])

    def test_stdout_and_stderr_to_file(self):
        import io
        fake_file0 = io.StringIO('line1\nline2\nline333')
        fake_file1 = io.StringIO()
        fake_file2 = io.StringIO()
        def prog(proc, *args):
            for line in proc[0]:
                proc[1].writeline('[' + line + ']')
                proc[2].writeline('{' + line + '}')
            return 42
        p = run(prog, stdin=fake_file0, stdout=fake_file1, stderr=fake_file2)
        self.eq(p.returncode, 42)
        self.eq(p.stdout.lines, [])
        self.eq(fake_file1.getvalue(), '[line1]\n[line2]\n[line333]\n')
        self.eq(fake_file2.getvalue(), '{line1}\n{line2}\n{line333}\n')

    def test_stdout_and_stderr_to_file_writeline(self):
        import io
        fake_file0 = io.StringIO('line1\nline2\nline333')

        class LineFile(io.StringIO):
            def __init__(self):
                self.lines = []
            def writeline(self, line):
                self.lines.append(line)

        fake_file1 = LineFile()
        fake_file2 = LineFile()
        def prog(proc, *args):
            for line in proc[0]:
                proc[1].writeline('[' + line + ']')
                proc[2].writeline('{' + line + '}')
            return 42
        p = run(prog, stdin=fake_file0, stdout=fake_file1, stderr=fake_file2)
        self.eq(p.returncode, 42)
        self.eq(p.stdout.lines, [])
        self.eq(fake_file1.lines, ['[line1]', '[line2]', '[line333]'])
        self.eq(fake_file2.lines, ['{line1}', '{line2}', '{line333}'])

    def test_rstrip_false(self):
        import io
        fake_file0 = io.StringIO('line1\nline2\nline333')
        fake_file1 = io.StringIO()
        def prog(proc, *args):
            for line in proc[0]:
                proc[1].writeline('[' + line + ']')
            return 42
        p = run(prog, stdin=fake_file0, stdout=fake_file1, rstrip=False)
        self.eq(p.returncode, 42)
        self.eq(p.stdout.lines, [])
        self.eq(fake_file1.getvalue(), '[line1\n]\n[line2\n]\n[line333]\n')

    def test_wait_early(self):
        p = command('seq 5'.split())
        p.wait()
        p.run(wait=False)
        p.wait()
        self.eq(p.stdout.lines, '1 2 3 4 5'.split())

    def test_already_running_error(self):
        checkpoint = self.checkpoint()

        def prog(proc, *args):
            checkpoint.wait()

        p = command(prog)
        p.run(wait=False)

        with self.raises(AlreadyRunningError) as e1:
            p.run()
        self.contains(str(e1.exception), 'prog()')

        checkpoint.set()
        p.wait()

        p = command(['sleep', 1])
        p.run(wait=False)
        with self.raises(AlreadyRunningError) as e2:
            p.run(wait=False)
        self.eq(str(e2.exception), 'sleep 1')
        p.kill()

    def test_word(self):
        p = run('true')
        self.eq(p.returncode, 0)

        p = run('false')
        self.eq(p.returncode, 1)

    def test_invalid_cmd(self):
        with self.raises(TypeError):
            p = command()
        with self.raises(TypeError):
            p = run()

        for i in ([], True, 3, None, queue.Queue()):
            with self.raises(ValueError):
                command(i)
            with self.raises(ValueError):
                run(i)

    def test_run_with_context_manager(self):
        barrier = threading.Barrier(2)

        def prog(proc, *args):
            barrier.wait()

        with command(prog) as p:
            barrier.wait()

    def test_keep_trailing_whitespaces(self):
        p = run(['echo', 'a b c '])
        self.eq(p.stdout.lines, ['a b c '])

    def test_callable(self):
        def prog(proc, *args):
            self.eq(args, ('arg1', 'arg2'))
            for idx, line in enumerate(proc[0]):
                proc[(idx % 2) + 1].writeline(line)
            return 2024

        p = run([prog, 'arg1', 'arg2'], stdin=['hello ', 'how are you ', 'im fine ', 'thank you '])
        self.eq(p.stdout.lines, ['hello ', 'im fine '])
        self.eq(p.stderr.lines, ['how are you ', 'thank you '])
        self.eq(p.returncode, 2024)

    def test_multi_ouptut_merge(self):
        def prog(proc, *args):
            for i in range(5):
                proc[1].write(i)
                proc[2].write(i)

        Q = queue.Queue()

        lines = []
        def callback(line):
            lines.append(line)

        self.true(hasattr(Q, 'put'))

        p = command(prog, stdout=(Q, callback, True), stderr=(Q, True))
        p.run()
        self.eq(p.stdout.lines, [0, 1, 2, 3, 4])
        self.eq(p.stderr.lines, [0, 1, 2, 3, 4])
        self.eq(lines, [0, 1, 2, 3, 4])

        self.eq(queue_to_list(Q), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_stdin_from_list(self):
        p = command('nl -w 1 -s :'.split(), stdin=['hello', 'world'])
        p.run()
        self.eq(p.stdout.lines, ['1:hello', '2:world'])

    def test_stdin_ignore_delayed_writes(self):
        data = ['hello', 'world']
        p = command('nl -w 1 -s :'.split(), stdin=data)

        data += ['wow', 'haha']     # ignored
        p.stdin.writeline('wah')    # allow write before run
        p.run(wait=False).wait()

        self.eq(p.stdout.lines, ['1:hello', '2:world', '3:wah'])

    def test_stdin_from_queue(self):
        Q = queue.Queue()
        p = command('nl -w 1 -s :'.split(), stdin=Q)

        Q.put('pre')

        p.run(wait=False)

        Q.join()
        Q.put('hello')
        Q.put('world')
        Q.join()
        Q.put('wah')
        Q.join()

        p.stdin.close()

        p.wait()
        self.eq(p.stdout.lines, ['1:pre', '2:hello', '3:world', '4:wah'])

    def test_stdin_from_file(self):
        import io
        fake_file = io.StringIO('line1\nline2\nline333')
        def prog(proc, *args):
            for line in proc[0]:
                proc[1].writeline(line)
            return 42
        p = run(prog, stdin=fake_file)
        self.eq(p.returncode, 42)
        self.eq(p.stdout.lines, ['line1', 'line2', 'line333'])

    def test_callable_with_pipe(self):
        def prog(proc, *args):
            for line in proc[0]:
                proc[1].writeline(line)
                proc[2].writeline(line)
            return 2024

        p1 = command(prog, stdin=['hello', 'world'])
        p2 = command('nl -w 1 -s :'.split(), stdin=True)
        p3 = command('nl -w 1 -s /'.split(), stdin=True)
        pipe(p1.stdout, p2.stdin)
        pipe(p1.stderr, p3.stdin)

        p1.run()
        p2.run()
        p3.run()

        self.eq(p1.returncode, 2024)
        self.eq(p2.stdout.lines, ['1:hello', '2:world'])
        self.eq(p3.stdout.lines, ['1/hello', '2/world'])

    def test_callable_raises_exception(self):
        def prog(proc, *args):
            # NameError
            n + 1

        with self.raises(NameError):
            p = run(prog)

    def test_loopback(self):
        # Collatz function
        def collatz_function(x):
            return (x // 2) if (x % 2 == 0) else (3 * x + 1)

        def prog(proc, *args):
            for line in proc[0]:
                x = line
                if x == 1:
                    break
                else:
                    proc[1].write(collatz_function(x))

        p = command(prog, stdin=True)

        # loopback
        pipe(p.stdout, p.stdin)

        import time
        t = int(time.time())
        p.stdin.write(t)
        p.run()

        # If this test fails, make sure to check the initial input
        self.eq(p.stdout.lines[-1], 1, p.stdin.lines[0])

    def test_wait_timeouts(self):
        checkpoint = self.checkpoint()

        def prog(proc, *args):
            checkpoint.wait()

        p = command(prog)
        self.false(p.alive)

        p.run(wait=False)
        self.true(p.alive)

        self.false(p.wait(False))
        self.true(p.alive)

        self.false(p.wait(timeout=0.01))
        self.true(p.alive)

        checkpoint.set()
        self.true(p.wait())
        self.true(p.wait(False))
        self.false(p.alive)

    def test_wait_invalid_types(self):
        def prog(proc, *args): # pragma: no cover
            pass

        p = command(prog)
        with self.raises(TypeError):
            p.run(wait='wah')
        self.eq(p.proc, None)
        self.eq(p.thread, None)

    def test_run_timeout(self):
        import time
        p = command(['sleep', 3])
        t1 = time.time()
        p.run(wait=0.1)
        self.true(p.alive)
        t2 = time.time()
        self.le(t2 - t1, 1)
        p.kill()

    def test_poll(self):
        p = command('true')
        self.eq(p.poll(), False)
        p.run()
        self.eq(p.poll(), 0)

        checkpoint = self.checkpoint()
        def prog(proc, *args):
            checkpoint.wait()
            return 1
        p = command(prog)
        self.eq(p.poll(), False)
        p.run(wait=False)
        self.eq(p.poll(), None)
        checkpoint.set()
        p.wait()
        self.eq(p.poll(), 1)

    def test_signal(self):
        p = run(['sleep', 86400], wait=False)
        import signal
        p.signal(signal.SIGINT)
        p.wait()
        self.eq(p.signaled, signal.SIGINT)
        self.eq(repr(p.signaled), '<IntegerEvent 2>')

    def test_signaled_externally(self):
        p = run(['sleep', 86400], wait=False)
        import signal
        os.kill(p.proc.pid, signal.SIGTERM)
        p.wait()
        self.eq(p.signaled, signal.SIGTERM)
        self.eq(repr(p.signaled), '<IntegerEvent 15>')

    def test_kill_callable(self):
        checkpoint = self.checkpoint()

        def prog(proc, *args):
            proc.killed.wait()
            checkpoint.set()

        p = run(prog, wait=False)
        self.eq(p.signaled, False)
        self.eq(p.signaled, None)
        p.kill()
        p.wait()

        import signal
        checkpoint.check()
        self.eq(p.signaled, signal.SIGTERM)

        p.signaled.clear()
        self.eq(p.signaled, False)
        self.eq(p.signaled, None)

        p.kill(signal.SIGKILL)
        self.eq(p.signaled, None)

    def test_kill_sigint_pass_to_child_process(self):
        import signal

        class MockPopen:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.received_signal = None
                self.returncode = None

                class MockStream:
                    def __init__(self):
                        pass
                    def __iter__(self):
                        return iter([])
                    def close(self):
                        pass
                self.stdin = MockStream()
                self.stdout = MockStream()
                self.stderr = MockStream()

            def wait(self, timeout=None):
                if not self.received_signal:
                    raise exception

            def send_signal(self, signal):
                self.received_signal = signal
                self.returncode = -int(signal)

            def poll(self):
                return self.returncode or None

        self.patch('subprocess.Popen', MockPopen)

        exception = KeyboardInterrupt()
        p = command(['sleep', '86400'])
        with self.raises(KeyboardInterrupt):
            p.run()
            self.fail()
        self.eq(p.signaled, signal.SIGINT)
        self.eq(p.proc.received_signal, signal.SIGINT)
        p.kill()

        exception = OSError()
        p = command(['sleep', '86400'])
        with self.raises(OSError):
            p.run()
            self.fail()
        self.eq(p.signaled, signal.SIGTERM)
        self.eq(p.proc.received_signal, signal.SIGTERM)
        p.kill()

    def test_read_stdout_twice(self):
        ans = '1 2 3 4 5'.split()
        p = run('seq 5'.split())

        lines = []
        for line in p.stdout:
            lines.append(line)
        self.eq(lines, ans)

        checkpoint = self.checkpoint()
        lines = []
        def may_stuck():
            for line in p.stdout:
                lines.append(line)
            checkpoint.set()

        with self.run_in_thread(may_stuck):
            checkpoint.check()
            self.eq(lines, ans)

    def test_encoding_false(self):
        pi = b''
        pi += b'\x31\x41\x59\x26\x53\x58\x97\x93\x23\x84\x62\x64\x33\x83\x27\x95'
        pi += b'\x02\x88\x41\x97\x16\x93\x99\x37\x51\x05\x82\x09\x74\x94\x45\x92'
        pi += b'\x30\x78\x16\x40\x62\x86\x20\x89\x98\x62\x80\x34\x82\x53\x42\x11'
        pi += b'\x70\x67\x90'
        p = run(['xxd'], encoding=False, stdin=pi)
        self.eq(p.returncode, 0)
        self.eq(p.stdin.lines, [pi])
        self.eq(len(p.stdout.lines), 1)
        hex_pi = p.stdout.lines[0].decode('utf-8')
        self.eq(hex_pi, '''
00000000: 3141 5926 5358 9793 2384 6264 3383 2795  1AY&SX..#.bd3.'.
00000010: 0288 4197 1693 9937 5105 8209 7494 4592  ..A....7Q...t.E.
00000020: 3078 1640 6286 2089 9862 8034 8253 4211  0x.@b. ..b.4.SB.
00000030: 7067 90                                  pg.
'''.lstrip())

    def test_encoding_true_but_write_binary(self):
        pi = b''
        pi += b'\x31\x41\x59\x26\x53\x58\x97\x93\x23\x84\x62\x64\x33\x83\x27\x95'
        pi += b'\x02\x88\x41\x97\x16\x93\x99\x37\x51\x05\x82\x09\x74\x94\x45\x92'
        pi += b'\x30\x78\x16\x40\x62\x86\x20\x89\x98\x62\x80\x34\x82\x53\x42\x11'
        pi += b'\x70\x67\x90'
        p = run(['xxd'], stdin=pi)
        self.eq(p.returncode, 0)
        self.eq(p.stdin.lines, [pi])
        self.eq(len(p.stdout.lines), 4)
        self.eq(p.stdout.lines, [
            "00000000: 3141 5926 5358 9793 2384 6264 3383 2795  1AY&SX..#.bd3.'.",
            "00000010: 0288 4197 1693 9937 5105 8209 7494 4592  ..A....7Q...t.E.",
            "00000020: 3078 1640 6286 2089 9862 8034 8253 4211  0x.@b. ..b.4.SB.",
            "00000030: 7067 90                                  pg.",
            ])

    def test_large_amount_of_retention_data(self):
        p = command(['echo', 'a lot of data'], encoding=False)
        def mock_poll():
            # block p.poll() until the p.proc writes all stdout and exits
            while p.proc.poll() is None:
                pass
            return p.proc.poll()

        p.poll = mock_poll
        p.run()
        self.eq(p.stdout.lines, [b'a lot of data\n'])

    def test_stdin_broken_pipe_error(self):
        class IntrovertStream:
            def __init__(self, which, *args, **kwargs):
                self.which = which

            def write(self, line):
                if self.which == scary:
                    raise BrokenPipeError()

            def flush(self):
                pass

            def close(self):
                if self.which == scary:
                    raise BrokenPipeError()

            def __iter__(self):
                if self.which == scary:
                    raise BrokenPipeError()
                yield f'stream {self.which}'

        class IntrovertProcess:
            def __init__(self, *args, **kwargs):
                self.received_signal = None
                self.returncode = None
                self.stdin = IntrovertStream(0)
                self.stdout = IntrovertStream(1)
                self.stderr = IntrovertStream(2)

            def wait(self, timeout=None):
                self.returncode = 0
                return None

            def send_signal(self, signal):
                self.received_signal = signal
                self.returncode = -int(signal)

            def poll(self):
                return self.returncode or None

        self.patch('subprocess.Popen', IntrovertProcess)

        scary = 0
        p = run('what', stdin=['line1', 'line2'])
        self.true(p.stdin.broken_pipe_error)
        self.false(p.stdout.broken_pipe_error)
        self.false(p.stderr.broken_pipe_error)

        scary = 1
        p = run('what')
        self.false(p.stdin.broken_pipe_error)
        self.true(p.stdout.broken_pipe_error)
        self.false(p.stderr.broken_pipe_error)

        scary = 2
        p = run('what')
        self.false(p.stdin.broken_pipe_error)
        self.false(p.stdout.broken_pipe_error)
        self.true(p.stderr.broken_pipe_error)


class TestPipe(TestCase):
    def test_pipe(self):
        p1 = command('nl -w 1 -s :'.split(), stdin=['hello', 'world'])
        p2 = command('nl -w 1 -s /'.split(), stdin=True)
        pp = pipe(p1.stdout, p2.stdin)

        p1.run()
        self.eq(p1.stdout.lines, ['1:hello', '2:world'])
        p2.run(wait=False)

        p2.wait()
        self.eq(p2.stdin.lines, ['1:hello', '2:world'])
        self.eq(p2.stdout.lines, ['1/1:hello', '2/2:world'])

        pp.join()

    def test_pipe_dont_start(self):
        p1 = command(lambda prog: prog.stdout.writeline('wah'))

        def cat(prog):
            for line in prog.stdin:
                prog.stdout.writeline(line)
        p2 = command(cat, stdin=True)

        pp = pipe(p1.stdout, p2.stdin, start=False)

        p1.run(wait=False)
        p2.run(wait=False)
        p1.wait()

        self.false(p2.stdout.lines)
        self.eq(p2.returncode, None)

        pp.start()
        p2.wait()
        pp.join()

        self.eq(p2.stdout.lines, ['wah'])

    def test_pipe_istream_already_closed(self):
        i = stream()
        o = stream()
        i.close()

        with self.raises(EOFError):
            pipe(i, o)

    def test_pipe_ostream_already_closed(self):
        i = stream()
        o = stream()
        o.close()

        with self.raises(BrokenPipeError):
            pipe(i, o)

    def test_pipe_exception(self):
        i = stream()
        o = stream()
        o.queue = None
        o.close = lambda: None

        p = pipe(i, o)
        i.writeline('wah')

        with self.raises(Exception):
            p.join()

        self.ne(p.exception, None)

    def test_pipe_merge_to_same_stream(self):
        i1 = stream()
        i2 = stream()
        o = stream()
        o.keep = True

        pp1 = pipe(i1, o)
        e1 = threading.Event()
        pp1.post_write = e1.set
        pp2 = pipe(i2, o)
        e2 = threading.Event()
        pp2.post_write = e2.set

        i1.writeline('wah1')
        e1.wait()
        e1.clear()
        self.eq(o.lines, ['wah1'])

        i2.writeline('wow1')
        e2.wait()
        e2.clear()
        self.eq(o.lines, ['wah1', 'wow1'])

        i1.writeline('wah2')
        e1.wait()
        e1.clear()
        self.eq(o.lines, ['wah1', 'wow1', 'wah2'])

        i1.close()
        pp1.join()
        self.false(o.closed)

        i2.writeline('wow2')
        e2.wait()
        e2.clear()
        self.eq(o.lines, ['wah1', 'wow1', 'wah2', 'wow2'])

        i2.close()
        pp2.join()

        self.true(o.closed)

class TestChildrenManagement(TestCase):
    def patch_pid_funcions(self):
        import random
        from signal import SIGUSR1, SIGUSR2, SIGKILL

        self.log = []

        def mock_sleep(duration):
            self.log.append(('time.sleep', duration))
        self.patch('time.sleep', mock_sleep)

        self.pid = random.randrange(10000, 65536)
        def mock_getpid():
            return self.pid
        self.patch('os.getpid', mock_getpid)

        def mock_getpgid(pid):
            self.eq(pid, self.pid)
            return self.pid
        self.patch('os.getpgid', mock_getpgid)

        def mock_kill(pid, signum):
            self.log.append(('os.kill', (pid, signum)))
        self.patch('os.kill', mock_kill)

        def mock_killpg(pgid, signum):
            self.log.append(('os.killpg', (pgid, signum)))
        self.patch('os.killpg', mock_killpg)

    def test_is_parant_process_alive(self):
        self.true(is_parant_process_alive())
        self.false(is_parant_process_dead())

    def test_term_self_without_signum(self):
        self.patch_pid_funcions()

        from signal import SIGUSR1, SIGUSR2, SIGKILL, SIGTERM
        terminate_self()
        self.eq(self.log, [
            ('os.killpg', (self.pid, SIGTERM)),
            ('time.sleep', 3),
            ('os.killpg', (self.pid, SIGKILL)),
            ('time.sleep', 3),
            ])

    def test_term_self_with_signum(self):
        self.patch_pid_funcions()

        from signal import SIGUSR1, SIGUSR2, SIGKILL
        terminate_self(SIGUSR1, SIGUSR2)
        self.eq(self.log, [
            ('os.killpg', (self.pid, SIGUSR1)),
            ('time.sleep', 3),
            ('os.killpg', (self.pid, SIGUSR2)),
            ('time.sleep', 3),
            ('os.killpg', (self.pid, SIGKILL)),
            ('time.sleep', 3),
            ])

    def test_term_self_with_how(self):
        self.patch_pid_funcions()

        from signal import SIGUSR1, SIGUSR2, SIGKILL
        terminate_self(SIGUSR2, how=os.getpid)
        self.eq(self.log, [
            ('os.kill', (self.pid, SIGUSR2)),
            ('time.sleep', 3),
            ('os.kill', (self.pid, SIGKILL)),
            ('time.sleep', 3),
            ])

    def test_term_self_with_timeout(self):
        self.patch_pid_funcions()

        from signal import SIGUSR1, SIGUSR2, SIGKILL
        terminate_self(SIGUSR1, SIGUSR2, timeout=86400)
        self.eq(self.log, [
            ('os.killpg', (self.pid, SIGUSR1)),
            ('time.sleep', 86400),
            ('os.killpg', (self.pid, SIGUSR2)),
            ('time.sleep', 86400),
            ('os.killpg', (self.pid, SIGKILL)),
            ('time.sleep', 86400),
            ])

    def test_term_children_when_no_children(self):
        self.patch_pid_funcions()

        self.eq(children(), [])

        from signal import SIGUSR1, SIGUSR2, SIGKILL
        terminate_children(SIGUSR1, SIGUSR2)
        self.eq(self.log, [])

    def test_term_children(self):
        from signal import SIGTERM, SIGKILL

        self.eq(children(), [])

        import time
        def prog(proc, *args):
            proc.signaled.wait()
        p1 = command(prog)
        p1.run(wait=False)
        self.eq(children(), [p1])

        p2 = command(['sleep', 86400])
        p2.run(wait=False)
        self.eq(children(), [p1, p2])

        terminate_children(timeout=0.1)
        self.false(p1.alive)
        self.false(p2.alive)
        self.eq(p1.signaled, SIGTERM)
        self.eq(p2.signaled, SIGTERM)
        self.eq(children(), [])

    def test_children_wait(self):
        from signal import SIGUSR1

        self.eq(children(), [])
        def prog(proc, *args):
            proc.signaled.wait()

        p1 = command(prog)
        p1.run(wait=False)
        self.eq(children(), [p1])

        self.false(children().wait(timeout=0.01))
        p1.signal(SIGUSR1)
        self.true(children().wait(timeout=0.01))

    def test_monitor_thread(self):
        parent_proc_alive = True
        def mock_is_parant_process_alive():
            return parent_proc_alive

        callback_checkpoint = self.checkpoint()
        def mock_terminate_self():
            callback_checkpoint.set()

        t = monitor_parant_process(interval=0.1,
                                   cond=mock_is_parant_process_alive,
                                   callback=mock_terminate_self)

        parent_proc_alive = False
        t.join()
        callback_checkpoint.check()
