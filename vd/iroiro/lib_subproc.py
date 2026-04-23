import time
import os
import queue
import subprocess as sub
import threading

from signal import SIGINT, SIGTERM, SIGKILL
from collections import UserList

from .lib_lang import AlreadyRunningError
from .lib_itertools import is_iterable

from .internal_utils import exporter
export, __all__ = exporter()


class EventBroadcaster:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def broadcast(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)


class QueueEventAdapter:
    def __init__(self, Q):
        self.Q = Q

    def __call__(self, line):
        self.Q.put(line)


class FileAdapter:
    def __init__(self, file, newline):
        self.file = file
        self.newline = newline

    def __call__(self, line):
        if self.newline:
            self.file.write(line + '\n')
        else:
            self.file.writeline(line)
        self.file.flush()


class stream:
    def __init__(self):
        self.queue = queue.Queue()
        self.keep = False
        self.lines = []
        self.eof = threading.Event()
        self.hub = EventBroadcaster()
        self.broken_pipe_error = False

        self.pipe_count_lock = threading.Lock()
        self.pipe_count = 0

    def welcome(self, subscriber):
        if isinstance(subscriber, (list, tuple)):
            for s in subscriber:
                self.welcome_one(s)
        else:
            self.welcome_one(subscriber)

    def welcome_one(self, subscriber):
        if subscriber is self:
            raise TypeError('Invalid subscriber value: {}'.format(repr(subscriber)))

        if subscriber is True:
            self.keep = True
            return

        handler = None
        if hasattr(subscriber, 'put') and callable(subscriber.put):
            handler = QueueEventAdapter(subscriber)
        elif hasattr(subscriber, 'writeline') and callable(subscriber.writeline):
            handler = FileAdapter(subscriber, newline=False)
        elif hasattr(subscriber, 'write') and callable(subscriber.write):
            handler = FileAdapter(subscriber, newline=True)
        elif callable(subscriber):
            handler = subscriber
        else:
            raise TypeError('Invalid subscriber: {}'.format(repr(subscriber)))

        self.hub += handler

    def pipe_attached(self):
        with self.pipe_count_lock:
            self.pipe_count += 1

    def pipe_detached(self):
        with self.pipe_count_lock:
            self.pipe_count -= 1
            if self.pipe_count <= 0:
                self.close()

    def read(self):
        data = self.queue.get()
        return data

    def readline(self):
        return self.read()

    def write(self, data, *, suppress=True):
        if self.closed:
            if suppress:
                return
            raise BrokenPipeError('stream already closed')

        if self.keep:
            self.lines.append(data)

        self.queue.put(data)
        self.hub.broadcast(data)

    def writeline(self, line, *, suppress=True):
        self.write(line, suppress=suppress)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def close(self):
        self.eof.set()
        self.queue.put(None)

    @property
    def closed(self):
        return self.eof.is_set()

    @property
    def empty(self):
        return not self.lines and self.queue.empty()

    def __bool__(self):
        return not self.empty

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        if self.closed:
            yield from self.lines

        else:
            while True:
                line = self.readline()
                if line is None:
                    break
                yield line


class IntegerEvent(threading.Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = None

    def __repr__(self):
        return f'<IntegerEvent {self.value}>'

    def set(self, value=None):
        self.value = value
        super().set()

    def clear(self):
        self.value = None
        super().clear()

    def __eq__(self, other):
        if not self.is_set():
            return other is False or other is None
        return self.value == other


@export
class command:
    def __init__(self, cmd, *,
                 cwd=None,
                 stdin=None, stdout=True, stderr=True,
                 encoding='utf8', rstrip='\r\n',
                 bufsize=-1,
                 env=None):

        if cmd and isinstance(cmd, str):
            cmd = [cmd]
        elif callable(cmd):
            cmd = [cmd]
        elif isinstance(cmd, (tuple, list)):
            pass
        else:
            raise ValueError('Invalid command:' + repr(cmd))

        if not cmd:
            raise ValueError('command is empty')

        if callable(cmd[0]):
            self.cmd = [token for token in cmd]
        else:
            self.cmd = [str(token) for token in cmd]

        self.encoding = encoding
        self.bufsize = bufsize
        self.rstrip = rstrip

        self.cwd = cwd
        self.env = env
        self.proc = None
        self.thread = None
        self.exception = None
        self.signaled = IntegerEvent()
        self.returncode = None

        if isinstance(stdin, (str, bytes, bytearray)):
            stdin = [stdin]

        # Initialize stdin stream
        self.stdin = stream()
        self.stdin.keep = True
        self.stdin_queue = None
        self.stdin_autoclose = False
        if stdin is None or stdin is False:
            self.proc_stdin = None
            self.stdin.close()
        else:
            self.proc_stdin = sub.PIPE
            if isinstance(stdin, queue.Queue):
                self.stdin_queue = stdin
            elif is_iterable(stdin):
                for line in stdin:
                    try:
                        line = line.rstrip(self.rstrip)
                    except TypeError:
                        pass
                    self.stdin.write(line)
                self.stdin_autoclose = True

        # Initialize stdout stream
        self.stdout = stream()
        if stdout is None:
            self.proc_stdout = None
            self.stdout.close()
        elif stdout is False:
            self.proc_stdout = sub.DEVNULL
            self.stdout.close()
        else:
            self.proc_stdout = sub.PIPE
            self.stdout.keep = False
            self.stdout.welcome(stdout)

        # Initialize stderr stream
        self.stderr = stream()
        if stderr is None:
            self.proc_stderr = None
            self.stderr.close()
        elif stderr is False:
            self.proc_stderr = sub.DEVNULL
            self.stderr.close()
        else:
            self.proc_stderr = sub.PIPE
            self.stderr.keep = False
            self.stderr.welcome(stderr)

        self.io_threads = []

    @property
    def killed(self):
        return self.signaled

    @property
    def alive(self):
        if self.proc:
            return self.proc.poll() is None
        if self.thread:
            return self.thread.is_alive()
        return False

    def __repr__(self):
        return f'<command [{self.cmd[0]}] ({hex(id(self))})>'

    def __getitem__(self, idx):
        return [self.stdin, self.stdout, self.stderr][idx]

    def __enter__(self):
        return self.run(wait=False)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stdin.close()
        self.stdout.close()
        self.stderr.close()
        self.wait()

    def run(self, wait=None):
        if wait is not None and not isinstance(wait, (int, bool, float)):
            raise TypeError('The type of "wait" should be NoneType, int, bool, or float')

        if self.proc or self.thread:
            if callable(self.cmd[0]):
                who = self.cmd[0].__name__ + '()'
            else:
                who = self.cmd[0]
            raise AlreadyRunningError(' '.join([who] + self.cmd[1:]))

        if callable(self.cmd[0]):
            def worker():
                try:
                    self.returncode = self.cmd[0](self, *self.cmd[1:])
                except Exception as e:
                    self.exception = e

                self.stdin.close()
                self.stdout.close()
                self.stderr.close()

            self.thread = threading.Thread(target=worker)
            self.thread.daemon = True
            self.thread.start()
            _children.append(self)

        else:
            if self.encoding == False:
                # binary mode
                kwargs = {
                        'bufsize': 2 if self.bufsize == 1 else self.bufsize,
                        'text': False,
                        }
            else:
                kwargs = {
                        'bufsize': 1,
                        'encoding': self.encoding,
                        'errors': 'backslashreplace',
                        }

            self.proc = sub.Popen(
                    self.cmd, cwd=self.cwd,
                    stdin=self.proc_stdin,
                    stdout=self.proc_stdout,
                    stderr=self.proc_stderr,
                    env=self.env, **kwargs)
            _children.append(self)

            def writer(self_stream, proc_stream):
                try:
                    for line in self_stream:
                        if self.encoding == False:
                            proc_stream.write(line)
                        elif isinstance(line, (bytes, bytearray)):
                            proc_stream.buffer.write(line)
                        else:
                            proc_stream.write(line + '\n')
                        proc_stream.flush()
                except BrokenPipeError:
                    self_stream.broken_pipe_error = True

                try:
                    proc_stream.close()
                except BrokenPipeError:
                    self_stream.broken_pipe_error = True

            def reader_text(self_stream, proc_stream):
                for line in proc_stream:
                    line = line.rstrip(self.rstrip)
                    self_stream.writeline(line)

            def reader_binary(self_stream, proc_stream):
                while self.poll() is None:
                    data = proc_stream.read(
                            -1
                            if self.bufsize < 0
                            else (self.bufsize or 1)
                            )

                    if not data:
                        continue

                    self_stream.write(data)

                # Read all remaining data left in stream
                data = proc_stream.read()
                if data:
                    self_stream.writeline(data)

            def reader(self_stream, proc_stream):
                try:
                    if self.encoding != False:
                        reader_text(self_stream, proc_stream)
                    else:
                        reader_binary(self_stream, proc_stream)
                except BrokenPipeError:
                    self_stream.broken_pipe_error = True

                self_stream.close()
                try:
                    proc_stream.close()
                except BrokenPipeError:
                    self_stream.broken_pipe_error = True

            for (worker, self_stream, proc_stream) in (
                    (writer, self.stdin, self.proc.stdin),
                    (reader, self.stdout, self.proc.stdout),
                    (reader, self.stderr, self.proc.stderr),
                    ):
                if self_stream is not None and proc_stream is not None:
                    t = threading.Thread(target=worker, args=(self_stream, proc_stream))
                    t.daemon = True
                    t.start()
                    self.io_threads.append(t)

        # Pull data from stdin_queue and feed into stdin stream
        if self.stdin_queue:
            def feeder():
                while True:
                    self.stdin.writeline(self.stdin_queue.get())
                    self.stdin_queue.task_done()

            t = threading.Thread(target=feeder)
            t.daemon = True
            t.start()

        elif self.stdin_autoclose:
            self.stdin.close()

        self.wait(wait)

        return self

    def poll(self):
        if self.proc:
            return self.proc.poll()
        if self.thread:
            return self.returncode
        return False

    def wait(self, timeout=None):
        if timeout is True:
            timeout = None
        elif timeout is False:
            return not self.alive

        # Wait too early
        if self.proc is None and self.thread is None:
            return False

        # Wait for child process to finish
        if self.proc:
            self.exception = None
            try:
                self.proc.wait(timeout)
                self.returncode = self.proc.returncode
                if self.returncode < 0 and not self.signaled.is_set():
                    self.signaled.set(-self.returncode)
                _children.discard(self)
            except sub.TimeoutExpired as e:
                return False
            except KeyboardInterrupt as e:
                self.signal(SIGINT)
                self.exception = e
            except Exception as e:
                self.signal(SIGTERM)
                self.exception = e

        if self.thread:
            self.thread.join(timeout=timeout)
            if self.alive:
                return False
            _children.discard(self)

        if self.exception:
            raise self.exception

        # Wait for all streams to close
        self.stdin.eof.wait()
        self.stdout.eof.wait()
        self.stderr.eof.wait()

        # Gracefully wait for threads to finish
        for t in self.io_threads:
            t.join()

        return True

    def signal(self, signal):
        if not self.alive:
            return
        if self.proc:
            self.proc.send_signal(signal)
        self.signaled.set(signal)

    def kill(self, signal=SIGTERM):
        self.signal(signal)
        if self.proc:
            self.wait()
        if self.thread:
            self.thread.join()


@export
def run(cmd, *,
        cwd=None,
        stdin=None, stdout=True, stderr=True,
        encoding='utf8', rstrip='\r\n',
        bufsize=-1,
        env=None,
        wait=True):
    ret = command(cmd, cwd=cwd,
                  stdin=stdin, stdout=stdout, stderr=stderr,
                  encoding=encoding,
                  rstrip=rstrip, env=env)
    ret.run(wait=wait)
    return ret


class Pipe:
    def __init__(self, istream, *ostreams):
        if istream.closed:
            raise EOFError('istream already closed')

        for ostream in ostreams:
            if ostream.closed:
                raise BrokenPipeError('ostream already closed')

        self.exception = None
        self.thread = None
        self.istream = istream
        self.ostreams = ostreams
        self.post_write = None

    def main(self):
        try:
            for line in self.istream:
                for ostream in self.ostreams:
                    ostream.write(line)
                if self.post_write:
                    self.post_write()

        except Exception as e:
            self.exception = e
            self.istream.close()

        self.istream.eof.wait()
        for ostream in self.ostreams:
            ostream.pipe_detached()

    def start(self):
        self.thread = threading.Thread(target=self.main)
        self.thread.daemon = True
        self.thread.start()

    def join(self):
        self.thread.join()
        if self.exception:
            raise self.exception


@export
def pipe(istream, *ostreams, start=True):
    p = Pipe(istream, *ostreams)
    for ostream in ostreams:
        ostream.pipe_attached()

    if start:
        p.start()
    return p


@export
def is_parant_process_alive():
    return os.getppid() != 1


@export
def is_parant_process_dead():
    return not is_parant_process_alive()


class Children(UserList):
    def __init__(self, data=None):
        super().__init__(data)
        self.rlock = threading.RLock()

    def __enter__(self):
        self.rlock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.rlock.release()

    def __len__(self):
        with self:
            return len(self.data)

    def append(self, child):
        with self:
            self.data.append(child)

    def discard(self, child):
        with self:
            try:
                self.data.remove(child)
            except: # pragma: no cover
                pass

    def refresh(self):
        with self:
            for child in list(self.data):
                if not child.alive:
                    self.discard(child)

    def wait(self, timeout=None):
        snapshot = list(self.data)
        ret = True
        for child in snapshot:
            ret = ret and child.wait(timeout=timeout)
        return ret


_children = Children()


@export
def children():
    _children.refresh()
    return _children


TERM_TIMEOUT = 3


def term_pids(who, signum=tuple(), timeout=TERM_TIMEOUT, how=None):
    who_list = list(who)
    if not who_list:
        return

    signum_list = list(signum)
    if not signum_list:
        signum_list = [SIGTERM]

    if how is os.getpid or how is os.kill:
        how = [lambda x: x, os.kill]
    else:
        how = [os.getpgid, os.killpg]

    for signum in signum_list + [SIGKILL]:
        for who in who_list:
            if isinstance(who, command):
                who.signal(signum)
            else:
                who = how[0](who)
                how[1](who, signum)
        time.sleep(timeout)


@export
def terminate_self(*signum_list, timeout=TERM_TIMEOUT, how=None):
    term_pids(who=[os.getpid()], signum=signum_list, timeout=timeout, how=how)


@export
def terminate_children(*signum_list, timeout=TERM_TIMEOUT, how=None):
    term_pids(who=_children, signum=signum_list, timeout=timeout, how=how)


@export
def monitor_parant_process(interval=TERM_TIMEOUT, cond=is_parant_process_alive, callback=terminate_self):
    def loop():
        while cond():
            time.sleep(interval)
        callback()

    t = threading.Thread(target=loop)
    t.start()
    return t
