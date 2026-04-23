from .lib_test_utils import *

from iroiro import getch, register_key, deregister_key
from iroiro import KEY_UP, KEY_HOME


class TestGetch(TestCase):
    def setUp(self):
        self.patch('sys.stdin.fileno', self.mock_stdin_fileno)
        self.patch('select.select', self.mock_select)
        self.patch('os.read', self.mock_read)
        self.patch('os.getpid', self.mock_getpid)
        self.patch('os.kill', self.mock_kill)
        self.patch('tty.setraw', self.mock_setraw)
        self.patch('termios.tcgetattr', self.mock_tcgetattr)
        self.patch('termios.tcsetattr', self.mock_tcsetattr)
        self.buffer = bytearray()
        self.default_term_attr = [
                'iflag', 'oflag', 'cflag', 'lflag',
                'ispeed', 'ospeed',
                [b'cc'] * 20]

        import termios
        self.default_term_attr[6][termios.VINTR] = b'\x03'
        self.default_term_attr[6][termios.VSUSP] = b'\x1c'
        self.default_term_attr[6][termios.VQUIT] = b'\x1a'

        self.term_attr = list(self.default_term_attr)
        self.killed = None

    def tearDown(self):
        self.eq(self.term_attr, self.default_term_attr)

    def press(self, key):
        if isinstance(key, str):
            key = key.encode('utf8')
        self.buffer += key

    def mock_stdin_fileno(self):
        return 0

    def mock_select(self, rlist, wlist, xlist, timeout=None):
        self.eq(self.term_attr[0], 'raw')
        if self.buffer:
            return (rlist, [], [])
        return ([], [], [])

    def mock_read(self, fd, n):
        self.eq(self.term_attr[0], 'raw')
        ret = self.buffer[:n]
        del self.buffer[:n]
        return ret

    def mock_getpid(self):
        return self

    def mock_kill(self, pid, sig):
        assert pid is self
        self.killed = sig

    def mock_setraw(self, fd, when=None):
        import termios
        self.eq(when, termios.TCSADRAIN)
        self.term_attr = ['raw', 'raw', 'raw', 'raw', 'raw', 'raw', 'cc']

    def mock_tcgetattr(self, fd):
        return self.term_attr

    def mock_tcsetattr(self, fd, when, attributes):
        import termios
        self.eq(when, termios.TCSADRAIN)
        self.term_attr = attributes

    def test_getch_basic(self):
        self.eq(getch(), None)
        self.press(b'abc')
        self.eq(getch(), 'a')
        self.eq(getch(), 'b')
        self.eq(getch(), 'c')
        self.eq(getch(), None)

    def test_getch_unicode(self):
        self.eq(getch(), None)
        self.press('測試')
        self.eq(getch(), '測')
        self.eq(getch(), '試')
        self.eq(getch(), None)

    def test_getch_escape_keys(self):
        self.eq(getch(), None)
        self.press('\033[AA')
        self.eq(getch(), 'up')
        self.eq(getch(), 'A')
        self.eq(getch(), None)

    def test_getch_unicode_error(self):
        self.eq(getch(), None)
        test_data = '測'.encode('utf8')[:-1]
        self.press(test_data)
        self.eq(getch(), test_data)
        self.eq(getch(), None)

    def test_register_key_empty_seq(self):
        with self.raises(ValueError):
            register_key('')

    def test_register_key_with_key_object(self):
        new_key = type(KEY_UP)(r'\033[[[[[[', 'wow')
        nkey = register_key(new_key, 'wah', 'haha')
        self.eq(new_key.seq, nkey.seq)
        self.eq(new_key, 'wow')
        self.eq(nkey, 'wah')
        self.eq(nkey, 'haha')
        self.eq(deregister_key(new_key), new_key)

    def test_register_deregister_key(self):
        self.eq(getch(), None)
        self.press('測試')
        self.eq(getch(), '測')
        self.eq(getch(), '試')
        self.eq(getch(), None)

        # Resigter keys
        TE = register_key('測'.encode('utf8'), 'TE')
        ST = register_key('試'.encode('utf8'), 'ST')
        ABCD = register_key('\033ABCD', 'ABCD')
        self.eq(TE, '測')
        self.eq(TE, '測'.encode('utf8'))
        self.eq(ST, '試')
        self.eq(ST, '試'.encode('utf8'))
        self.eq(ABCD, 'ABCD')
        self.eq(ABCD, '\033ABCD')

        self.press('測試\033ABCD')
        self.eq(getch(), TE)
        self.eq(getch(), ST)
        self.eq(getch(), 'ABCD')
        self.eq(getch(), None)

        # Deresigter keys
        TE = deregister_key(TE)
        ST = deregister_key(ST.seq)
        ABCD = deregister_key('\033ABCD')

        self.press('測試\033ABCD')
        self.eq(getch(), TE)
        self.eq(getch(), ST)
        self.eq(getch(), '\033A')
        self.eq(getch(), 'B')
        self.eq(getch(), 'C')
        self.eq(getch(), 'D')
        self.eq(getch(), None)

        MY_HOME = register_key(KEY_HOME.seq, 'MY_HOME')
        self.eq(MY_HOME, KEY_HOME)
        self.press(KEY_HOME.seq)
        self.eq(getch(), MY_HOME)

    def test_capture(self):
        import signal

        self.press('\x03')
        getch()

        self.press('\x03')
        getch(capture='unknown key')
        self.eq(self.killed, signal.SIGINT)

    def test_simultaneous_getch_error(self):
        def mock_select(rlist, wlist, xlist, timeout=None):
            got_it.wait()
            return (rlist, [], [])
        self.patch('select.select', mock_select)

        def mock_read(fd, n):
            got_it.wait()
            return 'q'.encode('utf8')
        self.patch('os.read', mock_read)

        from iroiro import ResourceError

        got_it = self.checkpoint()

        def run_getch():
            try:
                getch()
            except ResourceError:
                got_it.set()

        with self.run_in_thread(run_getch):
            run_getch()

        self.true(got_it.is_set())
