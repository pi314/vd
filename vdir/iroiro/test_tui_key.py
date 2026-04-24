from .lib_test_utils import *

from iroiro import *


class TestKey(TestCase):
    def test_builtin_key(self):
        self.eq(KEY_ESCAPE, b'\033')
        self.eq(KEY_ESCAPE, '\033')
        self.eq(KEY_ESCAPE, 'esc')
        self.eq(KEY_ESCAPE, 'escape')

        self.eq(KEY_BACKSPACE, b'\x7f')
        self.eq(KEY_BACKSPACE, 'backspace')

        self.eq(KEY_TAB, b'\t')
        self.eq(KEY_TAB, 'tab')
        self.eq(KEY_TAB, 'ctrl-i')
        self.eq(KEY_TAB, 'ctrl+i')
        self.eq(KEY_TAB, '^I')

        self.eq(KEY_ENTER, b'\r')
        self.eq(KEY_ENTER, '\r')
        self.eq(KEY_ENTER, 'enter')
        self.eq(KEY_ENTER, 'ctrl-m')
        self.eq(KEY_ENTER, 'ctrl+m')
        self.eq(KEY_ENTER, '^M')

        self.eq(KEY_SPACE, b' ')
        self.eq(KEY_SPACE, ' ')
        self.eq(KEY_SPACE, 'space')

        self.eq(KEY_UP, b'\033[A')
        self.eq(KEY_UP, '\033[A')
        self.eq(KEY_UP, 'up')

        self.eq(KEY_DOWN, b'\033[B')
        self.eq(KEY_DOWN, '\033[B')
        self.eq(KEY_DOWN, 'down')

        self.eq(KEY_RIGHT, b'\033[C')
        self.eq(KEY_RIGHT, '\033[C')
        self.eq(KEY_RIGHT, 'right')

        self.eq(KEY_LEFT, b'\033[D')
        self.eq(KEY_LEFT, '\033[D')
        self.eq(KEY_LEFT, 'left')

        self.eq(KEY_HOME, b'\033[1~')
        self.eq(KEY_HOME, '\033[1~')
        self.eq(KEY_HOME, 'home')

        self.eq(KEY_END, b'\033[4~')
        self.eq(KEY_END, '\033[4~')
        self.eq(KEY_END, 'end')

        self.eq(KEY_PGUP, b'\033[5~')
        self.eq(KEY_PGUP, '\033[5~')
        self.eq(KEY_PGUP, 'pgup')
        self.eq(KEY_PGUP, 'pageup')

        self.eq(KEY_PGDN, b'\033[6~')
        self.eq(KEY_PGDN, '\033[6~')
        self.eq(KEY_PGDN, 'pgdn')
        self.eq(KEY_PGDN, 'pagedown')

        self.eq(KEY_F1, b'\033OP')
        self.eq(KEY_F1, 'F1')

        self.eq(KEY_F2, b'\033OQ')
        self.eq(KEY_F2, 'F2')

        self.eq(KEY_F3, b'\033OR')
        self.eq(KEY_F3, 'F3')

        self.eq(KEY_F4, b'\033OS')
        self.eq(KEY_F4, 'F4')

        self.eq(KEY_F5, b'\033[15~')
        self.eq(KEY_F5, 'F5')

        self.eq(KEY_F6, b'\033[17~')
        self.eq(KEY_F6, 'F6')

        self.eq(KEY_F7, b'\033[18~')
        self.eq(KEY_F7, 'F7')

        self.eq(KEY_F8, b'\033[19~')
        self.eq(KEY_F8, 'F8')

        self.eq(KEY_F9, b'\033[20~')
        self.eq(KEY_F9, 'F9')

        self.eq(KEY_F10, b'\033[21~')
        self.eq(KEY_F10, 'F10')

        self.eq(KEY_F11, b'\033[23~')
        self.eq(KEY_F11, 'F11')

        self.eq(KEY_F12, b'\033[24~')
        self.eq(KEY_F12, 'F12')

        for c in 'abcdefghjklnopqrstuvwxyz':
            key = globals()['KEY_CTRL_' + c.upper()]
            self.eq(key, chr(ord(c) - ord('a') + 1))
            self.eq(key, 'ctrl-' + c)
            self.eq(key, 'ctrl+' + c)
            self.eq(key, '^' + c.upper())

    def test_key_invalid_seq_and_alias(self):
        Key = type(KEY_UP)
        with self.raises(TypeError):
            Key(['wah'], 'WAH')

        with self.raises(TypeError):
            Key('wah', ['WAH'])

    def test_key_hash(self):
        self.eq(hash(KEY_UP), hash(KEY_UP.seq))

    def test_key_nameit(self):
        Key = type(KEY_UP)
        TEST_KEY = Key('test_key')
        self.ne(TEST_KEY, 'wah')
        TEST_KEY.nameit('wah')
        self.eq(TEST_KEY, 'wah')
        TEST_KEY.nameit('wah')
        self.eq(TEST_KEY, 'wah')

    def test_key_repr(self):
        self.eq(repr(KEY_UP), 'Key(up)')

        Key = type(KEY_UP)
        new_key = Key('測')
        self.eq(repr(new_key), r"Key('測')")

        seq = '測'.encode('utf8')[:-2]
        new_key2 = Key(seq)
        self.eq(repr(new_key2), r'Key(' + repr(seq) + ')')
