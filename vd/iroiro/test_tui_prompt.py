import queue

import unittest.mock

from .lib_test_utils import *

from iroiro import prompt


def queue_to_list(Q):
    ret = []
    while not Q.empty():
        ret.append(Q.get())
    return ret


class TestPromotAskUser(TestCase):
    def setUp(self):
        self.input_queue = None
        self.print_queue = queue.Queue()

        self.patch('iroiro.tui.tui_print', self.mock_print)
        self.patch('iroiro.tui.tui_input', self.mock_input)

        self.mock_open = unittest.mock.mock_open()
        self.patch('builtins.open', self.mock_open)
        self.assert_called_open = True

    def tearDown(self):
        if self.assert_called_open:
            self.mock_open.assert_has_calls([
                    unittest.mock.call('/dev/tty'),
                    unittest.mock.call('/dev/tty', 'w'),
                    unittest.mock.call('/dev/tty', 'w'),
                    ])

            handle = self.mock_open()
            handle.close.assert_has_calls([
                    unittest.mock.call(),
                    unittest.mock.call(),
                    unittest.mock.call(),
                ])
        else:
            self.mock_open.assert_not_called()

    def set_input(self, *lines):
        self.input_queue = queue.Queue()
        for line in lines:
            self.input_queue.put(line)

    def mock_print(self, *args, **kwargs):
        self.print_queue.put(' '.join(args) + kwargs.get('end', '\n'))

    def mock_input(self, prompt=None):
        if prompt:
            self.print_queue.put(prompt)

        if self.input_queue.empty():
            self.fail('Expected more test input')

        s = self.input_queue.get()
        if isinstance(s, BaseException):
            raise s
        return s + '\n'

    def expect_output(self, *args):
        self.eq(queue_to_list(self.print_queue), list(args))

    def test_empty(self):
        with self.raises(TypeError):
            s = prompt()

        self.assert_called_open = False

    def test_continue(self):
        self.set_input('wah')
        yn = prompt('Input anything to continue>')
        self.expect_output('Input anything to continue> ')

        repr(yn)
        self.eq(yn.selected, 'wah')
        self.eq(str(yn), 'wah')
        self.eq(yn, 'wah')
        self.ne(yn, 'WAH')

    def test_coffee_or_tea(self):
        self.set_input('')
        yn = prompt('Coffee or tea?', 'coffee tea')
        self.expect_output('Coffee or tea? [(C)offee / (t)ea] ')

        self.eq(yn.ignorecase, True)
        self.eq(yn.selected, '')
        self.eq(yn, '')
        self.eq(yn, 'coffee')
        self.eq(yn, 'Coffee')
        self.eq(yn, 'COFFEE')
        self.ne(yn, 'tea')

    def test_coffee_or_tea_yes(self):
        self.set_input(
                'what',
                'WHAT?',
                'tea',
                )
        yn = prompt('Coffee or tea?', 'coffee tea both')
        self.expect_output(
                'Coffee or tea? [(C)offee / (t)ea / (b)oth] ',
                'Coffee or tea? [(C)offee / (t)ea / (b)oth] ',
                'Coffee or tea? [(C)offee / (t)ea / (b)oth] ',
                )

        self.eq(yn.selected, 'tea')
        self.eq(yn, 'tea')
        self.ne(yn, 'coffee')

    def test_eoferror(self):
        self.set_input(EOFError())
        yn = prompt('Coffee or tea?', 'coffee tea')
        self.expect_output(
                'Coffee or tea? [(C)offee / (t)ea] ',
                '\n',
                )

        self.eq(yn.selected, None)
        self.eq(yn, None)
        self.ne(yn, 'coffee')
        self.ne(yn, 'tea')
        self.ne(yn, 'water')
        self.ne(yn, 'both')

    def test_keyboardinterrupt(self):
        self.set_input(KeyboardInterrupt())
        yn = prompt('Coffee or tea?', 'coffee tea')
        self.expect_output(
                'Coffee or tea? [(C)offee / (t)ea] ',
                '\n',
                )

        self.eq(yn.selected, None)
        self.eq(yn, None)
        self.ne(yn, 'coffee')
        self.ne(yn, 'tea')
        self.ne(yn, 'water')
        self.ne(yn, 'both')

    def test_suppress(self):
        self.set_input(RuntimeError('wah'), TimeoutError('waaaaah'))
        yn = prompt('Question', suppress=RuntimeError)
        self.eq(yn, None)

        with self.raises(TimeoutError):
            yn = prompt('Question', suppress=RuntimeError)
        self.eq(yn, None)

    def test_sep(self):
        self.set_input('')
        yn = prompt('Coffee or tea?', 'coffee tea', sep='|')
        self.expect_output('Coffee or tea? [(C)offee|(t)ea] ')

    def test_noignorecase(self):
        self.set_input('tea')
        yn = prompt('Coffee or tea?', 'coffee tea', ignorecase=False)
        self.expect_output('Coffee or tea? [(c)offee / (t)ea] ')

        self.eq(yn, 'tea')
        self.ne(yn, 'TEA')

        self.set_input('coFFee', 'coffEE', 'coffee')
        yn = prompt('Coffee or tea?', 'coffee tea', ignorecase=False)
        self.expect_output(
                'Coffee or tea? [(c)offee / (t)ea] ',
                'Coffee or tea? [(c)offee / (t)ea] ',
                'Coffee or tea? [(c)offee / (t)ea] ',
                )

        self.ne(yn, 'coFFee')
        self.ne(yn, 'coffEE')
        self.eq(yn, 'coffee')

    def test_noabbr(self):
        self.set_input('t', 'tea')
        yn = prompt('Coffee or tea?', 'coffee tea', abbr=False)
        self.expect_output(
                'Coffee or tea? [coffee / tea] ',
                'Coffee or tea? [coffee / tea] ',
                )

        self.ne(yn, 't')
        self.eq(yn, 'tea')
        self.ne(yn, 'TEA')

    def test_noaccept_empty(self):
        self.set_input('', 'c')
        yn = prompt('Coffee or tea?', 'coffee tea', accept_empty=False)
        self.expect_output(
                'Coffee or tea? [(c)offee / (t)ea] ',
                'Coffee or tea? [(c)offee / (t)ea] ',
                )

        self.eq(yn.ignorecase, True)
        self.eq(yn.selected, 'c')
        self.eq(yn, 'c')
        self.eq(yn, 'coffee')
        self.eq(yn, 'Coffee')
        self.eq(yn, 'COFFEE')
        self.ne(yn, 'tea')
