from .lib_test_utils import *

from iroiro import Pager


class TestPager(TestCase):
    def setUp(self):
        from .lib_test_utils import FakeTerminal
        self.terminal = FakeTerminal()
        self.patch('shutil.get_terminal_size', lambda: self.terminal.get_terminal_size())
        self.patch('iroiro.tui.tui_print', lambda *args, **kwargs: self.terminal.print(*args, **kwargs))
        self.patch('iroiro.tui.tui_flush', lambda: None)

    def test_data_storing(self):
        pager = Pager()
        self.true(pager.empty)
        self.eq(pager.lines, tuple())

        pager.append('wah1')
        pager.append('wah2')
        pager.append('wah3')
        pager.extend(['wah4', 'wah5'])
        self.eq(len(pager), 5)
        self.eq(pager.lines, ('wah1', 'wah2', 'wah3', 'wah4', 'wah5'))
        self.false(pager.empty)

        self.eq(pager[1].text, 'wah2')

        pager[1] = 'wahwah'
        self.eq(pager[1].text, 'wahwah')

        pager[1:3] = ['slice1', 'slice2', 'slice3', 'slice4']
        self.eq(pager.lines,
                ('wah1', 'slice1', 'slice2', 'slice3', 'slice4', 'wah4', 'wah5')
                )

    def test_auto_append(self):
        pager = Pager()
        self.true(pager.empty)

        pager[2] = 'line3'
        pager[1] = 'line2'
        self.eq(len(pager), 3)

        pager[4] = 'line5'

        self.eq(pager.lines, (
            '',
            'line2',
            'line3',
            '',
            'line5',
            ))

    def test_render_basic(self):
        pager = Pager()

        self.eq(self.terminal.lines, [''])
        pager.render()
        self.eq(self.terminal.lines, [''])

        data = ['wah1', 'wah2', 'wah3']
        pager.extend(data)

        self.eq(self.terminal.lines, [''])
        pager.render()
        self.eq(self.terminal.lines, data)

        pager[1] = '哇啊'
        self.eq(self.terminal.lines, data)
        pager.render()
        self.eq(self.terminal.lines, ['wah1', '哇啊', 'wah3'])

    def test_render_horizontal_overflow(self):
        pager = Pager()
        self.eq(self.terminal.width, 80)
        self.eq(self.terminal.height, 24)

        pager.append('哇' * 50)
        pager.append('a' + '哇' * 50)
        pager.append('aa' + '哇' * 50)
        pager.render()
        self.eq(self.terminal.lines, [
            '哇' * 40,
            'a' + '哇' * 39,
            'aa' + '哇' * 39,
            ])

    def get_small_terminal_wah_pager(self, **kwargs):
        # Use a smaller terminal to make output less
        self.terminal = FakeTerminal(lines=5, columns=8)
        self.eq(self.terminal.width, 8)
        self.eq(self.terminal.height, 5)

        pager = Pager(**kwargs)

        for i in range(10):
            pager.append('哇 {}'.format(i))

        return pager

    def test_render_vertical_overflow(self):
        pager = self.get_small_terminal_wah_pager()

        self.terminal.recording = True
        pager.render()
        # Check output sequence
        self.eq(self.terminal.recording, [
            '\r哇 0\033[K\n',
            '\r哇 1\033[K\n',
            '\r哇 2\033[K\n',
            '\r哇 3\033[K\n',
            '\r哇 4\033[K',
            ])
        self.terminal.recording = False

        # Check terminal has 5 lines
        self.eq(len(self.terminal.lines), 5)
        for i in range(5):
            self.eq(self.terminal.lines[i], '哇 {}'.format(i))

        # Check cursor position
        self.eq(self.terminal.cursor.y, 4)

    def test_partial_re_render(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        # Update a visible line and an invisible line
        self.terminal.recording = True
        pager[2] = '哇 2 (new)'
        pager[17] = '哇 17 (new)'
        pager.render()

        # The last line is always updated in order to restore cursor position
        self.eq(self.terminal.recording, [
            '\r\033[2A',
            '\r哇 2 (ne\033[K\n',
            '\r\033[1B',
            '\r哇 4\033[K'])
        self.terminal.recording = False

    def test_hard_re_render(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        # A hard re-render
        self.terminal.recording = True
        pager[3] = '哇 3'
        pager.render(all=True)
        self.eq(self.terminal.recording, [
            '\r\033[4A',
            '\r哇 0\033[K\n',
            '\r哇 1\033[K\n',
            '\r哇 2\033[K\n',
            '\r哇 3\033[K\n',
            '\r哇 4\033[K'])
        self.terminal.recording = False

    def test_append_on_the_fly(self):
        self.terminal = FakeTerminal()
        self.terminal.print('previous line')

        pager = Pager()
        self.eq(pager.term_height, self.terminal.height)

        pager.render()
        self.eq(self.terminal.lines, [
            'previous line',
            '',
            ])
        self.eq(self.terminal.cursor.y, 1)

        pager.append('line1')
        self.terminal.recording = True
        pager.render()
        self.eq(self.terminal.recording, [
            '\rline1\033[K',
            ])
        self.eq(self.terminal.cursor.y, 1)
        self.eq(self.terminal.lines, [
            'previous line',
            'line1',
            ])
        self.terminal.recording = False

        self.terminal.recording = True
        pager.append('line2')
        pager.render()
        self.eq(self.terminal.recording, [
            '\n',
            '\rline2\033[K',
            ])
        self.eq(self.terminal.cursor.y, 2)
        self.eq(self.terminal.lines, [
            'previous line',
            'line1',
            'line2',
            ])
        self.terminal.recording = False

        pager.append('line3')
        self.terminal.recording = True
        pager.render()
        self.eq(self.terminal.recording, [
            '\n',
            '\rline3\033[K',
            ])
        self.eq(self.terminal.lines, [
            'previous line',
            'line1',
            'line2',
            'line3',
            ])
        self.terminal.recording = False

        pager[0] = 'line1 new'
        pager.append('line4')
        self.terminal.recording = True
        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[2A',
            '\rline1 new\033[K\n',
            '\r\033[1B\n',
            '\rline4\033[K',
            ])
        self.eq(self.terminal.lines, [
            'previous line',
            'line1 new',
            'line2',
            'line3',
            'line4',
            ])
        self.terminal.recording = False

    def test_pop_insert(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        # pop line[0] and line[2] and then re-render
        self.terminal.recording = True
        pager.pop(0)
        pager.pop(2)
        pager.insert(3, '哇 new')
        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[4A',
            '\r哇 1\033[K\n',
            '\r哇 2\033[K\n',
            '\r哇 4\033[K\n',
            '\r哇 new\033[K\n',
            '\r哇 5\033[K'])
        self.terminal.recording = False

    def test_scrolling(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        self.terminal.recording = True
        pager[6] = '哇 6 (new)'
        pager.scroll += 2
        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[4A',
            '\r哇 2\033[K\n',
            '\r哇 3\033[K\n',
            '\r哇 4\033[K\n',
            '\r哇 5\033[K\n',
            '\r哇 6 (ne\033[K'])
        self.terminal.recording = False

        # Scroll to end
        pager.scroll = pager.end
        self.eq(pager.preview, (
            '哇 5',
            '哇 6 (new)',
            '哇 7',
            '哇 8',
            '哇 9',
            ))

    def test_postition_aliases(self):
        pager = self.get_small_terminal_wah_pager()
        self.eq(pager.home, 0)
        self.eq(pager.end, len(pager) - 1)

    def test_clear(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        # Clear
        self.terminal.recording = True
        pager.clear()
        self.true(pager.empty)
        self.eq(pager.lines, tuple())
        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K',
            ])
        self.eq(pager.display, tuple())
        self.terminal.recording = False

    def test_clear_by_slice_assignment(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        # Clear by slice assignment
        self.terminal.recording = True
        pager[:] = []
        self.eq(pager.lines, tuple())
        pager.render()
        self.eq(pager.display, tuple())
        self.eq(self.terminal.recording, [
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K\033[A',
            '\r\033[K',
            ])
        self.terminal.recording = False

    def test_size_limits(self):
        self.terminal = FakeTerminal()
        self.eq(self.terminal.width, 80)
        self.eq(self.terminal.height, 24)

        pager = Pager()
        pager.max_height = 5
        pager.max_width = 8

        self.lt(pager.max_height, self.terminal.height)
        self.lt(pager.max_width, self.terminal.width)

        self.terminal.recording = True
        pager[0] = 'line0line0'
        pager[1] = 'line1line1'
        pager[2] = 'line2line2'
        pager[3] = 'line3line3'
        pager[4] = 'line4line4'
        pager[5] = 'line5line5'
        pager[6] = 'line6line6'
        self.eq(len(pager), 7)
        pager.render()
        self.eq(len(pager.display), 5)

        self.eq(self.terminal.recording, [
            '\rline0lin\033[K\n',
            '\rline1lin\033[K\n',
            '\rline2lin\033[K\n',
            '\rline3lin\033[K\n',
            '\rline4lin\033[K',
            ])
        self.terminal.recording = False

    def test_header(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        pager.header.append('header')

        self.terminal.recording = True
        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[4A',
            '\rheader\033[K\n',
            '\r哇 0\033[K\n',
            '\r哇 1\033[K\n',
            '\r哇 2\033[K\n',
            '\r哇 3\033[K'])
        self.terminal.recording = False

    def test_footer(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        pager.footer.append('footer')

        self.terminal.recording = True
        pager.render()
        self.eq(self.terminal.recording, [
            '\rfooter\033[K'])
        self.terminal.recording = False

    def test_header_and_footer(self):
        pager = self.get_small_terminal_wah_pager()
        pager.render()

        pager.header.append('header')
        pager.footer.append('footer')

        self.terminal.recording = True

        # scroll down 1 line to test partial update
        pager.scroll += 1

        pager.render()
        self.eq(self.terminal.recording, [
            '\r\033[4A',
            '\rheader\033[K\n',
            '\r\033[3B',
            '\rfooter\033[K'])
        self.terminal.recording = False

    def test_flex(self):
        pager = self.get_small_terminal_wah_pager(flex=True, max_height=5)
        self.true(pager.flex)
        self.eq(pager.height, 5)
        self.eq(pager.max_height, 5)

        pager.clear()
        self.eq(pager.preview, (
            '',
            '',
            '',
            '',
            '',
            ))
        self.eq(pager.height, 5)

        pager.append('line0')
        pager.append('line1')
        pager.footer.append('footer')

        self.eq(pager.preview, (
            'line0',
            'line1',
            '',
            '',
            'footer'
            ))

    def test_thick_header_and_footer(self):
        pager = self.get_small_terminal_wah_pager()

        pager.header.extend(['header' + str(i) for i in range(5)])
        pager.footer.extend(['footer' + str(i) for i in range(5)])

        # Both header and footer are guarenteed to print at least one line
        # header has higher priority
        pager.render()
        self.eq(pager.preview, (
            'header0',
            'header1',
            'header2',
            'header3',
            'footer0',
            ))

        # footer fills the remaining space
        pager.header.clear()
        pager.header.extend(['header0', 'header1'])
        self.eq(pager.preview, (
            'header0',
            'header1',
            'footer0',
            'footer1',
            'footer2',
            ))

        # footer fills all the space
        pager.header.clear()
        self.eq(pager.preview, (
            'footer0',
            'footer1',
            'footer2',
            'footer3',
            'footer4',
            ))

        # body starts to have space to print
        pager.header.append('header0')
        pager.footer.pop(0)
        pager.footer.pop(0)
        pager.footer.pop(0)
        self.eq(pager.preview, (
            'header0',
            '哇 0',
            '哇 1',
            'footer3',
            'footer4',
            ))

        pager.header.append('header1')
        self.eq(pager.preview, (
            'header0',
            'header1',
            '哇 0',
            'footer3',
            'footer4',
            ))

        pager.header.pop(0)
        pager.footer.pop(0)
        self.eq(pager.preview, (
            'header1',
            '哇 0',
            '哇 1',
            '哇 2',
            'footer4',
            ))

        self.eq(list(p.text for p in pager.header), ['header1'])
        self.eq(list(p.text for p in pager), ['哇 0', '哇 1', '哇 2', '哇 3', '哇 4', '哇 5', '哇 6', '哇 7', '哇 8', '哇 9'])
        self.eq(list(p.text for p in pager.footer), ['footer4'])
