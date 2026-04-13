from .lib_test_utils import *

import iroiro


class TestMenuData(TestCase):
    def test_basic(self):
        data = iroiro.tui.MenuData()
        data['key'] = 'value'
        self.eq(data.key, 'value')

        data.key = 42
        self.eq(data['key'], 42)
        self.eq(repr(data), "MenuData({'key': 42})")

        del data.what
        self.eq(repr(data), "MenuData({'key': 42})")

        del data.key
        self.eq(repr(data), 'MenuData({})')

        data.key = 52
        self.eq(repr(data), "MenuData({'key': 52})")

        data.key = None
        self.eq(repr(data), 'MenuData({})')


class TestMenuCursor(TestCase):
    def setUp(self):
        self.menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])

    def test_repr(self):
        cursor = self.menu.cursor
        self.eq(repr(self.menu.cursor), f'MenuCursor(pos={cursor.pos}, wrap={cursor.wrap})')

    def test_str(self):
        self.eq(str(self.menu.cursor), '>')

    def test_add_sub_to(self):
        cursor = self.menu.cursor
        self.eq(cursor, 0)

        cursor += 1
        self.eq(cursor, 1)

        cursor += 1
        self.eq(cursor, 2)

        cursor += 10
        self.eq(cursor, 2)

        cursor -= 100
        self.eq(cursor, 0)

        cursor.to(1 + cursor)
        self.eq(cursor, 1)

        cursor.to(1 - cursor)
        self.eq(cursor, 0)

        cursor.to(1)
        self.ne(cursor, 0)
        self.gt(cursor, 0)
        self.ge(cursor, 0)
        self.ge(cursor, 1)
        self.eq(cursor, 1)
        self.le(cursor, 1)
        self.le(cursor, 2)
        self.lt(cursor, 2)
        self.ne(cursor, 2)

        cursor.up()
        self.eq(cursor, 0)

        cursor.down()
        cursor.down()
        self.eq(cursor, 2)

        cursor.to(self.menu[1])
        self.ne(cursor, self.menu[0])
        self.gt(cursor, self.menu[0])
        self.ge(cursor, self.menu[0])
        self.ge(cursor, self.menu[1])
        self.eq(cursor, self.menu[1])
        self.le(cursor, self.menu[1])
        self.le(cursor, self.menu[2])
        self.lt(cursor, self.menu[2])
        self.ne(cursor, self.menu[2])
        self.eq(cursor.text, 'Option 2')

    def test_up_down_wrap(self):
        cursor = self.menu.cursor

        cursor.wrap = True
        cursor.to(31)
        self.eq(cursor, 1)

        cursor.down()
        self.eq(cursor, 2)

        cursor.down()
        self.eq(cursor, 0)

        cursor.up()
        self.eq(cursor, 2)

    def test_to_diff_menu(self):
        other_menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])

        with self.raises(ValueError):
            self.menu.cursor.to(other_menu[1])

    def test_attr(self):
        cursor = self.menu.cursor

        cursor.to(1)
        self.eq(cursor.text, 'Option 2')

        with self.raises(AttributeError):
            cursor.bau

    def test_attr_relay(self):
        cursor = self.menu.cursor
        with self.raises(AttributeError):
            cursor.wrong_attr = 3

        cursor.to(self.menu[0])

        self.eq(cursor.text, self.menu[0].text)

        cursor.text = 'wah'
        self.eq(self.menu[0].text, 'wah')

        self.menu[0].text = 'iroiro'
        self.eq(cursor.text, 'iroiro')

    def test_relay_select_unselect_toggle(self):
        cursor = self.menu.cursor
        cursor.to(self.menu[0])

        cursor.select()
        self.true(self.menu[0].selected)

        cursor.unselect()
        self.false(self.menu[0].selected)

        cursor.toggle()
        self.true(self.menu[0].selected)

    def test_feedkey(self):
        cursor = self.menu.cursor
        cursor.to(self.menu[0])

        import queue
        q = queue.Queue()

        def foo(item, key):
            q.put((item, key))
            return 42
        self.menu[0].bind('k', foo)
        self.eq(cursor.feedkey('k'), 42)
        self.eq(q.get(), (self.menu[0], 'k'))


class TestMenuKeyHandler(TestCase):
    def setUp(self):
        self.menu = iroiro.Menu('title', ['Option 1', 'Option 2'])

    def test_empty_handler(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)
        ret = handler.handle('a')
        self.eq(ret, None)

    def test_bool(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)
        self.false(handler)
        handler.bind(lambda: None)
        self.true(handler)

    def test_bind_without_handler(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)
        with self.raises(ValueError):
            handler.bind('a', 'b', 'c')

    def test_bind_with_wrong_signature(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda item, key: 'k')
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda hello, key: 'k')
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda key, hello: 'k')

        handler = iroiro.tui.MenuKeyHandler(self.menu[0])
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda menu, key: 'k')
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda hello, key: 'k')
        with self.raises(iroiro.SignatureError):
            handler.bind('k', lambda key, hello: 'k')

    def test_bind_unbind_handler(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)

        def foo(menu, key):
            pass
        def bar(menu, key):
            pass
        def baz(menu, key):
            pass

        # Pre-condition
        self.false(handler)

        # Bind single handler
        handler.bind(foo)
        self.eq(handler[None], [foo])

        # Bind multiple handlers
        handler.bind(bar, baz)
        self.eq(handler[None], [foo, bar, baz])

        # Unbind multiple handlers
        handler.unbind(foo, bar, baz)
        self.eq(handler[None], [])

        # Bind multiple handlers packed in a tuple
        handler.bind((foo, bar, baz))
        self.eq(handler[None], [foo, bar, baz])
        handler.unbind((foo, bar, baz))

        # Bind multiple handlers packed in a list
        handler.bind([foo, bar, baz])
        self.eq(handler[None], [foo, bar, baz])
        handler.unbind([foo, bar, baz])

        # Bind multiple handlers packed in a dict
        handler.bind({'k': [foo, bar, baz]})
        self.eq(handler['k'], [foo, bar, baz])
        handler.unbind([foo, bar, baz])

        # Bind with a specified mapping
        handler.bind({'f': foo, 'b': [bar, baz]})
        self.eq(handler['f'], [foo])
        self.eq(handler['b'], [bar, baz])

        # Unbind with a specified mapping
        handler.unbind({'f': foo})
        self.eq(handler['f'], [])
        self.eq(handler['b'], [bar, baz])

        handler.unbind({'b': [bar, baz]})
        self.eq(handler['f'], [])
        self.eq(handler['b'], [])

        # __iadd__ single function
        handler += foo
        self.eq(handler[None], [foo])

        # __iadd__ multiple functions packed in a tuple
        handler += (bar, baz)
        self.eq(handler[None], [foo, bar, baz])
        handler.unbind(foo, bar, baz)

        # __iadd__ a MenuKeyHandler
        h2 = iroiro.tui.MenuKeyHandler(self.menu)
        h2 += (bar, foo)
        handler += h2
        self.eq(handler[None], [bar, foo])
        handler.unbind(foo, bar, baz)
        handler.bind(foo, bar, baz)

        # __isub__
        handler -= bar
        self.eq(handler[None], [foo, baz])
        handler -= (baz, foo)
        self.eq(handler[None], [])

        handler['k'] += foo
        self.eq(handler['k'], [foo])
        handler['w'] += (bar, baz)
        self.eq(handler['w'], [bar, baz])

        handler['k'] = None
        self.eq(handler['k'], [])
        handler -= (baz, foo)
        self.eq(handler['k'], [])
        self.eq(handler['w'], [bar])
        handler['w'] -= [bar]
        self.eq(handler['w'], [])

        handler['k'] = None
        self.eq(handler['k'], [])

        handler['k'] = foo
        self.eq(handler['k'], [foo])
        handler['k'] = bar
        self.eq(handler['k'], [bar])
        handler -= 'k'
        self.eq(handler['k'], [])

    def test_ignore_duplicated_bind(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)

        def foo(menu, key):
            by.append(foo)

        handler.bind(foo)
        handler.bind(foo)

        by = []
        ret = handler.handle('f')
        self.eq(by, [foo])

    def test_key_bubbling(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)

        def foo(menu, key):
            by.append(foo)
            if key == 'f':
                return foo

        def bar(menu, key):
            by.append(bar)
            if key == 'r':
                return bar

        def baz(menu, key):
            by.append(baz)
            if key == 'z':
                return baz

        handler.bind(foo)

        handler.bind('b', bar)
        handler.bind('b', baz)

        handler.bind('f', bar)
        handler.bind('f', baz)

        handler.bind('r', bar)

        handler.bind('z', bar)
        handler.bind('z', baz)

        by = []
        ret = handler.handle('w')
        self.eq(ret, None)
        self.eq(by, [foo])

        by = []
        ret = handler.handle('b')
        self.eq(ret, None)
        self.eq(by, [bar, baz, foo])

        by = []
        ret = handler.handle('f')
        self.eq(ret, foo)
        self.eq(by, [bar, baz, foo])

        by = []
        ret = handler.handle('r')
        self.eq(ret, bar)
        self.eq(by, [bar])

        by = []
        ret = handler.handle('z')
        self.eq(ret, baz)
        self.eq(by, [bar, baz])

    def test_attach_to_menu_item(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu[0])

        by = []
        def foo(item, key):
            by.append(foo)
        handler += foo

        handler.handle('f')
        self.eq(by, [foo])

    def test_attach_to_something_else(self):
        handler = iroiro.tui.MenuKeyHandler(iroiro)

        by = []
        def foo(key):
            by.append(foo)
            return key
        handler += foo

        ret = handler.handle('f')
        self.eq(ret, 'f')
        self.eq(by, [foo])

    def test_handler_flexible_signatures(self):
        handler = iroiro.tui.MenuKeyHandler(self.menu)
        def empty():
            return 'e'
        handler += empty
        self.eq(handler.handle('k'), 'e')

        handler = iroiro.tui.MenuKeyHandler(self.menu)
        def key_only(key):
            return key
        handler += key_only
        self.eq(handler.handle('g'), 'g')

        handler = iroiro.tui.MenuKeyHandler(self.menu)
        def menu_only(menu):
            self.eq(menu, self.menu)
            return 'm'
        handler += menu_only
        self.eq(handler.handle('k'), 'm')

        handler = iroiro.tui.MenuKeyHandler(self.menu)
        def key_and_menu(key, menu):
            self.eq(menu, self.menu)
            return key
        handler += key_and_menu
        self.eq(handler.handle('%'), '%')

        handler = iroiro.tui.MenuKeyHandler(self.menu[0])
        def empty():
            return 'e'
        handler += empty
        self.eq(handler.handle('k'), 'e')

        handler = iroiro.tui.MenuKeyHandler(self.menu[0])
        def key_only(key):
            return key
        handler += key_only
        self.eq(handler.handle('k'), 'k')

        handler = iroiro.tui.MenuKeyHandler(self.menu[0])
        def item_only(item):
            self.eq(item, self.menu[0])
            return 'i'
        handler += item_only
        self.eq(handler.handle('k'), 'i')

        handler = iroiro.tui.MenuKeyHandler(self.menu[0])
        def key_and_item(key, item):
            self.eq(item, self.menu[0])
            return key
        handler += key_and_item
        self.eq(handler.handle('u'), 'u')


class TestMenuItem(TestCase):
    def setUp(self):
        self.menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])

    def test_repr(self):
        self.eq(repr(self.menu[0]), f"MenuItem(index=0, selected=False, text='Option 1')")

    def test_index(self):
        for i in range(len(self.menu)):
            self.eq(self.menu[i].index, i)

    def test_select_unselect_toggle(self):
        i = self.menu[1]
        self.eq(i.selected, False)
        i.select()
        self.eq(i.selected, True)
        i.unselect()
        self.eq(i.selected, False)
        i.toggle()
        self.eq(i.selected, True)
        i.selected = False
        self.eq(i.selected, False)
        i.selected = True
        self.eq(i.selected, True)

    def test_meta_item_select_unselect_toggle(self):
        i = self.menu.append('meta', meta=True)
        self.eq(i.selected, False)
        i.select()
        self.eq(i.selected, False)
        i.unselect()
        self.eq(i.selected, False)
        i.toggle()
        self.eq(i.selected, False)
        i.selected = True
        self.eq(i.selected, False)

    def test_moveto(self):
        i = self.menu[1]
        self.eq(i.index, 1)
        i.moveto(2)
        self.eq(i.index, 2)
        i.moveto(0)
        self.eq(i.index, 0)

    def test_cmp(self):
        self.eq(self.menu[0], 0)
        self.eq(self.menu[0], 'Option 1')
        self.ne(self.menu[0], None)

    def test_onkey(self):
        self.false(self.menu[0].onkey)

        self.menu[0].bind('a', lambda item, key: 'a')
        self.true(self.menu[0].onkey)

        self.menu[0].unbind('a')
        self.false(self.menu[0].onkey)

        self.menu[0].onkey = ('b', lambda item, key: 'b')
        self.true(self.menu[0].onkey)

        self.eq(self.menu[0].feedkey('a'), None)
        self.eq(self.menu[0].feedkey('b'), 'b')

    def test_proxy_class(self):
        def foo(item, key):
            return 'k'
        item = self.menu.Item(text='wah', checkbox='[]')
        self.true(isinstance(item, iroiro.tui.MenuItem))

        item.onkey += ('a', foo)
        self.eq(item.onkey['a'], [foo])
        self.eq(item.menu, self.menu)
        self.eq(item.text, 'wah')
        self.eq(item.check, None)
        self.eq(item.box, '[]')
        self.eq(item.feedkey('a'), 'k')

        item = self.menu.Item(text='wah', check='v', box='//')
        self.eq(item.check, None)
        self.eq(item.box, '//')

        item.select()
        self.eq(item.check, 'v')
        self.eq(item.box, '//')

    def test_meta_item(self):
        item = self.menu.Item(text='wah', meta=True)
        self.eq(item.menu, self.menu)
        self.eq(item.text, 'wah')
        self.eq(item.check, None)
        self.eq(item.box, '{}')

        def meta_check(item):
            return '_-='[item.data.state]
        def meta_box(item):
            return ['||', '|}', '{|'][item.data.state]

        item = self.menu.Item(text='iro', meta=True, checkbox='{+}', check='.')
        self.eq(item.text, 'iro')
        self.eq(item.check, None)
        self.eq(item.box, '{}')

        item.check = meta_check
        item.box = meta_box

        item.data.state = 0
        self.eq(item.check, '_')
        self.eq(item.box, '||')

        item.data.state = 1
        self.eq(item.check, '-')
        self.eq(item.box, '|}')

        item.data.state = 2
        self.eq(item.check, '=')
        self.eq(item.box, '{|')


class TestMenuThread(TestCase):
    def test_menu_thread(self):
        menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])

        checkpoint = self.checkpoint()

        def foo(*args, **kwargs):
            self.eq(args, (1, 2, 3))
            self.eq(kwargs, {'key': 'value'})
            checkpoint.wait()

        t = menu.Thread(target=foo, args=[1, 2, 3], kwargs={'key': 'value'})
        self.false(t.is_alive())

        t.start()
        self.true(t.thread.daemon)
        self.true(t.is_alive())

        with self.raises(iroiro.AlreadyRunningError) as e:
            t.start()

        self.contains(str(e.exception), 'foo()')

        checkpoint.set()

        menu.threads.join()
        self.false(t.is_alive())


class TestMenuStdoutNotTTY(TestCase):
    def test_menu_stdout_not_tty(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'])
        with self.raises(iroiro.Menu.StdoutIsNotAtty):
            menu.interact()


class TestIdleMenuRendering(TestCase):
    def test_idle_menu_rendering(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'])

        class ExtremelyUntouchableClass:
            def __getattr__(self):
                raise Exception('Dont touch me')

        menu.pager = ExtremelyUntouchableClass()

        menu.do_render()


class TestMenuFixture(TestCase):
    def setUp(self):
        from .lib_test_utils import FakeTerminal
        self.terminal = FakeTerminal()
        self.patch('sys.stdout.isatty', lambda *args, **kargs: True)
        self.patch('shutil.get_terminal_size', self.terminal.get_terminal_size)
        self.patch('iroiro.tui.tui_print', lambda *args, **kwargs: self.terminal.print(*args, **kwargs))
        self.patch('iroiro.tui.tui_flush', lambda: None)

        from contextlib import nullcontext
        self.patch('iroiro.tui.HijackStdio', nullcontext)

        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'yes yes', 'surely yes'])
        self.menu_ret = None

        import threading
        self.is_waiting_user = threading.Event()
        self.menu_thread = None

        import queue
        self.key_queue = queue.Queue()
        self.patch('iroiro.tui.getch', self.mock_getch)

    def tearDown(self):
        self.eq(self.menu_thread, None)

    def mock_getch(self, *args, **kwargs):
        self.is_waiting_user.set()
        ret = self.key_queue.get()
        if isinstance(ret, Exception):
            raise ret
        if isinstance(ret, type) and issubclass(ret, Exception):
            raise ret()
        if callable(ret):
            return ret()
        return ret

    def start_menu(self, *args, **kwargs):
        self.eq(self.menu_thread, None)
        self.is_waiting_user.clear()
        def menu_runner(*args, **kwargs):
            try:
                self.menu_ret = self.menu.interact(*args, **kwargs)
            finally:
                self.menu_thread = None
                self.is_waiting_user.set()
        import threading
        self.menu_thread = threading.Thread(target=menu_runner, args=args, kwargs=kwargs)
        self.menu_thread.daemon = True
        self.menu_thread.start()
        self.is_waiting_user.wait()

    def feedkey(self, key):
        menu_thread = self.menu_thread
        self.ne(menu_thread, None)
        self.is_waiting_user.clear()
        self.key_queue.put(key)
        if isinstance(key, EOFError):
            menu_thread.join()
        self.is_waiting_user.wait()


class TestMenuAttributes(TestMenuFixture):
    def test_menu_option_type_error(self):
        with self.raises(TypeError):
            menu = iroiro.Menu(3)
        with self.raises(TypeError):
            menu = iroiro.Menu('probably typo')

    def test_menu_without_title(self):
        menu = iroiro.Menu(['option1', 'option2'])
        self.eq(menu.title, None)
        self.eq(menu[0].text, 'option1')
        self.eq(menu[1].text, 'option2')

        menu = iroiro.Menu(options=['option1', 'option2'])
        self.eq(menu.title, None)
        self.eq(menu[0].text, 'option1')
        self.eq(menu[1].text, 'option2')

    def test_menu_with_message(self):
        menu = iroiro.Menu(['option1', 'option2'], message='message1')
        self.eq(menu.message, 'message1')
        menu.message = 'message2'
        self.eq(menu.message, 'message2')

    def test_menu_wrap(self):
        menu = iroiro.Menu(['option1'], wrap=True)
        self.true(menu.wrap)
        self.true(menu.cursor.wrap)

        menu.wrap = False
        self.false(menu.wrap)
        self.false(menu.cursor.wrap)

    def test_menu_max_height(self):
        menu = iroiro.Menu(['option1'])
        self.eq(menu.max_height, None)

        menu = iroiro.Menu(['option1'], max_height=42)
        self.eq(menu.max_height, 42)
        self.eq(menu.pager.max_height, 42)

        menu.max_height = 53
        self.eq(menu.max_height, 53)
        self.eq(menu.pager.max_height, 53)


class TestMenuDefaultKeyHandler(TestMenuFixture):
    def test_basic_menu_default_key_handler_enter(self):
        self.start_menu()

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            '',
            ])
        self.eq(self.menu.selected.text, 'Yes')

    def test_basic_menu_default_key_handler_q(self):
        self.start_menu()

        self.feedkey('q')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            '',
            ])
        self.eq(self.menu.selected, None)

    def test_basic_menu_default_key_handler_up_down(self):
        self.start_menu()

        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            ])

        self.feedkey(iroiro.KEY_UP)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])

        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            ])

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            '',
            ])
        self.eq(self.menu.selected.text, 'yes yes')

    def test_single_select_menu_default_key_handler_enter(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='()')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> ( ) Yes',
            '  ( ) no',
            ])
        self.eq(self.menu.selected, None)

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> (*) Yes',
            '  ( ) no',
            ])
        self.eq(self.menu.selected, 'Yes')

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> (*) Yes',
            '  ( ) no',
            '',
            ])
        self.eq(self.menu.selected, 'Yes')

        self.terminal.reset()
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='()')
        self.start_menu()
        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  ( ) Yes',
            '> ( ) no',
            ])

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  ( ) Yes',
            '> (*) no',
            ])
        self.eq(self.menu.selected, 'no')
        self.true(self.menu.active)

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  ( ) Yes',
            '> (*) no',
            '',
            ])
        self.eq(self.menu.selected, 'no')
        self.false(self.menu.active)

    def test_single_select_menu_default_key_handler_space(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='()')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> ( ) Yes',
            '  ( ) no',
            ])
        self.eq(self.menu.selected, None)

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> (*) Yes',
            '  ( ) no',
            ])
        self.eq(self.menu.selected, 'Yes')

        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  (*) Yes',
            '> ( ) no',
            ])
        self.eq(self.menu.selected, 'Yes')

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  ( ) Yes',
            '> (*) no',
            ])
        self.eq(self.menu.selected, 'no')

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  ( ) Yes',
            '> ( ) no',
            ])
        self.eq(self.menu.selected, None)

        self.feedkey(EOFError)

    def test_multi_select_menu_default_key_handler_enter(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [ ] Yes',
            '  [ ] no',
            ])
        self.eq(self.menu.selected, [])

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [*] Yes',
            '  [ ] no',
            ])
        self.eq(self.menu.selected, ['Yes'])
        self.true(self.menu.active)

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [*] Yes',
            '  [ ] no',
            '',
            ])
        self.eq(self.menu.selected, ['Yes'])
        self.false(self.menu.active)

        self.terminal.reset()
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]')
        self.start_menu()
        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [ ] Yes',
            '> [ ] no',
            ])

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [ ] Yes',
            '> [*] no',
            ])
        self.eq(self.menu.selected, ['no'])
        self.true(self.menu.active)

        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [ ] Yes',
            '> [*] no',
            '',
            ])
        self.eq(self.menu.selected, ['no'])
        self.false(self.menu.active)

    def test_multi_select_menu_default_key_handler_space(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [ ] Yes',
            '  [ ] no',
            ])
        self.eq(self.menu.selected, [])

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [*] Yes',
            '  [ ] no',
            ])
        self.eq(self.menu.selected, ['Yes'])

        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [*] Yes',
            '> [ ] no',
            ])
        self.eq(self.menu.selected, ['Yes'])

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [*] Yes',
            '> [*] no',
            ])
        self.eq(self.menu.selected, ['Yes', 'no'])

        self.feedkey(iroiro.KEY_UP)
        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [ ] Yes',
            '  [*] no',
            ])
        self.eq(self.menu.selected, ['no'])

        self.feedkey(EOFError)


class TestMenuKeyBinding(TestMenuFixture):
    def test_menu_init_with_key_handler(self):
        def foo(menu):
            menu.quit()
        self.menu = iroiro.Menu('Do you like iroiro?',
                                ['Yes', 'yes yes', 'surely yes'],
                                onkey={'a': foo})
        self.start_menu()
        self.feedkey(iroiro.KEY_ENTER)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])

        self.feedkey('a')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            '',
            ])
        self.eq(self.menu.selected, None)

    def test_menu_bind_unbind_key_handler(self):
        menu = self.menu
        self.false(menu.onkey)
        menu.bind('k', menu.cursor.up)
        menu.bind('j', menu.cursor.down)
        self.true(menu.onkey)

        self.start_menu()
        self.eq(menu.onkey[iroiro.KEY_ENTER], [])
        self.eq(menu.onkey['q'], [])
        self.eq(menu.onkey['k'], [menu.cursor.up])
        self.eq(menu.onkey['j'], [menu.cursor.down])

        self.feedkey('a')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])

        self.feedkey('j')
        self.feedkey('j')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '  yes yes',
            '> surely yes',
            ])

        self.feedkey('k')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            ])

        menu.unbind('k')
        self.eq(menu.onkey['k'], [])

        self.feedkey('k')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            ])

        self.feedkey(EOFError)

    def test_menu_overwrite_key_handler(self):
        menu = self.menu
        menu.onkey = menu.cursor.down

        self.start_menu()
        self.eq(menu.onkey[iroiro.KEY_ENTER], [])
        self.eq(menu.onkey['q'], [])
        self.eq(menu.onkey[None], [menu.cursor.down])

        self.feedkey('a')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '> yes yes',
            '  surely yes',
            ])
        self.feedkey('q')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  Yes',
            '  yes yes',
            '> surely yes',
            ])
        self.feedkey(EOFError)


class TestMenuFormatting(TestMenuFixture):
    def test_menu_builtin_checkbox_types(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'])
        self.eq(menu.check, None)
        self.eq(menu.box, None)

        for checkbox in ('()', 'single', 'radio'):
            menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox=checkbox)
            self.eq(menu.check, '*')
            self.eq(menu.box, '()')

        for checkbox in ('[]', 'multi', 'multiple', 'checkbox'):
            menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox=checkbox)
            self.eq(menu.check, '*')
            self.eq(menu.box, '[]')

        # Unknown type
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='what')
        self.eq(menu.check, None)
        self.eq(menu.box, None)

    def test_menu_custom_radio_checkbox(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='(check)')
        self.eq(self.menu.check, 'check')
        self.eq(self.menu.box, '()')

        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> (     ) Yes',
            '  (     ) no',
            ])

        self.feedkey(iroiro.KEY_SPACE)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> (check) Yes',
            '  (     ) no',
            ])

        self.feedkey(EOFError)

    def test_menu_custom_multiple_checkbox(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[check]')
        self.eq(self.menu.check, 'check')
        self.eq(self.menu.box, '[]')

        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [     ] Yes',
            '  [     ] no',
            ])

        self.feedkey(iroiro.KEY_SPACE)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [check] Yes',
            '  [     ] no',
            ])

        self.feedkey(EOFError)

    def test_menu_custom_cursor(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], cursor='哇')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '哇 Yes',
            '   no',
            ])
        self.feedkey(EOFError)

    def test_menu_custom_format_string(self):
        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]',
                                format='{box[0]}{box[1]} {check} {item.text} {cursor}')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '[]   Yes >',
            '[]   no',
            ])

        self.feedkey(iroiro.KEY_SPACE)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '[] * Yes >',
            '[]   no',
            ])

        self.feedkey(iroiro.KEY_DOWN)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '[] * Yes',
            '[]   no >',
            ])

        self.feedkey(EOFError)

    def test_menu_custom_format_callable(self):
        def format(menu, cursor, item, check, box):
            return f'{cursor} {box[0]}{(item.index + 5) if item.selected else " "}{box[1]} {item.text}'

        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]',
                                format=format)
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [ ] Yes',
            '  [ ] no',
            ])

        self.feedkey(iroiro.KEY_SPACE)
        self.feedkey(iroiro.KEY_DOWN)
        self.feedkey(iroiro.KEY_SPACE)
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [5] Yes',
            '> [6] no',
            ])

        self.feedkey(EOFError)


class TestMenuItemManipulation(TestMenuFixture):
    def test_menu_getitem(self):
        menu = self.menu
        self.eq(menu[0], 'Yes')
        self.eq(menu[1], 'yes yes')
        self.eq(menu[2], 'surely yes')

        menu.cursor = 0
        self.eq(menu[menu.cursor], 'Yes')

        menu.cursor += 1
        self.eq(menu[menu.cursor], 'yes yes')

        menu.cursor += 1
        self.eq(menu[menu.cursor], 'surely yes')

        other_menu = iroiro.Menu('other menu', ['unrelated', 'menu'])
        with self.raises(ValueError):
            menu[other_menu.cursor]

    def test_menu_setitem_str(self):
        menu = self.menu
        self.eq(menu[1], 'yes yes')

        menu[1] = 'yes yes yes'
        self.eq(menu[1], 'yes yes yes')

        menu.cursor = 2
        self.eq(menu[menu.cursor], 'surely yes')

        menu[menu.cursor] = 'of course yes'
        self.eq(menu[2], 'of course yes')

        other_menu = iroiro.Menu('other menu', ['unrelated', 'menu'])
        with self.raises(ValueError):
            menu[other_menu.cursor] = 'ValueError'

    def test_menu_first_last(self):
        menu = self.menu
        self.eq(menu.first, 'Yes')
        self.eq(menu.last, 'surely yes')

    def test_menu_index(self):
        menu = self.menu
        self.eq(menu.index(menu[0]), 0)
        self.eq(menu.index(menu[1]), 1)
        self.eq(menu.index(menu[2]), 2)

        self.eq(menu.index('Yes'), 0)
        self.eq(menu.index('yes yes'), 1)
        self.eq(menu.index('surely yes'), 2)

        menu.cursor = 0
        self.eq(menu.index(menu.cursor), 0)

        menu.cursor = 1
        self.eq(menu.index(menu.cursor), 1)

        menu.cursor = 2
        self.eq(menu.index(menu.cursor), 2)

    def test_menu_index_with_invalid_obj(self):
        self.eq(self.menu.index(self.menu), -1)

    def test_menu_insert(self):
        menu = self.menu
        def foo(item, key):
            return key.upper()
        menu.insert(1, text='text', onkey={'k': foo})
        menu.insert(1, text='text2')

        self.eq(menu[0].text, 'Yes')
        self.eq(menu[1].text, 'text2')
        self.eq(menu[2].text, 'text')
        self.eq(menu[3].text, 'yes yes')
        self.eq(menu[4].text, 'surely yes')

        menu.cursor = 2
        self.eq(menu.feedkey('k'), 'K')

    def test_menu_append(self):
        menu = self.menu
        def foo(item, key):
            return key.upper()
        i3 = menu.append(text='text', onkey={'k': foo})
        i4 = menu.append(text='text2')

        self.eq(menu[0].text, 'Yes')
        self.eq(menu[1].text, 'yes yes')
        self.eq(menu[2].text, 'surely yes')
        self.eq(menu[3].text, 'text')
        self.eq(menu[4].text, 'text2')

        self.eq(i3, menu[3])
        self.eq(i4, menu[4])

        menu.cursor = 3
        self.eq(menu.feedkey('k'), 'K')

    def test_menu_extend(self):
        menu = self.menu
        def foo(item, key):
            return key.upper()
        menu.extend(['text'], onkey={'k': foo})
        menu.extend(['text2'])

        self.eq(menu[0].text, 'Yes')
        self.eq(menu[1].text, 'yes yes')
        self.eq(menu[2].text, 'surely yes')
        self.eq(menu[3].text, 'text')
        self.eq(menu[4].text, 'text2')

        menu.cursor = 3
        self.eq(menu.feedkey('k'), 'K')

    def test_menu_swap(self):
        menu = self.menu
        self.eq(menu[0], 'Yes')
        self.eq(menu[1], 'yes yes')
        self.eq(menu[2], 'surely yes')

        menu.swap(0, 1)
        self.eq(menu[0], 'yes yes')
        self.eq(menu[1], 'Yes')
        self.eq(menu[2], 'surely yes')

        menu.swap(menu.cursor, menu[1])
        self.eq(menu[0], 'Yes')
        self.eq(menu[1], 'yes yes')
        self.eq(menu[2], 'surely yes')

    def test_menu_moveto(self):
        menu = self.menu
        menu.extend(['text1', 'text2'])
        self.eq(menu[0], 'Yes')
        self.eq(menu[1], 'yes yes')
        self.eq(menu[2], 'surely yes')
        self.eq(menu[3], 'text1')
        self.eq(menu[4], 'text2')

        menu[1].moveto(3)
        self.eq(menu[0], 'Yes')
        self.eq(menu[1], 'surely yes')
        self.eq(menu[2], 'text1')
        self.eq(menu[3], 'yes yes')
        self.eq(menu[4], 'text2')

        menu[3].moveto(menu[0])
        self.eq(menu[0], 'yes yes')
        self.eq(menu[1], 'Yes')
        self.eq(menu[2], 'surely yes')
        self.eq(menu[3], 'text1')
        self.eq(menu[4], 'text2')

        menu.moveto(menu[2], 4)
        self.eq(menu[0], 'yes yes')
        self.eq(menu[1], 'Yes')
        self.eq(menu[2], 'text1')
        self.eq(menu[3], 'text2')
        self.eq(menu[4], 'surely yes')

        menu.moveto(menu[4], 0)
        self.eq(menu[0], 'surely yes')
        self.eq(menu[1], 'yes yes')
        self.eq(menu[2], 'Yes')
        self.eq(menu[3], 'text1')
        self.eq(menu[4], 'text2')

        with self.raises(TypeError):
            menu.moveto(1, 2)

    def test_basic_menu_select_all(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'yes yes', 'surely yes'])
        menu.select_all()
        self.eq(menu.selected, None)

    def test_single_menu_select_all(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='()')
        menu.select_all()
        self.eq(menu.selected, None)

    def test_multi_select_menu_select_all(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]')
        menu.select_all()
        self.eq(menu.selected, ['Yes', 'no'])

    def test_multi_select_menu_with_meta_item_select_all(self):
        menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'no'], checkbox='[]')
        menu.append('meta', meta=True)
        menu.select_all()
        self.eq(menu.selected, ['Yes', 'no'])


class TestMenuRendering(TestMenuFixture):
    def test_basic_menu(self):
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])
        self.feedkey(EOFError)

    def test_menu_without_title(self):
        self.menu = iroiro.Menu(['option1', 'option2'])
        self.start_menu()
        self.eq(self.terminal.lines, [
            '> option1',
            '  option2',
            ])
        self.feedkey(EOFError)

    def test_reuse_menu(self):
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])
        self.feedkey('q')

        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])
        self.feedkey('q')

        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])
        self.feedkey('q')

    def test_menu_term_cursor_invisible(self):
        self.true(self.terminal.cursor.visible)

        self.menu = iroiro.Menu('Do you like iroiro?', ['Yes', 'yes yes', 'surely yes'],
                                term_cursor_invisible=True)
        self.start_menu()
        self.false(self.terminal.cursor.visible)

        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> Yes',
            '  yes yes',
            '  surely yes',
            ])
        self.feedkey(EOFError)

        self.true(self.terminal.cursor.visible)


class TestMenuEventDispatcher(TestCase):
    def setUp(self):
        self.menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])
        def foo(): pass
        def bar(): pass
        self.foo = foo
        self.bar = bar

    def test_bind_to_wrong_target(self):
        with self.raises(TypeError):
            iroiro.tui.MenuEventDispatcher(iroiro)

    def test_bound_target(self):
        menu = self.menu
        self.true(menu.onevent.target is menu)
        item = self.menu[0]
        self.true(item.onevent.target is item)

    def test_initial_empty(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        self.false(ed)

    def test_event_handler_installer_repr(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        self.eq(repr(ed.iroiro), "MenuEventHandlerInstaller(event='iroiro')")

    def test_bind_unbind_default_handler_by_call(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed(self.foo)
        self.true(ed)
        self.eq(ed[None], self.foo)

        ed(None)
        self.false(ed)
        self.eq(ed[None], None)

    def test_bool_and_eq(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        self.false(ed)

        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed.set_to(self.foo)
        self.true(ed)
        self.eq(ed, self.foo)

        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed.set_to(('iroiro', self.bar))
        self.true(ed)
        self.ne(ed, self.foo)
        self.ne(ed, self.bar)

    def test_bind_unbind_handler_by_set_to(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)

        # Bind default handler
        ed.set_to(self.bar)
        self.eq(ed[None], self.bar)

        # Unbind default handler
        ed.set_to(None)
        self.eq(ed[None], None)

        # Bind event handler
        ed.set_to(('iroiro', self.bar))
        self.eq(ed.iroiro, self.bar)

        # Short circuit prevention
        ed.set_to(ed)
        self.eq(ed.iroiro, self.bar)

    def test_bind_unbind_by_call(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed('iroiro', self.foo)
        self.eq(ed['iroiro'], self.foo)

        ed('iroiro', None)
        self.eq(ed['iroiro'], None)

    def test_bind_unbind_by_setattr_getattr(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed.iroiro = self.bar
        self.eq(ed.iroiro, self.bar)

        ed.iroiro = None
        self.eq(ed.iroiro, None)

    def test_bind_unbind_by_bind_unbind(self):
        ed = iroiro.tui.MenuEventDispatcher(self.menu)
        ed.bind('iroiro', self.foo)
        self.eq(ed.iroiro, self.foo)

        ed.bind('iroiro', self.bar)
        self.eq(ed.iroiro, self.bar)

        ed.unbind('iroiro')
        self.eq(ed.iroiro, None)

        ed.unbind('iroiro')
        self.eq(ed.iroiro, None)


class TestMenuEventHandler(TestCase):
    def setUp(self):
        self.menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])
        self.ed = iroiro.tui.MenuEventDispatcher(self.menu)
        def foo(): pass
        def bar(): pass
        self.foo = foo
        self.bar = bar

    def test_bool_and_eq(self):
        eh = iroiro.tui.MenuEventHandler()
        self.false(eh)
        self.eq(eh, None)

        eh = iroiro.tui.MenuEventHandler(self.foo)
        self.true(eh)
        self.eq(eh, self.foo)

    def test_set_event_handler_by_call(self):
        eh = iroiro.tui.MenuEventHandler()
        eh(self.foo)
        self.eq(eh, self.foo)

        eh(None)
        self.eq(eh, None)

    def test_set_to(self):
        eh = iroiro.tui.MenuEventHandler()

        eh.set_to(self.foo)
        self.eq(eh, self.foo)

        eh.set_to(eh)
        self.eq(eh, self.foo)

        eh.set_to(None)
        self.eq(eh, None)

        with self.raises(ValueError):
            eh.set_to(42)

        eh.set_to(self.bar)
        self.eq(eh, self.bar)

        eh2 = iroiro.tui.MenuEventHandler(self.foo)
        eh.set_to(eh2)
        self.eq(eh, self.foo)


class TestMenuEvent(TestCase):
    def test_menu_menu_onevent_attrs(self):
        menu = iroiro.Menu('title', ['Option 1', 'Option 2', 'Option 3'])
        def foo():
            pass

        self.false(menu.onevent)
        menu.onevent = foo
        self.true(menu.onevent)

        self.eq(menu.onsubmit, None)
        menu.onsubmit = foo
        self.eq(menu.onsubmit, foo)
        self.eq(menu.onsubmit, menu.onevent['submit'])

        self.eq(menu.onquit, None)
        menu.onquit = foo
        self.eq(menu.onquit, foo)
        self.eq(menu.onquit, menu.onevent['quit'])

        self.eq(menu.onselect, None)
        menu.onselect = foo
        self.eq(menu.onselect, foo)
        self.eq(menu.onselect, menu.onevent['select'])

        self.eq(menu.onunselect, None)
        menu.onunselect = foo
        self.eq(menu.onunselect, foo)
        self.eq(menu.onunselect, menu.onevent['unselect'])

    def test_menu_item_onevent_attrs(self):
        menu = iroiro.Menu('title', ['item1', 'item2', 'item3'])
        def foo():
            pass

        self.false(menu[0].onevent)
        menu[0].onevent = foo
        self.true(menu[0].onevent)

        self.eq(menu[1].onselect, None)
        menu[1].onselect = foo
        self.eq(menu[1].onselect, foo)
        self.eq(menu[1].onselect, menu[1].onevent['select'])

        self.eq(menu[2].onunselect, None)
        menu[2].onunselect = foo
        self.eq(menu[2].onunselect, foo)
        self.eq(menu[2].onunselect, menu[2].onevent['unselect'])

    def test_menu_event_bubbling_onselect(self):
        menu = iroiro.Menu('title', ['through', 'block', 'reject'], checkbox='[]')

        # item onselect handler
        checkpoint_item = self.checkpoint()
        def item_onselect(item):
            checkpoint_item.set()
            if item == 'block':
                return True
            if item == 'reject':
                return False
        for item in menu:
            item.onselect(item_onselect)

        # menu onselect handler
        checkpoint_menu = self.checkpoint()
        def menu_onselect(menu):
            checkpoint_menu.set()
        menu.onselect(menu_onselect)

        # Event bubble to menu
        menu[0].select()
        checkpoint_item.verify()
        checkpoint_menu.verify()

        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked by item level onselect and select succ
        menu[1].select()
        self.true(menu[1].selected)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)

        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked by item level onselect and select fail
        menu[2].select()
        self.false(menu[2].selected)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)

    def test_menu_event_bubbling_onunselect(self):
        menu = iroiro.Menu('title', ['through', 'block', 'reject'], checkbox='[]')

        # item onunselect handler
        checkpoint_item = self.checkpoint()
        def item_onunselect(item):
            checkpoint_item.set()
            if item == 'block':
                return True
            if item == 'reject':
                return False
        for item in menu:
            item.onunselect(item_onunselect)

        # menu onunselect handler
        checkpoint_menu = self.checkpoint()
        def menu_onunselect(menu):
            checkpoint_menu.set()
        menu.onunselect(menu_onunselect)

        # select all items for later test
        for item in menu:
            item.select()
            self.true(item.selected)
        self.eq(menu.selected, [menu[0], menu[1], menu[2]])

        # Event bubble to menu
        menu[0].unselect()
        checkpoint_item.verify()
        checkpoint_menu.verify()

        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked by item level onunselect and unselect succ
        menu[1].unselect()
        self.false(menu[1].selected)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)

        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked by item level onunselect and unselect fail
        menu[2].unselect()
        self.true(menu[2].selected)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)

    def test_menu_event_bubbling_onsubmit(self):
        menu = iroiro.Menu('title', ['item', 'key'], checkbox='[]')

        # menu onsubmit handler
        checkpoint_menu = self.checkpoint()
        def menu_onsubmit(menu):
            checkpoint_menu.set()
            if not menu[1].selected:
                return False
        menu.onsubmit(menu_onsubmit)

        # Submit rejected
        self.eq(menu.submit(), False)
        checkpoint_menu.verify()
        checkpoint_menu.clear()

        # Submit accepted
        with self.raises(iroiro.Menu.DoneSelection):
            menu[1].select()
            menu.submit()
        checkpoint_menu.verify()
        checkpoint_menu.clear()

    def test_menu_event_bubbling_onquit(self):
        menu = iroiro.Menu('title', ['item', 'key'], checkbox='[]')

        # menu onquit handler
        checkpoint_menu = self.checkpoint()
        def menu_onquit(menu):
            checkpoint_menu.set()
            return False
        menu.onquit(menu_onquit)

        # Quit is not stoppable
        with self.raises(iroiro.Menu.GiveUpSelection):
            menu.quit()
        checkpoint_menu.verify()

    def test_menu_event_bubbling_user_defined_event(self):
        menu = iroiro.Menu('title', ['skipped', 'through', 'block', 'reject'], checkbox='[]')

        # item oniroiro handler
        checkpoint_item = self.checkpoint()
        expect_kwargs = {}
        expect_item = None
        def item_oniroiro(event, item, **kwargs):
            self.eq(expect_kwargs, kwargs)
            checkpoint_item.set()
            if item == 'block':
                return True
            if item == 'reject':
                return False
        for item in menu[1:]:
            item.onevent('iroiro', item_oniroiro)

        # menu oniroiro handler
        checkpoint_menu = self.checkpoint()
        def menu_oniroiro(event, menu, item, **kwargs):
            self.eq(expect_item, item)
            self.eq(expect_kwargs, kwargs)
            checkpoint_menu.set()
        menu.onevent('iroiro', menu_oniroiro)

        # Event bubble to menu
        expect_kwargs = {'key': 'value'}
        expect_item = menu[0]
        menu[0].emit('iroiro', key='value')
        checkpoint_item.verify(False)
        checkpoint_menu.verify()
        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event bubble to menu
        expect_kwargs = {'key': 'value2'}
        expect_item = menu[1]
        ret = menu[1].emit('iroiro', key='value2')
        checkpoint_item.verify()
        checkpoint_menu.verify()
        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked
        expect_kwargs = {'key': 'value3'}
        expect_item = menu[2]
        ret = menu[2].emit('iroiro', key='value3')
        self.eq(ret, True)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)
        checkpoint_item.clear()
        checkpoint_menu.clear()

        # Event blocked
        expect_kwargs = {'key': 'value4'}
        expect_item = menu[3]
        ret = menu[3].emit('iroiro', key='value4')
        self.eq(ret, False)
        checkpoint_item.verify()
        checkpoint_menu.verify(False)
        checkpoint_item.clear()
        checkpoint_menu.clear()


class TestMenuScrolling(TestMenuFixture):
    def test_menu_top_bottom_none(self):
        self.menu = iroiro.Menu('Do you like iroiro?', options=['item1'])
        self.eq(self.menu.top, None)
        self.eq(self.menu.bottom, None)

    def test_menu_with_message(self):
        self.menu = iroiro.Menu('Do you like iroiro?', options=['item1', 'item2'], message='message')
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item1',
            '  item2',
            'message'
            ])
        self.feedkey(EOFError)

    def test_menu_with_limited_height(self):
        self.menu = iroiro.Menu('Do you like iroiro?', [f'item{i}' for i in range(10)], max_height=6)
        self.start_menu()

        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item0',
            '  item1',
            '  item2',
            '  item3',
            '  item4',
            ])
        self.eq(self.menu.top, 'item0')
        self.eq(self.menu.bottom, 'item4')

        self.feedkey(EOFError)

    def test_menu_scroll_pull_window(self):
        self.menu = iroiro.Menu('Do you like iroiro?', [f'item{i}' for i in range(10)], max_height=6)
        self.menu.onkey('g', lambda menu: menu.cursor.to(menu.first))
        self.menu.onkey('H', lambda menu: menu.cursor.to(menu.top))
        self.menu.onkey('L', lambda menu: menu.cursor.to(menu.bottom))
        self.menu.onkey('G', lambda menu: menu.cursor.to(menu.last))
        self.start_menu()

        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item0',
            '  item1',
            '  item2',
            '  item3',
            '  item4',
            ])
        self.eq(self.menu.top, 'item0')
        self.eq(self.menu.bottom, 'item4')

        self.feedkey('L')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item0',
            '  item1',
            '  item2',
            '  item3',
            '> item4',
            ])
        self.eq(self.menu.top, 'item0')
        self.eq(self.menu.bottom, 'item4')

        self.feedkey('G')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item5',
            '  item6',
            '  item7',
            '  item8',
            '> item9',
            ])
        self.eq(self.menu.top, 'item5')
        self.eq(self.menu.bottom, 'item9')

        self.feedkey('H')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item5',
            '  item6',
            '  item7',
            '  item8',
            '  item9',
            ])
        self.eq(self.menu.top, 'item5')
        self.eq(self.menu.bottom, 'item9')

        self.feedkey('g')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item0',
            '  item1',
            '  item2',
            '  item3',
            '  item4',
            ])
        self.eq(self.menu.top, 'item0')
        self.eq(self.menu.bottom, 'item4')

        self.feedkey(EOFError)

    def test_menu_scroll_pull_cursor(self):
        self.menu = iroiro.Menu('Do you like iroiro?', [f'item{i}' for i in range(10)], max_height=6)
        self.menu.onkey('e', lambda menu: menu.scroll())
        self.menu.onkey('y', lambda menu: menu.scroll(-1))
        self.menu.onkey('G', lambda menu: menu.cursor.to(menu.last))
        self.start_menu()

        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item0',
            '  item1',
            '  item2',
            '  item3',
            '  item4',
            ])
        self.eq(self.menu.top, 'item0')
        self.eq(self.menu.bottom, 'item4')

        self.feedkey('e')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item1',
            '  item2',
            '  item3',
            '  item4',
            '  item5',
            ])
        self.eq(self.menu.top, 'item1')
        self.eq(self.menu.bottom, 'item5')

        self.feedkey('e')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> item2',
            '  item3',
            '  item4',
            '  item5',
            '  item6',
            ])
        self.eq(self.menu.top, 'item2')
        self.eq(self.menu.bottom, 'item6')

        self.feedkey('y')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item1',
            '> item2',
            '  item3',
            '  item4',
            '  item5',
            ])
        self.eq(self.menu.top, 'item1')
        self.eq(self.menu.bottom, 'item5')

        self.feedkey('G')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item5',
            '  item6',
            '  item7',
            '  item8',
            '> item9',
            ])
        self.eq(self.menu.top, 'item5')
        self.eq(self.menu.bottom, 'item9')

        self.feedkey('y')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item4',
            '  item5',
            '  item6',
            '  item7',
            '> item8',
            ])
        self.eq(self.menu.top, 'item4')
        self.eq(self.menu.bottom, 'item8')

        self.feedkey('y')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  item3',
            '  item4',
            '  item5',
            '  item6',
            '> item7',
            ])
        self.eq(self.menu.top, 'item3')
        self.eq(self.menu.bottom, 'item7')

        self.feedkey(EOFError)


class TestMenuMetaItems(TestMenuFixture):
    def test_render_meta_item(self):
        self.menu = iroiro.Menu('Do you like iroiro?', options=['item1', 'item2'], checkbox='[]')

        def meta_check(item):
            if all(i.selected for i in item.menu if not i.meta):
                return '*'
            elif all(not i.selected for i in item.menu if not i.meta):
                return ' '
            else:
                return '+'

        def meta_box(item):
            if all(i.selected for i in item.menu if not i.meta):
                return '||'
            elif all(not i.selected for i in item.menu if not i.meta):
                return '__'
            else:
                return '|}'

        self.menu.append('item meta', meta=True, check=meta_check, box=meta_box)
        self.start_menu()
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [ ] item1',
            '  [ ] item2',
            '  _ _ item meta',
            ])

        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '> [*] item1',
            '  [ ] item2',
            '  |+} item meta',
            ])

        self.feedkey(iroiro.KEY_DOWN)
        self.feedkey(' ')
        self.eq(self.terminal.lines, [
            'Do you like iroiro?',
            '  [*] item1',
            '> [*] item2',
            '  |*| item meta',
            ])

        self.feedkey(EOFError)
