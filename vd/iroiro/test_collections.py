from .lib_test_utils import *

from .lib_collections import namablelist


class TestNamableList(TestCase):
    def setUp(self):
        self.nl = namablelist([10, 11, 12, 13])
        self.nl.nameit(0, 'apple')
        self.nl.nameit(1, 'banana')
        self.nl.nameit(2, 'canana')
        self.nl.nameit(3, 'danana')
        self.eq(self.nl, [10, 11, 12, 13])
        self.eq(len(self.nl), 4)

    def test_init_with_kwargs(self):
        nl = namablelist(apple=10, banana=11, canana=12, danana=13)
        self.eq(nl, self.nl)

        self.eq(nl.indexof('apple'), 0)
        self.eq(nl.indexof('banana'), 1)
        self.eq(nl.indexof('canana'), 2)
        self.eq(nl.indexof('danana'), 3)

        with self.raises(ValueError):
            namablelist([1, 2, 3], apple=10)

    def test_unname(self):
        self.eq(self.nl.canana, 12)
        self.eq(self.nl.indexof.canana, 2)
        self.nl.unname('canana')
        with self.raises(KeyError):
            self.nl['canana']
        with self.raises(AttributeError):
            self.nl.canana
        with self.raises(AttributeError):
            self.nl.indexof.canana

    def test_access_through_attr(self):
        nl = self.nl
        self.eq(nl.apple, 10)
        self.eq(nl.banana, 11)
        self.eq(nl.canana, 12)
        self.eq(nl.danana, 13)
        self.true('canana' in dir(nl))
        with self.raises(AttributeError):
            nl.eanana = 14

    def test_keys(self):
        self.eq(self.nl.keys(), ['apple', 'banana', 'canana', 'danana'])

    def test_values(self):
        self.eq(self.nl.values(), list(self.nl))

    def test_indexof_method(self):
        nl = self.nl
        self.eq(nl.indexof('apple'), 0)
        self.eq(nl.indexof('banana'), 1)
        self.eq(nl.indexof('canana'), 2)
        self.eq(nl.indexof('danana'), 3)
        self.eq(nl.indexof('F'), None)
        self.eq(nl.indexof(3), 3)
        self.eq(nl.indexof(10), None)

    def test_indexof_nums(self):
        nl = self.nl
        self.eq(nl.indexof.apple, 0)
        self.eq(nl.indexof.banana, 1)
        self.eq(nl.indexof.canana, 2)
        self.eq(nl.indexof.danana, 3)

    def test_nameof(self):
        nl = self.nl
        self.eq(nl.nameof(0), 'apple')
        self.eq(nl.nameof(1), 'banana')
        self.eq(nl.nameof(2), 'canana')
        self.eq(nl.nameof(3), 'danana')
        self.eq(nl.nameof(4), None)
        self.eq(nl.nameof('danana'), 'danana')
        self.eq(nl.nameof('wah'), None)

    def test_getitem(self):
        nl = self.nl
        self.eq(nl[1], 11)
        self.eq(nl.banana, 11)
        self.eq(nl['banana'], 11)
        self.eq(nl[1:'danana'], [11, 12])

        with self.raises(TypeError):
            nl[nl]

    def test_setitem(self):
        nl = self.nl

        nl[1] = 111
        self.eq(nl[1], 111)

        nl.banana = 1111
        self.eq(nl[1], 1111)

        nl['banana'] = 1111
        self.eq(nl[1], 1111)

        nl[1:'danana'] = [111, 112]
        self.eq(nl[1:3], [111, 112])

        with self.raises(TypeError):
            nl[nl] = 1
