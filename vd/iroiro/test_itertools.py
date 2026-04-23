from .lib_test_utils import *

from iroiro import *


class TestItertools(TestCase):
    def test_unwrap_one(self):
        self.eq(unwrap_one(1), 1)
        self.eq(unwrap_one((1,)), (1,))
        self.eq(unwrap_one(False), False)
        self.eq(unwrap_one((False,)), (False,))
        self.eq(unwrap_one('text'), 'text')
        self.eq(unwrap_one([1, 2, 3]), [1, 2, 3])
        self.eq(unwrap_one([[1, 2, 3]]), [1, 2, 3])
        self.eq(unwrap_one([[[1, 2, 3]]]), [1, 2, 3])
        self.eq(unwrap_one([[[[1, 2, 3]]]]), [1, 2, 3])
        self.eq(unwrap_one([(1, 2, 3)]), (1, 2, 3))
        self.eq(unwrap_one(((1, 2, 3))), (1, 2, 3))
        self.eq(unwrap_one(([1, 2, 3])), [1, 2, 3])

    def test_unwrap(self):
        def wrap(obj, type=tuple):
            return type((obj,))

        self.eq(unwrap(), None)

        self.eq(unwrap(1), 1)
        self.eq(unwrap(False), False)
        self.eq(unwrap('text'), 'text')

        self.eq(unwrap(wrap(1)), 1)
        self.eq(unwrap(wrap(False)), False)
        self.eq(unwrap(wrap('text')), 'text')

        self.eq(unwrap([1, 2, 3]), [1, 2, 3])
        self.eq(unwrap([[1, 2, 3]]), [1, 2, 3])
        self.eq(unwrap([[[1, 2, 3]]]), [1, 2, 3])
        self.eq(unwrap([[[[1, 2, 3]]]]), [1, 2, 3])

        self.eq(unwrap([(1, 2, 3),]), (1, 2, 3))
        self.eq(unwrap(((1, 2, 3),)), (1, 2, 3))
        self.eq(unwrap(([1, 2, 3],)), [1, 2, 3])

    def test_flatten(self):
        self.eq(flatten(False), False)
        self.eq(flatten('text'), 'text')

        self.eq(
                flatten([[1, 2, 3], [4, 5, 6], [7], [8, 9]]),
                [1, 2, 3, 4, 5, 6, 7, 8, 9]
                )

        self.eq(
                flatten(([1, 2, 3], [4, 5, 6], [7], [8, 9])),
                (1, 2, 3, 4, 5, 6, 7, 8, 9)
                )

        self.eq(
                flatten(([[1, 2, [[]], 3], [4, ([5], 6)], 7, [8, 9]],)),
                (1, 2, 3, 4, 5, 6, 7, 8, 9)
                )

    def test_lookahead(self):
        data = [1, 2, 3, 4, 5]
        self.eq([(val, is_last) for val, is_last in lookahead(data)], [
            (1, False),
            (2, False),
            (3, False),
            (4, False),
            (5, True),
            ])

        data = [1]
        self.eq([(val, is_last) for val, is_last in lookahead(data)], [
            (1, True),
            ])

        data = []
        self.eq([(val, is_last) for val, is_last in lookahead(data)], [])

    def test_zip_longest(self):
        self.eq(list(zip_longest('ABCD', [1, 2])),
                [
                    ('A', 1),
                    ('B', 2),
                    ('C', None),
                    ('D', None),
                    ]
                )

        self.eq(list(zip_longest('AB', [1, 2, 3, 4])),
                [
                    ('A', 1),
                    ('B', 2),
                    (None, 3),
                    (None, 4),
                    ]
                )

        self.eq(list(zip_longest('ABCD', [1, 2], fillvalues='#')),
                [
                    ('A', 1),
                    ('B', 2),
                    ('C', '#'),
                    ('D', '#'),
                    ]
                )

        self.eq(list(zip_longest('AB', [1, 2, 3, 4], fillvalues='#')),
                [
                    ('A', 1),
                    ('B', 2),
                    ('#', 3),
                    ('#', 4),
                    ]
                )

        self.eq(list(zip_longest('ABCD', [1, 2], fillvalues=('#', 0))),
                [
                    ('A', 1),
                    ('B', 2),
                    ('C', 0),
                    ('D', 0),
                    ]
                )

        self.eq(list(zip_longest('AB', [1, 2, 3, 4], fillvalues=('#', 0))),
                [
                    ('A', 1),
                    ('B', 2),
                    ('#', 3),
                    ('#', 4),
                    ]
                )


class TestChain(TestCase):
    def test_chaining_map(self):
        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        res = seq.map(lambda x: x * 2).eval()
        self.isinstance(res, list)
        self.eq(res, [2, 2, 4, 6, 10, 16, 26])

        seq = chaining((1, 1, 2, 3, 5, 8, 13))
        res = seq.map(lambda x: x * 2).eval()
        self.isinstance(res, tuple)
        self.eq(res, (2, 2, 4, 6, 10, 16, 26))

        seq = chaining({1, 1, 2, 3, 5, 8, 13})
        res = seq.map(lambda x: x * 2).eval()
        self.isinstance(res, set)
        self.eq(res, {2, 4, 6, 10, 16, 26})

    def test_chaining_enumerate(self):
        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        self.eq(seq.enumerate().eval(), [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 5),
            (5, 8),
            (6, 13),
            ])

        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        self.eq(seq.enumerate(start=5).eval(), [
            (5, 1),
            (6, 1),
            (7, 2),
            (8, 3),
            (9, 5),
            (10, 8),
            (11, 13),
            ])

    def test_chaining_zip(self):
        import itertools
        seq = chaining([1, 1, 2, 3, 5, 8, 13]).zip(itertools.cycle([0, 1, 2]))
        self.eq(seq.eval(), [
            (1, 0),
            (1, 1),
            (2, 2),
            (3, 0),
            (5, 1),
            (8, 2),
            (13, 0),
            ])

    def test_chaining_zipleft(self):
        import itertools
        seq = chaining([1, 1, 2, 3, 5, 8, 13]).zipleft(itertools.cycle([0, 1, 2]))
        self.eq(seq.eval(), [
            (0, 1),
            (1, 1),
            (2, 2),
            (0, 3),
            (1, 5),
            (2, 8),
            (0, 13),
            ])

    def test_chaining_zip_with_shorter_seq(self):
        import itertools
        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        seq = seq.zip(['foo', 'bar', 'baz'], ['foofoo', 'barbar', 'bazbaz'], fill=True)
        self.eq(seq.eval(), [
            (1, 'foo', 'foofoo'),
            (1, 'bar', 'barbar'),
            (2, 'baz', 'bazbaz'),
            (3, True, True),
            (5, True, True),
            (8, True, True),
            (13, True, True),
            ])

        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        seq = seq.zipleft(['foo', 'bar', 'baz'], ['foofoo', 'barbar', 'bazbaz'], fill=True)
        self.eq(seq.eval(), [
            ('foo', 'foofoo', 1),
            ('bar', 'barbar', 1),
            ('baz', 'bazbaz', 2),
            (True, True, 3),
            (True, True, 5),
            (True, True, 8),
            (True, True, 13),
            ])

    def test_chaining_sort(self):
        data = [1, 1, 2, 3, 5, 8, 13]
        seq = chaining(data[::-1])
        self.eq(seq.sort().eval(), data)

    def test_chaining_filter(self):
        seq = chaining([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 0])
        self.eq(seq.filter().eval(), [1, 2, 3, 4, 5])

        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        self.eq(seq.filter(lambda x: x % 2).eval(), [1, 1, 3, 5, 13])

    def test_chaining_starfilter(self):
        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        self.eq(seq
                .enumerate()
                .starfilter(lambda idx, elem: idx >= 4)
                .starmap(lambda idx, elem: elem)
                .eval(),
                [5, 8, 13])

    def test_chaining_reduce(self):
        seq = chaining([1, 0, 1, 2, 3, 0, 5, 8, 13])
        self.eq(seq.reduce(lambda a, b: a + b), 33)

        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        self.eq(seq.reduce(lambda a, b: a + b, initial=42), 33 + 42)

    def test_chaining_join(self):
        s = chaining('iroiro')
        self.eq(s.map(str.upper).join('.'), 'I.R.O.I.R.O')

    def test_chaining_min_max(self):
        seq = chaining([3, 5, 34, 8, 13, 21])
        self.eq(seq.min(), 3)
        self.eq(seq.max(), 34)
        self.eq(seq.min(key=lambda x: -x), 34)
        self.eq(seq.max(key=lambda x: -x), 3)

    def test_chaining_iter(self):
        seq = chaining([1, 1, 2, 3, 5, 8, 13])
        i = seq.iter()
        i2 = iter(seq)
        self.ne(type(i), list)
        self.eq(list(i), seq.data)
        self.eq(list(i2), seq.data)

    def test_chaining_concat(self):
        res = (chaining([1, 2, 3])
               .concat([4, 5, 6, 7], (8, 9, 10))
               .map(lambda x: x * 2)
               ).eval()
        self.eq(res, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    def test_chaining_in_action(self):
        res = (chaining([1, 0, 1, 2, 3, 0, 5, 8, 13])
               .filter()
               .map(lambda x: x * 2)
               .enumerate()
               .starmap(lambda idx, elem: idx + elem)
               .reduce(lambda a, b: a + b)
               )
        self.eq(res, 87)

    def test_chaining_map_on_dict(self):
        seq = chaining({'foo': 1, 'bar': 2, 'baz': 3, 'qux': 5, 'quux': 8, 'corge': 13})
        res = seq.map(lambda key, value: (key + key, value * 2)).eval()
        self.eq(res, {'foofoo': 2, 'barbar': 4, 'bazbaz': 6, 'quxqux': 10, 'quuxquux': 16, 'corgecorge': 26})

    def test_chaining_filter_on_dict(self):
        seq = chaining({'foo': 1, 'bar': 2, 'baz': 3, 'qux': 5, 'quux': 8, 'corge': 13})
        res = seq.filter(lambda key, value: 'b' not in key).eval()
        self.eq(res, {'foo': 1, 'qux': 5, 'quux': 8, 'corge': 13})

    def test_chaining_reduce_on_dict(self):
        seq = chaining({'foo': 1, 'bar': 2, 'baz': 3, 'qux': 5, 'quux': 8, 'corge': 13})
        res = seq.reduce(lambda acc, item: (acc[0] + item[0], acc[1] + item[1]))
        self.eq(res, ('foobarbazquxquuxcorge', 32))

        seq = chaining({'foo': 1, 'bar': 2, 'baz': 3, 'qux': 5, 'quux': 8, 'corge': 13})
        res = seq.reduce(lambda acc, item: (acc[0] + item[0], acc[1] + item[1]), initial=('+', 5))
        self.eq(res, ('+foobarbazquxquuxcorge', 37))

    def test_chaining_keys_values_items_on_dict(self):
        seq = chaining({'foo': 1, 'bar': 2, 'baz': 3, 'qux': 5, 'quux': 8, 'corge': 13})
        res = seq.keys().map(lambda x: f'({x})').eval()
        self.eq(res, ('(foo)', '(bar)', '(baz)', '(qux)', '(quux)', '(corge)'))

        res = seq.values().map(lambda x: x * 3).eval()
        self.eq(res, (3, 6, 9, 15, 24, 39))

        res = seq.items().map(lambda x: (x[1], x[0])).eval()
        self.eq(res, ((1, 'foo'), (2, 'bar'), (3, 'baz'), (5, 'qux'), (8, 'quux'), (13, 'corge')))

    def test_chaining_casting(self):
        seq = chaining(((1, 'foo'), (2, 'bar'), (3, 'baz'), (5, 'qux'), (8, 'quux'), (13, 'corge')))

        res = seq.to_dict()
        self.eq(res, {1: 'foo', 2: 'bar', 3: 'baz', 5: 'qux', 8: 'quux', 13: 'corge'})

        res = seq.to_list()
        self.eq(res, [(1, 'foo'), (2, 'bar'), (3, 'baz'), (5, 'qux'), (8, 'quux'), (13, 'corge')])

        res = seq.to_tuple()
        self.eq(res, ((1, 'foo'), (2, 'bar'), (3, 'baz'), (5, 'qux'), (8, 'quux'), (13, 'corge')))

        res = seq.to_set()
        self.eq(res, {(1, 'foo'), (2, 'bar'), (3, 'baz'), (5, 'qux'), (8, 'quux'), (13, 'corge')})
