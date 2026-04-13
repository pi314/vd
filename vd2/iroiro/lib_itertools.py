import itertools
import collections

from .internal_utils import exporter
export, __all__ = exporter()


@export
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


@export
def unwrap_one(obj):
    try:
        while True:
            if len(obj) == 1 and iter(obj[0]) and not isinstance(obj[0], str):
                obj = obj[0]
            else:
                return obj
    except TypeError:
        pass

    return obj


@export
def unwrap(obj=None):
    try:
        while True:
            if isinstance(obj, str):
                return obj

            if len(obj) == 1:
                obj = obj[0]
                continue

            return obj
    except TypeError:
        pass

    return obj


@export
def flatten(tree):
    if not is_iterable(tree) or isinstance(tree, str):
        return tree

    wrapper_type = type(tree)
    return wrapper_type(itertools.chain.from_iterable(
        flatten(i) if is_iterable(i) and not isinstance(i, str) else [i]
        for i in tree
        ))


@export
def lookahead(iterable):
    it = iter(iterable)
    try:
        lookahead = next(it)
    except StopIteration:
        return

    for val in it:
        yield lookahead, False
        lookahead = val

    yield lookahead, True


@export
def zip_longest(*iterables, fillvalues=None):
    if not isinstance(fillvalues, (tuple, list)):
        fillvalues = (fillvalues,) * len(iterables)

    iterators = list(map(iter, iterables))

    while True:
        values = []
        cont = False
        for idx, iterator in enumerate(iterators):
            try:
                value = next(iterator)
                cont = True
            except:
                value = fillvalues[idx]

            values.append(value)

        if not cont:
            break

        yield tuple(values)


class Chained:
    def __init__(self, data, type=None):
        self.data = data
        self.type = type

    def __iter__(self):
        return iter(self.data)

    def iter(self):
        return iter(self)

    def eval(self):
        if self.type:
            self.data = (self.type)(self.data)
            return self.data
        else:
            return self.data

    def items(self):
        return Chained(self.data.items(), type=tuple)

    def keys(self):
        return Chained(self.data.keys(), type=tuple)

    def values(self):
        return Chained(self.data.values(), type=tuple)

    def to_dict(self):
        return dict(self.data)

    def to_list(self):
        return list(self.data)

    def to_tuple(self):
        return tuple(self.data)

    def to_set(self):
        return set(self.data)

    def map(self, func):
        if isinstance(self.data, (dict, collections.UserDict)):
            data_type = type(self.data)
            return Chained(data_type(func(key, value) for key, value in self.data.items()))
        else:
            return Chained(map(func, self.data),
                           type=self.type or type(self.data))

    def starmap(self, func):
        return Chained(itertools.starmap(func, self.data),
                       type=self.type or type(self.data))

    def enumerate(self, start=0):
        counter = itertools.count(start=start)
        return self.map(lambda x: (next(counter), x))

    def zip(self, *others, fill=None):
        return Chained(zip(self.data, *[itertools.chain(o, itertools.repeat(fill)) for o in others]),
                       type=self.type or type(self.data))

    def zipleft(self, *others, fill=None):
        return Chained(zip(*[itertools.chain(o, itertools.repeat(fill)) for o in others], self.data),
                       type=self.type or type(self.data))

    def sort(self, key=None):
        new_seq = sorted(self.eval(), key=key)
        return Chained(new_seq)

    def filter(self, func=None):
        if isinstance(self.data, (dict, collections.UserDict)):
            data_type = type(self.data)
            items = self.data.items()
            return Chained(data_type((key, value) for key, value in items if func(key, value)))
        else:
            return Chained(filter(func, self.data),
                           type=self.type or type(self.data))

    def starfilter(self, func=None):
        return Chained(filter(lambda x: func(*x), self.data),
                       type=self.type or type(self.data))

    def reduce(self, func, **kwargs):
        if isinstance(self.data, (dict, collections.UserDict)):
            data_type = type(self.data)
            items = iter(self.data.items())
            ret = kwargs['initial'] if 'initial' in kwargs else next(items)
            for item in items:
                ret = func(ret, item)
            return ret
        else:
            import functools
            args = (kwargs['initial'],) if 'initial' in kwargs else tuple()
            return functools.reduce(func, self.eval(), *args)

    def join(self, sep=' '):
        return sep.join(self.data)

    def max(self, key=lambda x: x):
        return max(self.data, key=key)

    def min(self, key=lambda x: x):
        return min(self.data, key=key)

    def concat(self, *other):
        return Chained(itertools.chain(self.data, *other), type=self.type or type(self.data))


@export
def chaining(data):
    return Chained(data)
