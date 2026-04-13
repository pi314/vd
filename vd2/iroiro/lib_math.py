from .lib_itertools import unwrap_one

from .internal_utils import exporter
export, __all__ = exporter()


@export
def is_uint8(i):
    return isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 256


@export
def sgn(i):
    return (i > 0) - (i < 0)


@export
def lerp(a, b, t):
    if t == 0:
        return a
    if t == 1:
        return b
    return a + t * (b - a)


@export
def clamp(A, x, B):
    import builtins
    min, max = builtins.min(A, B), builtins.max(A, B)
    return builtins.max(min, builtins.min(x, max))


@export
class vector:
    def __init__(self, *args):
        args = unwrap_one(args)

        if isinstance(args, vector):
            self.data = list(args.data)
        else:
            self.data = list(args)

        for i in self.data:
            if not isinstance(i, (int, float, complex)):
                raise ValueError('Invalid value: {}'.format(repr(tuple(self.data))))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if not isinstance(other, (vector, tuple, list)):
            raise TypeError('Cannot compare with type {}'.format(repr(type(other))))
        return tuple(self.data) == tuple(other)

    def __repr__(self):
        return '(' + ', '.join(map(str, self.data)) + ')'

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return vector(i + other for i in self)
        if len(self) != len(other):
            raise ValueError('Cannot operate on vector(len={}) and {}'.format(len(self), other))
        return vector(map(lambda x: x[0] + x[1], zip(self, other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return vector(i - other for i in self)
        if len(self) != len(other):
            raise ValueError('Cannot operate on vector(len={}) and {}'.format(len(self), other))
        return vector(map(lambda x: x[0] - x[1], zip(self, other)))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return vector(i * other for i in self)
        if len(self) != len(other):
            raise ValueError('Cannot operate on vector(len={}) and {}'.format(len(self), other))
        return vector(map(lambda x: x[0] * x[1], zip(self, other)))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return vector(i / other for i in self)
        raise TypeError('Cannot operate on vector(len={}) and {}'.format(len(self), other))

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return vector(i // other for i in self)
        raise TypeError('Cannot operate on vector(len={}) and {}'.format(len(self), other))

    def map(self, func):
        return vector(func(i) for i in self)


@export
def interval(a, b, *, close=True):
    direction = sgn(b - a)
    if direction == 0:
        return [a] if close else []

    ret = range(a, b + direction, direction)
    if close:
        return list(ret)
    return list(ret[1:-1])


@export
def resample(samples, N):
    if N is None:
        return samples

    n = len(samples)

    if N == n:
        return samples

    if N < n:
        # Averaging skipped samples into N-1 gaps
        skip_count = n - N
        gap_count = N - 1

        probe = 0
        dup, rem = divmod(skip_count, gap_count)

        ret = [samples[0]]
        for i in range(gap_count):
            probe += 1 + dup + (i < rem)
            ret.append(samples[probe])

    if N > n:
        # Duplicate samples to match N
        ret = []
        dup, rem = divmod(N, n)
        for i in range(n):
            for d in range(dup + (i < rem)):
                ret.append(samples[i])

    return tuple(ret)
