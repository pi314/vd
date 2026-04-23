import builtins

from .internal_utils import exporter
export, __all__ = exporter()


class LineFileWrapper:
    def __init__(self, path, mode, rstrip, newline, **kwargs):
        self.path = path
        self.mode = mode
        self.rstrip = rstrip
        self.newline = newline
        self.kwargs = kwargs

        self.file = builtins.open(self.path, mode=self.mode, **self.kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        cleanup = getattr(self.file, '__exit__')
        if cleanup:
            return cleanup(*args, **kwargs)

        else: # pragma: no cover
            cleanup = getattr(self.file, 'close')
            if cleanup:
                cleanup()

    def __getattr__(self, attr):
        return getattr(self.file, attr)

    def writeline(self, *args):
        self.file.write(' '.join(str(arg) for arg in args) + self.newline)

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def readline(self):
        return self.file.readline().rstrip(self.rstrip)

    def readlines(self):
        return [line for line in self]

    def __iter__(self):
        for line in self.file:
            yield line.rstrip(self.rstrip)


@export
def open(path, mode='rt', rstrip='\r\n', newline='\n', **kwargs):
    # Skip for binary mode
    if 'b' in mode:
        return builtins.open(path, mode=mode, **kwargs)

    kwargs['encoding'] = kwargs.get('encoding', 'utf-8')
    kwargs['errors'] = kwargs.get('errors', 'backslashreplace')

    return LineFileWrapper(path, mode, rstrip=rstrip, newline=newline, **kwargs)


@export
def natsorted(iterable, key=None):
    import re
    def filename_as_key(name):
        def int_or_not(x):
            if x and x[0] in '1234567890':
                return int(x)
            return x
        return tuple(int_or_not(x) for x in re.split(r'([0-9]+)', str(name)))

    sort_key = (filename_as_key
                if key is None
                else lambda x: filename_as_key(key(x))
                )

    return sorted(iterable, key=sort_key)
