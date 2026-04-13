import shlex
import threading

from .iroiro import *


def log(tag, *args, **kwargs):
    if not args and not kwargs:
        print()
        return

    if tag == 'debug':
        color = magenta
    elif tag == 'info':
        color = cyan
    elif tag == 'warning':
        color = yellow
    elif tag == 'error':
        color = red
    else:
        color = nocolor

    if tag:
        print(color(f'[{tag}]'.format(tag=tag)), *args, **kwargs)
    else:
        print(*args, **kwargs)


def stdout(*args, **kwargs):
    print(*args, **kwargs)


def debug(*args, **kwargs):
    log('debug', *args, **kwargs)


def info(*args, **kwargs):
    log('info', *args, **kwargs)


def warning(*args, **kwargs):
    log('warning', *args, **kwargs)


_error_lines = []
_has_err = threading.Event()
def errorq(*args, **kwargs):
    _has_err.set()
    _error_lines.append((args, kwargs))


def errorflush():
    for a, ka in _error_lines:
        log('error', *a, **ka)
    _error_lines.clear()


def error(*args, **kwargs):
    _has_err.set()
    errorflush()
    log('error', *args, **kwargs)


def errorclear():
    _has_err.clear()


def has_error():
    return _has_err.is_set()


def cmd(c, tag=None, **kwargs):
    tokens = [
            murasaki('$'),
            cyan(c[0]),
            ]

    for arg in c[1:]:
        qarg = shlex.quote(str(arg))
        if qarg.startswith(("'", '"')):
            qcolor = orange
        else:
            qcolor = nocolor

        tokens.append(qcolor(qarg))

    log(tag, ' '.join(tokens))
