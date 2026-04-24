from .internal_utils import exporter
export, __all__ = exporter()


@export
def getter(func):
    return property(func)


@export
def setter(func):
    import inspect
    frame = inspect.stack()[1]
    return frame[0].f_locals[func.__name__].setter(func)


@export
class AlreadyRunningError(RuntimeError):
    pass


@export
class ResourceError(RuntimeError):
    pass


@export
class SignatureError(ValueError):
    pass
