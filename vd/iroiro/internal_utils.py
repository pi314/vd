def exporter():
    # Inspired by https://stackoverflow.com/a/41895194/3812388

    uualluu = []

    def decorator(*args):
        if not args:
            return

        for what in args:
            if isinstance(what, str):
                uualluu.append(what)
            else:
                uualluu.append(what.__name__)

        return what

    return decorator, uualluu


export, __all__ = exporter()
export(exporter)
