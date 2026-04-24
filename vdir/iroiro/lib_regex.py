import re

from .internal_utils import exporter
export, __all__ = exporter()


@export
class rere:
    def __init__(self, text):
        self.text = text
        self.cache = None

    def search(self, pattern, flags=0):
        self.cache = re.search(pattern, self.text, flags=flags)
        return self.cache

    def match(self, pattern, flags=0):
        self.cache = re.match(pattern, self.text, flags=flags)
        return self.cache

    def fullmatch(self, pattern, flags=0):
        self.cache = re.fullmatch(pattern, self.text, flags=flags)
        return self.cache

    def sub(self, pattern, repl):
        return re.sub(pattern, repl, self.text)

    def __getattr__(self, attr):
        if hasattr(re, attr):
            re_attr = getattr(re, attr)
            return lambda *args, **kwargs: re_attr(*args, self.text, **kwargs)

        return getattr(self.cache, attr)
