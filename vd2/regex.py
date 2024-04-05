import re


class rere:
    def __init__(self, text):
        self.text = text
        self.cache = None

    def match(self, pattern):
        self.cache = re.match(pattern, self.text)
        return self.cache

    def __getattr__(self, attr):
        return getattr(self.cache, attr)

    def sub(self, pattern, repl):
        return re.sub(pattern, repl, self.text)


def selftest():
    from . import selftest
    EXPECT_EQ = selftest.EXPECT_EQ

    rec = rere('my smol tools')

    m = rec.match(r'^(\w+) (\w+)$')
    EXPECT_EQ(m, None)

    m = rec.match(r'^(\w+) (\w+) (\w+)$')
    EXPECT_EQ(m.groups(), ('my', 'smol', 'tools'))
    EXPECT_EQ(m.group(2), 'smol')

    EXPECT_EQ(rec.sub(r'smol', 'SMOL'), 'my SMOL tools')
