from .lib_test_utils import *

from iroiro import *


class TestRegex(TestCase):
    def test_match(self):
        rec = rere('iro i ro')

        m = rec.match(r'^(\w+) (\w+)$')
        self.eq(m, None)

        m = rec.match(r'^(\w+) (\w+) (\w+)$')
        self.eq(m.groups(), ('iro', 'i', 'ro'))
        self.eq(m.group(2), 'i')

        self.eq(rec.groups(), m.groups())
        self.eq(rec.group(2), m.group(2))

        self.eq(rec.sub(r'iro', 'IRO'), 'IRO i ro')

        rec.search(r'\bi\b')
        self.eq(rec.start(), 4)
        self.eq(rec.end(), 5)

        self.ne(rec.fullmatch(r'(\w+) (\w+) (\w+)'), None)

        self.eq(rec.split(' +'), rec.text.split(' '))

        self.eq(rec.findall(r'\b\w+\b'), ['iro', 'i', 'ro'])

        self.eq(rec.subn(r'i', 'I'), ('Iro I ro', 2))
