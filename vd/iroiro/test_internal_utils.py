from .lib_test_utils import *

from .internal_utils import exporter


class TestInternalUtils(TestCase):
    def test_exporter(self):
        export, uualluu = exporter()

        self.eq(uualluu, [])

        export('iro')
        self.eq(uualluu, ['iro'])

        @export
        def test_func():
            pass

        self.eq(uualluu, ['iro', 'test_func'])

        export()
        self.eq(uualluu, ['iro', 'test_func'])
