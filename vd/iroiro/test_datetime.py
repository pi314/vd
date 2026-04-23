from .lib_test_utils import *

import datetime


class TestDatetime(TestCase):
    def test_isoweekday(self):
        from .lib_datetime import weekday

        self.eq(weekday(1582, 1, 1), 5)
        self.eq(weekday(1583, 1, 1), 6)
        self.eq(weekday(1582, 13, 1), 6)
        self.eq(weekday(1582, 25, 1), 0)
        self.eq(weekday(1789, 7, 14), 2)
        self.eq(weekday(1900, 1, 1), 1)
        self.eq(weekday(1945, 4, 30), 1)
        self.eq(weekday(1969, 7, 20), 0)
        self.eq(weekday(2013, 6, 15), 6)
        self.eq(weekday(2025, 5, 12), 1)
        self.eq(weekday(2025, 2, 34), 4)
        self.eq(weekday(2025, 5, -2), 1)
        self.eq(weekday(2025, 5, 35), 3)
        self.eq(weekday(2025, 1, 1), 3)
        self.eq(weekday(2025, 12, 31), 3)
        self.eq(weekday(2025, 1, 366), 4)
        self.eq(weekday(2026, 3, 3), 2)

    def test_is_leap_year(self):
        from .lib_datetime import is_leap_year

        self.eq(is_leap_year(1), False)
        self.eq(is_leap_year(1582), False)
        self.eq(is_leap_year(1600), True)
        self.eq(is_leap_year(1700), False)
        self.eq(is_leap_year(1800), False)
        self.eq(is_leap_year(1900), False)
        self.eq(is_leap_year(1945), False)
        self.eq(is_leap_year(1969), False)
        self.eq(is_leap_year(2000), True)
        self.eq(is_leap_year(2013), False)
        self.eq(is_leap_year(2024), True)
        self.eq(is_leap_year(2025), False)
        self.eq(is_leap_year(2026), False)

    def test_to_abs_days(self):
        from .lib_datetime import to_abs_days

        def check(year, month, day):
            my = to_abs_days(year, month, day)
            ans = (datetime.date(year, month, day) - datetime.date(1, 1, 1)).days + 1
            self.eq((year, month, day, my), (year, month, day, ans))

        check(1582, 1, 1)
        check(1583, 1, 1)
        self.eq(to_abs_days(1582, 13, 1), to_abs_days(1582, 12, 31) + 1)
        self.eq(to_abs_days(1582, 25, 1), to_abs_days(1584, 1, 1))
        check(1789, 7, 14)
        check(1900, 1, 1)
        check(1945, 4, 30)
        check(1969, 7, 20)
        check(2013, 6, 15)
        check(2025, 5, 12)
        self.eq(to_abs_days(2025, 2, 34), to_abs_days(2025, 2, 28) + 6)
        self.eq(to_abs_days(2025, 5, -2), to_abs_days(2025, 5, 1) - 3)
        self.eq(to_abs_days(2025, 5, 35), to_abs_days(2025, 5, 31) + 4)
        check(2025, 1, 1)
        check(2025, 12, 31)
        self.eq(to_abs_days(2025, 1, 366), to_abs_days(2025, 1, 1) + 365)
        check(2026, 3, 3)

    def test_from_abs_days(self):
        from .lib_datetime import to_abs_days, from_abs_days

        def check(year, month, day):
            abs_days = (datetime.date(year, month, day) - datetime.date(1, 1, 1)).days + 1
            my = from_abs_days(abs_days)
            self.eq((year, month, day), my)

        check(1582, 1, 1)
        check(1583, 1, 1)
        check(1789, 7, 14)
        check(1900, 1, 1)
        check(1945, 4, 30)
        check(1969, 7, 20)
        check(2013, 6, 15)
        check(2025, 5, 12)
        check(2025, 1, 1)
        check(2025, 12, 31)
        check(2026, 3, 3)
