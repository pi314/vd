from .internal_utils import exporter
export, __all__ = exporter()


@export
def weekday(year, month, day):
    return to_abs_days(year, month, day) % 7


@export
def is_leap_year(year):
    return (year % 4 == 0) - (year % 100 == 0) + (year % 400 == 0)


@export
def to_abs_days(year, month, day):
    # Normalize to 0
    y = year - 1
    m = month - 1
    d = day - 1

    quo, rem = divmod(m, 12)
    y, m = y + quo, rem

    days_year = (y * 365) + (y // 4) - (y // 100) + (y // 400)
    days_month = (30 * m) + ((m + 1 + (m > 7)) // 2) - (m >= 2) * (2 - is_leap_year(year))
    days_day = day

    return days_year + days_month + days_day


@export
def from_abs_days(days):
    cycles, days = divmod(days, 146097)
    year = cycles * 400
    month = 0
    day = 0

    while True:
        year_len = 365 + is_leap_year(year + 1)
        if days <= year_len:
            break
        days -= year_len
        year += 1

    for mlen in [31, 28 + is_leap_year(year + 1), 31, 30, 31, 30, 31, 31, 30, 31, 30]:
        if days <= mlen:
            break
        month += 1
        days -= mlen

    return (year + 1, month + 1, days)
