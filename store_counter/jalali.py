"""Gregorian <-> Jalali (Persian) calendar conversion, no external dependency."""
import datetime as dt

MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
          "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]

# Python weekday(): Monday=0 ... Sunday=6
WEEKDAYS = {5: "شنبه", 6: "یکشنبه", 0: "دوشنبه", 1: "سه‌شنبه",
            2: "چهارشنبه", 3: "پنجشنبه", 4: "جمعه"}


def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
            - 80 + gd + g_d_m[gm - 1])
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_to_gregorian(jy, jm, jd):
    if jy > 979:
        gy = 1600
        jy -= 979
    else:
        gy = 621
    days = (365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + 78 + jd
            + ((jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186))
    gy += 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    sal_a = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 13 and gd > sal_a[gm]:
        gd -= sal_a[gm]
        gm += 1
    return gy, gm, gd


def to_jalali(d: dt.date):
    return gregorian_to_jalali(d.year, d.month, d.day)


def jalali_str(d: dt.date) -> str:
    jy, jm, jd = to_jalali(d)
    return f"{jy:04d}/{jm:02d}/{jd:02d}"


def jalali_long(d: dt.date) -> str:
    jy, jm, jd = to_jalali(d)
    return f"{WEEKDAYS[d.weekday()]} {jd} {MONTHS[jm - 1]} {jy}"


def jalali_month_range(d: dt.date):
    """(first_day, last_day, title) of the Jalali month containing d."""
    jy, jm, _ = to_jalali(d)
    start = dt.date(*jalali_to_gregorian(jy, jm, 1))
    ny, nm = (jy + 1, 1) if jm == 12 else (jy, jm + 1)
    nxt = dt.date(*jalali_to_gregorian(ny, nm, 1))
    return start, nxt - dt.timedelta(days=1), f"{MONTHS[jm - 1]} {jy}"


def week_range(d: dt.date):
    """Iranian week: Saturday .. Friday."""
    start = d - dt.timedelta(days=(d.weekday() - 5) % 7)
    return start, start + dt.timedelta(days=6)
