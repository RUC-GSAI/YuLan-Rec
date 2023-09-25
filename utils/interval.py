from datetime import datetime, timedelta
import calendar


class Interval:
    """
    Interval class for each round.
    """
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit


def parse_interval(string):
    value = int(string[:-1])
    unit = string[-1]

    if unit == "s":
        return Interval(value, "seconds")
    elif unit == "m":
        return Interval(value, "minutes")
    elif unit == "h":
        return Interval(value, "hours")
    elif unit == "d":
        return Interval(value, "days")
    elif unit == "M":
        return Interval(value, "months")
    elif unit == "y":
        return Interval(value, "years")
    else:
        raise ValueError("Invalid unit: {}".format(unit))


def add_interval(date, interval):
    if interval.unit == "seconds":
        return date + timedelta(seconds=interval.value)
    elif interval.unit == "minutes":
        return date + timedelta(minutes=interval.value)
    elif interval.unit == "hours":
        return date + timedelta(hours=interval.value)
    elif interval.unit == "days":
        return date + timedelta(days=interval.value)
    elif interval.unit == "months":
        # Adding months to a date is a bit more complex
        new_year = date.year + (date.month + interval.value - 1) // 12
        new_month = (date.month + interval.value - 1) % 12 + 1
        new_day = min(date.day, calendar.monthrange(new_year, new_month)[1])
        return date.replace(year=new_year, month=new_month, day=new_day)
    elif interval.unit == "years":
        # Adding years to a date is similar to adding months
        new_year = date.year + interval.value
        new_month = date.month
        new_day = min(date.day, calendar.monthrange(new_year, new_month)[1])
        return date.replace(year=new_year, month=new_month, day=new_day)
    else:
        raise ValueError("Invalid interval unit: {}".format(interval.unit))


def format_time(date):
    return date.strftime("%B %d, %Y, %I:%M %p")


def get_weekday(date):
    return date.strftime("%A")
