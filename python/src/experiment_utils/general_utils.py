"""I thought I was going to use this more but no, it's noly used in the unicommodity benchmark."""
from math import floor, log10


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))
