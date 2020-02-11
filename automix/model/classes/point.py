from enum import Enum
from typing import List


class Shape(Enum):
    """
    Enum for the shape available for the points
    0: linear, 1: square, 2: slow start/end, 3: fast start, 4: fast end, 5: bezier
    """
    LINEAR = 0
    SQUARE = 1
    SIGMOID = 2
    FASTSTART = 3
    FASTEND = 4
    BEZIER = 5

    def jsonEncode(self):
        return str(self)


class Point(object):
    """
    Represents a Reaper's point for automation
    """

    def __init__(self, position=0, amplitude=0, shape=Shape(5), curve=0):
        """
        Position in s
        Amplitude in dB
        Shape as described in Shape Enum
        curve only implemented for the bezier shape. (-1 = fast start, 0 = linear, 1 = fast end)
        """
        self.position = position
        self.amplitude = amplitude
        self.shape = shape
        self.curve = curve