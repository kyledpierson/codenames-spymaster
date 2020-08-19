from math import sqrt

import numpy as np


class Point:
    x: float = 0
    y: float = 0

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def distance(self, other: 'Point') -> float:
        return sqrt(pow(self.x - other.x, 2) + pow(self.y - other.y, 2))

    def midpoint(self, other: 'Point') -> 'Point':
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)

    def divide(self, other: 'Point', dim: int = 5) -> np.array:
        xs: np.array = np.histogram_bin_edges([self.x, other.x], dim)
        ys: np.array = np.histogram_bin_edges([self.y, other.y], dim)
        return np.array([Point(xs[i], ys[i]) for i in range(len(xs))], dtype=Point)
