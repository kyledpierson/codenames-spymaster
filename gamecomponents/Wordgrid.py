import numpy as np

from math import floor


class Wordgrid:
    grid: np.array = None

    def __init__(self, dim: int):
        self.grid = np.array([["" for col in range(dim)] for row in range(dim)], dtype=str)

    def set1D(self, pos: int, text: str):
        rows, cols = self.grid.shape
        self.grid[int(floor(pos / cols)), pos % cols] = text

    def set2D(self, row: int, col: int, text: str):
        self.grid[row, col] = text
