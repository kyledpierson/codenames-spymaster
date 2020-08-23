import numpy as np

from math import floor


class Wordgrid:
    grid: np.array = None

    def __init__(self, dim: int = 5):
        self.grid = np.array([["" for col in range(dim)] for row in range(dim)], dtype=str)

    def set1D(self, pos: int, word: str):
        rows, cols = self.grid.shape
        self.grid[int(floor(pos / cols)), pos % cols] = word

    def set2D(self, row: int, col: int, word: str):
        self.grid[row, col] = word
