import numpy as np

from math import floor

Grid = np.array


class Wordgrid:
    grid: Grid = None

    def __init__(self, dim: int):
        self.grid = Grid([["" for col in range(dim)] for row in range(dim)], dtype="U20")

    def set1D(self, pos: int, text: str):
        rows, cols = self.grid.shape
        self.grid[int(floor(pos / cols)), pos % cols] = text

    def set2D(self, row: int, col: int, text: str):
        self.grid[row, col] = text
