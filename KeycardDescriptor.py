import numpy as np

from enum import Enum
from math import floor


class Team(Enum):
    NEUTRAL = 0
    RED = 1
    BLUE = 2
    ASSASSIN = 3


class KeycardDescriptor:
    grid: np.array = None

    def __init__(self, dim: int = 5):
        self.grid = np.array([[Team.NEUTRAL for col in range(dim)] for row in range(dim)], dtype=Team)

    def set1D(self, pos: int, team: Team):
        rows, cols = self.grid.shape
        self.grid[int(floor(pos / cols)), pos % cols] = team

    def set2D(self, row: int, col: int, team: Team):
        self.grid[row, col] = team
