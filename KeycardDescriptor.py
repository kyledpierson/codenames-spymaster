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

    def __init__(self, dim: int):
        self.grid = np.array([[Team.NEUTRAL for col in range(dim)] for row in range(dim)])

    def set1D(self, pos: int, team: Team):
        self.grid[int(floor(pos / 5)), pos % 5] = team

    def set2D(self, row: int, col: int, team: Team):
        self.grid[row, col] = team
