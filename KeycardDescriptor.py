from enum import Enum
from math import floor


class Team(Enum):
    NEUTRAL = 0
    RED = 1
    BLUE = 2
    ASSASSIN = 3


class KeycardDescriptor:
    grid = [[Team.NEUTRAL for i in range(5)] for j in range(5)]

    def __init__(self):
        pass

    def set1D(self, pos: int, team: Team):
        self.grid[int(floor(pos / 5))][pos % 5] = team

    def set2D(self, row: int, col: int, team: Team):
        self.grid[row][col] = team
