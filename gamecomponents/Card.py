from enum import IntEnum


class Team(IntEnum):
    NEUTRAL = 0
    RED = 1
    BLUE = 2
    ASSASSIN = 3


class Card:
    def __init__(self):
        self.team: Team = Team.NEUTRAL
        self.text: str = ""
        self.visible: bool = True

    def __str__(self):
        return str(self.text) + ", " + str(self.team) + ", " + str(self.visible)
