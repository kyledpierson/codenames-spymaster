import numpy as np

from scipy import spatial
from typing import Tuple

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE

Grid = np.array


class ClueGenerator:
    def __init__(self, dictionaryFileName: str):
        self.dictionary: dict = {}
        with open(dictionaryFileName, 'r', encoding="utf8") as dictionaryFile:
            for line in dictionaryFile:
                values: list = line.split()
                word: str = values[0]
                vector: np.array = np.asarray(values[1:], "float32")
                self.dictionary[word] = vector

    def distance(self, firstWord: str, secondWord: str) -> float:
        return spatial.distance.cosine(self.dictionary[firstWord], self.dictionary[secondWord])

    def closestClues(self, word: str) -> list:
        return sorted(self.dictionary.keys(), key=lambda clue: self.distance(word, clue))

    def goodness(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        return sum([self.distance(word, negativeWord) for negativeWord in negativeWords]) - \
               4.0 * sum([self.distance(word, positiveWord) for positiveWord in positiveWords])

    def minimax(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        return min([self.distance(word, negativeWord) for negativeWord in negativeWords]) - \
               max([self.distance(word, positiveWord) for positiveWord in positiveWords])

    def candidates(self, positiveWords: np.array, negativeWords: np.array, size: int = 100) -> list:
        best: list = sorted(self.dictionary.keys(),
                            key=lambda word: -1 * self.goodness(word, positiveWords, negativeWords))
        clues: list = [(str(i + 1), "{0:.2f}".format(self.minimax(word, positiveWords, negativeWords)), word)
                       for i, word in enumerate(sorted(
                best[:200], key=lambda word: -1 * self.minimax(word, positiveWords, negativeWords))[:size])]
        return [(". ".join([clue[0], clue[2]]) + " (" + clue[1] + ")") for clue in clues]

    def getWordsFromCardGrid(self, cardGrid: Grid, team: Team) -> Tuple:
        positiveWords: np.array = []
        negativeWords: np.array = []
        assassinWord: str = ""

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                card: Card = cardGrid[row, col]
                if card.visible:
                    if card.team == team:
                        np.append(positiveWords, card.text)
                    if card.team == team:
                        np.append(negativeWords, card.text)
                    if card.team == team:
                        assassinWord = card.text

        return positiveWords, negativeWords, assassinWord

    def getClueFromLists(self, positiveWords: np.array, negativeWords: np.array, assassinWord: str) -> str:
        return self.candidates(positiveWords, np.append(negativeWords, assassinWord))[0]

    def getClue(self, cardGrid: Grid, team: Team) -> str:
        (positiveWords, negativeWords, assassinWord) = self.getWordsFromCardGrid(cardGrid, team)

        return self.getClueFromLists(positiveWords, negativeWords, assassinWord)
