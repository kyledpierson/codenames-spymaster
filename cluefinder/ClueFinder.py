import itertools
import numpy as np

from typing import Tuple

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE

Grid = np.array


class ClueFinder:
    # ========== OVERRIDE (PROTECTED) ========== #
    def __init__(self, vocabularySize: int):
        self.vocabularySize: int = vocabularySize

    def _textInVocabulary(self, text: str) -> bool:
        pass

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        pass

    # ========== PUBLIC ========== #
    def checkVocabulary(self, cardGrid: Grid) -> bool:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                text: str = cardGrid[row, col].text
                if not self._textInVocabulary(text):
                    print("ERROR: " + text + " is outside my vocabulary")
                    return False
        return True

    def getClue(self, cardGrid: Grid, team: Team) -> str:
        (positiveWords, negativeWords) = ClueFinder.__getWordsFromCardGrid(cardGrid, team)

        bestClue: str = ""
        bestScore: float = float("-inf")
        bestWords: np.array = np.array([])
        for k in range(1, positiveWords.size + 1):
            combinations: np.array = itertools.combinations(positiveWords, k)
            for connectedWords in combinations:
                (clue, score) = self._getBestClue(connectedWords, negativeWords)
                score = score * k  # TODO: tune k
                if score > bestScore:
                    bestClue = clue
                    bestScore = score
                    bestWords = connectedWords

        print(bestClue + ", " + str(bestScore) + " " + str(bestWords))
        return bestClue + ", " + str(len(bestWords))

    # ========== PROTECTED ========== #
    def _validate(self, clue: str, positiveWords: np.array, negativeWords: np.array) -> bool:
        return clue not in np.append(positiveWords, negativeWords)

    # ========== PRIVATE ========== #
    @staticmethod
    def __getWordsFromCardGrid(cardGrid: Grid, team: Team) -> Tuple:
        positiveWords: np.array = np.array([])
        negativeWords: np.array = np.array([])

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                card: Card = cardGrid[row, col]
                if card.visible:
                    if card.team == team:
                        positiveWords = np.append(positiveWords, card.text)
                    else:
                        negativeWords = np.append(negativeWords, card.text)

        return positiveWords, negativeWords
