import itertools
import numpy as np

from pattern.vector import stem
from pattern.text.en import singularize, pluralize
from scipy import spatial
from typing import Tuple

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE
from .heuristics import weightedSum
from .risks import linearRisk

Grid = np.array


class ClueFinder:
    # ========== OVERRIDE (PROTECTED) ========== #
    def __init__(self, vocabularySize: int):
        self.vocabularySize: int = vocabularySize
        self.previousClues: np.array = np.array([])

        self.distance = spatial.distance.cosine
        self.heuristic = weightedSum
        self.normalize: bool = False
        self.riskFunction = linearRisk

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

    def getClue(self, cardGrid: Grid, team: Team, risk: int) -> str:
        (positiveWords, negativeWords) = ClueFinder.__getWordsFromCardGrid(cardGrid, team)

        clues: list = []
        for k in range(1, positiveWords.size + 1):
            clues += [self._getBestClue(connectedWords, negativeWords)
                      for connectedWords in itertools.combinations(positiveWords, k)]

        if self.normalize:
            clues = ClueFinder.__normalizeScores(clues)

        clues = [(clue[0], self.riskFunction(clue[1], len(clue[2]), risk), clue[2]) for clue in clues]
        clue: Tuple = max(clues, key=lambda clue: clue[1])

        self.previousClues = np.append(self.previousClues, clue[0])

        print(clue[0] + str(clue[2]))
        return clue[0] + ", " + str(len(clue[2]))

    # ========== PROTECTED ========== #
    def _validate(self, clue: str, positiveWords: np.array, negativeWords: np.array) -> bool:
        clue = clue.lower()

        invalidWords: np.array = np.append(self.previousClues, np.append(positiveWords, negativeWords))
        stemmedClue: str = stem(clue)
        singularClue: str = singularize(clue)
        pluralClue: str = pluralize(clue)

        if not clue.isalpha() or not clue.isascii() or set("aeiouy").isdisjoint(clue) or not 2 <= len(clue) <= 12:
            return False

        for word in invalidWords:
            stemmedWord = stem(word)
            singularWord = singularize(word)
            pluralWord = pluralize(word)
            if clue in word or word in clue or stemmedClue in word or stemmedWord in clue or \
                    singularClue in word or singularWord in clue or pluralClue in word or pluralWord in clue:
                return False

        return True

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

    @staticmethod
    def __printTopClues(clues: list, numClues: int = 10):
        clues = sorted(clues, key=lambda clue: clue[1], reverse=True)
        print(clues[:numClues])

    @staticmethod
    def __normalizeScores(clues: list) -> list:
        minScore: float = min(clues, key=lambda clue: clue[1])[1]
        normalizer: float = max(clues, key=lambda clue: clue[1])[1] - minScore
        clues = [(clue[0], (clue[1] - minScore) / normalizer, clue[2]) for clue in clues]
        return clues
