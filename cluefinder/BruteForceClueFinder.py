import numpy as np

from typing import Tuple

from .ClueFinder import ClueFinder
from .heuristics import minimax, weightedSum


class BruteForceClueFinder(ClueFinder):
    def __init__(self, vocabularySize: int):
        super().__init__(vocabularySize)

        self.vocabulary: dict = {}
        with open("data/glove." + str(vocabularySize) + ".txt", "r", encoding="utf8") as vocabularyFile:
            for line in vocabularyFile:
                wordVector: np.array = line.split()
                word: str = wordVector[0]
                vector: np.array = np.asarray(wordVector[1:], float)
                self.vocabulary[word] = vector

    def _textInVocabulary(self, text: str) -> bool:
        return text in self.vocabulary

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        goodClues: np.array = sorted(self.vocabulary.keys(), reverse=True,
                                     key=lambda word: weightedSum(word, positiveWords, negativeWords, self.__distance))
        bestClues: np.array = [(word, minimax(word, positiveWords, negativeWords, self.__distance)) for word in
                               sorted(goodClues[:100], reverse=True,
                                      key=lambda word: minimax(word, positiveWords, negativeWords, self.__distance))]

        for (clue, score) in bestClues:
            if self._validate(clue, positiveWords, negativeWords):
                return clue, score, positiveWords

        return "", 0, np.array([])

    # ========== PRIVATE ========== #
    def __distance(self, firstWord: str, secondWord: str) -> float:
        return self.distance(self.vocabulary[firstWord], self.vocabulary[secondWord])
