import numpy as np

from scipy import spatial
from typing import Tuple

from .ClueFinder import ClueFinder


class BruteForceClueFinder(ClueFinder):
    def __init__(self, vocabularySize: int):
        super().__init__(vocabularySize)

        self.vocabulary: dict = {}
        with open("wordvectors/glove." + str(vocabularySize) + ".txt", "r", encoding="utf8") as vocabularyFile:
            for line in vocabularyFile:
                wordVector: np.array = line.split()
                word: str = wordVector[0]
                vector: np.array = np.asarray(wordVector[1:], float)
                self.vocabulary[word] = vector

    def _textInVocabulary(self, text: str) -> bool:
        return text in self.vocabulary

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        goodClues: np.array = sorted(self.vocabulary.keys(),
                                     key=lambda word: self.__goodness(word, positiveWords, negativeWords), reverse=True)
        bestClues: np.array = [(word, self.__minimax(word, positiveWords, negativeWords)) for word in sorted(
            goodClues[:100], key=lambda word: self.__minimax(word, positiveWords, negativeWords), reverse=True)]

        for (clue, score) in bestClues:
            if self._validate(clue, positiveWords, negativeWords):
                return clue, score

    # ========== PRIVATE ========== #
    def __distance(self, firstWord: str, secondWord: str) -> float:
        return spatial.distance.cosine(self.vocabulary[firstWord], self.vocabulary[secondWord])

    def __goodness(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        else:
            return sum([self.__distance(word, negativeWord) for negativeWord in negativeWords]) - \
                   4.0 * sum([self.__distance(word, positiveWord) for positiveWord in positiveWords])

    def __minimax(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        else:
            return min([self.__distance(word, negativeWord) for negativeWord in negativeWords]) - \
                   max([self.__distance(word, positiveWord) for positiveWord in positiveWords])
