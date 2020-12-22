import numpy as np

from scipy import spatial
from sklearn import preprocessing
from typing import Tuple

from .ClueFinder import ClueFinder


class ApproximationClueFinder(ClueFinder):
    def __init__(self, vocabularySize: int):
        super().__init__(vocabularySize)

        self.vocabulary: dict = {}
        with open("wordvectors/glove." + str(vocabularySize) + ".txt", "r", encoding="utf8") as vocabularyFile:
            for line in vocabularyFile:
                wordVector: np.array = line.split()
                word: str = wordVector[0]
                vector: np.array = np.asarray(wordVector[1:], float)
                self.vocabulary[word] = preprocessing.normalize([vector])[0]
        self.gloveTree = spatial.KDTree(np.array(list(self.vocabulary.values())))

    def _textInVocabulary(self, text: str) -> bool:
        return text in self.vocabulary

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        positiveWordVectors: np.array = [self.vocabulary[word] for word in positiveWords]
        negativeWordVectors: np.array = [self.vocabulary[word] for word in negativeWords]
        positiveWordMean: np.array = np.mean(positiveWordVectors, axis=0)

        nearestPoints: np.array = self.gloveTree.query([positiveWordMean], k=16)

        for nearestPointIndex in nearestPoints[1][0]:
            clue: str = list(self.vocabulary.keys())[nearestPointIndex]
            if self._validate(clue, positiveWords, negativeWords):
                clueVector: np.array = self.vocabulary[clue]
                score = ApproximationClueFinder.__calculateScore(clueVector, positiveWordVectors, negativeWordVectors)
                if score > float("-inf"):
                    return clue, score

        return "", float("-inf")

    # ========== PRIVATE ========== #
    @staticmethod
    def __calculateScore(clueVector: np.array, positiveWordVectors: np.array, negativeWordVectors: np.array) -> float:
        sumOfSquares: float = 0
        maxPositiveDistance: float = float("-inf")
        for positiveWordVector in positiveWordVectors:
            positiveDistance: float = spatial.distance.euclidean(positiveWordVector, clueVector)
            sumOfSquares += positiveDistance ** 2
            maxPositiveDistance = max(maxPositiveDistance, positiveDistance)

        for j in range(len(negativeWordVectors)):
            negativeWordVector: np.array = negativeWordVectors[j]
            negativeDistance: float = spatial.distance.euclidean(negativeWordVector, clueVector)
            sumOfSquares -= negativeDistance ** 2
            if negativeDistance < maxPositiveDistance:
                return float("-inf")

        return -sumOfSquares
