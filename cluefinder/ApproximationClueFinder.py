import numpy as np

from sklearn import preprocessing, neighbors
from typing import Tuple

from .ClueFinder import ClueFinder


class ApproximationClueFinder(ClueFinder):
    def __init__(self, vocabularySize: int):
        super().__init__(vocabularySize)
        self.vocabulary: dict = {}

        with open("data/glove." + str(vocabularySize) + ".txt", "r", encoding="utf8") as vocabularyFile:
            for line in vocabularyFile:
                self.__addWordVectorToVocabulary(line)

        self.gloveTree = neighbors.NearestNeighbors(n_neighbors=16, algorithm="ball_tree", n_jobs=-1)
        self.gloveTree.fit(np.array(list(self.vocabulary.values())))

    def _textInVocabulary(self, text: str) -> bool:
        return text in self.vocabulary

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        positiveWordVectors: np.array = [self.vocabulary[word] for word in positiveWords]
        negativeWordVectors: np.array = [self.vocabulary[word] for word in negativeWords]
        positiveWordMean: np.array = np.mean(positiveWordVectors, axis=0)

        nearestPoints: np.array = self.gloveTree.kneighbors([positiveWordMean], return_distance=False)[0]

        for nearestPointIndex in nearestPoints:
            clue: str = list(self.vocabulary.keys())[nearestPointIndex]
            if self._validate(clue, positiveWords, negativeWords):
                clueVector: np.array = self.vocabulary[clue]
                score: float = self.heuristic(clueVector, positiveWordVectors, negativeWordVectors, self.distance)
                if score > 0:
                    return clue, score, positiveWords

        return "", 0, np.array([])

    # ========== PRIVATE ========== #
    def __addWordVectorToVocabulary(self, line: str):
        wordVector: np.array = line.split()
        word: str = wordVector[0]
        vector: np.array = np.asarray(wordVector[1:], float)
        self.vocabulary[word] = preprocessing.normalize([vector])[0]
