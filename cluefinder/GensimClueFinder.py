import gensim
import numpy as np

from typing import Tuple

from .ClueFinder import ClueFinder


class GensimClueFinder(ClueFinder):
    def __init__(self, vocabularySize: int):
        super().__init__(vocabularySize)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            "wordvectors/GoogleNews-vectors-negative300.bin", binary=True, limit=vocabularySize)

    def _textInVocabulary(self, text: str) -> bool:
        return text in self.model.vocab

    def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        bestClues: np.array = self.model.most_similar(positive=positiveWords, negative=negativeWords,
                                                      restrict_vocab=self.vocabularySize, topn=16)

        for (clue, score) in bestClues:
            if self._validate(clue, positiveWords, negativeWords):
                return clue, score, positiveWords

        return "", 0, np.array([])
