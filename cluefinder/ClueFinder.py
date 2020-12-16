import gensim
import numpy as np

from scipy import spatial
from sklearn import preprocessing
from typing import Tuple

from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE

Grid = np.array


class ClueFinder:
    def __init__(self, vocabularySize: int):
        self.vocabularySize: int = vocabularySize

        # GLOVE
        self.gloveDictionary: dict = {}
        with open("wordvectors/glove." + str(vocabularySize) + ".txt", 'r', encoding="utf8") as dictionaryFile:
            for line in dictionaryFile:
                values: list = line.split()
                word: str = values[0]
                vector: np.array = np.asarray(values[1:], float)
                self.gloveDictionary[word] = preprocessing.normalize([vector])[0]
        self.gloveTree = spatial.KDTree(np.array(list(self.gloveDictionary.values())))

        # WORD2VEC
        self.gensimModel = gensim.models.KeyedVectors.load_word2vec_format(
            'wordvectors/GoogleNews-vectors-negative300.bin', binary=True, limit=vocabularySize)

    def checkVocabulary(self, cardGrid: Grid) -> bool:
        for card in cardGrid:
            if card.text not in self.gloveDictionary:
                print("ERROR: " + card.text + " is outside my vocabulary")
                return False
        return True

    def distance(self, firstWord: str, secondWord: str) -> float:
        return spatial.distance.cosine(self.gloveDictionary[firstWord], self.gloveDictionary[secondWord])

    def goodness(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        else:
            return sum([self.distance(word, negativeWord) for negativeWord in negativeWords]) - \
                   4.0 * sum([self.distance(word, positiveWord) for positiveWord in positiveWords])

    def minimax(self, word: str, positiveWords: np.array, negativeWords: np.array) -> float:
        if word in np.append(positiveWords, negativeWords):
            return float("-inf")
        else:
            return min([self.distance(word, negativeWord) for negativeWord in negativeWords]) - \
                   max([self.distance(word, positiveWord) for positiveWord in positiveWords])

    def gloveFunction(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        goodClues: list = sorted(self.gloveDictionary.keys(),
                                 key=lambda word: self.goodness(word, positiveWords, negativeWords), reverse=True)
        bestClues: list = [(word, self.minimax(word, positiveWords, negativeWords)) for word in sorted(
            goodClues[:100], key=lambda word: self.minimax(word, positiveWords, negativeWords), reverse=True)]

        return bestClues[0]

    def gloveTreeFunction(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        wordVectors: np.array = [self.gloveDictionary[word] for word in positiveWords]
        wordMean: np.array = np.mean(wordVectors, axis=0)

        nearestPoint: np.array = self.gloveTree.query([wordMean], k=10)[1][0]

        i: int = 0
        nearestPointIndex: int = nearestPoint[i]
        clue: str = list(self.gloveDictionary.keys())[nearestPointIndex]
        while clue in np.append(positiveWords, negativeWords):
            i = i + 1
            nearestPointIndex = nearestPoint[i]
            clue = list(self.gloveDictionary.keys())[nearestPointIndex]

        clueVector: np.array = self.gloveDictionary[clue]
        sumOfSquares: float = sum([spatial.distance.euclidean(wordVector, clueVector) ** 2
                                   for wordVector in wordVectors])

        return clue, -sumOfSquares

    def gensimFunction(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
        bestClues: list = self.gensimModel.most_similar(positive=positiveWords, negative=negativeWords,
                                                        restrict_vocab=self.vocabularySize, topn=1)

        return bestClues[0]

    def getClueFromLists(self, positiveWords: np.array, negativeWords: np.array, clueFunction) -> str:
        bestWords: np.array = ["", "", ""]
        bestClue: Tuple = ("", float("-inf"))

        for i in range(0, positiveWords.size - 1):
            for j in range(i + 1, positiveWords.size - 0):
                connectedWords: np.array = [positiveWords[i], positiveWords[j]]
                candidate: Tuple = clueFunction(connectedWords, negativeWords)
                if candidate[1] > bestClue[1]:
                    bestWords = connectedWords
                    bestClue = candidate

        return bestClue[0] + " (" + bestWords[0] + ", " + bestWords[1] + ") " + str(bestClue[1])

    def getClue(self, cardGrid: Grid, team: Team, model: str) -> str:
        (positiveWords, negativeWords) = self.getWordsFromCardGrid(cardGrid, team)

        def clueFunction(positiveWords: np.array, negativeWords: np.array) -> Tuple:
            return "", 0.0

        if model is "gensim":
            clueFunction = self.gensimFunction
        elif model is "glove":
            clueFunction = self.gloveFunction
        elif model is "gloveTree":
            clueFunction = self.gloveTreeFunction

        return self.getClueFromLists(positiveWords, negativeWords, clueFunction)

    def getWordsFromCardGrid(self, cardGrid: Grid, team: Team) -> Tuple:
        positiveWords: np.array = []
        negativeWords: np.array = []

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                card: Card = cardGrid[row, col]
                if card.visible:
                    if card.team == team:
                        positiveWords = np.append(positiveWords, card.text)
                    else:
                        negativeWords = np.append(negativeWords, card.text)

        return positiveWords, negativeWords
