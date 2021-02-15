import numpy as np


def leastSquares(word, positiveWords: np.array, negativeWords: np.array, distance) -> float:
    sumOfSquares: float = 0

    maxPositiveDistance: float = 0
    for positiveWord in positiveWords:
        positiveDistance: float = distance(positiveWord, word)
        sumOfSquares -= positiveDistance ** 2
        maxPositiveDistance = max(maxPositiveDistance, positiveDistance)
    sumOfSquares /= len(positiveWords)

    minNegativeDistance: float = float("inf")
    for negativeWord in negativeWords:
        negativeDistance: float = distance(negativeWord, word)
        if negativeDistance < maxPositiveDistance:
            return 0
        minNegativeDistance = min(minNegativeDistance, negativeDistance)
    sumOfSquares += minNegativeDistance ** 2

    return sumOfSquares


def minimax(word, positiveWords: np.array, negativeWords: np.array, distance) -> float:
    return min([distance(word, negativeWord) for negativeWord in negativeWords]) - \
           max([distance(word, positiveWord) for positiveWord in positiveWords])


def weightedSum(word, positiveWords: np.array, negativeWords: np.array, distance) -> float:
    return sum([distance(word, negativeWord) for negativeWord in negativeWords]) - \
           4.0 * sum([distance(word, positiveWord) for positiveWord in positiveWords])
