from math import log, log10


def linearRisk(score: float, numWords: int, risk: int) -> float:
    return score + score * (numWords - 1) * (risk / 5)


def quadraticRisk(score: float, numWords: int, risk: int) -> float:
    if numWords > 1 and risk > 0:
        score = numWords * (score - score * numWords * (1 / (18 * risk)))
    return score


def geometricRisk(score: float, numWords: int, risk: int) -> float:
    for i in range(2, numWords):
        score = score * (i / (i - (risk / 5)))
    return score


def logBaseRisk(score: float, numWords: int, risk: int) -> float:
    if numWords > 0 and risk > 0:
        score = score + score * log(numWords ** 2, 12 - risk)
    return score


def logCoefficientRisk(score: float, numWords: int, risk: int) -> float:
    if numWords > 0 and risk > 0:
        score = score + score * risk * log10(numWords)
    return score
