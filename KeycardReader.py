import cv2
import numpy as np
from typing import Tuple

from imageprocessing.Filterer import Filterer
from imageprocessing.Morpher import Morpher
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Point, BoundingBox, Segmenter
from KeycardDescriptor import Team, KeycardDescriptor

Image = np.ndarray

blueMean = [200, 125, 25]
blueRange = ((120, 0, 0), (255, 180, 80))
redMean = [50, 50, 200]
redRange = ((0, 0, 150), (100, 70, 255))
blackMean = [0, 0, 0]
blackRange = ((0, 0, 0), (50, 50, 50))
distThreshold = (155, 255)


def xCoord(box):
    return box[0][0]


def yCoord(box):
    return box[0][1]


class KeycardReader:
    def __init__(self, referenceImageFileName: str):
        self.referenceImage = cv2.imread(referenceImageFileName)
        self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycardDescriptor(self, path: str) -> Tuple:
        descriptor = KeycardDescriptor()

        image = cv2.imread(path)
        rows, cols = image.shape[:2]
        segmentedImage = Segmenter.grabcut(image, 'rectangular')
        segmentedImage, _ = Preprocessor.resizeToSameSize(segmentedImage, self.referenceImage, 500, 500)

        blackResponseImage = segmentedImage[:, :, 0].copy()
        for i in range(rows):
            for j in range(cols):
                blackResponseImage[i, j] = max(0, 255 - np.linalg.norm(
                    np.array(blackMean) - np.array(segmentedImage[i, j])
                ))

        thresholdedImage = Segmenter.threshold(segmentedImage, blackResponseImage, 'otsu', distThreshold)
        rectangleImage, outerBox, innerBoxes = Segmenter.drawRectangles(blackResponseImage)

        # innerBoxes = sorted(sorted(innerBoxes, key=xCoord), key=yCoord)
        # b = 0
        # g = 0
        # r = 0
        # for i in range(len(innerBoxes)):
        #     box = innerBoxes[i]
        #
        #     b += 10
        #     g += 10
        #     r += 10
        #     rectangleImage[box[0][1]:box[1][1], box[0][0]:box[1][0]] = (b, g, r)
        #
        #     cell = segmentedImage[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        #     cellMean = np.mean(np.mean(cell, 0), 0)
        #     blueSim = np.linalg.norm(np.array(blueMean) - np.array(cellMean))
        #     redSim = np.linalg.norm(np.array(redMean) - np.array(cellMean))
        #     team = Team.BLUE if blueSim < redSim else Team.RED
        #     descriptor.set1D(i, team)

        return descriptor, rectangleImage
