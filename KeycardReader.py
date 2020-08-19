import cv2
import imutils
import numpy as np
from typing import Tuple

from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Point, BoundingBox, Segmenter
from KeycardDescriptor import Team, KeycardDescriptor

Image = np.ndarray

blackMean: np.array = np.array([0, 0, 0])
blueMean: np.array = np.array([200, 125, 25])
redMean: np.array = np.array([50, 50, 200])
tanMean: np.array = np.array([215, 220, 210])

blackRange: np.array = np.array([[0, 0, 0], [50, 50, 50]])
blueRange: np.array = np.array([[120, 0, 0], [255, 180, 80]])
redRange: np.array = np.array([[0, 0, 150], [100, 70, 255]])

distThreshold: Tuple = (155, 255)


class KeycardReader:
    referenceImage: Image = None

    def __init__(self, referenceImageFileName: str = ''):
        if referenceImageFileName:
            self.referenceImage = cv2.imread(referenceImageFileName)
            self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycardDescriptor(self, path: str) -> KeycardDescriptor:
        descriptor: KeycardDescriptor = KeycardDescriptor()

        image: Image = cv2.imread(path)
        segmentedImage: Image = Segmenter.grabcut(image, 'rectangular')
        if self.referenceImage is None:
            segmentedImage = imutils.resize(segmentedImage, 500, 500)
        else:
            segmentedImage, _ = Preprocessor.resizeToSameSize(segmentedImage, self.referenceImage, 500, 500)

        rows, cols = segmentedImage.shape[:2]
        blackResponseImage: Image = segmentedImage[:, :, 0].copy()
        for row in range(rows):
            for col in range(cols):
                blackResponseImage[row, col] = max(0, 255 - np.linalg.norm(blackMean - segmentedImage[row, col]))

        # thresholdedImage: Image = Segmenter.threshold(segmentedImage, blackResponseImage, 'otsu', distThreshold)
        # innerBoxes: np.array = Segmenter.inferBoxesFromBlackRegions(blackResponseImage)
        innerBoxes: np.array = Segmenter.inferBoxesFromGridlines()

        rows, cols = innerBoxes.shape
        for row in range(rows):
            for col in range(cols):
                box: BoundingBox = innerBoxes[row, col]
                topLeft: Point = box.topLeft()
                bottomRight: Point = box.bottomRight()
                cell: np.array = segmentedImage[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

                cellMean: np.array = np.mean(np.mean(cell, 0), 0)
                similarity: np.array = np.array([np.linalg.norm(blackMean - cellMean),
                                                 np.linalg.norm(blueMean - cellMean),
                                                 np.linalg.norm(redMean - cellMean),
                                                 np.linalg.norm(tanMean - cellMean)])
                teams: np.array = np.array([Team.ASSASSIN, Team.BLUE, Team.RED, Team.NEUTRAL])
                team: Team = teams[np.argmin(similarity)]

                descriptor.set2D(row, col, team)

        return descriptor
