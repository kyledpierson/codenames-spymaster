import cv2
import imutils
import numpy as np
from typing import Tuple

from imageprocessing.Detector import Detector
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter
from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point
from gamecomponents.Keycard import Team, Keycard
from gamecomponents.Wordgrid import Wordgrid

Image = np.ndarray


class ComponentReader:
    referenceImage: Image = None

    blackMean: np.array = np.array([0, 0, 0])
    blueMean: np.array = np.array([200, 125, 25])
    redMean: np.array = np.array([50, 50, 200])
    tanMean: np.array = np.array([215, 220, 210])

    blackRange: np.array = np.array([[0, 0, 0], [50, 50, 50]])
    blueRange: np.array = np.array([[120, 0, 0], [255, 180, 80]])
    redRange: np.array = np.array([[0, 0, 150], [100, 70, 255]])

    distThreshold: Tuple = (155, 255)

    def __init__(self, referenceImageFileName: str = ''):
        if referenceImageFileName:
            self.referenceImage = cv2.imread(referenceImageFileName)
            self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycard(self, path: str) -> Keycard:
        keycard: Keycard = Keycard()

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
                blackResponseImage[row, col] = max(0, 255 - np.linalg.norm(self.blackMean - segmentedImage[row, col]))

        # thresholdedImage: Image = Segmenter.threshold(segmentedImage, blackResponseImage, 'otsu', distThreshold)
        # innerBoxes: np.array = Segmenter.inferBoxesFromBlackRegions(blackResponseImage)
        innerBoxes: np.array = Segmenter.inferBoxesFromGridlines()

        rows, cols = innerBoxes.shape
        for row in range(rows):
            for col in range(cols):
                box: Box = innerBoxes[row, col]
                topLeft: Point = box.topLeft()
                bottomRight: Point = box.bottomRight()
                cell: np.array = segmentedImage[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

                cellMean: np.array = np.mean(np.mean(cell, 0), 0)
                similarity: np.array = np.array([np.linalg.norm(self.blackMean - cellMean),
                                                 np.linalg.norm(self.blueMean - cellMean),
                                                 np.linalg.norm(self.redMean - cellMean),
                                                 np.linalg.norm(self.tanMean - cellMean)])
                teams: np.array = np.array([Team.ASSASSIN, Team.BLUE, Team.RED, Team.NEUTRAL])
                team: Team = teams[np.argmin(similarity)]

                keycard.set2D(row, col, team)

        return keycard

    def extractWordgrid(self, path: str) -> Wordgrid:
        wordgrid: Wordgrid = Wordgrid()

        image: Image = cv2.imread(path)
        boxes: np.array = Segmenter.inferCardsFromGridlines(image)

        rows, cols = boxes.shape
        for row in range(rows):
            for col in range(cols):
                box: Box = boxes[row, col]
                topLeft: Point = box.topLeft()
                bottomRight: Point = box.bottomRight()
                cell: np.array = image[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

                detector: Detector = Detector()
                text = detector.readWordsOnCard(cell)

                print(text)
                # cv2.imshow("cell-" + str(row) + "-" + str(col), cell)
                # cv2.waitKey()

        return wordgrid
