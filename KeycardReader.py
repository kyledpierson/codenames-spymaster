import cv2
import numpy as np
from typing import Tuple

from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter
from KeycardDescriptor import Team, KeycardDescriptor

Image = np.ndarray

blueMean = [200, 125, 25]
blueRange = ((120, 0, 0), (255, 180, 80))
redMean = [50, 50, 200]
redRange = ((0, 0, 150), (100, 70, 255))
blackMean = [0, 0, 0]
blackRange = ((0, 0, 0), (50, 50, 50))
distThreshold = (155, 255)


class KeycardReader:
    referenceImage: Image = None

    def __init__(self, referenceImageFileName: str):
        self.referenceImage = cv2.imread(referenceImageFileName)
        self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycardDescriptor(self, path: str) -> Tuple:
        descriptor: KeycardDescriptor = KeycardDescriptor(5)

        image: Image = cv2.imread(path)
        rows, cols = image.shape[:2]
        segmentedImage = Segmenter.grabcut(image, 'rectangular')
        segmentedImage, _ = Preprocessor.resizeToSameSize(segmentedImage, self.referenceImage, 500, 500)

        blackResponseImage = segmentedImage[:, :, 0].copy()
        for row in range(rows):
            for col in range(cols):
                blackResponseImage[row, col] = max(0, 255 - np.linalg.norm(
                    np.array(blackMean) - np.array(segmentedImage[row, col])))

        thresholdedImage = Segmenter.threshold(segmentedImage, blackResponseImage, 'otsu', distThreshold)
        squareImage, innerBoxes = Segmenter.drawSquares(blackResponseImage)

        for row in range(5):
            for col in range(5):
                box = innerBoxes[row, col]
                topLeft = box.topLeft()
                bottomRight = box.bottomRight()
                cell = segmentedImage[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

                cellMean = np.mean(np.mean(cell, 0), 0)
                blueSim = np.linalg.norm(np.array(blueMean) - np.array(cellMean))
                redSim = np.linalg.norm(np.array(redMean) - np.array(cellMean))
                team = Team.BLUE if blueSim < redSim else Team.RED

                descriptor.set2D(row, col, team)

        return descriptor, squareImage
