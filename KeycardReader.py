import cv2
import numpy as np
from typing import Tuple

from imageprocessing.Filterer import Filterer
from imageprocessing.Morpher import Morpher
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter

blueMean = [200, 125, 25]
blueRange = ((120, 0, 0), (255, 180, 80))
redMean = [50, 50, 200]
redRange = ((0, 0, 150), (100, 70, 255))
blackMean = [0, 0, 0]
blackRange = ((0, 0, 0), (50, 50, 50))
distThreshold = (155, 255)


class KeycardReader:
    def __init__(self, referenceImageFileName: str):
        self.referenceImage = cv2.imread(referenceImageFileName)
        self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycardDescriptor(self, path: str, method: str = 'auto') -> Tuple:
        image = cv2.imread(path)
        image = Segmenter.grabcut(image, 'rectangular')
        image, resizedTargetImage = Preprocessor.resizeToSameSize(image, self.referenceImage, 500, 500)

        if method is 'adaptive' or method is 'otsu' or method is 'range':
            dist = image[:, :, 0].copy()

            rows, cols = image.shape[:2]
            for i in range(rows):
                for j in range(cols):
                    dist[i, j] = max(0, 255 - np.linalg.norm(np.array(blackMean) - np.array(image[i, j])))

            thresholded = Segmenter.threshold(image, dist, method, distThreshold)
        elif method is 'ranges':
            thresholded = Segmenter.threshold(image, image, 'range', blackRange)
        else:
            raise ValueError(method + ' method not implemented')

        return image, thresholded
