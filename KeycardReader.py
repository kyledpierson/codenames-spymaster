import cv2
import numpy as np
from typing import Tuple

from imageprocessing.Filterer import Filterer
from imageprocessing.Morpher import Morpher
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter


class KeycardReader:
    def __init__(self, referenceImageFileName: str):
        self.referenceImage = cv2.imread(referenceImageFileName)
        self.referenceImage = Segmenter.grabcut(self.referenceImage, 'rectangular')

    def extractKeycardDescriptor(self, path: str, method: str = 'auto') -> Tuple:
        image = cv2.imread(path)
        image = Segmenter.grabcut(image, 'rectangular')
        image, resizedTargetImage = Preprocessor.resizeToSameSize(image, self.referenceImage, 500, 500)

        if method is 'ranges':
            bRange = ((120, 0, 0), (255, 180, 80))
            rRange = ((0, 0, 150), (100, 70, 255))

            bThresholded = Segmenter.threshold(image, image, 'range', bRange)
            rThresholded = Segmenter.threshold(image, image, 'range', rRange)
        else:
            bDist = image[:, :, 0].copy()
            rDist = image[:, :, 2].copy()

            rows, cols = image.shape[:2]
            for i in range(rows):
                for j in range(cols):
                    bDist[i, j] = max(0, 255 - np.linalg.norm(np.array([200, 125, 25]) - np.array(image[i, j])))
                    rDist[i, j] = max(0, 255 - np.linalg.norm(np.array([50, 50, 200]) - np.array(image[i, j])))

            bThresholded = Segmenter.threshold(image, bDist, 'range', (155, 255))
            rThresholded = Segmenter.threshold(image, rDist, 'range', (155, 255))

        return image, bThresholded, rThresholded
