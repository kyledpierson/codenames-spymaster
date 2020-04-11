import cv2
import numpy as np

from imageprocessing.Filterer import Filterer
from imageprocessing.Morpher import Morpher
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter


class KeycardReader:
    def __init__(self, referenceImageFileName):
        self.referenceImage = cv2.imread(referenceImageFileName)
        self.referenceImage = Segmenter.grabcut(self.referenceImage)
        self.referenceImage = Filterer.equalizeHistogram(self.referenceImage,
                                                         cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, (0, 0, 1))

    def extractKeycardDescriptor(self, path):
        image = cv2.imread(path)
        image = Segmenter.grabcut(image)
        image = Filterer.equalizeHistogram(image, cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, (0, 0, 1))

        image, resizedTargetImage = Preprocessor.resizeToSameSize(image, self.referenceImage, 500, 500)
        image = Morpher.registerWithEcc(image, resizedTargetImage)
        image = Filterer.mapHistogram(image, resizedTargetImage, cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR)
        image = Filterer.enhanceEdges(image)

        b = image[:, :, 0].copy()
        r = image[:, :, 2].copy()

        rows, cols = image.shape[:2]
        for i in range(rows):
            for j in range(cols):
                b[i, j] = max(0, 255 - np.linalg.norm(np.array([255, 0, 0]) - np.array(image[i, j])))
                r[i, j] = max(0, 255 - np.linalg.norm(np.array([0, 0, 255]) - np.array(image[i, j])))

        bThresholded = Segmenter.threshold(image, b)
        rThresholded = Segmenter.threshold(image, r)

        return image, b, r, bThresholded, rThresholded
