import cv2
import numpy as np

from imageprocessing.Filterer import Filterer
from imageprocessing.Morpher import Morpher
from imageprocessing.Preprocessor import Preprocessor
from imageprocessing.Segmenter import Segmenter

from KeycardDescriptor import KeycardDescriptor


class KeycardReader:
    referenceImage = None
    filterer = Filterer()
    morpher = Morpher()
    segmenter = Segmenter()
    preprocessor = Preprocessor()

    def __init__(self):
        self.referenceImage = cv2.imread("keycard-reference.webp")
        self.referenceImage, _ = self.segmenter.grabcut(self.referenceImage)
        self.referenceImage = self.filterer.equalizeHistogram(self.referenceImage)

    def extractKeycardDescriptor(self, path):
        image = cv2.imread(path)
        image, _ = self.segmenter.grabcut(image)
        image = self.filterer.equalizeHistogram(image)

        image, resizedTargetImage = self.preprocessor.resizeToSameSize(image, self.referenceImage, 500, 500)
        image = self.morpher.registerWithEcc(image, resizedTargetImage)
        image = self.filterer.mapHistogram(image, resizedTargetImage)
        image = self.filterer.enhanceEdges(image)

        b = image[:, :, 0].copy()
        r = image[:, :, 2].copy()

        rows, cols = image.shape[:2]
        for i in range(rows):
            for j in range(cols):
                b[i, j] = max(0, 255 - np.linalg.norm(np.array([255, 0, 0]) - np.array(image[i, j])))
                r[i, j] = max(0, 255 - np.linalg.norm(np.array([0, 0, 255]) - np.array(image[i, j])))

        bThresholded = self.segmenter.threshold(image, b, 'otsu')
        rThresholded = self.segmenter.threshold(image, r, 'otsu')

        return image, b, r, bThresholded, rThresholded
