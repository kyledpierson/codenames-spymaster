import cv2
import numpy as np

from typing import Tuple


class Filterer:
    def __init__(self):
        pass

    def equalizeHistogram(self, image):
        """ Performs CLAHE (Contrast Limited Adaptive Histogram Equalization).
        :param image: BGR image
        :return equalized BGR image """

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsvImage)

        clahe = cv2.createCLAHE()
        equalizedV = clahe.apply(v)

        equalizedHsvImage = cv2.merge((h, s, equalizedV))
        equalizedImage = cv2.cvtColor(equalizedHsvImage, cv2.COLOR_HSV2BGR)

        return equalizedImage

    def mapHistogram(self, image, targetImage):
        """ Maps color histogram of targetImage onto image.
        :param image: BGR image
        :param targetImage: BGR target image
        :return BGR image with targetImage color """

        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        labTargetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2LAB).astype("float32")

        lMean, lStd, aMean, aStd, bMean, bStd = self.__computeImageStats(labImage)
        lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget = self.__computeImageStats(
            labTargetImage)

        l, a, b = cv2.split(labImage)
        l -= lMean
        a -= aMean
        b -= bMean

        l = (lStd / lStdTarget) * l
        a = (aStd / aStdTarget) * a
        b = (bStd / bStdTarget) * b

        l += lMeanTarget
        a += aMeanTarget
        b += bMeanTarget

        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        coloredLabImage = cv2.merge([l, a, b])
        coloredImage = cv2.cvtColor(coloredLabImage.astype("uint8"), cv2.COLOR_LAB2BGR)

        return coloredImage

    def enhanceEdges(self, image, action: str = 'enhanceEdges', kernel=None):
        """ Applies a filter to an image, accentuating edges
        :param image: BGR image to be filtered
        :param action: desired action (sharpen, excessiveSharpen, enhanceEdges)
        :param kernel: override kernel to apply to image
        :return BGR sharpened image """
        enhancedImage = cv2.edgePreservingFilter(image)
        # Alternative: cv2.bilateralFilter(image, 5, 50, 50)

        if not kernel:
            if action == 'sharpen':
                kernel = np.array([[-1, -1, -1],
                                   [-1, +9, -1],
                                   [-1, -1, -1]])
            elif action == 'excessiveSharpen':
                kernel = np.array([[+1, +1, +1],
                                   [+1, -7, +1],
                                   [+1, +1, +1]])
            elif action == 'enhanceEdges':
                kernel = np.array([[-1, -1, -1, -1, -1],
                                   [-1, +2, +2, +2, -1],
                                   [-1, +2, +8, +2, -1],
                                   [-1, +2, +2, +2, -1],
                                   [-1, -1, -1, -1, -1]]) / 8.0
            else:
                raise ValueError(action + ' not implemented')

        enhancedImage = cv2.filter2D(enhancedImage, -1, kernel)
        return enhancedImage

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    def __computeImageStats(self, image) -> Tuple:
        """ Splits image into channels, and computes mean and std on each channel.
        :param image: image for which stats are to be computed
        :return stats (a.mean, a.std, b.mean, b.std, c.mean, c.std) """
        a, b, c = cv2.split(image)

        aNoBlack = []
        bNoBlack = []
        cNoBlack = []

        for i, row in enumerate(a):
            for j, pixel in enumerate(row):
                if a[i][j] != 0:
                    aNoBlack.append(a[i][j])
                    bNoBlack.append(b[i][j])
                    cNoBlack.append(c[i][j])

        aNoBlack = np.array(aNoBlack)
        bNoBlack = np.array(bNoBlack)
        cNoBlack = np.array(cNoBlack)
        return aNoBlack.mean(), aNoBlack.std(), bNoBlack.mean(), bNoBlack.std(), cNoBlack.mean(), cNoBlack.std()
