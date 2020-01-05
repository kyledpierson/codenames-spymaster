import cv2
import imutils
import numpy as np
from math import floor, ceil

from typing import Tuple


class ImagePreprocessor:
    def __init__(self):
        pass

    def resizeToSameSize(self, image, targetImage, width: int, height: int) -> Tuple:
        """ Resizes both images to the same size, padding when necessary
        :param image: first image
        :param targetImage: second image
        :param width: max width of the new images
        :param height: max height of the new images
        :return: both images, resized to the same size """
        resizedImage = imutils.resize(image, width, height)
        resizedTargetImage = imutils.resize(targetImage, width, height)

        height, width = resizedImage.shape[:2]
        targetHeight, targetWidth = resizedTargetImage.shape[:2]
        vertical = height - targetHeight
        horizontal = width - targetWidth

        top = floor(vertical / 2)
        bottom = ceil(vertical / 2)
        left = floor(horizontal / 2)
        right = ceil(horizontal / 2)
        if vertical < 0:
            resizedImage = cv2.copyMakeBorder(resizedImage, borderType=cv2.BORDER_CONSTANT,
                                              top=abs(top), bottom=abs(bottom), left=0, right=0)
        else:
            resizedTargetImage = cv2.copyMakeBorder(resizedTargetImage, borderType=cv2.BORDER_CONSTANT,
                                                    top=top, bottom=bottom, left=0, right=0)
        if horizontal < 0:
            resizedImage = cv2.copyMakeBorder(resizedImage, borderType=cv2.BORDER_CONSTANT,
                                              top=0, bottom=0, left=abs(left), right=abs(right))
        else:
            resizedTargetImage = cv2.copyMakeBorder(resizedTargetImage, borderType=cv2.BORDER_CONSTANT,
                                                    top=0, bottom=0, left=left, right=right)

        return resizedImage, resizedTargetImage

    def computeImageStats(self, image) -> Tuple:
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

    def equalizeHistogram(self, image, targetImage):
        """ Maps color histogram of targetImage onto image.
        :param image: BGR image
        :param targetImage: BGR target image
        :return BGR image with targetImage color """

        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        labTargetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2LAB).astype("float32")

        lMean, lStd, aMean, aStd, bMean, bStd = self.computeImageStats(labImage)
        lMeanTarget, lStdTarget, aMeanTarget, aStdTarget, bMeanTarget, bStdTarget = self.computeImageStats(
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

    def performClahe(self, image, clipLimit: int, tileGridSize: Tuple):
        """ Performs CLAHE (Contrast Limited Adaptive Histogram Equalization).
        :param image: BGR image
        :param clipLimit: threshold for contrast limiting
        :param tileGridSize: size of grid for histogram equalization
        :return equalized BGR image """

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsvImage)

        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        equalizedV = clahe.apply(v)

        equalizedHsvImage = cv2.merge((h, s, equalizedV))
        equalizedImage = cv2.cvtColor(equalizedHsvImage, cv2.COLOR_HSV2BGR)

        return equalizedImage

    def enhanceEdges(self, image, action: str = 'enhanceEdges', kernel=None, sigma_s=None, sigma_r=None):
        """ Applies a filter to an image, accentuating edges
        :param image: BGR image to be filtered
        :param action: desired action (sharpen, excessiveSharpen, enhanceEdges)
        :param kernel: override kernel to apply to image
        :return BGR sharpened image """
        enhancedImage = cv2.edgePreservingFilter(image, sigma_s=sigma_s, sigma_r=sigma_r)

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

    def registerWithCorrespondences(self, image, points, targetImage, targetPoints):
        """ Warps image into targetImage, based on correspondences
        :param image: BGR image
        :param points: ordered coordinates of landmarks in image
        :param targetImage: BGR target image
        :param targetPoints: ordered coordinates of landmarks in targetImage
        :return: BGR image warped into targetImage """
        rows, cols = targetImage.shape[:2]

        for (x, y) in points:
            cv2.circle(image, (x, y), 3, 0, -1)
        for (x, y) in targetPoints:
            cv2.circle(targetImage, (x, y), 3, 0, -1)

        # warpMatrix = cv2.getPerspectiveTransform(points, targetPoints)
        warpMatrix, status = cv2.findHomography(points, targetPoints)
        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows))

        return registeredImage

    def registerWithEcc(self, image, targetImage, iterations: int, terminationEpsilon: float):
        """ Performs a perspective transformation of image to the foreground orientation of targetImage
        :param image: BGR image
        :param targetImage: BGR image with which image should be aligned
        :param iterations: number of iterations
        :param terminationEpsilon: threshold of increment in correlation coefficient between iterations
        :return BGR image aligned to targetImage """
        rows, cols = targetImage.shape[:2]

        grayTargetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        warpMatrix = np.eye(3, 3, dtype=np.float32)
        motionType = cv2.MOTION_HOMOGRAPHY

        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, terminationEpsilon)

        cc, warpMatrix = cv2.findTransformECC(grayTargetImage, grayImage, warpMatrix, motionType, criteria, None, 1)

        registeredImage = cv2.warpPerspective(image, warpMatrix, (cols, rows),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return registeredImage
