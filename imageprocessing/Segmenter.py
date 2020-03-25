import cv2
import maxflow
import numpy as np
from math import exp, pow

from typing import Tuple


class Segmenter:
    def __init__(self):
        pass

    def grabcut(self, image, maskType: bool = 'rectangular', customMask: Tuple = None):
        """ Segments image, assuming the outer portion is representative of the background
        :param image: image to segment
        :param maskType: type of mask to apply ('rectangular', 'circular')
        :param customMask: bounding box that contains the foreground (minX, minY, maxX, maxY)
        :return: segmented image, mask image """
        rows, cols = image.shape[:2]

        mask = np.zeros((rows, cols), np.uint8)
        maskImage = np.zeros((rows, cols), np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        mode = cv2.GC_INIT_WITH_RECT

        if customMask:
            minX, minY, width, height = customMask
            maskImage[minY:minY + height, minX:minX + width] = 255
        else:
            midRow = rows / 2.
            midCol = cols / 2.
            if maskType is 'rectangular':
                bgXmin = (1. / 12.) * cols
                bgXmax = cols - bgXmin
                bgYmin = (1. / 12.) * rows
                bgYmax = rows - bgYmin

                likelyBgXmin = (1. / 6.) * cols
                likelyBgXmax = cols - likelyBgXmin
                likelyBgYmin = (1. / 6.) * rows
                likelyBgYmax = rows - likelyBgYmin

                likelyFgXmin = (1. / 4.) * cols
                likelyFgXmax = cols - likelyFgXmin
                likelyFgYmin = (1. / 4.) * rows
                likelyFgYmax = rows - likelyFgYmin

                isBg = lambda i, j, dist: i < bgYmin or i > bgYmax or j < bgXmin or j > bgXmax
                isLikelyBg = lambda i, j, dist: i < likelyBgYmin or i > likelyBgYmax or \
                                                j < likelyBgXmin or j > likelyBgXmax
                isLikelyFg = lambda i, j, dist: i < likelyFgYmin or i > likelyFgYmax or \
                                                j < likelyFgXmin or j > likelyFgXmax

            else:
                likelyBg = midRow if midRow > midCol else midCol
                likelyFg = (2. / 3.) * likelyBg
                fg = (1. / 3.) * likelyBg

                isBg = lambda i, j, dist: dist > likelyBg
                isLikelyBg = lambda i, j, dist: dist > likelyFg
                isLikelyFg = lambda i, j, dist: dist > fg

            for i in range(rows):
                for j in range(cols):
                    dist = np.sqrt((i - midRow) ** 2 + (j - midCol) ** 2)
                    if isBg(i, j, dist):
                        mask[i, j] = 0
                    elif isLikelyBg(i, j, dist):
                        mask[i, j] = 2
                    elif isLikelyFg(i, j, dist):
                        mask[i, j] = 3
                    else:
                        mask[i, j] = 1

            maskImage = mask * 85
            mode = cv2.GC_INIT_WITH_MASK

        cv2.grabCut(image, mask, customMask, bgModel, fgModel, 2, mode)

        mask = np.where((mask == 0) | (mask == 2), 0, 1).astype(np.uint8)
        segmentedImage = image * mask[:, :, np.newaxis]

        return segmentedImage, maskImage

    def mrf(self, image, costImage, mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3, pairwiseCost):
        """ Segments image as a Markov random field
        :param image: image to segment
        :param costImage: image on which to base the costs
        :param mu_1: mean of the first channel
        :param mu_2: mean of the second channel
        :param mu_3: mean of the third channel
        :param sigma_1: std of the first channel
        :param sigma_2: std of the second channel
        :param sigma_3: std of the third channel
        :param pairwiseCost: fraction to use as pairwise cost
        :return: segmented image """
        costs = [[self.__gaussianKernel3D(pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3)
                  for pixel in row] for row in costImage]
        costs = np.array(costs)

        graph = maxflow.Graph[float]()
        nodes = graph.add_grid_nodes(image.shape[:2])
        graph.add_grid_edges(nodes, pairwiseCost * 255)
        graph.add_grid_tedges(nodes, 255 - costs, costs)

        graph.maxflow()

        segments = graph.get_grid_segments(nodes)
        labels = np.array(segments, dtype=np.uint8)
        segmentedImage = cv2.bitwise_and(image, image, mask=labels)

        return segmentedImage

    def threshold(self, image, colorSpaceImage, method, ranges=None):
        """
        Performs range-based, adaptive or otsu thresholding on image
        :param image: image to segment
        :param colorSpaceImage: image in the desired color space used for thresholding
        :param method: method to use ('range', 'adaptive' or 'otsu')
        :param range: range for range-based thresholding
        :return: thresholded image
        """
        """
        if colorSpace is cv2.COLOR_BGR2RGB:
            lightRed1 = (150, 0, 0)
            darkRed1 = (255, 70, 100)

            lightRed2 = (150, 0, 0)
            darkRed2 = (255, 70, 100)

            lightBlue = (0, 0, 120)
            darkBlue = (80, 180, 255)
        elif colorSpace is cv2.COLOR_BGR2HLS:
            lightRed1 = (0, 100, 150)
            darkRed1 = (40, 150, 255)

            lightRed2 = (215, 100, 150)
            darkRed2 = (255, 150, 255)

            lightBlue = (115, 75, 120)
            darkBlue = (155, 130, 255)
        """

        if method == 'range':
            mask = None
            for range in ranges:
                lower, upper = range
                newMask = cv2.inRange(colorSpaceImage, lower, upper)
                mask = mask + newMask if mask else newMask

        elif method == 'adaptive':
            mask = cv2.adaptiveThreshold(colorSpaceImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         11, 3)
        elif method == 'otsu':
            _, mask = cv2.threshold(colorSpaceImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(method + ' not implemented')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        thresholdedImage = cv2.bitwise_and(image, image, mask=mask)
        return thresholdedImage

    def shape(self, image):
        """ Segments all squares of a keycard
        :param image: an image of the keycard
        :return: An image with contours drawn on the square perimeters """
        processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processedImage = cv2.GaussianBlur(processedImage, (3, 3), 0)
        _, processedImage = cv2.threshold(processedImage, 100, 255, cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(processedImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)

                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)

                    if 0.95 <= ar <= 1.05:
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        return image

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    def __gaussianKernel2D(self, pixel, mu_1, sigma_1, mu_2, sigma_2):
        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[2], 2) / (2 * pow(sigma_2, 2))))
        return d * 255

    def __gaussianKernel3D(self, pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3):
        n = 1  # No normalization
        # n = 1 / (pow(2 * pi, 3 / 2) * y_sigma * cr_sigma * cb_sigma)

        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[1], 2) / (2 * pow(sigma_2, 2)) +
                pow(mu_3 - pixel[2], 2) / (2 * pow(sigma_3, 2))))
        return n * d * 255
