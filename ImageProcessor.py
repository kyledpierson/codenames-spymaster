import cv2
import maxflow
import numpy as np
from math import exp, pow

from typing import Tuple


class ImageProcessor:
    def __init__(self):
        pass

    def segmentGrabcut(self, image, rect: Tuple = None, useRectMask: bool = True):
        """ Segments image, assuming the outer portion is representative of the background
        :param image: image to segment
        :param rect: bounding box that contains the foreground (minX, minY, maxX, maxY)
                     None means use a radial mask in the center of the image
        :param useRectMask: use a rectangular mask instead of a radial one
        :return: segmented image, mask image """
        rows, cols = image.shape[:2]

        mask = np.zeros((rows, cols), np.uint8)
        maskImage = np.zeros((rows, cols), np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        mode = cv2.GC_INIT_WITH_RECT

        if rect:
            minX, minY, width, height = rect
            maskImage[minY:minY + height, minX:minX + width] = 255
        else:
            midRow = rows / 2.
            midCol = cols / 2.
            if useRectMask:
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

        cv2.grabCut(image, mask, rect, bgModel, fgModel, 4, mode)

        mask = np.where((mask == 0) | (mask == 2), 0, 1).astype(np.uint8)
        segmentedImage = image * mask[:, :, np.newaxis]

        return segmentedImage, maskImage

    def segmentMrf(self, image, costImage, mu_1, mu_2, mu_3, sigma_1, sigma_2, sigma_3, pairwiseCost):
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
        costs = [[self.gaussianKernel3D(pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3)
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

    def segmentThreshold(self, image, colorSpaceImage, method, range=None):
        """
        Performs range-based, adaptive or otsu thresholding on image
        :param image: image to segment
        :param colorSpaceImage: image in the desired color space used for thresholding
        :param method: method to use ('range', 'adaptive' or 'otsu')
        :param blockSize:
        :param C:
        :param range: range for range-based thresholding
        :return: thresholded image
        """
        if method == 'range':
            lower, upper = range
            mask = cv2.inRange(colorSpaceImage, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.erode(mask, kernel, iterations=3)
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
        elif method == 'adaptive':
            mask = cv2.adaptiveThreshold(colorSpaceImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         11, 3)
        elif method == 'otsu':
            _, mask = cv2.threshold(colorSpaceImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(method + ' not implemented')

        thresholdedImage = cv2.bitwise_and(image, image, mask=mask)
        return thresholdedImage

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    def gaussianKernel2D(self, pixel, mu_1, sigma_1, mu_2, sigma_2):
        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[2], 2) / (2 * pow(sigma_2, 2))))
        return d * 255

    def gaussianKernel3D(self, pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3):
        n = 1  # No normalization
        # n = 1 / (pow(2 * pi, 3 / 2) * y_sigma * cr_sigma * cb_sigma)

        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[1], 2) / (2 * pow(sigma_2, 2)) +
                pow(mu_3 - pixel[2], 2) / (2 * pow(sigma_3, 2))))
        return n * d * 255
