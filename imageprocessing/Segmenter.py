from math import exp, pow
from typing import Tuple

import cv2
import maxflow
import numpy as np

Image = np.ndarray


class Point:
    x: float = 0
    y: float = 0

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def midpoint(self, other: 'Point') -> 'Point':
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)

    def divide(self, other: 'Point', dim: int) -> np.array:
        x = np.histogram_bin_edges([self.x, other.x], dim)
        y = np.histogram_bin_edges([self.y, other.y], dim)
        return np.array([Point(x[i], y[i]) for i in range(len(x))])

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"


class BoundingBox:
    box: np.array = np.array([Point(), Point()], dtype=Point)

    def __init__(self, topLeft: Point = Point(), bottomRight: Point = Point()):
        self.box = np.array([topLeft, bottomRight], dtype=Point)

    def topLeft(self) -> Point:
        return self.box[0]

    def bottomRight(self) -> Point:
        return self.box[1]

    def centroid(self) -> Point:
        return self.topLeft().midpoint(self.bottomRight())

    def split(self, axis: str = 'x') -> 'BoundingBox':
        centroid: Point = self.centroid()
        if axis is 'x':
            self.box[1].x = centroid.x
            return BoundingBox(Point(centroid.x, self.topLeft().y), self.bottomRight())
        elif axis is 'y':
            self.box[1].y = centroid.y
            return BoundingBox(Point(self.topLeft().x, centroid.y), self.bottomRight())
        else:
            raise ValueError(axis + ' axis not implemented, must specify x or y')

    def divide(self, dim: int) -> np.array:
        edges: np.array = self.topLeft().divide(self.bottomRight(), dim)
        boxes: np.array = np.array([[BoundingBox() for i in range(dim)] for j in range(dim)], dtype=BoundingBox)
        for i in range(dim):
            for j in range(dim):
                boxes[i, j].box = np.array([Point(edges[j].x, edges[i].y), Point(edges[j + 1].x, edges[i + 1].y)])
        return boxes

    def __str__(self):
        return "[" + str(self.topLeft()) + ", " + str(self.bottomRight()) + "]"


class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def grabcut(image: Image, maskType: str, customMask: Tuple = None) -> Image:
        segmentedImage = image.copy()
        rows, cols = segmentedImage.shape[:2]

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

            elif maskType is 'circular':
                likelyBg = midRow if midRow > midCol else midCol
                likelyFg = (2. / 3.) * likelyBg
                fg = (1. / 3.) * likelyBg

                isBg = lambda i, j, dist: dist > likelyBg
                isLikelyBg = lambda i, j, dist: dist > likelyFg
                isLikelyFg = lambda i, j, dist: dist > fg
            else:
                raise ValueError(maskType + ' mask not implemented')

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

        cv2.grabCut(segmentedImage, mask, customMask, bgModel, fgModel, 2, mode)

        mask = np.where((mask == 0) | (mask == 2), 0, 1).astype(np.uint8)
        segmentedImage[mask == 0] = (255, 255, 255)

        return segmentedImage

    @staticmethod
    def mrf(image: Image, costImage: Image, mu_1: float, mu_2: float, mu_3: float,
            sigma_1: float, sigma_2: float, sigma_3: float, pairwiseCost: float) -> Image:
        costs = [[Segmenter.__gaussianKernel3D(pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3)
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

    @staticmethod
    def threshold(image: Image, thresholdImage: Image, method: str, colorRange: Tuple = None) -> Image:
        mask = None
        if method == 'adaptive':
            mask = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        elif method == 'otsu':
            _, mask = cv2.threshold(thresholdImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'range':
            lower, upper = colorRange
            newMask = cv2.inRange(thresholdImage, lower, upper)
            mask = mask + newMask if mask else newMask
        else:
            raise ValueError(method + ' threshold not implemented')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        thresholdedImage = cv2.bitwise_and(image, image, mask=mask)
        return thresholdedImage

    @staticmethod
    def drawRectangles(image: Image) -> Tuple:
        if len(image.shape) < 3:
            binaryImage = image.copy()
            processedImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            binaryImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processedImage = image.copy()
        binaryImage = cv2.GaussianBlur(binaryImage, (3, 3), 0)
        _, binaryImage = cv2.threshold(binaryImage, 55, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        outerBox = BoundingBox()
        innerBoxes = np.array([], dtype=BoundingBox)
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                squareRatio = w / float(h)

                if 0.95 <= squareRatio <= 1.05:
                    if 5000 < area:
                        outerBox.box = np.array([Point(x, y), Point(x + w - 1, y + h - 1)])
                    elif 1000 < area < 5000:
                        np.append(innerBoxes, [BoundingBox(Point(x, y), Point(x + w - 1, y + h - 1))])
                    cv2.drawContours(processedImage, [c], -1, (0, 255, 0), 2)

        return processedImage, outerBox, innerBoxes

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    @staticmethod
    def __gaussianKernel2D(pixel: Tuple, mu_1: float, sigma_1: float, mu_2: float, sigma_2: float):
        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[2], 2) / (2 * pow(sigma_2, 2))))
        return d * 255

    @staticmethod
    def __gaussianKernel3D(pixel: Tuple,
                           mu_1: float, sigma_1: float, mu_2: float, sigma_2: float, mu_3: float, sigma_3: float):
        n = 1  # No normalization
        # n = 1 / (pow(2 * pi, 3 / 2) * y_sigma * cr_sigma * cb_sigma)

        d = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[1], 2) / (2 * pow(sigma_2, 2)) +
                pow(mu_3 - pixel[2], 2) / (2 * pow(sigma_3, 2))))
        return n * d * 255
