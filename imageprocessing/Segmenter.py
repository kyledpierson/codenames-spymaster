from math import exp, pow
from typing import Tuple

import cv2
import maxflow
import numpy as np

from globalvariables import GRID_SIZE
from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point

Grid = np.array
Image = np.ndarray


class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def grabcut(image: Image, maskType: str, customMask: Tuple = None) -> Image:
        (rows, cols) = image.shape[:2]

        mask: Image = np.zeros((rows, cols), dtype=np.uint8)
        maskImage: Image = np.zeros((rows, cols), dtype=np.uint8)
        bgModel: np.array = np.zeros((1, 65), dtype=np.float64)
        fgModel: np.array = np.zeros((1, 65), dtype=np.float64)
        mode: int = cv2.GC_INIT_WITH_RECT

        if customMask:
            (minX, minY, width, height) = customMask
            maskImage[minY:minY + height, minX:minX + width] = 255
        else:
            midRow: float = rows / 2.
            midCol: float = cols / 2.
            if maskType is "rectangular":
                bgXmin: float = (1. / 12.) * cols
                bgXmax: float = cols - bgXmin
                bgYmin: float = (1. / 12.) * rows
                bgYmax: float = rows - bgYmin

                likelyBgXmin: float = (1. / 6.) * cols
                likelyBgXmax: float = cols - likelyBgXmin
                likelyBgYmin: float = (1. / 6.) * rows
                likelyBgYmax: float = rows - likelyBgYmin

                likelyFgXmin: float = (1. / 4.) * cols
                likelyFgXmax: float = cols - likelyFgXmin
                likelyFgYmin: float = (1. / 4.) * rows
                likelyFgYmax: float = rows - likelyFgYmin

                isBg = lambda i, j, dist: i < bgYmin or i > bgYmax or j < bgXmin or j > bgXmax
                isLikelyBg = lambda i, j, dist: i < likelyBgYmin or i > likelyBgYmax or \
                                                j < likelyBgXmin or j > likelyBgXmax
                isLikelyFg = lambda i, j, dist: i < likelyFgYmin or i > likelyFgYmax or \
                                                j < likelyFgXmin or j > likelyFgXmax

            elif maskType is "circular":
                likelyBg: float = midRow if midRow > midCol else midCol
                likelyFg: float = (2. / 3.) * likelyBg
                fg: float = (1. / 3.) * likelyBg

                isBg = lambda i, j, dist: dist > likelyBg
                isLikelyBg = lambda i, j, dist: dist > likelyFg
                isLikelyFg = lambda i, j, dist: dist > fg
            else:
                raise ValueError(maskType + "mask not implemented")

            for i in range(rows):
                for j in range(cols):
                    dist: float = np.sqrt((i - midRow) ** 2 + (j - midCol) ** 2)
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
        image[mask == 0] = (255, 255, 255)

        return image

    @staticmethod
    def mrf(image: Image, costImage: Image, mu_1: float, mu_2: float, mu_3: float,
            sigma_1: float, sigma_2: float, sigma_3: float, pairwiseCost: float) -> Image:
        costs: np.array = np.array([[Segmenter.__gaussianKernel3D(pixel, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3)
                                     for pixel in row] for row in costImage], dtype=float)

        graph = maxflow.Graph[float]()
        nodes = graph.add_grid_nodes(image.shape[:2])
        graph.add_grid_edges(nodes, pairwiseCost * 255)
        graph.add_grid_tedges(nodes, 255 - costs, costs)

        graph.maxflow()

        segments = graph.get_grid_segments(nodes)
        labels: np.array = np.array(segments, dtype="uint8")
        image = cv2.bitwise_and(image, image, mask=labels)

        return image

    @staticmethod
    def threshold(image: Image, thresholdImage: Image, method: str, colorRange: Tuple = None) -> Tuple:
        if method == "adaptive":
            mask = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        elif method == "otsu":
            (_, mask) = cv2.threshold(thresholdImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif method == "range":
            (lower, upper) = colorRange
            mask = cv2.inRange(thresholdImage, lower, upper)
        else:
            raise ValueError(method + " threshold not implemented")

        kernel: Image = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        image = cv2.bitwise_and(image, image, mask=mask)
        return image, mask

    @staticmethod
    def inferBoxesFromBlackRegions(blackResponseImage: Image) -> Grid:
        if len(blackResponseImage.shape) < 3:
            binaryImage = blackResponseImage.copy()
            processedImage = cv2.cvtColor(blackResponseImage, cv2.COLOR_GRAY2BGR)
        else:
            binaryImage = cv2.cvtColor(blackResponseImage, cv2.COLOR_BGR2GRAY)
            processedImage = blackResponseImage.copy()
        binaryImage = cv2.GaussianBlur(binaryImage, (3, 3), 0)
        (_, binaryImage) = cv2.threshold(binaryImage, 100, 255, cv2.THRESH_BINARY)

        (contours, hierarchy) = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        outerBoxes: Grid = Grid([[Box()]], dtype=Box)
        Segmenter.__createBoxesFromContours(outerBoxes, outerBoxes, contours, 5000, 300000, processedImage)

        referenceBoxes: Grid = outerBoxes[0, 0].divide(GRID_SIZE)
        innerBoxes: Grid = Grid([[Box() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Box)
        Segmenter.__createBoxesFromContours(innerBoxes, referenceBoxes, contours, 1000, 5000, processedImage)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                box: Box = innerBoxes[row, col]
                if box.bottomRight().x == 0:
                    Segmenter.__inferBoxFromSurroundingBoxes(innerBoxes, row, col)

        return innerBoxes

    # -------------------------------------------------------- #
    # -------------------- Helper Methods -------------------- #
    # -------------------------------------------------------- #
    @staticmethod
    def __gaussianKernel2D(pixel: Tuple, mu_1: float, sigma_1: float, mu_2: float, sigma_2: float):
        d: float = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[2], 2) / (2 * pow(sigma_2, 2))))
        return d * 255

    @staticmethod
    def __gaussianKernel3D(pixel: Tuple,
                           mu_1: float, sigma_1: float, mu_2: float, sigma_2: float, mu_3: float, sigma_3: float):
        n: int = 1  # No normalization
        # n = 1 / (pow(2 * pi, 3 / 2) * y_sigma * cr_sigma * cb_sigma)

        d: float = exp(-(
                pow(mu_1 - pixel[0], 2) / (2 * pow(sigma_1, 2)) +
                pow(mu_2 - pixel[1], 2) / (2 * pow(sigma_2, 2)) +
                pow(mu_3 - pixel[2], 2) / (2 * pow(sigma_3, 2))))
        return n * d * 255

    @staticmethod
    def __createBoxesFromContours(boxes: Grid, referenceBoxes: Grid, contours: np.array, minArea: int,
                                  maxArea: int, processedImage: Image):
        for c in contours:
            area: int = cv2.contourArea(c)
            if minArea < area < maxArea:
                perimeter: int = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)

                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    squareRatio: float = w / float(h)

                    if 0.95 <= squareRatio <= 1.05:
                        box: Box = Box(Point(x, y), Point(x + w - 1, y + h - 1))
                        index: Tuple = Segmenter.__nearestBox(box, referenceBoxes)
                        boxes[index] = box
                        cv2.drawContours(processedImage, [c], -1, (0, 255, 0), 2)

    @staticmethod
    def __nearestBox(box: Box, boxes: Grid) -> Tuple:
        nearestDistance: float = float("inf")
        nearestIndex: Tuple = (0, 0)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                distance: float = box.centroid().distance(boxes[row, col].centroid())
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearestIndex = (row, col)

        return nearestIndex

    @staticmethod
    def __inferBoxFromSurroundingBoxes(innerBoxes: Grid, row: int, col: int):
        topLeftXs: np.array = np.array([])
        topLeftYs: np.array = np.array([])
        bottomRightXs: np.array = np.array([])
        bottomRightYs: np.array = np.array([])

        # ABOVE
        if row > 0:
            newRow = row - 1
            if col > 0:
                box = innerBoxes[newRow, col - 1]
                if box.bottomRight().x is not 0:
                    topLeftXs = np.append(topLeftXs, [box.bottomRight().x])
                    topLeftYs = np.append(topLeftYs, [box.bottomRight().y])

            box = innerBoxes[newRow, col]
            if box.bottomRight().x is not 0:
                topLeftXs = np.append(topLeftXs, [box.topLeft().x])
                bottomRightXs = np.append(bottomRightXs, [box.bottomRight().x])
                topLeftYs = np.append(topLeftYs, [box.bottomRight().y])

            if col < 4:
                box = innerBoxes[newRow, col + 1]
                if box.bottomRight().x is not 0:
                    bottomRightXs = np.append(bottomRightXs, [box.topLeft().x])
                    topLeftYs = np.append(topLeftYs, [box.bottomRight().y])

        # MIDDLE
        if col > 0:
            box = innerBoxes[row, col - 1]
            if box.bottomRight().x is not 0:
                topLeftXs = np.append(topLeftXs, [box.bottomRight().x])
                topLeftYs = np.append(topLeftYs, [box.topLeft().y])
                bottomRightYs = np.append(bottomRightYs, [box.bottomRight().y])

        if col < 4:
            box = innerBoxes[row, col + 1]
            if box.bottomRight().x is not 0:
                bottomRightXs = np.append(bottomRightXs, [box.topLeft().x])
                topLeftYs = np.append(topLeftYs, [box.topLeft().y])
                bottomRightYs = np.append(bottomRightYs, [box.bottomRight().y])

        # BELOW
        if row < 4:
            newRow = row + 1
            if col > 0:
                box = innerBoxes[newRow, col - 1]
                if box.bottomRight().x is not 0:
                    topLeftXs = np.append(topLeftXs, [box.bottomRight().x])
                    bottomRightYs = np.append(bottomRightYs, [box.topLeft().y])

            box = innerBoxes[newRow, col]
            if box.bottomRight().x is not 0:
                topLeftXs = np.append(topLeftXs, [box.topLeft().x])
                bottomRightXs = np.append(bottomRightXs, [box.bottomRight().x])
                bottomRightYs = np.append(bottomRightYs, [box.topLeft().y])

            if col < 4:
                box = innerBoxes[newRow, col + 1]
                if box.bottomRight().x is not 0:
                    bottomRightXs = np.append(bottomRightXs, [box.topLeft().x])
                    bottomRightYs = np.append(bottomRightYs, [box.topLeft().y])

        topLeftX: float = float(np.mean(topLeftXs))
        topLeftY: float = float(np.mean(topLeftYs))
        bottomRightX: float = float(np.mean(bottomRightXs))
        bottomRightY: float = float(np.mean(bottomRightYs))

        innerBoxes[row, col] = Box(Point(round(topLeftX), round(topLeftY)),
                                   Point(round(bottomRightX), round(bottomRightY)))
