import cv2
import numpy as np

from typing import Tuple

from globalvariables import GRID_SIZE, BOX_WIDTH_RATIO
from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point

Grid = np.array
Image = np.ndarray


class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def threshold(image: Image, thresholdImage: Image, method: str, colorRange: Tuple = None) -> Tuple:
        if method is "adaptive":
            mask = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        elif method is "otsu":
            (_, mask) = cv2.threshold(thresholdImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif method is "range":
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
    def inferKeyBoxesFromOverlay(image: Image) -> Grid:
        (height, width) = image.shape[:2]
        boxes: Grid = Segmenter.iterateBoxes(width, height, 1, 1, Segmenter.generateBoxes)
        return boxes

    @staticmethod
    def inferCardBoxesFromOverlay(image: Image) -> Grid:
        (height, width) = image.shape[:2]
        boxes: Grid = Segmenter.iterateBoxes(width, height, BOX_WIDTH_RATIO, 1, Segmenter.generateBoxes)
        return boxes

    @staticmethod
    def iterateBoxes(width: int, height: int, boxWidthRatio: float, boxHeightRatio: float, boxFunction):
        squareSize: int = int(min(height, width) / (GRID_SIZE + 1))
        boxWidth: int = int(squareSize * boxWidthRatio)
        boxHeight: int = int(squareSize * boxHeightRatio)
        gridWidth: int = GRID_SIZE * boxWidth
        gridHeight: int = GRID_SIZE * boxHeight
        minX: int = int((width - gridWidth) / 2)
        minY: int = int((height - gridHeight) / 2)

        return boxFunction(width, height, minX, minY, boxWidth, boxHeight, gridWidth, gridHeight)

    @staticmethod
    def generateBoxes(width: int, height: int, minX: int, minY: int,
                      boxWidth: int, boxHeight: int, gridWidth: int, gridHeight: int) -> Grid:
        return Grid([[Box(Point(minX + boxWidth * col, minY + boxHeight * row),
                          Point(minX + boxWidth * (col + 1), minY + boxHeight * (row + 1)))
                      for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Box)

    @staticmethod
    def generateOverlay(width: int, height: int, minX: int, minY: int,
                        boxWidth: int, boxHeight: int, gridWidth: int, gridHeight: int) -> Grid:
        maxX: int = minX + gridWidth
        maxY: int = minY + gridHeight
        overlay: np.array = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(GRID_SIZE + 1):
            offset = minX + boxWidth * i
            overlay[minY:maxY, offset, :] = 0xFF

        for i in range(GRID_SIZE + 1):
            offset = minY + boxHeight * i
            overlay[offset, minX:maxX, :] = 0xFF

        return overlay
