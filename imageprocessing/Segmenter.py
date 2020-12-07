import cv2
import numpy as np

from typing import Tuple

from globalvariables import GRID_SIZE
from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point

Grid = np.array
Image = np.ndarray


class Segmenter:
    def __init__(self):
        pass

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
    def inferKeyBoxesFromOverlay(image: Image) -> Grid:
        (height, width) = image.shape[:2]
        squareSize: int = int(min(height, width) / (GRID_SIZE + 1))
        boxes: Grid = Segmenter.__generateBoxes(width, height, squareSize, squareSize)
        return boxes

    @staticmethod
    def inferCardBoxesFromOverlay(image: Image) -> Grid:
        (height, width) = image.shape[:2]
        squareSize: int = int(min(height, width) / (GRID_SIZE + 1))
        rectangleHeight: int = squareSize
        rectangleWidth: int = int(squareSize * 1.5357)
        boxes: Grid = Segmenter.__generateBoxes(width, height, rectangleWidth, rectangleHeight)
        return boxes

    @staticmethod
    def __generateBoxes(width: int, height: int, boxWidth: int, boxHeight: int) -> Grid:
        gridWidth: int = GRID_SIZE * boxWidth
        gridHeight: int = GRID_SIZE * boxHeight
        minX: int = int((width - gridWidth) / 2)
        minY: int = int((height - gridHeight) / 2)

        boxes: Grid = Grid([[Box() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Box)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                boxes[row, col] = Box(Point(minX + boxWidth * col, minY + boxHeight * row),
                                      Point(minX + boxWidth * (col + 1), minY + boxHeight * (row + 1)))
        return boxes
