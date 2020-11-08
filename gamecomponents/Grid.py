import numpy as np

from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point

Grid = np.array
Image = np.array


def generateBoxes(width: int, height: int, boxWidth: int, boxHeight: int, numBoxes: int) -> np.array:
    gridWidth: int = numBoxes * boxWidth
    gridHeight: int = numBoxes * boxHeight
    minX: int = int((width - gridWidth) / 2)
    minY: int = int((height - gridHeight) / 2)

    boxes: np.array = np.array([[Box() for col in range(numBoxes)] for row in range(numBoxes)], dtype=Box)
    for row in range(numBoxes):
        for col in range(numBoxes):
            boxes[row, col] = Box(Point(minX + boxWidth * col, minY + boxHeight * row),
                                  Point(minX + boxWidth * (col + 1), minY + boxHeight * (row + 1)))
    return boxes


def inferSquareBoxesFromGridlines(image: Image, numBoxes: int) -> np.array:
    height, width = image.shape[:2]
    squareSize = int(min(height, width) / (numBoxes + 1))
    boxes: np.array = generateBoxes(width, height, squareSize, squareSize, numBoxes)
    return boxes


def inferCardBoxesFromGridlines(image: Image, numBoxes: int) -> np.array:
    height, width = image.shape[:2]
    squareSize = int(min(height, width) / (numBoxes + 1))
    rectangleHeight: int = squareSize
    rectangleWidth: int = int(squareSize * 1.5357)
    boxes: np.array = generateBoxes(width, height, rectangleWidth, rectangleHeight, numBoxes)
    return boxes


def iterateCellsInImage(image: Image, grid: Grid, func):
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            box: Box = grid[row, col]
            topLeft: Point = box.topLeft()
            bottomRight: Point = box.bottomRight()
            cell: np.array = image[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

            func(row, col, cell)
