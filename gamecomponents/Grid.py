import numpy as np

from globalvariables import GRID_SIZE
from imageprocessing.geometry.Box import Box
from imageprocessing.geometry.Point import Point

Grid = np.array
Image = np.array


def generateBoxes(width: int, height: int, boxWidth: int, boxHeight: int) -> Grid:
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


def inferKeyBoxesFromOverlay(image: Image) -> Grid:
    (height, width) = image.shape[:2]
    squareSize: int = int(min(height, width) / (GRID_SIZE + 1))
    boxes: Grid = generateBoxes(width, height, squareSize, squareSize)
    return boxes


def inferCardBoxesFromOverlay(image: Image) -> Grid:
    (height, width) = image.shape[:2]
    squareSize: int = int(min(height, width) / (GRID_SIZE + 1))
    rectangleHeight: int = squareSize
    rectangleWidth: int = int(squareSize * 1.5357)
    boxes: Grid = generateBoxes(width, height, rectangleWidth, rectangleHeight)
    return boxes


def iterateCellsInImage(image: Image, boxes: Grid, func):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            box: Box = boxes[row, col]
            topLeft: Point = box.topLeft()
            bottomRight: Point = box.bottomRight()
            cell: Image = image[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)]

            func(row, col, cell)
