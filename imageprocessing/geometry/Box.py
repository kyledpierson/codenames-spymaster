import numpy as np

from .Point import Point

Grid = np.array


class Box:
    def __init__(self, topLeft: Point = Point(), bottomRight: Point = Point()):
        self.box: np.array = np.array([topLeft, bottomRight], dtype=Point)

    def __str__(self):
        return "[" + str(self.topLeft()) + ", " + str(self.bottomRight()) + "]"

    def topLeft(self) -> Point:
        return self.box[0]

    def bottomRight(self) -> Point:
        return self.box[1]

    def centroid(self) -> Point:
        return self.topLeft().midpoint(self.bottomRight())

    def split(self, axis: str = "x") -> "Box":
        centroid: Point = self.centroid()
        if axis is "x":
            self.box[1].x = centroid.x
            return Box(Point(centroid.x, self.topLeft().y), self.bottomRight())
        elif axis is "y":
            self.box[1].y = centroid.y
            return Box(Point(self.topLeft().x, centroid.y), self.bottomRight())
        else:
            raise ValueError(axis + " axis not implemented, must specify x or y")

    def divide(self, dim: int) -> Grid:
        edges: np.array = self.topLeft().divide(self.bottomRight(), dim)
        boxes: Grid = Grid([[Box() for col in range(dim)] for row in range(dim)], dtype=Box)
        for row in range(dim):
            for col in range(dim):
                boxes[row, col].box = np.array([
                    Point(edges[col].x, edges[row].y), Point(edges[col + 1].x, edges[row + 1].y)], dtype=Point)
        return boxes
