import cv2
import numpy as np

from gamecomponents.Card import Team
from gamecomponents.Grid import inferCardBoxesFromOverlay, inferKeyBoxesFromOverlay, iterateCellsInImage
from globalvariables import GRID_SIZE
from imageprocessing.Detector import Detector
from imageprocessing.Filterer import Filterer

Color = np.array
Grid = np.array
Image = np.ndarray


class ComponentReader:
    def __init__(self):
        self.blueMean: Color = Color([200, 125, 25])
        self.redMean: Color = Color([50, 50, 200])
        self.tanMean: Color = Color([215, 220, 210])

        self.black: Color = Color([0, 0, 0])
        self.blue: Color = Color([255, 0, 0])
        self.red: Color = Color([0, 0, 255])
        self.green: Color = Color([0, 255, 0])

        self.teams: np.array = np.array([Team.ASSASSIN, Team.BLUE, Team.RED, Team.NEUTRAL], dtype=Team)

    def readKeycard(self, path: str, cardGrid: Grid):
        image: Image = cv2.imread(path)
        image = Filterer.equalizeHistogram(image, cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, (True, False, False))
        boxes: Grid = inferKeyBoxesFromOverlay(image)
        similarities: np.array = np.zeros((GRID_SIZE * GRID_SIZE, self.teams.size), dtype=float)

        def inferTeamFromCell(row: int, col: int, cell: Image):
            cellMean: Color = np.mean(cell)
            similarity: np.array = np.array([np.linalg.norm(self.black - cellMean),
                                             np.linalg.norm(self.blue - cellMean),
                                             np.linalg.norm(self.red - cellMean),
                                             (np.max(cellMean) - np.min(cellMean)) / np.max(cellMean)], dtype=float)
            similarities[row * GRID_SIZE + col] = similarity

        iterateCellsInImage(image, boxes, inferTeamFromCell)

        # Reserve blackest cell for the assassin
        blackIndex: int = int(np.argmin(similarities[:, 0]))
        similarities[:, 0] = float("inf")
        similarities[blackIndex, 0] = 0.0
        similarities[blackIndex, 3] = float("inf")

        # Reserve 7 most tan cells for neutral
        indices: np.array = np.argsort(similarities, axis=0)
        similarities[indices[:7, 3], 3] = 0.0
        similarities[indices[7:, 3], 3] = float("inf")

        def setTeam(row: int, col: int, cell: Image):
            similarity: np.array = similarities[row * GRID_SIZE + col]
            team: Team = self.teams[np.argmin(similarity)]
            cardGrid[row, col].setTeam(team)

        iterateCellsInImage(image, boxes, setTeam)

    def readWordgrid(self, path: str, cardGrid: Grid):
        image: Image = cv2.imread(path)
        boxes: Grid = inferCardBoxesFromOverlay(image)
        detector: Detector = Detector()

        def inferTextFromCell(row: int, col: int, cell: Image):
            text: str = detector.readTextOnCard(cell)
            cardGrid[row, col].setText(text)

        iterateCellsInImage(image, boxes, inferTextFromCell)
