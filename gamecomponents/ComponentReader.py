import cv2
import numpy as np

from imageprocessing.Detector import Detector
from gamecomponents.Keycard import Team, Keycard
from gamecomponents.Grid import inferCardBoxesFromGridlines, inferSquareBoxesFromGridlines, iterateCellsInImage
from gamecomponents.Wordgrid import Wordgrid

Color = np.array
Grid = np.array
Image = np.ndarray


class ComponentReader:
    blackMean: Color = Color([0, 0, 0])
    blueMean: Color = Color([200, 125, 25])
    redMean: Color = Color([50, 50, 200])
    tanMean: Color = Color([215, 220, 210])

    def __init__(self):
        pass

    def extractKeycard(self, path: str, numBoxes: int) -> Keycard:
        keycard: Keycard = Keycard(numBoxes)
        image: Image = cv2.imread(path)
        grid: Grid = inferSquareBoxesFromGridlines(image, numBoxes)

        def inferTeamFromCell(row: int, col: int, cell: Image):
            # cv2.imshow("Cell", cell)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            cellMean: Color = np.mean(np.mean(cell, 0), 0)
            similarity: np.array = np.array([np.linalg.norm(self.blackMean - cellMean),
                                             np.linalg.norm(self.blueMean - cellMean),
                                             np.linalg.norm(self.redMean - cellMean),
                                             np.linalg.norm(self.tanMean - cellMean)])
            teams: np.array = np.array([Team.ASSASSIN, Team.BLUE, Team.RED, Team.NEUTRAL])
            team: Team = teams[np.argmin(similarity)]

            keycard.set2D(row, col, team)

        iterateCellsInImage(image, grid, inferTeamFromCell)

        return keycard

    def extractWordgrid(self, path: str, numBoxes: int) -> Wordgrid:
        wordgrid: Wordgrid = Wordgrid(numBoxes)
        image: Image = cv2.imread(path)
        boxes: Grid = inferCardBoxesFromGridlines(image, numBoxes)

        def inferTextFromCell(row: int, col: int, cell: Image):
            # cv2.imshow("Cell", cell)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            detector: Detector = Detector()
            text: str = detector.readTextOnCard(cell)

            wordgrid.set2D(row, col, text)

        iterateCellsInImage(image, boxes, inferTextFromCell)

        return wordgrid
