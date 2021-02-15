import numpy as np

from globalvariables import GRID_SIZE

from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Card import Card, Team
from iomanager.CodenamesGUI import CodenamesGUI

from cluefinder.ClueFinder import ClueFinder
from cluefinder.ApproximationClueFinder import ApproximationClueFinder

"""
TODO:
 - Add detection for covered clues
 - have cardGrid be its own class to iterate over cards and maintain game state (team)
 
 - Implement "related words" strategy
 - Pre-compute distances between words
 - Pre-compute nearest neighbors?
"""

# ==================================================
inDir: str = "inImages/"
outDir: str = "outImages/"
# ==================================================

Grid = np.array
Image = np.array


def printCardGrid(cardGrid: Grid):
    output: str = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            output += "[{:^30}]".format(str(cardGrid[row, col]))
        output += "\n"
    print(output)


if __name__ == "__main__":
    gui: CodenamesGUI = CodenamesGUI()
    reader: ComponentReader = ComponentReader()
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    gui.captureKeycard()
    reader.readKeycard(inDir + "keycard.jpg", cardGrid)
    team: Team = gui.verifyKeycard(cardGrid)

    gameOver: bool = False
    clueFinder: ClueFinder = ApproximationClueFinder(vocabularySize=50000)
    while not gameOver:
        gui.captureWordgrid()
        reader.readWordgrid(inDir + "wordgrid.jpg", cardGrid)
        risk: int = gui.verifyWordgrid(cardGrid)

        while not clueFinder.checkVocabulary(cardGrid):
            risk = gui.verifyWordgrid(cardGrid)

        clue: str = clueFinder.getClue(cardGrid, team, risk)
        # Read clue aloud
        gameOver = gui.displayClueAndWait(clue)
