import numpy as np

from globalvariables import GRID_SIZE

from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Card import Card, Team
from iomanager.CodenamesGUI import CodenamesGUI

from cluefinder.ClueFinder import ClueFinder
from cluefinder.ApproximationClueFinder import ApproximationClueFinder
from cluefinder.BruteForceClueFinder import BruteForceClueFinder
from cluefinder.GensimClueFinder import GensimClueFinder

"""
if colorSpace is cv2.COLOR_BGR2HLS:
    lightRed1 = (0, 100, 150)
    darkRed1 = (40, 150, 255)
    lightRed2 = (215, 100, 150)
    darkRed2 = (255, 150, 255)
    lightBlue = (115, 75, 120)
    darkBlue = (155, 130, 255)
    
cv2.imshow("Image", image)
cv2.waitKey()
cv2.destroyAllWindows()

x = np.zeros((100000, 300))
index = 0
for key in embeddings:
    x[index] = embeddings[key]
    index = index + 1

pca = PCA()
pca.fit(x)

[(w, ", ".join(closest_words(w)[1:10]) + "...") for w in ["magic", "sport", "scuba", "sock"]]

TODO:
 - Implement validate method
 - Add detection for covered clues
 
 - update public/private methods in all classes
 - have cardGrid be its own class to iterate over cards and maintain game state (team)
 
 - Add risk level
 - Tune k in getClue
 - Pre-compute distances between words
"""

# ==================================================
inDir: str = "inImages/"
outDir: str = "outImages/"
# ==================================================

Grid = np.array
Image = np.array

if __name__ == "__main__":
    gui: CodenamesGUI = CodenamesGUI()
    reader: ComponentReader = ComponentReader()
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    gui.captureKeycard()
    reader.readKeycard(inDir + "keycard.jpg", cardGrid)
    team: Team = gui.verifyKeycard(cardGrid)

    gameOver: bool = False
    clueFinder: ClueFinder = ApproximationClueFinder(vocabularySize=100000)
    while gameOver is False:
        gui.captureWordgrid()
        reader.readWordgrid(inDir + "wordgrid.jpg", cardGrid)
        gui.verifyWordgrid(cardGrid)

        output: str = ""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                output += "[{:^30}]".format(str(cardGrid[row, col]))
            output += "\n"
        print(output)
        print(team)

        while not clueFinder.checkVocabulary(cardGrid):
            gui.verifyWordgrid(cardGrid)

        clue: str = clueFinder.getClue(cardGrid, team)
        gameOver = gui.displayClueAndWait(clue)
