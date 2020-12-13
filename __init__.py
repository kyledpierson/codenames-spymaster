import numpy as np

from globalvariables import GRID_SIZE

from cluegenerator.ClueGenerator import ClueGenerator
from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Card import Card, Team
from iomanager.CodenamesGUI import CodenamesGUI

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
"""

# ==================================================
inDir: str = 'inImages/'
outDir: str = 'outImages/'
keycardImages: list = [
    "keycard.jpg"
]
wordgridImages: list = [
    "wordgrid.jpg"
]
# ==================================================

Grid = np.array

if __name__ == '__main__':
    reader: ComponentReader = ComponentReader()
    codenamesGUI: CodenamesGUI = CodenamesGUI()
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    for keycardImage in keycardImages:
        reader.readKeycard(inDir + keycardImage, cardGrid)
    codenamesGUI.validateKeyCard(cardGrid)

    for wordgridImage in wordgridImages:
        reader.readWordgrid(inDir + wordgridImage, cardGrid)
    codenamesGUI.validateWordGrid(cardGrid)

    output: str = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            output += "[{:^30}]".format(str(cardGrid[row, col]))
        output += "\n"
    print(output)

    """
    clueGenerator: ClueGenerator = ClueGenerator("glove.42B.300d/top_100000.txt")
    redClue: str = clueGenerator.getClue(cardGrid, Team.RED)
    print(redClue)
    blueClue = clueGenerator.getClue(cardGrid, Team.BLUE)
    print(blueClue)
    """
