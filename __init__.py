import numpy as np

from globalvariables import GRID_SIZE

from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Card import Card, Team

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
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    for keycardImage in keycardImages:
        reader.readKeycard(inDir + keycardImage, cardGrid)
    for wordgridImage in wordgridImages:
        reader.readWordgrid(inDir + wordgridImage, cardGrid)

    output: str = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            output += "[{:^30}]".format(str(cardGrid[row, col]))
        output += "\n"
    print(output)
