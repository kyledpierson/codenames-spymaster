from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Keycard import Keycard
from gamecomponents.Wordgrid import Wordgrid

"""
TODO
 - Evaluate all the places I'm doing a gaussian blur

Ideas
 - Canny

if colorSpace is cv2.COLOR_BGR2HLS:
    lightRed1 = (0, 100, 150)
    darkRed1 = (40, 150, 255)
    lightRed2 = (215, 100, 150)
    darkRed2 = (255, 150, 255)
    lightBlue = (115, 75, 120)
    darkBlue = (155, 130, 255)
"""

# ==================================================
inDir: str = 'inImages/'
outDir: str = 'outImages/'
keycardImages: list = [
    'keycard-1.jpg',
    # 'keycard-2.jpg',
    # 'keycard-3.jpg',
    # 'keycard-light-1.jpg',
    # 'keycard-dark-1.jpg',
    # 'keycard-dark-2.jpg',
    # 'keycard-noborder-1.jpg',
    # 'keycard-noborder-2.jpg',
    # 'keycard-noborder-3.jpg',
    # 'keycard-offcenter-1.jpg',
    # 'keycard-offcenter-2.jpg',
    # 'keycard-offcenter-3.jpg',
    # 'keycard-offcenter-4.jpg'
]
wordgridImages = [
    'word-grid-1.png',
    # 'word-grid-2.JPG',
    # 'word-grid-3.jpg',
    # 'word-grid-4.png',
]
referenceImageFileName: str = 'keycard-reference.webp'
# ==================================================

if __name__ == '__main__':
    reader: ComponentReader = ComponentReader(inDir + referenceImageFileName)

    for keycardImage in keycardImages:
        for wordgridImage in wordgridImages:
            keycard: Keycard = reader.extractKeycard(inDir + keycardImage)
            wordgrid: Wordgrid = reader.extractWordgrid(inDir + wordgridImage)
            print(keycard.grid)
            print(wordgrid.grid)
