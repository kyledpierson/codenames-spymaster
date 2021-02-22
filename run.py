import numpy as np
import os
import sys

from gtts import gTTS

from cluefinder.ClueFinder import ClueFinder
from cluefinder.ApproximationClueFinder import ApproximationClueFinder
from gamecomponents.ComponentReader import ComponentReader
from gamecomponents.Card import Card, Team
from globalvariables import GRID_SIZE
from iomanager.CodenamesGUI import CodenamesGUI

"""
TODO:
 - Allow for setting of specific number of connected words
 - Add detection for covered clues
 - have cardGrid be its own class to iterate over cards and maintain game state (team)
 
 - Implement "related words" strategy
 - Pre-compute distances between words
 - Pre-compute nearest neighbors?
"""

Grid = np.array
Image = np.array

if __name__ == "__main__":
    # COMMAND LINE ARGUMENTS
    inDir: str = ""
    outDir: str = ""
    loadImages: bool = False

    if len(sys.argv) > 1:
        inDir = sys.argv[1] + "/"
        loadImages = True
        if len(sys.argv) > 2:
            outDir = sys.argv[2] + "/"

    # SETUP
    gui: CodenamesGUI = CodenamesGUI()
    reader: ComponentReader = ComponentReader()
    clueFinder: ClueFinder = ApproximationClueFinder(vocabularySize=50000)
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    if not loadImages:
        gui.captureKeycard()
    reader.readKeycard(inDir + "keycard.jpg", cardGrid)
    team: Team = gui.verifyKeycard(cardGrid)

    # MAIN GAME LOOP
    gameOver: bool = False
    while not gameOver:
        if not loadImages:
            gui.captureWordgrid()
        reader.readWordgrid(inDir + "wordgrid.jpg", cardGrid)
        risk: int = gui.verifyWordgrid(cardGrid)
        while not clueFinder.checkVocabulary(cardGrid):
            risk = gui.verifyWordgrid(cardGrid)

        clue: str = clueFinder.getClue(cardGrid, team, risk)

        gttsObj = gTTS(text=clue, lang="en")
        gttsObj.save("gtts.mp3")
        os.system("mpg321 gtts.mp3")

        gameOver = gui.displayClueAndWait(clue)
