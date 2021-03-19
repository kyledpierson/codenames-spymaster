import logging
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


def printUsage(arg: str):
    print("Invalid option: " + arg + "                                                                           \n" +
          "Usage: python run.py [options]                                                                        \n" +
          "Options:                                                                                              \n" +
          "  -noPi                     Don't run Raspberry Pi code for capturing images and dictating clues      \n" +
          "  -keycard <path>           Location of an existing image of the key card                             \n" +
          "  -wordgrid <path>          Location of an existing image of the grid of words                        \n" +
          "  -loadInitialState <path>  Location of the object from saveInitialState (bypass all image processing)\n" +
          "  -saveInitialState <path>  Location to save the game's initial state object                            ")
    quit(-1)


def stringifyCardgrid(cellString) -> str:
    output: str = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            output += "[{:^20}]".format(cellString(row, col))
        output += "\n"
    return output


if __name__ == "__main__":
    logging.basicConfig(filename="codenames.log", filemode="w", format="%(message)s", level=logging.INFO)

    # COMMAND LINE ARGUMENTS
    usePi: bool = True
    keycard: str = ""
    wordgrid: str = ""
    loadInitialState: str = ""
    saveInitialState: str = ""

    i: int = 0
    while i + 1 < len(sys.argv):
        i = i + 1
        arg = sys.argv[i]
        if arg == "-noPi":
            usePi = False
        elif arg == "-keycard" and i + 1 < len(sys.argv):
            i = i + 1
            keycard = sys.argv[i]
        elif arg == "-wordgrid" and i + 1 < len(sys.argv):
            i = i + 1
            wordgrid = sys.argv[i]
        elif arg == "-loadInitialState" and i + 1 < len(sys.argv):
            i = i + 1
            loadInitialState = sys.argv[i]
        elif arg == "-saveInitialState" and i + 1 < len(sys.argv):
            i = i + 1
            saveInitialState = sys.argv[i]
        else:
            printUsage(arg)

    # SETUP
    gui: CodenamesGUI = CodenamesGUI()
    reader: ComponentReader = ComponentReader()
    clueFinder: ClueFinder = ApproximationClueFinder(vocabularySize=50000)
    cardGrid: Grid = Grid([[Card() for col in range(GRID_SIZE)] for row in range(GRID_SIZE)], dtype=Card)

    if loadInitialState != "":
        cardGrid = np.load(loadInitialState, allow_pickle=True)
    else:
        if usePi and keycard == "":
            keycard = gui.captureKeycard()
        reader.readKeycard(keycard, cardGrid)

        if usePi and wordgrid == "":
            wordgrid = gui.captureWordgrid()
        reader.readWordgrid(wordgrid, cardGrid)

    team: Team = gui.verifyKeycard(cardGrid)
    logging.info("=========================\n======== KEYCARD ========\n=========================")
    logging.info(stringifyCardgrid(lambda row, col: str(cardGrid[row, col].team)))

    # MAIN GAME LOOP
    roundNumber: int = 0
    gameOver: bool = False
    while not gameOver:
        roundNumber = roundNumber + 1
        logging.info(
            "=========================\n======== ROUND " + str(roundNumber) + " ========\n=========================")

        risk: int = gui.verifyWordgrid(cardGrid)
        while not clueFinder.checkVocabulary(cardGrid):
            risk = gui.verifyWordgrid(cardGrid)
        logging.info(stringifyCardgrid(lambda row, col: str(cardGrid[row, col].text)))
        if saveInitialState != "":
            np.save(saveInitialState, cardGrid, allow_pickle=True)
            saveInitialState = ""

        clue: str = clueFinder.getClue(cardGrid, team, risk)

        if usePi:
            gttsObj = gTTS(text=clue, lang="en")
            gttsObj.save("gtts.mp3")
            os.system("mpg321 gtts.mp3")

        gameOver = gui.displayClueAndWait(clue)
