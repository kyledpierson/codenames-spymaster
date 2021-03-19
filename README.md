# Codenames Spymaster
An AI for processing visual input from a game of codenames and using it to
generate clues for a given team. This was originally written to run on a
Raspberry Pi with a camera module, but can be tested on any machine by using 
the `-noPi` option (this disables image capture and thus requires the user to
provide paths for input images or saved game state, see below).

## Requirements
- Python 3 (probably 3.7+)
```
pip install -r requirements
```

## Running
```
Usage: python run.py [options]
Options:
  -noPi                     Don't run Raspberry Pi code for capturing images and dictating clues
  -keycard <path>           Location of an existing image of the key card
  -wordgrid <path>          Location of an existing image of the grid of words
  -loadInitialState <path>  Location of the object from saveInitialState
  -saveInitialState <path>  Location to save the game's initial state object
```
*Examples*
```
python run.py -noPi -keycard images/keycard.jpg -wordgrid images/wordgrid.jpg
python run.py -noPi -loadInitialState data/initialState.npy
```

