# Codenames Spymaster
An AI for processing visual input from a game of codenames and using it to generate clues for a given team. This was
originally written to run on a Raspberry Pi with a camera module, but can be tested on any machine by using the `-noPi`
option (this disables image capture and thus requires the user to provide paths for input images or saved game state,
see below).

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
  -loadInitialState <path>  Location of the object from saveInitialState (bypass all image processing)
  -saveInitialState <path>  Location to save the game's initial state object
```
*Examples*
```
python run.py -noPi -keycard images/keycard.jpg -wordgrid images/wordgrid.jpg
python run.py -noPi -loadInitialState data/initialState.npy
```

## Adding new clue-finding strategies
If you want to experiment with a new strategy for finding optimal clues, you'll need to create a new child class of
ClueFinder. Your child class needs to implement the following functions:
```
def __init__(self, vocabularySize: int):
    # Constructor that must invoke the base class constructor
    super().__init__(vocabularySize)
    ...

def _textInVocabulary(self, text: str) -> bool:
    # Returns True if the given text exists in this clue finder's vocabulary
    ...

def _getBestClue(self, positiveWords: np.array, negativeWords: np.array) -> Tuple:
    # Finds a clue that tries to connect all the positive words while being unrelated to all the negative words
    # Returns a tuple of (clue: str, score: float, positiveWords: np.array)
    ...
    bool isValidClue = self._validate(clue, positiveWords, negativeWords)
    ...
```
The ClueFinder base class takes care of iterating through all combinations of your team's words and calling
`_getBestClue` for each such subset, meaning the `positiveWords` argument represents *all* words your derived class
needs to try and connect (i.e. your derived class does not need to subdivide `positiveWords` to try and find an optimal
grouping).

The `score` you return should represent how good your clue is, without taking into account the number of words it tries
to connect. This `score` is usually determined by a function of how "close" it is to the positive words and how "far" it
is from the negative words. This `score` does not need to be normalized, but it should always be non-negative. The
`heuristics.py` module contains some such functions, but feel free to add more. The base class will determine the final
clue using these scores weighted by the number of words they try to connect, with weights determined by the amount of
"risk" desired by the team.

The base class also provides a protected helper function called `_validate`, which should be invoked by `_getBestClue`
and should return `True` for any clue that your derived class returns. This ensures that your clue follows the rules of
the game.

#### Using your new ClueFinder
Once you have created your class, you can substitute the instantiation of the ClueFinder object in `run.py` with an
instance of your class.
```
...
# clueFinder: ClueFinder = ApproximationClueFinder(vocabularySize=50000)
clueFinder: ClueFinder = MyClueFinder(vocabularySize=100000)
...
```
