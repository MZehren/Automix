"""
Utility package with functions used by multiple packages
"""

import numpy as np

from automix.utils import quantization


def groupEventsInBar(beats, events):
    """
        Compute the number of event (i.e. kicks) occuring during this bar.
        returns a 2D array of the form : [[downbeatTimestampInSeconds, #events],...] 
    """
    beatsTime = [beat[0] for beat in beats]
    events = quantization.quantize(beatsTime, events)
    downbeat = [beat[0] for beat in beats if beat[1] == 1]

    result = [[downbeat[i], len([event for event in events if event > downbeat[i] and event < downbeat[i + 1]])]
              for i in range(len(downbeat) - 1)]
    return result


def getStructureFromEvents(eventCountPerBar, minBarSpawn=4):
    """
        try to find boundaries where there is a big change in the input
        TODO: make this function more general
    """
    flagBars = [[eventCountPerBar[i][0], eventCountPerBar[i][1] > 0] for i in range(len(eventCountPerBar))]
    changeIndexes = [i for i in range(len(flagBars) - 1) if flagBars[i][1] != flagBars[i + 1][1]]
    groupLengths = np.diff(changeIndexes)
    boundaryIndexes = [
        changeIndexes[i + 1] for i in range(len(groupLengths) - 1)
        if groupLengths[i] >= minBarSpawn and groupLengths[i + 1] >= minBarSpawn
    ]
    boundaryTimes = [flagBars[i + 1][0] for i in boundaryIndexes]

    return boundaryTimes


def KnuthMorrisPratt(text, pattern):
    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos - shift]:
            shift += shifts[pos - shift]
        shifts[pos + 1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    result = []
    for c in text:
        while matchLen == len(pattern) or \
                matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            result.append(startPos)

    return result


def hertzToNote(freq):
    """
    see https://pages.mtu.edu/~suits/NoteFreqCalcs.html
    returns the closest note corresponding to the freq in Hz
    i.e. : hertzToNote(130)='C'
    """
    raise DeprecationWarning()
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "E#", "F", "G", "G#"]
    semitonesDiff = range(-36, 37)
    computedFreqs = []
    if not computedFreqs:
        A4 = 440
        computedFreqs = [A4 * pow(1.059463094359, i) for i in semitonesDiff]
    ClosestSemitomeDiff = np.argmin([abs(freq - computedFreq) for computedFreq in computedFreqs])

    return notes[semitonesDiff[ClosestSemitomeDiff] % len(notes)]


def normalize(array):
    """
    normalize an array by changing its values to 0 to 1 
    """
    # TODO move to a better location
    max = np.max(array)
    min = np.min(array)
    return ((array - min) / (max - min)).tolist()
