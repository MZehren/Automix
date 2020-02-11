"""
Utility functions to do quantization 
"""
import math
from bisect import bisect_left
from typing import List
import itertools

import numpy as np
from scipy.stats import mode
from automix.model.classes.signal import Signal


def separateInEvenGrids(ticks, regularityThreshold=0.1, tickLength=None):
    '''
    from [0,1,2,3,4,10,11,12]
    returns [[0,1,2,3,4],[10,11,12]]

    Create a new group of values if the step between two values is deviated by 10% or more of the average step between each tick   
    the regularityThreshold can be used to fine tune the threshold 
    '''
    ticksGroups = [[ticks[0]]]
    ticksDiffs = np.diff(ticks)
    if tickLength is None:
        tickLength = mode(ticksDiffs)[0][0]
    # std =  np.std(ticksDiffs)
    precision = regularityThreshold * tickLength
    for i in range(len(ticks) - 1):
        if abs(ticksDiffs[i] - tickLength) < precision:
            ticksGroups[-1].append(ticks[i + 1])
        else:
            ticksGroups.append([ticks[i + 1]])

    return [group for group in ticksGroups if len(group) > 1]


def quantizeOne(grid, value):
    """
    return the closest grid tick to the value
    """
    closestTick = np.argmin([abs(tick - value) for tick in grid])
    return grid[closestTick]


# def quantize(grid, values, maxThreshold=-1, removeOutOfBound=True, removeDuplicatedValues=True):
#     """
#     Align values to a grid
#     Parameters:
#         maxThreshold specify the maximum distance between the value and the closest tick in the grid.
#             by default it'll take half of the most common difference in the grid
#         removeOutOfBound: if there is no tick near the value AND removeOutOfBound, the value is removed.
#             if removeOutOfBound is False, then the original value is kept
#         removeDuplicatedValues: If two values are quantized to the same grid tick, it's going to get removed

#     return the list of values quantized
#     """
#     # computes the max threshold if not specified
#     if maxThreshold == -1:
#         if len(grid) > 1:
#             maxThreshold = float(mode(np.diff(grid))[0][0]) / 2
#         else:
#             maxThreshold = 0

#     # creates a dict associating each value to his closest ground truth
#     alignementValues = findGridNeighbor(grid, values)

#     # replace the value by its closest grid tick, as long as it's close enough
#     alignementValues = ((originalValue, newValue if abs(originalValue - newValue) <= maxThreshold else originalValue)
#                         for originalValue, newValue in alignementValues
#                         if not (removeOutOfBound and abs(originalValue - newValue) > maxThreshold))

#     # compute the list of results from the dictionnary
#     if removeDuplicatedValues:
#         return list(sorted(set([newValue for originalValue, newValue in alignementValues])))
#     else:
#         return [newValue for originalValue, newValue in alignementValues]


def diff(grid, values, maxThreshold=-1):
    """
    get the difference between the ground truth values (grid) and the values.
    if the difference is above the maxThreshold, then the difference is considered to be zero.
    By default the maxThreshold is going to be the half the mean distance between to ticks in the GT values (grid)
    This is usefull for looking at the difference between events in two tracks.
    TODO: Include that in signal class ?
    """
    gridSignal = Signal(1, times=grid)
    valuesSignal = Signal(1, times=values)
    valuesSignal.quantizeTo(gridSignal, maxThreshold=maxThreshold, removeOutOfBound=False, removeDuplicatedValues=False)
    return valuesSignal.times - values


# def findGridNeighbor(grid, values, isSorted=False):
#     """
#     return a list of tuples indicating the position of the closest
#     """
#     if not isSorted:
#         grid = sorted(grid)

#     # return ((value, grid[np.argmin([abs(value - tick) for tick in grid])]) for value in values)
#     # return ((value, min(grid, key=lambda x:abs(x-value))) for value in values)
#     return [(value, findNeighboor(grid, value)) for value in values]

# def findNeighboor(grid, value):
#     """
#     Assumes grid is sorted. Returns closest value to value.
#     If two numbers are equally close, return the smallest number.
#     """
#     pos = bisect_left(grid, value)

#     if pos == 0:
#         return grid[0]
#     if pos == len(grid):
#         return grid[-1]
#     before = grid[pos - 1]
#     after = grid[pos]
#     if after - value < value - before:
#         return after
#     else:
#         return before


def extendGrid(refTick, ticks, trackLength, approximateTickDuration, SnapDistance=0.05):
    """
    Extends an array of ticks. fills holes and make the grid even by snapping ticks too far away from the point of reference
    """
    joinThreshold = approximateTickDuration * SnapDistance
    iT = refTick[0]  # index Time
    iL = refTick[1]  # index Label
    result = []
    while iT < trackLength:
        # if there is a beat next to what we expect
        closeBeat = [beat for beat in ticks if math.fabs(beat[0] - iT) < joinThreshold]
        if len(closeBeat) == 1:
            iT = closeBeat[0][0]
            result.append([iT, iL])
        else:
            result.append([iT, iL])

        iT = iT + approximateTickDuration
        iL = iL + 1
        if iL == 5:
            iL = 1

    # TODO: factorise the code
    iT = refTick[0] - approximateTickDuration  # index Time
    iL = refTick[1] - 1  # index Label
    if iL == 0:
        iL = 4

    while iT >= joinThreshold * -1:
        # if there is a beat next to what we expect
        closeBeat = [beat for beat in ticks if math.fabs(beat[0] - iT) < joinThreshold]
        if len(closeBeat) == 1:
            iT = closeBeat[0][0]
            result.insert(0, [iT, iL])
        elif iT >= 0:
            result.insert(0, [iT, iL])

        iT = iT - approximateTickDuration
        iL = iL - 1
        if iL == 0:
            iL = 4

    return result


def clusterValues(values: List[float], minDistance=0.5):
    """
    Returns a list of cluster points based on the minDistance
    The cluster center are at the position having the most occurences

    clusterValues([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) -> 1

    """
    raise DeprecationWarning()
    uniqueValues = {}
    for value in values:
        uniqueValues[value] = 1 if value not in uniqueValues else uniqueValues[value] + 1

    results = []
    for i in uniqueValues.keys():
        topFlag = True
        for j in uniqueValues.keys():
            if abs(i - j) < minDistance and uniqueValues[i] <= uniqueValues[j]:
                if uniqueValues[i] == uniqueValues[j]:
                    if i < j:
                        topFlag = False
                else:
                    topFlag = False
        if topFlag:
            results.append(i)

    results.sort()
    return results


def findPhase(signal: Signal, grid: Signal, period: int, toleranceWindow=0):
    """
    find the phase of the signal based on it's amplitude at the grid positions and the number of peaks
    - signal: works best with a discrete signal as no aglomeration is done
    - grid: positions of the beats
    - period: the periodicity to test
    - tolerance window: if not at 0, returns the closest value in the signal to the grid, within the tolerance window
    
    test:
    # result = findPhase(Signal(np.ones(5), times=np.array([0, 4, 8, 9, 12])+1), Signal(np.ones(16), times=range(16)), period=4)
    # print(result) = 1
    """
    phases = []
    for phase in range(period):
        values = [signal.getValue(grid.times[i], toleranceWindow=0) for i in range(phase, len(grid), period)]
        phases.append((np.sum([v for v in values if v is not None]) * len(values)))

    bestPhase = np.argmax(phases)
    return bestPhase
