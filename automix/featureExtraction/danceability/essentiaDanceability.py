"""
container for the ReadFile estimator
"""
import essentia.standard as estd

from featureExtraction.estimator import Estimator
from model.classes.signal import Signal


class EssentiaDanceability(Estimator):
    """
    estimator infering the danceability of a track from the samples
    The signal is split in function of the grid
    """

    def __init__(self, inputGrid="downbeats"):
        self.parameters = {}
        self.inputs = ["samples", inputGrid]
        self.outputs = ["danceability", "dfa"]
        self.cachingLevel = 0

    def predictOne(self, samples, grid):
        myDanceability = estd.Danceability()
        danceabilityDfaList = [myDanceability(samples.getValues(
            grid[i], grid[i+1])) for i in range(len(grid)-1)]

        return (Signal([danceabilityDfa[0] for danceabilityDfa in danceabilityDfaList], times=grid[:-1], duration=samples.duration),
                Signal([danceabilityDfa[1]for danceabilityDfa in danceabilityDfaList], times=grid[:-1], duration=samples.duration))
