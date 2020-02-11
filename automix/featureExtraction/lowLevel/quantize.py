"""
container for the Periodicity estimator
"""
import copy
from typing import List

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Quantize(Estimator):
    """
    Estimator infering the periodicity of a track

    Parameters:
        - maxThreshold specify the maximum distance between the value and the closest tick in the grid.  
         by default at -1, it'll take half of the most common difference in the grid  
         at 0, all the values not exactly on the grid are out of bound

         All the values further away than the threshold from a grid tick are removed
    """

    def __init__(self,
                 inputSignal="cqtAmplitudeCheckerboardPeaks",
                 inputGrid="period",
                 outputSignal="cqtAmplitudeCheckerboardQuantized",
                 parameterMaxThreshold=-1,
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.inputs = [inputSignal, inputGrid]
        self.outputs = [outputSignal]
        self.parameters = {"maxThreshold": Parameter(parameterMaxThreshold)}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, inputSignal: Signal, inputGrid: Signal):
        output = copy.deepcopy(inputSignal)
        output.quantizeTo(inputGrid, maxThreshold=self.parameters["maxThreshold"].value)
        return (output, )
