import copy
from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator
from automix.model.classes.signal import Signal


class Normalize(Estimator):
    """
    Estimator computing the hpss of given audio.
    """

    def __init__(self, inputSamples="barMSE", outputNormalizedSamples="normalizedBarMSE", cachingLevel=2,
                 forceRefreshCache=False):
        super().__init__()
        self.inputs = [inputSamples]
        self.outputs = [outputNormalizedSamples]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal) -> Signal:
        """
        Normalize the signal's values
        """
        max = np.max(samples.values)
        min = np.min(samples.values)
        result = copy.copy(samples)

        result.values = ((samples.values - min) / (max - min)).tolist()
        return (result, )
