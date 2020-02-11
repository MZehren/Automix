from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class OnsetDetection(Estimator):
    """
    Estimator calculating the cqt of given audio.
    """

    def __init__(self,
                 inputSamples="samples",
                 outputOnsetDetection="onsetDetection",
                 parameterHopLength=512,
                 parameterBacktrack="True",
                 cachingLevel=2,
                 forceRefreshCache=False):
        self.parameters = {"hopLength": Parameter(parameterHopLength), "backtrack": Parameter(parameterBacktrack)}
        self.inputs = [inputSamples]
        self.outputs = [outputOnsetDetection]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, samples: Signal):
        """TODO
        """
        hopLength = self.parameters["hopLength"].value
        onsets = librosa.onset.onset_detect(y=samples.values, sr=samples.sampleRate, hop_length=hopLength, backtrack=self.parameters["backtrack"].value)
        result = Signal(samples[onsets], times=[samples.getTime(onset * hopLength) for onset in onsets], sparse=True)
        return (result, )
